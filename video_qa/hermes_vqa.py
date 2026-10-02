import bisect
import math
import json
import random
import os
import torch
from logzero import logger
from tqdm import tqdm

from video_qa.base import BaseVQA, work


class HermesVQA(BaseVQA):
    """
    Unified VQA class for both offline and streaming benchmarks.
    
    Streaming mode is auto-detected per conversation sample:
    if a sample has 'end_time', frames are encoded up to that timestamp;
    otherwise all frames are encoded before answering.
    """

    def _relevance_zoom(self, video_path, coarse_video, coarse_times, rel, q_end, keep_ratio, encode_chunk_size,
                        video_sample, sample):
        """Relevance-guided temporal importance sampling (second pass of the uniform branch).

        The coarse memory has been pruned and read once; ``rel`` holds the grounding heads' attention per coarse frame.
        Time is split into one bin per coarse frame. The allocation q over bins mixes uniform coverage with the
        heads' relevance: q = mix / N + (1 - mix) * p^temp / sum(p^temp), where p is the per-token attention of each
        frame. zoom_frames extra frames are drawn by inverse-CDF (deterministic quantile) sampling from q; the coarse
        and extra frames are re-encoded in time order and pruned to the same absolute budget, each bin receiving a
        share q of it (split over its frames; most salient tokens kept within a frame). Returns the new first frame
        id and frame times.
        """
        n = len(coarse_times)
        dens = [rel["mass"].get(i, 0.0) / max(rel["tokens"].get(i, 0), 1) for i in range(n)]
        total = sum(dens) or 1.0
        p = [d / total for d in dens]
        temp, mix = float(getattr(self, 'zoom_temp', 1.0)), float(getattr(self, 'zoom_mix', 0.5))
        pt = [x ** temp for x in p]
        ptot = sum(pt) or 1.0
        q = [mix / n + (1.0 - mix) * x / ptot for x in pt]
        edges = [0.0] + [(coarse_times[i - 1] + coarse_times[i]) / 2 for i in range(1, n)] + [max(q_end, coarse_times[-1])]
        # Inverse-CDF sampling of the extra frame times (quantiles (k + 0.5) / K).
        cdf, acc = [], 0.0
        for x in q:
            acc += x
            cdf.append(acc)
        extra_times = []
        k_total = int(getattr(self, 'zoom_frames', 32))
        for k in range(k_total):
            u = (k + 0.5) / k_total * cdf[-1]
            b = bisect.bisect_left(cdf, u)
            lo = cdf[b - 1] if b else 0.0
            frac = (u - lo) / max(q[b], 1e-12)
            extra_times.append(edges[b] + frac * (edges[b + 1] - edges[b]))
        reader = self._open_reader(video_path)
        fps = float(reader.get_avg_fps())
        idx = sorted({min(len(reader) - 1, int(round(t * fps))) for t in extra_times})
        extra = torch.from_numpy(reader.get_batch(idx).asnumpy())
        times_all = list(coarse_times) + [i / fps for i in idx]
        frames_all = torch.cat([coarse_video, extra])
        keep, last = [], None
        for i in sorted(range(len(times_all)), key=times_all.__getitem__):
            if last is None or times_all[i] - last > 1e-3:
                keep.append(i)
                last = times_all[i]
        frames_all, times_all = frames_all[keep], [times_all[i] for i in keep]
        # Second pass: re-encode and prune with per-bin budget shares.
        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        first_frame = getattr(self.qa_model, 'total_processed_frames', 0)
        for start in range(0, len(frames_all), encode_chunk_size):
            stop = min(start + encode_chunk_size, len(frames_all))
            if hasattr(self.qa_model, 'next_frame_times'):
                self.qa_model.next_frame_times = times_all[start:stop]
            self.qa_model.encode_video_chunk(frames_all[start:stop])
        bins = [min(n - 1, max(0, bisect.bisect_right(edges, t) - 1)) for t in times_all]
        per_bin = {}
        for b in bins:
            per_bin[b] = per_bin.get(b, 0) + 1
        self.qa_model.frame_budget_share = {first_frame + j: q[b] / per_bin[b] for j, b in enumerate(bins)}
        lengths = self.qa_model._get_cache_seq_len_per_layer()
        visual = int((max(lengths) - self.qa_model.visual_start_idx) * self.uniform_num_frames / len(times_all))
        score_before, budget_before = self.qa_model.prune_score, self.qa_model.kv_size
        self.qa_model.prune_score, self.qa_model.kv_size = 'zoom', max(1, int(keep_ratio * visual))
        print(f"Relevance zoom: {len(times_all)} frames ({len(idx)} sampled), keeping {self.qa_model.kv_size} tokens")
        try:
            self.qa_model.predict_and_compress()
        finally:
            self.qa_model.prune_score, self.qa_model.kv_size = score_before, budget_before
            self.qa_model.frame_budget_share = None
        with open(os.path.join(self.save_dir, f"rzoom-{self.chunk_idx}.jsonl"), "a") as out:
            out.write(json.dumps({"video_id": video_sample['video_id'], "question_id": sample.get('question_id'),
                                  "coarse_times": coarse_times, "q": [round(x, 5) for x in q],
                                  "extra_times": [round(i / fps, 2) for i in idx]}) + "\n")
        return first_frame, times_all

    def _counterfactual_memory(self, question_video, frame_times, keep_ratio, encode_chunk_size, video_sample, sample):
        """Build the counterfactual memory for contrastive decoding and return its snapshot (the real memory is kept).

        contrastive_mode "stamps": the same frames with their frame-pair timestamps permuted (seeded per question), so
        what happened is kept but when it happened is broken; "blind": no frames (language prior only). The memory is
        pruned with the same selector and budget as the real one.
        """
        qa = self.qa_model
        real, frames_before = qa.memory_snapshot(), qa.total_processed_frames
        qa.clear_cache()
        qa.encode_init_prompt()
        mode = getattr(self, 'contrastive_mode', 'stamps')
        if mode == 'stamps':
            pairs = list(range(len(frame_times) // 2))
            order = pairs[:]
            random.Random(f"cf-{video_sample['video_id']}-{sample.get('question_id')}").shuffle(order)
            idx = [2 * p + k for p in order for k in (0, 1)] + list(range(2 * len(pairs), len(frame_times)))
            times = [frame_times[i] for i in idx]
            for start in range(0, len(question_video), encode_chunk_size):
                stop = min(start + encode_chunk_size, len(question_video))
                qa.next_frame_times = times[start:stop]
                qa.encode_video_chunk(question_video[start:stop])
            if keep_ratio < 1.0:
                lengths = qa._get_cache_seq_len_per_layer()
                budget_before = qa.kv_size
                qa.kv_size = max(1, int(keep_ratio * (max(lengths) - qa.visual_start_idx)))
                qa.predict_and_compress()
                qa.kv_size = budget_before
        negative = qa.memory_snapshot()
        qa.memory_restore(real)
        qa.total_processed_frames = frames_before
        return negative

    @torch.inference_mode()
    def analyze_a_video(self, video_sample, encode_chunk_size=16):
        encode_chunk_size = getattr(self, 'encode_chunk_size', encode_chunk_size)
        frame_sampling = getattr(self, 'frame_sampling', 'incremental')
        # Source times of streamed frames when a variable-rate schedule is used; None means
        # frames are evenly spaced at sample_fps.
        stream_frame_times = None
        # GridPlan only: temporal-group start time of each streamed frame, and frames already evicted.
        stream_anchors, evicted_frames = None, set()
        video_path = video_sample['video_path']

        video_fps = video_sample.get('fps', None)
        clip = video_sample.get('clip', None)

        if frame_sampling == 'incremental':
            if video_path.endswith('.npy'):
                video = self.load_video(video_path, clip=clip)
                video_tensor = torch.from_numpy(video)
            elif os.path.isdir(video_path):
                if video_fps is None:
                    raise ValueError(f"video_fps must be provided for image-based video: {video_path}")
                video = self.load_video_frames(video_path, video_fps, clip=clip)
                video_tensor = torch.from_numpy(video)
            elif getattr(self, 'sample_schedule', None):
                if clip is not None:
                    raise ValueError("sample_schedule does not support clipped videos")
                video, stream_frame_times, stream_anchors = self.load_scheduled_video(
                    video_path, self.sample_schedule
                )
                video_tensor = torch.from_numpy(video)
            else:
                video = self.load_video(video_path, clip=clip)
                video_tensor = torch.from_numpy(video)
        else:
            video_tensor = None

        if getattr(self.qa_model, 'token_trace_enabled', False):
            self.qa_model.set_token_trace_video(video_sample['video_id'])
        if frame_sampling == 'incremental':
            self.qa_model.clear_cache()
            self.qa_model.encode_init_prompt()

        current_frame_idx = 0

        # import pdb; pdb.set_trace();

        for sample in tqdm(video_sample['conversations']):
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']

            selected_frame_indices = None
            if frame_sampling == 'uniform':
                self.qa_model.clear_cache()
                self.qa_model.encode_init_prompt()
                oracle_window = getattr(self, 'oracle_window', False)
                window_start = window_end = None
                if oracle_window:
                    # Diagnostic upper bound: all frames inside the gold interval (clipped to the question time).
                    q_end = float(sample.get('end_time', sample['answer_end_time']))
                    window_start = float(sample['answer_start_time'])
                    window_end = min(float(sample['answer_end_time']), q_end)
                    if window_end - window_start < 1.0:  # keep at least ~1 s of frames
                        window_start = max(0.0, min(window_start, q_end - 1.0))
                        window_end = min(q_end, window_start + 1.0)
                video, selected_frame_indices = self.load_uniform_video(
                    video_path,
                    num_frames=self.uniform_num_frames,
                    end_time=window_end if oracle_window else sample.get('end_time'),
                    video_fps=video_fps,
                    duration=video_sample.get('duration'),
                    start_time=window_start,
                )
                question_video = torch.from_numpy(video)
                shuffle = getattr(self, 'shuffle_mode', 'none')
                if shuffle != 'none':
                    # Diagnostic: permute Qwen3's frame pairs (seeded per question). "frames": pairs move with
                    # their timestamps; "stamps": frames stay in order and only the timestamps are permuted.
                    pairs = list(range(len(question_video) // 2))
                    order = pairs[:]
                    random.Random(f"{video_sample['video_id']}-{sample.get('question_id')}").shuffle(order)
                    idx = [2 * p + k for p in order for k in (0, 1)] + list(range(2 * len(pairs), len(question_video)))
                    times = list(self.last_uniform_frame_times)
                    if shuffle == 'frames':
                        question_video = question_video[idx]
                    self.last_uniform_frame_times = [times[i] for i in idx]
                zoom_windows = (getattr(self, 'zoom_windows', None) or {}).get(sample.get('question_id'))
                if zoom_windows:
                    # Coverage-then-zoom: add zoom_frames frames sampled densely inside the zoom window(s)
                    # (split by window length) to the coarse uniform frames, in time order.
                    clips, times_all = [question_video], list(self.last_uniform_frame_times)
                    total = sum(max(e - s, 0.0) for s, e in zoom_windows) or 1.0
                    for s, e in zoom_windows:
                        if e - s < 1.0:
                            continue
                        k = max(2, int(round(self.zoom_frames * (e - s) / total)))
                        extra, _ = self.load_uniform_video(video_path, num_frames=k, end_time=e, video_fps=video_fps,
                                                           duration=video_sample.get('duration'), start_time=s)
                        clips.append(torch.from_numpy(extra))
                        times_all += list(self.last_uniform_frame_times)
                    merged = torch.cat(clips)
                    keep, last = [], None
                    for i in sorted(range(len(times_all)), key=times_all.__getitem__):
                        if last is None or times_all[i] - last > 1e-3:
                            keep.append(i)
                            last = times_all[i]
                    question_video = merged[keep]
                    self.last_uniform_frame_times = [times_all[i] for i in keep]
                logger.debug(
                    "Uniform baseline selected %d source frames through %.3fs: %s",
                    len(selected_frame_indices),
                    float(sample.get('end_time', 0.0)),
                    selected_frame_indices,
                )
                # Frame ids keep counting across questions; ids first_frame + i <-> frame_times[i].
                first_frame = getattr(self.qa_model, 'total_processed_frames', 0)
                frame_times = list(getattr(self, 'last_uniform_frame_times', []))
                if getattr(self, 'blind', False):
                    question_video = question_video[:0]  # Diagnostic: answer without any video.
                if oracle_window:
                    # Timestamp text alone every 5 s (0.2 fps) before the evidence window.
                    self.qa_model.encode_timestamp_text([5.0 * k for k in range(int(math.ceil(window_start / 5.0)))
                                                         if 5.0 * k < window_start])
                for start in range(0, len(question_video), encode_chunk_size):
                    stop = min(start + encode_chunk_size, len(question_video))
                    print(f"Encoding uniform frames {start} to {stop-1}")
                    # Models that encode time (Qwen) need the real source times of
                    # uniformly spaced frames; sample_fps does not describe them.
                    if hasattr(self.qa_model, 'next_frame_times'):
                        self.qa_model.next_frame_times = self.last_uniform_frame_times[start:stop]
                    self.qa_model.encode_video_chunk(question_video[start:stop])
                if oracle_window:
                    # ... and after it, up to the question time.
                    k0 = int(math.floor(window_end / 5.0)) + 1
                    self.qa_model.encode_timestamp_text([5.0 * k for k in range(k0, int(q_end // 5.0) + 1)])
                if getattr(self, 'question_attention', False):
                    # Diagnostic: where the real question attends, over the unpruned cache.
                    profile = self.qa_model.question_attention_profile(question, first_frame=first_frame)
                    with open(os.path.join(self.save_dir, f"qattn-{self.chunk_idx}.jsonl"), "a") as out:
                        out.write(json.dumps({
                            "video_id": video_sample['video_id'],
                            "question_id": sample.get('question_id'),
                            "frame_times": frame_times,
                            **profile,
                        }) + "\n")
                if getattr(self.qa_model, 'prune_score', 'hermes') == 'oracle':
                    # Diagnostic: frames inside the gold interval (nearest frame if none falls inside).
                    lo, hi = float(sample['answer_start_time']), float(sample['answer_end_time'])
                    inside = {first_frame + i for i, t in enumerate(frame_times) if lo <= t <= hi}
                    if not inside:
                        mid = (lo + hi) / 2
                        inside = {first_frame + min(range(len(frame_times)), key=lambda i: abs(frame_times[i] - mid))}
                    self.qa_model.oracle_frame_ids = inside
                if getattr(self.qa_model, 'prune_score', 'hermes') == 'zoom':
                    self.qa_model.zoom_frame_ids = {
                        first_frame + i for i, t in enumerate(frame_times)
                        if any(s <= t <= e for s, e in (zoom_windows or []))}
                keep_ratio = float(getattr(self, 'offline_keep_ratio', 1.0))
                if keep_ratio < 1.0:
                    # Offline token pruning: one compression pass over all encoded frames,
                    # keeping keep_ratio of the visual tokens (the budget counts visual tokens).
                    lengths = self.qa_model._get_cache_seq_len_per_layer()
                    visual = max(lengths) - self.qa_model.visual_start_idx
                    if zoom_windows:
                        # Same absolute budget as pruning the uniform_num_frames coarse frames alone.
                        visual = int(visual * self.uniform_num_frames / max(len(frame_times), 1))
                    full_budget = self.qa_model.kv_size
                    self.qa_model.kv_size = max(1, int(keep_ratio * visual))
                    print(f"Offline pruning: keeping {self.qa_model.kv_size} of {visual} visual tokens")
                    self.qa_model.predict_and_compress()
                    self.qa_model.kv_size = full_budget
                if getattr(self, 'relevance_heads', None):
                    # Diagnostic: per-frame attention of the given heads while reading the prompt, over the memory
                    # the model actually has (after pruning).
                    rel = self.qa_model.head_relevance(self.qa_model.get_prompt(sample.get('prompt') or question),
                                                       self.relevance_heads, first_frame=first_frame)
                    with open(os.path.join(self.save_dir, f"hrel-{self.chunk_idx}.jsonl"), "a") as out:
                        out.write(json.dumps({"video_id": video_sample['video_id'],
                                              "question_id": sample.get('question_id'),
                                              "frame_times": frame_times, **rel}) + "\n")
                    if getattr(self, 'relevance_zoom', False):
                        first_frame, frame_times = self._relevance_zoom(
                            video_path, question_video, frame_times, rel, float(sample.get('end_time', frame_times[-1])),
                            keep_ratio, encode_chunk_size, video_sample, sample)
                negative_memory = None
                if getattr(self, 'contrastive_mode', 'none') != 'none' and sample.get('benchmark') == 'sember_grounding':
                    negative_memory = self._counterfactual_memory(question_video, frame_times, keep_ratio,
                                                                  encode_chunk_size, video_sample, sample)
            else:
                negative_memory = None
                if 'end_time' in sample and stream_frame_times is not None:
                    end_frame_idx = bisect.bisect_left(stream_frame_times, sample['end_time'])
                    if (stream_anchors is not None and 0 < end_frame_idx < len(stream_anchors)
                            and stream_anchors[end_frame_idx] == stream_anchors[end_frame_idx - 1]
                            and end_frame_idx - 1 >= current_frame_idx):
                        # Keep Qwen temporal pairs whole: hold back a pair's first frame until
                        # its partner (at most pair_gap seconds later) has streamed in.
                        end_frame_idx -= 1
                elif 'end_time' in sample:
                    end_frame_idx = min(
                        len(video_tensor),
                        math.ceil(sample['end_time'] * self.sample_fps),
                    )
                else:
                    end_frame_idx = len(video_tensor)

                while current_frame_idx < end_frame_idx:
                    next_encode_end = min(current_frame_idx + encode_chunk_size, end_frame_idx)
                    if next_encode_end > current_frame_idx:
                        print(f"Encoding frames {current_frame_idx} to {next_encode_end-1}")
                        video_chunk = video_tensor[current_frame_idx:next_encode_end]
                        if stream_frame_times is not None and hasattr(self.qa_model, 'next_frame_times'):
                            self.qa_model.next_frame_times = stream_frame_times[current_frame_idx:next_encode_end]
                        self.qa_model.encode_video_chunk(video_chunk)
                        current_frame_idx = next_encode_end

                        if stream_anchors is not None:
                            # Thin old frames onto the grid for the current stream time.
                            now = stream_frame_times[next_encode_end - 1]
                            evict = [
                                i for i in range(next_encode_end)
                                if i not in evicted_frames
                                and not self.sample_schedule.on_grid(stream_anchors[i], now)
                            ]
                            evicted_frames.update(evict)
                            if evict:
                                print(f"Grid thinning at {now:.0f}s (spacing {self.sample_schedule.spacing(now):g}s): "
                                      f"evicting {len(evict)} frames, {next_encode_end - len(evicted_frames)} kept")
                            self.qa_model.evict_frame_ids = evict or None

                        logger.info(f"Triggering question prediction and KV compression")
                        self.qa_model.predict_and_compress()
                        self.qa_model.evict_frame_ids = None

            # Memory at question time: tokens kept per streamed frame (for retention analysis).
            snapshot = (
                self.qa_model.retention_snapshot()
                if getattr(self, 'retention_snapshot', False) else None
            )

            if 'choices' in sample:
                choices = sample['choices']
                if answer is None:
                    answer = choices[0]
                correct_choice = self.choice_letters[choices.index(answer)]
                qa_results = self.video_close_qa(
                    question,
                    choices,
                    correct_choice,
                    prompt=sample.get('prompt'),
                )
                print("Pred Answer: ", qa_results['pred_answer'])

                record_entry = {
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'choices': choices,
                    'answer': answer,
                    'correct_choice': correct_choice,
                    'pred_answer': qa_results['pred_answer'],
                    'pred_choice': qa_results['pred_choice'],
                    'qa_acc': qa_results['acc'] * 100,
                }
                if sample.get('benchmark') == 'sember_mcq':
                    record_entry['pred_raw'] = qa_results['pred_answer']
            else:
                is_sember = sample.get('benchmark') == 'sember_grounding'
                trace_answer = getattr(self, 'answer_attention', False) and frame_sampling == 'uniform'
                if trace_answer:
                    cache_len = self.qa_model._get_cache_seq_len_per_layer()[0]
                    offsets = list(self.qa_model._get_next_global_offset_per_layer())
                if negative_memory is not None:
                    # Contrastive decoding against the counterfactual memory.
                    qa_results = {'pred_answer': self.qa_model.contrastive_answering(
                        self.qa_model.get_prompt(sample.get('prompt') or question), negative_memory,
                        alpha=float(getattr(self, 'contrastive_alpha', 1.0)),
                        beta=float(getattr(self, 'contrastive_beta', 0.1)),
                        max_new_tokens=getattr(self, 'max_new_tokens', 256),
                        repetition_penalty=getattr(self, 'repetition_penalty', 1.1))}
                else:
                    qa_results = self.video_open_qa(
                        question,
                        max_new_tokens=getattr(self, 'max_new_tokens', 256),
                        prompt=sample.get('prompt'),
                        preserve_newlines=is_sember,
                    )
                print("Pred Answer: ", qa_results['pred_answer'])
                if trace_answer:
                    # Diagnostic: gold-frame attention while reading the prompt and writing the answer.
                    lo, hi = float(sample['answer_start_time']), float(sample['answer_end_time'])
                    gold = [i for i, t in enumerate(frame_times) if lo <= t <= hi]
                    if gold:
                        prompt_text = self.qa_model.get_prompt(sample.get('prompt') or question)
                        profile = self.qa_model.answer_attention_profile(
                            prompt_text, qa_results['pred_answer'], cache_len, offsets, first_frame, gold)
                        with open(os.path.join(self.save_dir, f"aattn-{self.chunk_idx}.jsonl"), "a") as out:
                            out.write(json.dumps({"video_id": video_sample['video_id'],
                                                  "question_id": sample.get('question_id'),
                                                  "gold_frames": gold, "num_frames": len(frame_times),
                                                  **profile}) + "\n")

                record_entry = {
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'answer': answer,
                    'pred_answer': qa_results['pred_answer'],
                }
                if is_sember:
                    record_entry['pred_raw'] = qa_results['pred_answer']

            task = sample.get('task', sample.get('question_type', video_sample.get('task', None)))
            if task is not None:
                record_entry['task'] = task

            duration_category = video_sample.get('duration_category', None)
            if duration_category is not None:
                record_entry['duration_category'] = duration_category

            if sample.get('benchmark') == 'sember_grounding':
                for field in (
                    'question_id',
                    'question_time',
                    'question_category',
                    'memory_recency',
                    'answer_start_time',
                    'answer_end_time',
                    'answer_range',
                    'duration',
                    'video_category',
                    'video_category_broad',
                ):
                    if field in sample:
                        record_entry[field] = sample[field]
                record_entry['answers_json'] = json.dumps(
                    sample.get('answers', []), ensure_ascii=False
                )

            if snapshot is not None and frame_sampling == 'uniform':
                # Offline: frame ids are relative to this question's first encoded frame.
                shift = lambda d: {int(k) - first_frame: v for k, v in d.items()}
                snapshot = {key: shift(val) for key, val in snapshot.items()}
                times = frame_times
            elif snapshot is not None:
                times = (
                    stream_frame_times[:current_frame_idx] if stream_frame_times is not None
                    else [i / float(self.sample_fps) for i in range(current_frame_idx)]
                )
            if snapshot is not None:
                with open(os.path.join(self.save_dir, f"retention-{self.chunk_idx}.jsonl"), "a") as out:
                    out.write(json.dumps({
                        "video_id": video_sample['video_id'],
                        "question_id": sample.get('question_id'),
                        "question_time": sample.get('question_time', sample.get('end_time')),
                        "frame_times": times,
                        **snapshot,
                    }) + "\n")

            if selected_frame_indices is not None:
                record_entry['frame_sampling'] = 'uniform'
                record_entry['num_input_frames'] = len(selected_frame_indices)
                record_entry['source_frame_indices_json'] = json.dumps(
                    selected_frame_indices
                )

            if sample.get('benchmark') == 'sember_mcq':
                for field in (
                    'question_id',
                    'question_time',
                    'question_category',
                    'correct_index',
                    'correct_letter',
                    'correct_option_source',
                    'answer_start_time',
                    'answer_end_time',
                    'duration',
                    'video_category',
                    'video_category_broad',
                ):
                    if field in sample:
                        record_entry[field] = sample[field]
                record_entry['options_json'] = json.dumps(
                    sample.get('choices', []), ensure_ascii=False
                )
                record_entry['ground_truths_json'] = json.dumps(
                    sample.get('ground_truths', []), ensure_ascii=False
                )

            self.record.append(record_entry)


if __name__ == "__main__":
    work(HermesVQA)
