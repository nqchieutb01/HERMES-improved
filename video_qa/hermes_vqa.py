import bisect
import math
import json
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
                video, selected_frame_indices = self.load_uniform_video(
                    video_path,
                    num_frames=self.uniform_num_frames,
                    end_time=sample.get('end_time'),
                    video_fps=video_fps,
                    duration=video_sample.get('duration'),
                )
                question_video = torch.from_numpy(video)
                logger.debug(
                    "Uniform baseline selected %d source frames through %.3fs: %s",
                    len(selected_frame_indices),
                    float(sample.get('end_time', 0.0)),
                    selected_frame_indices,
                )
                for start in range(0, len(question_video), encode_chunk_size):
                    stop = min(start + encode_chunk_size, len(question_video))
                    print(f"Encoding uniform frames {start} to {stop-1}")
                    # Models that encode time (Qwen) need the real source times of
                    # uniformly spaced frames; sample_fps does not describe them.
                    if hasattr(self.qa_model, 'next_frame_times'):
                        self.qa_model.next_frame_times = self.last_uniform_frame_times[start:stop]
                    self.qa_model.encode_video_chunk(question_video[start:stop])
                keep_ratio = float(getattr(self, 'offline_keep_ratio', 1.0))
                if keep_ratio < 1.0:
                    # Offline token pruning: one compression pass over all encoded frames,
                    # keeping keep_ratio of the visual tokens (the budget counts visual tokens).
                    lengths = self.qa_model._get_cache_seq_len_per_layer()
                    visual = max(lengths) - self.qa_model.visual_start_idx
                    full_budget = self.qa_model.kv_size
                    self.qa_model.kv_size = max(1, int(keep_ratio * visual))
                    print(f"Offline pruning: keeping {self.qa_model.kv_size} of {visual} visual tokens")
                    self.qa_model.predict_and_compress()
                    self.qa_model.kv_size = full_budget
            else:
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
                if getattr(self, 'retention_snapshot', False) and frame_sampling == 'incremental'
                else None
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
                qa_results = self.video_open_qa(
                    question,
                    max_new_tokens=getattr(self, 'max_new_tokens', 256),
                    prompt=sample.get('prompt'),
                    preserve_newlines=is_sember,
                )
                print("Pred Answer: ", qa_results['pred_answer'])

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

            if snapshot is not None:
                times = (
                    stream_frame_times[:current_frame_idx] if stream_frame_times is not None
                    else [i / float(self.sample_fps) for i in range(current_frame_idx)]
                )
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
