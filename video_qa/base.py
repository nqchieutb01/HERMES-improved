import csv
import warnings
import random
import json
import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from decord import VideoReader
from transformers import (
    logging,
)

import logzero
from logzero import logger

from inference.llavaov_hermes import load_model as llavaov_hermes_load_model
from video_qa.adapters import load_annotations
from video_qa.sampling import (
    DupSchedule,
    GridPlan,
    frame_end_exclusive,
    parse_sample_schedule,
    schedule_frame_times,
    uniform_frame_indices,
)


def qwenvl_hermes_load_model(*args, **kwargs):
    try:
        from inference.qwenvl_hermes import load_model as _load_model
    except Exception as exc:
        raise ImportError(
            "Failed to import inference.qwenvl_hermes. "
            "Qwen models require a newer transformers version. "
            "Please use llava models on old transformers, or upgrade for qwen."
        ) from exc
    return _load_model(*args, **kwargs)


def qwen3vl_hermes_load_model(*args, **kwargs):
    try:
        from inference.qwen3vl_hermes import load_model as _load_model
    except Exception as exc:
        raise ImportError(
            "Failed to import inference.qwen3vl_hermes. Qwen3-VL requires "
            "the dedicated Qwen environment with transformers>=4.57.1; "
            "do not run it from the LLaVA environment."
        ) from exc
    return _load_model(*args, **kwargs)

MODELS = {
    'llava_ov_0.5b': {
        'load_func': llavaov_hermes_load_model,
        'model_path': 'models/llava-onevision-qwen2-0.5b-ov-hf',
    },
    'llava_ov_7b': {
        'load_func': llavaov_hermes_load_model,
        'model_path': 'models/llava-onevision-qwen2-7b-ov-hf',
    },
    'llava_ov_72b': {
        'load_func': llavaov_hermes_load_model,
        'model_path': 'models/llava-onevision-qwen2-72b-ov-hf',
    },
    'qwen2.5_vl_3b': {
        'load_func': qwenvl_hermes_load_model,
        'model_path': 'models/Qwen2.5-VL-3B-Instruct',
    },
    'qwen2.5_vl_7b': {
        'load_func': qwenvl_hermes_load_model,
        'model_path': 'models/Qwen2.5-VL-7B-Instruct',
    },
    'qwen2.5_vl_32b': {
        'load_func': qwenvl_hermes_load_model,
        'model_path': 'models/Qwen2.5-VL-32B-Instruct',
    },
    'qwen3_vl_8b': {
        'load_func': qwen3vl_hermes_load_model,
        'model_path': '/l/users/chieu.nguyen/models/Qwen3-VL-8B-Instruct',
    },
}

class BaseVQA:
    def __init__(self, anno, save_dir, sample_fps,
                 qa_model, qa_processor=None,
                 num_chunks=None, chunk_idx=None) -> None:
        self.sample_fps = sample_fps
        self.qa_model = qa_model
        self.qa_processor = qa_processor

        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        if num_chunks is not None:
            anno = self.get_chunk(anno, num_chunks, chunk_idx)
        self.anno = anno

        self.save_dir = save_dir
        self.choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.record = []

    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]
    
    def load_video(self, video_path, clip=None):
        """
        Load video from file.
        
        Args:
            video_path: Path to the video file (.npy or regular video)
            clip: Optional [start_time, end_time] in seconds to extract a specific segment
            
        Returns:
            For .npy files: numpy array of frames
            For regular videos: (numpy array of frames, resized_height, resized_width)
        """
        if video_path.endswith('.npy'):
            video = np.load(video_path)
            num_frames = len(video)
            frame_idx = np.linspace(0, num_frames-1, int(num_frames*self.sample_fps), dtype=int).tolist()
            video = video[frame_idx]
            return video
        else:
            vr = self._open_reader(video_path)
            fps = round(vr.get_avg_fps())
            total_frames = len(vr)
            
            if clip is not None:
                # Calculate frame range based on clip times
                start_frame = max(0, int(clip[0] * fps))
                end_frame = min(total_frames, int(clip[1] * fps) + 1)
                print(f"start_frame: {start_frame}")
                print(f"end_frame: {end_frame}")
            else:
                start_frame = 0
                end_frame = total_frames
            
            # Sample frames at target fps within the clip range
            sample_step = int(fps / self.sample_fps)
            frame_idx = [i for i in range(start_frame, end_frame, sample_step)]
            video = vr.get_batch(frame_idx).asnumpy()
            return video
    
    def _open_reader(self, video_path):
        """VideoReader that decodes at ``frame_scale`` x the source resolution (1.0 = native)."""
        scale = float(getattr(self, "frame_scale", 1.0))
        if scale == 1.0:
            return VideoReader(video_path, num_threads=1)
        h, w = VideoReader(video_path, num_threads=1)[0].shape[:2]
        return VideoReader(video_path, num_threads=1,
                           width=max(32, int(w * scale) // 2 * 2), height=max(32, int(h * scale) // 2 * 2))

    def load_scheduled_video(self, video_path, schedule):
        """Decode streamed frames for a sample schedule; returns (frames, times, anchors).

        ``schedule`` is a list of (rate, until) stages or a GridPlan. ``anchors`` gives each
        frame's temporal-group start time for GridPlan thinning (None for rate stages).
        """
        reader = self._open_reader(video_path)
        source_fps = float(reader.get_avg_fps())
        total = len(reader)
        if isinstance(schedule, DupSchedule):
            # Each frame twice in a row: Qwen's temporal patch then pairs a frame with itself
            # instead of blending two moments seconds apart.
            times = [t for t in schedule_frame_times(schedule.stages, total / source_fps) for _ in (0, 1)]
            anchors = None
        elif isinstance(schedule, GridPlan):
            times, anchors = schedule.frame_times(total / source_fps)
        else:
            times, anchors = schedule_frame_times(schedule, total / source_fps), None
        frame_idx = [min(total - 1, int(round(t * source_fps))) for t in times]
        return reader.get_batch(frame_idx).asnumpy(), times, anchors

    def load_video_frames(self, video_path, video_fps, clip=None):
        """
        Load video from a directory of image frames (for OVBench image-based videos).
        
        Args:
            video_path: Path to the directory containing image frames
            video_fps: Original FPS of the video (from annotation)
            clip: Optional [start_time, end_time] to extract a specific segment
            
        Returns:
            video: numpy array of frames
        """
        # Get sorted list of image files
        img_files = sorted(os.listdir(video_path))
        # Filter only image files
        img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_frames = len(img_files)
        
        # Calculate frame indices based on clip times
        if clip is not None:
            start_time, end_time = clip
            start_frame = max(0, int(start_time * video_fps))
            end_frame = min(num_frames - 1, int(end_time * video_fps))
            print(f"start_frame: {start_frame}")
            print(f"end_frame: {end_frame}")
        else:
            start_frame = 0
            end_frame = num_frames - 1
        
        # Generate sampled frame indices based on sample_fps
        sample_step = max(1, int(video_fps / self.sample_fps))
        frame_idx = list(range(start_frame, end_frame + 1, sample_step))
        
        # Load images
        frames = []
        for i in frame_idx:
            if i < len(img_files):
                img_path = os.path.join(video_path, img_files[i])
                img = Image.open(img_path).convert('RGB')
                frames.append(np.array(img))
        
        video = np.stack(frames, axis=0)
        return video

    def load_uniform_video(
        self,
        video_path,
        *,
        num_frames,
        end_time=None,
        video_fps=None,
        duration=None,
        start_time=None,
    ):
        """Decode uniform source frames from time zero (or ``start_time``) through a question time."""
        frames, frame_idx, times = self.uniform_frames(video_path, num_frames=num_frames, end_time=end_time,
                                                       video_fps=video_fps, duration=duration, start_time=start_time)
        self.last_uniform_frame_times = times
        return frames, frame_idx

    def uniform_frames(self, video_path, *, num_frames, end_time=None, video_fps=None, duration=None,
                       start_time=None):
        """``load_uniform_video`` without side effects (safe in worker threads); also returns the frame times."""
        if video_path.endswith('.npy'):
            source = np.load(video_path, mmap_mode='r')
            total_frames = len(source)
            source_fps = video_fps
            if source_fps is None and duration is not None and float(duration) > 0:
                source_fps = max(1, total_frames - 1) / float(duration)
            if source_fps is None:
                raise ValueError(
                    "video_fps or duration is required for time-bounded uniform "
                    f"sampling from {video_path}"
                )
            end_frame = frame_end_exclusive(
                total_frames=total_frames,
                fps=float(source_fps),
                end_time=end_time,
            )
            frame_idx = uniform_frame_indices(
                total_frames=total_frames,
                num_frames=num_frames,
                end_frame_exclusive=end_frame,
            )
            return np.asarray(source[frame_idx]), frame_idx, [i / float(source_fps) for i in frame_idx]

        if os.path.isdir(video_path):
            if video_fps is None:
                raise ValueError(
                    f"video_fps must be provided for image-based video: {video_path}"
                )
            img_files = sorted(
                filename
                for filename in os.listdir(video_path)
                if filename.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            end_frame = frame_end_exclusive(
                total_frames=len(img_files),
                fps=float(video_fps),
                end_time=end_time,
            )
            frame_idx = uniform_frame_indices(
                total_frames=len(img_files),
                num_frames=num_frames,
                end_frame_exclusive=end_frame,
            )
            frames = []
            for index in frame_idx:
                img_path = os.path.join(video_path, img_files[index])
                with Image.open(img_path) as image:
                    frames.append(np.array(image.convert('RGB')))
            return np.stack(frames, axis=0), frame_idx, [i / float(video_fps) for i in frame_idx]

        reader = self._open_reader(video_path)
        source_fps = float(reader.get_avg_fps())
        end_frame = frame_end_exclusive(
            total_frames=len(reader),
            fps=source_fps,
            end_time=end_time,
        )
        # Diagnostic: skip the first uniform_start_frac of the window (0 = start at frame zero).
        first = int(float(getattr(self, "uniform_start_frac", 0.0)) * end_frame)
        if start_time is not None:
            # Diagnostic (evidence-window oracle): sample only from start_time on.
            first = min(max(0, int(float(start_time) * source_fps)), max(0, end_frame - 1))
        frame_idx = [first + i for i in uniform_frame_indices(
            total_frames=len(reader) - first,
            num_frames=num_frames,
            end_frame_exclusive=end_frame - first,
        )]
        if not frame_idx:
            raise ValueError(
                f"No frames are available through time {end_time!r} in {video_path}"
            )
        return reader.get_batch(frame_idx).asnumpy(), frame_idx, [i / source_fps for i in frame_idx]
    
    def format_mcqa_prompt(self, question, candidates):
        assert len(question) > 0, f"Q: {question}"

        formatted_choices = "\n".join(["(" + self.choice_letters[i] + ") " + candidate for i, candidate in enumerate(candidates)])
        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."

        return {
            "question": f"{question}",
            "formatted_question": formatted_question,
            "prompt": self.qa_model.get_prompt(formatted_question, mc=True)
        }

    def extract_characters_regex(self, s):
        s = s.strip()
        if ")" in s:
            index = s.index(")")
            pred = s[index - 1 : index]
            return pred
        else:
            try:
                return s[0]
            except:
                return s

    def video_open_qa(
        self,
        question,
        max_new_tokens=1024,
        retrieved_indices=None,
        *,
        prompt=None,
        preserve_newlines=False,
    ):
        model_query = prompt if prompt is not None else question
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(model_query)
        }
        pred_answer = self.qa_model.question_answering(
            input_text, max_new_tokens=max_new_tokens,
            repetition_penalty=getattr(self, 'repetition_penalty', 1.1))
        return {
            'pred_answer': pred_answer if preserve_newlines else pred_answer.replace('\n', ''),
        }

    def video_close_qa(
        self,
        question,
        candidates,
        correct_choice,
        retrieved_indices=None,
        *,
        prompt=None,
    ):
        if prompt is None:
            input_text = self.format_mcqa_prompt(question, candidates)
        else:
            input_text = {
                "question": question,
                "formatted_question": prompt,
                "prompt": self.qa_model.get_prompt(prompt),
            }
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    def pseudo_qa(self, prediction_prompt=None):
        if prediction_prompt is None:
            prediction_prompt = "<|im_end|><|im_start|>assistant\n"
        input_text = {
            "question": prediction_prompt,
            "prompt": self.qa_model.get_prompt(prediction_prompt)
        }
        self.qa_model.pseudo_forward(input_text)

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        pass

    def analyze(self, debug=False):
        video_annos = self.anno[:1] if debug else self.anno
        for video_sample in tqdm(video_annos):
            logger.debug(f'video_id: {video_sample["video_id"]}')
            self.analyze_a_video(video_sample)
        if hasattr(self, '_flush_answers'):
            self._flush_answers()  # answers still waiting for batched decoding

        final_df = pd.DataFrame(self.record)
        final_df.to_csv(f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def work(QA_CLASS):
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument(
        "--frame_sampling",
        choices=("incremental", "uniform"),
        default="incremental",
        help="Incremental FPS stream or fixed-count uniform context per question",
    )
    parser.add_argument(
        "--offline_keep_ratio",
        type=float,
        default=1.0,
        help="Uniform sampling only: prune once after encoding, keeping this fraction of visual tokens",
    )
    parser.add_argument(
        "--prune_score",
        choices=("hermes", "random", "recent", "stratified", "oracle", "hermes_exact", "zoom"),
        default="hermes",
        help="Token score used by compression: HERMES attention+recency, random, or most recent; "
             "oracle keeps gold-interval frames first (diagnostic); hermes_exact propagates the "
             "probe questions through every layer when scoring",
    )
    parser.add_argument(
        "--retention_snapshot",
        type=str2bool,
        default=False,
        help="Write per-question token retention per frame to retention-<chunk>.jsonl",
    )
    parser.add_argument(
        "--question_attention",
        type=str2bool,
        default=False,
        help="Uniform sampling: write the real question's attention per frame (before pruning) "
             "to qattn-<chunk>.jsonl (diagnostic)",
    )
    parser.add_argument("--answer_attention", type=str2bool, default=False,
                        help="Uniform sampling: write per-head gold-frame attention while reading the prompt and "
                             "writing the answer to aattn-<chunk>.jsonl (diagnostic)")
    parser.add_argument("--relevance_heads", type=str, default=None,
                        help="Diagnostic: JSON [[layer, head], ...]; write their per-frame prompt attention after pruning "
                             "to hrel-<chunk>.jsonl")
    parser.add_argument("--contrastive_mode", choices=("none", "stamps", "blind", "noise"), default="none",
                        help="Grounding answers by contrastive decoding against a counterfactual memory: the same frames "
                             "with permuted timestamps ('stamps') or no frames ('blind')")
    parser.add_argument("--contrastive_alpha", type=float, default=1.0, help="Contrastive decoding strength")
    parser.add_argument("--contrastive_adaptive", type=str2bool, default=False,
                        help="Confidence-adaptive contrast: strength alpha * (1 - max p(. | real memory)) at each step")
    parser.add_argument("--contrastive_rule", choices=("pmi", "against"), default="pmi",
                        help="pmi: (1+a) log p_M - a log p_C; against: log p_M - a max(0, log p_C - log p_M)")
    parser.add_argument("--contrastive_trace", type=str2bool, default=False,
                        help="Write top tokens with both log-probabilities at time-value steps to cdtrace-<chunk>.jsonl")
    parser.add_argument("--contrastive_scope", choices=("all", "time", "time+answer"), default="all",
                        help="Tokens the contrast applies to: all, time values only, or time values and the answer line")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Decode the answers of this many questions together (memories are built one at a time)")
    parser.add_argument("--prefetch_videos", type=int, default=0,
                        help="Uniform sampling: decode the frames of this many upcoming questions in background threads")
    parser.add_argument("--force_batched_decoder", type=str2bool, default=False,
                        help="Diagnostic: use the batched decoder even with batch_size 1")
    parser.add_argument("--decode_log", type=str2bool, default=False,
                        help="Diagnostic (batched decoder): write answer token ids and top-1/top-2 score margins per step "
                             "to decode-<chunk>.jsonl")
    parser.add_argument("--contrastive_beta", type=float, default=0.1, help="Adaptive plausibility cut-off")
    parser.add_argument("--visual_attention_gain", type=float, default=1.0,
                        help="Answer generation: multiply the attention mass on visual-memory tokens by this factor "
                             "(additive log-gain on their logits; 1 = off)")
    parser.add_argument("--visual_attention_layers", type=str, default="all",
                        help="Layers for visual_attention_gain: 'all' or 'start-end' (inclusive)")
    parser.add_argument("--relevance_zoom", type=str2bool, default=False,
                        help="Relevance-guided importance sampling: after the coarse pruned pass, allocate extra frames and "
                             "the token budget by q = mix/N + (1-mix) p^temp from the relevance heads")
    parser.add_argument("--zoom_mix", type=float, default=0.5, help="Relevance zoom: weight of uniform coverage in q")
    parser.add_argument("--zoom_temp", type=float, default=1.0, help="Relevance zoom: temperature on the head relevance")
    parser.add_argument("--zoom_windows", type=str, default=None,
                        help="Coverage-then-zoom: JSON {question_id: [[start, end], ...]} of zoom windows (seconds)")
    parser.add_argument("--zoom_frames", type=int, default=32,
                        help="Coverage-then-zoom: extra frames sampled inside the zoom window(s)")
    parser.add_argument("--zoom_share", type=float, default=0.6,
                        help="Coverage-then-zoom: share of the token budget given to frames inside the window(s)")
    parser.add_argument("--oracle_window", type=str2bool, default=False,
                        help="Diagnostic upper bound: sample the uniform frames only inside the gold evidence "
                             "interval; outside it, show timestamp text alone every 5 s (0.2 fps)")
    parser.add_argument("--blind", type=str2bool, default=False,
                        help="Diagnostic: answer without encoding any video frame")
    parser.add_argument("--shuffle_mode", choices=("none", "frames", "stamps"), default="none",
                        help="Diagnostic: permute frame pairs with their timestamps, or only the timestamps")
    parser.add_argument("--time_offset", type=float, default=0.0,
                        help="Diagnostic: add this many seconds to every Qwen3 timestamp")
    parser.add_argument("--drop_timestamps", type=str2bool, default=False,
                        help="Diagnostic: omit Qwen3's per-group timestamp text")
    parser.add_argument("--uniform_start_frac", type=float, default=0.0,
                        help="Diagnostic: uniform frames start at this fraction of the question time")
    parser.add_argument(
        "--frame_scale",
        type=float,
        default=1.0,
        help="Decode video frames at this fraction of the source resolution (fewer visual tokens per frame)",
    )
    parser.add_argument(
        "--sample_schedule",
        type=str,
        default=None,
        help="Variable-rate incremental sampling, e.g. '1.0:60,0.5:160,0.2' "
        "(fps until seconds, last stage open-ended), or 'grid:N:smax' for dense-early sampling "
        "with progressive thinning of old frames (see GridPlan); overrides --sample_fps",
    )
    parser.add_argument(
        "--uniform_num_frames",
        type=int,
        default=None,
        help="Number of source frames per question when --frame_sampling=uniform",
    )
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--dataset_adapter", type=str, default=None)
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument(
        "--question_categories",
        nargs="+",
        default=None,
        help="Optional S-EMBER question-category IDs to retain",
    )
    parser.add_argument("--model", type=str, default="llava_ov_7b")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional checkpoint path overriding the built-in model registry",
    )
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--kv_size", type=int)
    parser.add_argument("--encode_chunk_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--recency_weight_start", type=float, default=0.75)
    parser.add_argument("--recency_weight_decay", type=float, default=0.6)
    parser.add_argument("--reindex_margin", type=int, default=1024)
    parser.add_argument("--use_history", type=str2bool, default=True)
    parser.add_argument(
        "--token_trace_path",
        type=str,
        default=None,
        help="Optional CSV path for per-frame retained visual-token counts",
    )
    parser.add_argument(
        "--verbose_token_trace",
        type=str2bool,
        default=False,
        help="Print detailed token-retention messages during inference",
    )
    parser.add_argument(
        "--min_tokens_per_frame",
        type=int,
        default=0,
        help=(
            "Minimum visual tokens retained per frame during compression; "
            "k=1 applies frame_summary_strategy when no patch survives "
            "(0 disables the floor)"
        ),
    )
    parser.add_argument(
        "--frame_summary_strategy",
        choices=(
            "mean",
            "top_patch",
            "top_attention_patch",
            "attention_weighted",
            "softmax_score_weighted",
        ),
        default="mean",
        help=(
            "Representation for a frame with no selected patch when "
            "min_tokens_per_frame=1"
        ),
    )
    parser.add_argument(
        "--frame_summary_temperature",
        type=float,
        default=0.1,
        help="Softmax temperature for softmax_score_weighted summaries",
    )
    parser.add_argument(
        "--counting_prompt",
        choices=("official", "count_first"),
        default="official",
        help="S-EMBER MCQ prompt for counting questions: letter only or count then letter",
    )
    parser.add_argument(
        "--grounding_prompt",
        choices=("official", "full_span", "timeline"),
        default="official",
        help="S-EMBER grounding prompt: official; full_span (ask for the whole event span); or timeline "
        "(list the relevant timestamped moments first, then derive the answer and interval)",
    )
    parser.add_argument(
        "--keep_time_tokens",
        type=lambda v: {"false": "none", "true": "all"}.get(str(v).lower(), str(v).lower()),
        choices=("none", "all", "surviving"),
        default="none",
        help=(
            "Qwen3 timestamp/marker tokens: prune normally (none), never prune "
            "(all), or keep only for groups that still have a visual token (surviving)"
        ),
    )
    parser.add_argument("--streaming", type=str2bool, nargs='?', const=True, default=False,
                        help="Streaming (online) mode. If False (default), uses offline mode where should_compact is always True.")
    args = parser.parse_args()
    if args.encode_chunk_size <= 0 or args.max_new_tokens <= 0 or args.repetition_penalty <= 0:
        parser.error("Chunk size, answer length and repetition penalty must be positive")
    if args.min_tokens_per_frame < 0:
        parser.error("min_tokens_per_frame must be nonnegative")
    if args.frame_summary_temperature <= 0:
        parser.error("frame_summary_temperature must be positive")
    if args.frame_sampling == "uniform" and (
        args.uniform_num_frames is None or args.uniform_num_frames <= 0
    ):
        parser.error("uniform_num_frames must be positive for uniform frame sampling")
    if not 0 <= args.recency_weight_decay <= args.recency_weight_start <= 1:
        parser.error("Require 0 <= recency_weight_decay <= recency_weight_start <= 1")
    if args.reindex_margin < 0:
        parser.error("reindex_margin must be nonnegative")

    if not args.debug:
        logzero.loglevel(logging.INFO)
        warnings.filterwarnings('ignore')

    os.makedirs(args.save_dir, exist_ok=True)

    # Seed every RNG used by the inference pipeline. Greedy decoding should be
    # deterministic, but exposing the seed makes repeated-run checks explicit
    # and also covers any stochastic preprocessing/backend behavior.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f'seed: {args.seed}')

    # VideoQA model
    if args.model not in MODELS:
        parser.error(
            f"Unknown model {args.model!r}; choose from {sorted(MODELS)}"
        )
    model_path = args.model_path or MODELS[args.model]['model_path']
    load_func = MODELS[args.model]['load_func']
    logger.info(f"Loading VideoQA model: {model_path}")
    videoqa_model, videoqa_processor = load_func(
        model_path=model_path,
        kv_size=args.kv_size,
        streaming=args.streaming,
        sample_fps=args.sample_fps,
    )
    if args.token_trace_path:
        videoqa_model.enable_token_trace(args.token_trace_path)
    videoqa_model.set_token_trace_verbose(args.verbose_token_trace)
    videoqa_model.set_frame_summary_strategy(
        args.frame_summary_strategy, args.frame_summary_temperature
    )
    videoqa_model.keep_time_tokens = args.keep_time_tokens
    videoqa_model.set_min_tokens_per_frame(args.min_tokens_per_frame)
    for name in ('recency_weight_start', 'recency_weight_decay', 'reindex_margin', 'use_history'):
        setattr(videoqa_model, name, getattr(args, name))
    logger.info(f'Effective inference settings: {vars(args)}')

    # Load ground truth file, adapting external benchmark formats when needed.
    anno = load_annotations(
        args.anno_path,
        adapter=args.dataset_adapter,
        video_root=args.video_root,
        max_videos=args.max_videos,
        question_categories=args.question_categories,
        counting_prompt=args.counting_prompt,
        grounding_prompt=args.grounding_prompt,
    )

    analyzer = QA_CLASS(
        anno=anno,
        sample_fps=args.sample_fps,
        qa_model=videoqa_model,
        qa_processor=videoqa_processor,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        save_dir=args.save_dir,
    )

    analyzer.encode_chunk_size = args.encode_chunk_size
    analyzer.max_new_tokens = args.max_new_tokens
    analyzer.repetition_penalty = args.repetition_penalty
    analyzer.frame_sampling = args.frame_sampling
    analyzer.uniform_num_frames = args.uniform_num_frames
    if args.frame_scale <= 0:
        parser.error("frame_scale must be positive")
    analyzer.frame_scale = args.frame_scale
    analyzer.retention_snapshot = args.retention_snapshot
    if not 0 < args.offline_keep_ratio <= 1:
        parser.error("offline_keep_ratio must be in (0, 1]")
    analyzer.offline_keep_ratio = args.offline_keep_ratio
    videoqa_model.prune_score = args.prune_score
    videoqa_model.exact_attention = args.prune_score == "hermes_exact"
    analyzer.zoom_windows = json.load(open(args.zoom_windows)) if args.zoom_windows else None
    analyzer.zoom_frames = args.zoom_frames
    analyzer.relevance_heads = json.load(open(args.relevance_heads)) if args.relevance_heads else None
    analyzer.relevance_zoom, analyzer.zoom_mix, analyzer.zoom_temp = args.relevance_zoom, args.zoom_mix, args.zoom_temp
    analyzer.contrastive_mode, analyzer.contrastive_alpha, analyzer.contrastive_beta = (
        args.contrastive_mode, args.contrastive_alpha, args.contrastive_beta)
    analyzer.contrastive_scope = args.contrastive_scope
    analyzer.contrastive_adaptive = args.contrastive_adaptive
    analyzer.contrastive_rule, analyzer.contrastive_trace = args.contrastive_rule, args.contrastive_trace
    if args.batch_size < 1:
        parser.error("batch_size must be positive")
    if args.batch_size > 1 and args.frame_sampling != "uniform" and args.use_history:
        parser.error("batched answering in streaming needs use_history false (later questions would see the answers)")
    analyzer.batch_size = args.batch_size
    analyzer.force_batched_decoder, analyzer.decode_log = args.force_batched_decoder, args.decode_log
    if args.prefetch_videos < 0:
        parser.error("prefetch_videos must be nonnegative")
    analyzer.prefetch_videos = args.prefetch_videos
    videoqa_model.visual_attention_gain = args.visual_attention_gain
    if args.visual_attention_layers != "all":
        lo, hi = (int(x) for x in args.visual_attention_layers.split("-"))
        videoqa_model.visual_attention_layers = list(range(lo, hi + 1))
    if args.visual_attention_gain != 1.0:
        videoqa_model.force_token_provenance = True
    if args.relevance_zoom and not (args.relevance_heads and args.offline_keep_ratio < 1):
        parser.error("relevance_zoom needs relevance_heads and offline_keep_ratio < 1")
    if args.relevance_heads:
        videoqa_model.force_token_provenance = True
    videoqa_model.zoom_share = args.zoom_share
    if args.prune_score in ("stratified", "oracle", "zoom"):
        videoqa_model.force_token_provenance = True
    if args.retention_snapshot or args.question_attention:
        videoqa_model.force_token_provenance = True
    analyzer.question_attention = args.question_attention
    analyzer.answer_attention = args.answer_attention
    analyzer.blind = args.blind
    analyzer.oracle_window = args.oracle_window
    analyzer.shuffle_mode = args.shuffle_mode
    if args.answer_attention:
        videoqa_model.force_token_provenance = True
    videoqa_model.time_offset = args.time_offset
    videoqa_model.drop_timestamps = args.drop_timestamps
    if not 0 <= args.uniform_start_frac < 1:
        parser.error("uniform_start_frac must be in [0, 1)")
    analyzer.uniform_start_frac = args.uniform_start_frac
    if not args.sample_schedule:
        analyzer.sample_schedule = None
    elif args.sample_schedule.startswith("dup:"):
        analyzer.sample_schedule = DupSchedule(parse_sample_schedule(args.sample_schedule[len("dup:"):]))
    elif args.sample_schedule.startswith("grid:"):
        analyzer.sample_schedule = GridPlan.parse(args.sample_schedule)
        videoqa_model.force_token_provenance = True
    else:
        analyzer.sample_schedule = parse_sample_schedule(args.sample_schedule)
    analyzer.analyze(debug=args.debug)
    videoqa_model.write_token_trace()
