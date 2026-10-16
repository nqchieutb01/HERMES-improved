"""Dump a per-frame embedding index for S-EMBER grounding videos (M3: retrieval-based boundary refinement).

Frames are decoded at --fps and --scale. Each frame is fed alone to Qwen3-VL's vision encoder (duplicated
to fill the temporal patch, so no two moments are blended), and its visual tokens are mean-pooled into
one vector. Output per video: <out>/<video_id>.npz with `times` [T] and `emb` [T, D] (float16).

This index lives outside the KV cache, so token pruning never touches it; a 1 fps index can be
subsampled to 0.2 fps to match the streaming frame rate.
Usage (from the repo root, on a GPU node, in the Qwen env):
  python3 logs/phase10/dump_frame_embeddings.py --shard 0 --num_shards 4
"""
import argparse
import os
import sys

import numpy as np
import torch
from decord import VideoReader

sys.path.insert(0, os.getcwd())
from inference.qwen3vl_hermes import load_model  # noqa: E402
from video_qa.adapters import load_annotations  # noqa: E402

ANNO = "/nfs-stor/chieu.nguyen/s-ember/sember_grounding.jsonl"
VIDEOS = "/nfs-stor/chieu.nguyen/s-ember/videos"
MODEL = "/nfs-stor/chieu.nguyen/models/Qwen3-VL-8B-Instruct"


def decode(path, fps, scale):
    vr = VideoReader(path, num_threads=2)
    h, w = vr[0].shape[:2]
    vr = VideoReader(path, num_threads=2, width=int(w * scale) // 2 * 2, height=int(h * scale) // 2 * 2)
    src_fps, total = float(vr.get_avg_fps()), len(vr)
    times = np.arange(0.0, total / src_fps, 1.0 / fps)
    idx = np.minimum(np.round(times * src_fps).astype(int), total - 1)
    return times, vr, idx


@torch.inference_mode()
def embed(model, processor, frames):
    """frames: [B, H, W, 3] uint8 -> [B, D] mean-pooled visual tokens, one frame per temporal group."""
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).repeat_interleave(2, dim=0)
    inputs = processor(text=[""], videos=video, do_sample_frames=False, return_tensors="pt")
    pixel = inputs["pixel_values_videos"].to(model.device, torch.bfloat16)
    grid = inputs["video_grid_thw"].to(model.device)
    groups, _ = model.get_video_features(pixel, grid)
    feats = torch.cat(groups, dim=0)
    t = int(grid[0, 0])
    return feats.view(t, -1, feats.shape[-1]).float().mean(dim=1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_videos", type=int, default=300)
    ap.add_argument("--out", default="results/qwen3_vl_8b/frame_embeddings/fps1-scale0.5")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Same selection as the grounding runs (experiment sember_grounding_time_count_location).
    videos = load_annotations(ANNO, adapter="sember_grounding", video_root=VIDEOS, max_videos=args.max_videos,
                              question_categories=["time_duration", "counting_objects_events", "location_trace"])
    videos = videos[args.shard::args.num_shards]
    model, processor = load_model(model_path=MODEL, kv_size=6000)
    for i, v in enumerate(videos):
        out_path = os.path.join(args.out, f"{v['video_id']}.npz")
        if os.path.exists(out_path):
            continue
        times, vr, idx = decode(v["video_path"], args.fps, args.scale)
        embs = [embed(model, processor, vr.get_batch(idx[s:s + args.batch].tolist()).asnumpy())
                for s in range(0, len(idx), args.batch)]
        np.savez(out_path + ".tmp.npz", times=times.astype(np.float32), emb=np.concatenate(embs).astype(np.float16))
        os.replace(out_path + ".tmp.npz", out_path)
        print(f"[{i + 1}/{len(videos)}] {v['video_id']}: {len(times)} frames", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
