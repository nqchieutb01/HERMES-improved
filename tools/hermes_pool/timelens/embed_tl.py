"""Phase 37: SigLIP 2 index (1 fps) for the TimeLens-Bench QVHighlights subset (copy of logs/phase28/embed_charades.py), same recipe as logs/phase23/embed_siglip.py.
Writes /nfs-stor/chieu.nguyen/timelens-bench/siglip2_1fps/<video>.npz and questions_grounding.npz; runs read it with
HERMES_RETRIEVAL_INDEX pointing there. Usage (1 GPU): python logs/phase28/embed_charades.py
"""
import json
import os

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import AutoModel, AutoProcessor

D = "/nfs-stor/chieu.nguyen/timelens-bench"
OUT = f"{D}/siglip2_1fps"
MODEL = "google/siglip2-so400m-patch14-384"


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(f"{D}/qvh_tl_subset.jsonl")]
    embs = []
    for s in range(0, len(rows), 128):
        t = proc(text=[r["question"] for r in rows[s:s + 128]], padding="max_length", max_length=64, truncation=True,
                 return_tensors="pt").to(dev)
        with torch.no_grad():
            embs.append(torch.nn.functional.normalize(model.get_text_features(**t).float(), dim=-1).cpu())
    np.savez(f"{OUT}/questions_grounding.npz", question_id=np.array([r["question_id"] for r in rows]),
             emb=torch.cat(embs).half().numpy())
    for n, vid in enumerate(sorted({r["video_id"] for r in rows})):
        out = f"{OUT}/{vid}.npz"
        if os.path.exists(out):
            continue
        vr = VideoReader(f"{D}/videos_flat/{vid}.mp4", ctx=cpu(0), width=384, height=384, num_threads=4)
        fps = vr.get_avg_fps()
        times = np.arange(0.0, len(vr) / fps, 1.0)
        frames = vr.get_batch(np.minimum((times * fps).round().astype(int), len(vr) - 1)).asnumpy()
        pix = proc(images=list(frames), return_tensors="pt")["pixel_values"].to(dev, model.dtype)
        with torch.no_grad():
            e = torch.nn.functional.normalize(model.get_image_features(pixel_values=pix).float(), dim=-1).cpu()
        np.savez(out, times=times.astype(np.float32), emb=e.half().numpy())
    print("EMBED DONE", flush=True)


if __name__ == "__main__":
    main()
