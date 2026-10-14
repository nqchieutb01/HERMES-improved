"""Extract a filmstrip for the teaser (QVHighlights 6SdLxtywo4s_60.0_210.0, gold [64, 94] s)."""
import os

from decord import VideoReader
from PIL import Image

VIDEO = "/nfs-stor/chieu.nguyen/qvhighlights/videos_flat/6SdLxtywo4s_60.0_210.0.mp4"
OUT = "/home/chieu.nguyen/HERMES/.claude/worktrees/paper-draft/docs/paper/latex/fig/teaser"
TIMES = [5, 25, 45, 70, 85, 110, 130, 145]

os.makedirs(OUT, exist_ok=True)
vr = VideoReader(VIDEO)
fps = vr.get_avg_fps()
for t in TIMES:
    img = Image.fromarray(vr[min(len(vr) - 1, int(t * fps))].asnumpy())
    img.thumbnail((320, 180))
    img.save(f"{OUT}/t{t:03d}.jpg", quality=85)
print(len(vr) / fps, "s;", [f"t{t:03d}.jpg" for t in TIMES])
