| Budget | Method | Visual tokens | R1@0.3 | R1@0.5 | R1@0.7 | mIoU | ΔmIoU vs budget baseline |
|---|---|---|---|---|---|---|---|
| full | Qwen3-VL-8B, TimeLens script (anchor) | 13,650 | 75.8 | 64.7 | 50.7 | 60.5 | – |
| full | Ours, unpruned (bridge) | 13,650 | 77.0 | 64.9 | 49.5 | 60.5 | -0.0 [-1.3, +1.3] |
| ~4.2k | TimeLens script 2 fps, lower resolution | 4,200 | 74.5 | 62.0 | 47.0 | 58.0 | – |
| ~4.2k | TimeLens script 1 fps | 4,500 | 72.1 | 58.7 | 41.3 | 54.5 | -3.5 [-6.1, -1.0] * |
| ~4.2k | Random, 300 frames | 3,271 | 74.7 | 60.8 | 47.0 | 58.2 | +0.1 [-2.0, +2.4] |
| ~4.2k | SemVID, 300 frames | 4,232 | 75.6 | 62.4 | 48.3 | 58.8 | +0.8 [-1.5, +3.1] |
| ~4.2k | Cover, 300 frames | 3,271 | 78.0 | 66.7 | 50.9 | 61.7 | +3.7 [+1.3, +6.0] * |
| ~4.2k | Cover + PCD, 300 frames | 3,271 | 79.1 | 66.9 | 50.9 | 61.6 | +3.6 [+1.1, +6.0] * |
| ~4.2k | Retrieve + random, 128 frames | 3,876 | 76.4 | 66.1 | 47.4 | 59.7 | +1.7 [-0.7, +4.3] |
| ~4.2k | Retrieve + Cover, 128 frames | 3,876 | 78.4 | 68.0 | 49.5 | 61.5 | +3.5 [+1.1, +6.0] * |
| ~4.2k | Retrieve + Cover + PCD, 128 frames | 3,876 | 80.1 | 68.4 | 51.5 | 61.2 | +3.2 [+0.7, +5.8] * |
| ~1.1k | TimeLens script 1 fps, lower resolution | 1,125 | 64.7 | 53.0 | 37.4 | 49.5 | – |
| ~1.1k | Random, 300 frames (timestamps fill the budget) | ~0 | 4.5 | 1.8 | 1.2 | 4.2 | -45.3 [-48.6, -42.2] * |
| ~1.1k | Retrieve + random, 64 frames | 897 | 73.5 | 60.8 | 43.1 | 55.5 | +6.0 [+3.4, +8.4] * |
| ~1.1k | Retrieve + Cover, 64 frames | 897 | 76.6 | 63.7 | 45.4 | 58.3 | +8.8 [+6.3, +11.3] * |
| ~1.1k | Retrieve + Cover + PCD, 64 frames | 897 | 78.2 | 64.7 | 45.0 | 57.8 | +8.3 [+5.7, +11.0] * |
| 300 frames: Cover vs Random | +3.5 [+1.6, +5.4] * |
| 300 frames: Cover vs SemVID | +2.9 [+0.9, +4.9] * |
| 128 frames: Retrieve+Cover vs Retrieve+random | +1.8 [+0.6, +3.1] * |
| 64 frames: Retrieve+Cover vs Retrieve+random | +2.8 [+1.2, +4.5] * |
| PCD on Cover, 300 frames | -0.1 [-2.1, +1.9] |
| PCD on Retrieve+Cover, 128 | -0.3 [-2.0, +1.5] |
