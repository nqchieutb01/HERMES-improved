| Method | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed | ΔmIoU vs random | ΔR@0.5 vs random |
|---|---|---|---|---|---|---|---|---|
| Unpruned (100%, bs1) | 500 | 24.8 | 14.2 | 6.6 | 16.6 | 2 | -0.8 [-2.7, +0.9] | +0.4 [-2.4, +3.0] |
| Random | 500 | 26.6 | 13.8 | 5.4 | 17.4 | 2 | – | – |
| FastVID (NeurIPS'25) | 500 | 20.6 | 11.6 | 4.4 | 13.3 | 0 | -4.1 [-6.1, -2.0] * | -2.2 [-5.4, +1.0] |
| VidCom² (EMNLP'25) | 500 | 18.6 | 8.4 | 5.0 | 12.3 | 1 | -5.2 [-7.0, -3.3] * | -5.4 [-8.4, -2.4] * |
| FlashVID (ICLR'26) | 500 | 21.4 | 10.8 | 4.6 | 14.1 | 3 | -3.3 [-5.2, -1.4] * | -3.0 [-6.0, +0.0] |
| SemVID (ECCV'26) | 500 | 22.2 | 11.2 | 5.4 | 14.6 | 2 | -2.8 [-4.7, -0.7] * | -2.6 [-5.6, +0.6] |
| Cover: question top-k, uniform frames | 500 | 24.6 | 13.6 | 5.8 | 16.4 | 2 | -1.0 [-2.8, +0.7] | -0.2 [-3.0, +2.6] |
| Retrieve + Cover | 500 | 23.0 | 12.0 | 6.2 | 15.4 | 1 | -2.1 [-3.8, -0.4] * | -1.8 [-4.8, +0.8] |
| Random + PCD | 500 | 35.2 | 21.8 | 10.8 | 23.4 | 42 | – | – |
| SemVID + PCD | 500 | 23.2 | 13.8 | 5.8 | 16.0 | 68 | -7.4 [-9.6, -5.1] * | -8.0 [-11.6, -4.4] * |
| Retrieve + Cover + PCD | 500 | 32.0 | 18.6 | 7.8 | 20.6 | 28 | -2.8 [-4.9, -0.8] * | -3.2 [-6.4, +0.0] |

Paired PCD gain (same selector, with vs without PCD), 95% bootstrap, * = excludes 0:
- random10:     mIoU +6.0 [+3.6, +8.2] *  R@0.5 +8.0 [+4.4, +11.6] *
- semvid10:     mIoU +1.4 [-0.8, +3.6]    R@0.5 +2.6 [-0.6, +5.8]
- rc64-qtopk10: mIoU +5.3 [+3.0, +7.6] *  R@0.5 +6.6 [+3.4, +10.2] *

Notes: unpruned ran at batch size 1 (bs 8 and 3 ran out of memory on 880-token-per-pair videos); all other runs bs 8.
Charades-STA 500-query test subset, 413 videos, mean 29.7 s; 64 uniform frames; 10% of visual tokens.
