# Results, phases 28-34 (Qwen3-VL-8B, 64 frames) — official parser

### S-EMBER grounded QA, 10% of visual tokens (475 questions; one judging session)

| Method | n | Acc. | GQ@0.5 | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed | ΔmIoU vs Random | ΔR@0.5 vs Random | ΔAcc. vs Random |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Unpruned (100%) | 475 | 19.8 | 8.8 | 38.7 | 26.7 | 15.8 | 28.0 | 6 | +2.5 [+0.6, +4.5] * | +2.1 [-0.8, +5.5] | +2.5 [-0.8, +6.1] |
| Random | 475 | 17.3 | 7.4 | 36.2 | 24.6 | 13.3 | 25.6 | 4 | – | – | – |
| FastVID | 475 | 15.2 | 4.8 | 23.8 | 15.8 | 9.5 | 17.9 | 0 | -7.6 [-10.2, -5.2] * | -8.8 [-12.6, -5.1] * | -2.1 [-5.7, +1.3] |
| VidCom² | 475 | 16.2 | 5.9 | 27.6 | 18.7 | 10.7 | 20.4 | 1 | -5.2 [-7.4, -2.8] * | -5.9 [-9.7, -2.3] * | -1.1 [-4.4, +2.5] |
| FlashVID | 475 | 17.3 | 6.1 | 32.8 | 21.5 | 10.9 | 22.7 | 0 | -2.9 [-4.9, -0.9] * | -3.2 [-6.3, +0.0] | +0.0 [-3.4, +3.2] |
| SemVID | 475 | 19.8 | 7.6 | 32.6 | 20.4 | 12.4 | 23.0 | 6 | -2.6 [-4.8, -0.4] * | -4.2 [-7.8, -0.8] * | +2.5 [-1.1, +6.1] |
| Cover (question top-k, uniform frames) | 475 | 22.5 | 11.2 | 44.2 | 32.4 | 19.4 | 31.7 | 4 | +6.1 [+4.1, +8.1] * | +7.8 [+4.4, +11.2] * | +5.3 [+2.1, +8.4] * |
| Retrieve + random | 475 | 21.1 | 10.9 | 42.5 | 29.5 | 15.6 | 29.4 | 3 | +3.8 [+1.7, +6.0] * | +4.8 [+1.3, +8.2] * | +3.8 [+0.2, +7.6] * |
| Retrieve + Cover | 475 | 22.3 | 11.2 | 45.9 | 34.1 | 21.1 | 32.5 | 3 | +6.9 [+4.6, +9.3] * | +9.5 [+5.7, +13.5] * | +5.1 [+1.5, +8.6] * |
| Random + PCD | 475 | 19.4 | 11.4 | 47.6 | 31.4 | 20.6 | 32.6 | 34 | +7.0 [+4.6, +9.7] * | +6.7 [+2.5, +10.9] * | +2.1 [-1.7, +5.9] |
| Random + PCD + reserve | 475 | 20.4 | 12.0 | 50.3 | 33.3 | 21.7 | 34.4 | 8 | +8.9 [+6.4, +11.4] * | +8.6 [+4.4, +12.8] * | +3.2 [-0.6, +6.9] |
| SemVID + PCD | 475 | 17.9 | 9.7 | 43.8 | 31.8 | 20.2 | 31.1 | 51 | +5.6 [+2.8, +8.3] * | +7.2 [+2.7, +11.6] * | +0.6 [-3.2, +4.6] |
| SemVID + PCD + reserve (not judged in this session) | 475 | – | – | 47.6 | 34.5 | 21.5 | 33.7 | 13 | +8.1 [+5.4, +10.9] * | +9.9 [+5.3, +14.5] * | – |
| Retrieve + Cover + PCD | 475 | 21.7 | 13.3 | 51.4 | 38.5 | 23.2 | 36.8 | 39 | +11.2 [+8.4, +14.1] * | +13.9 [+9.5, +18.3] * | +4.4 [+0.2, +8.8] * |
| Retrieve + Cover + PCD + reserve (final) | 475 | 21.7 | 13.3 | 53.9 | 40.2 | 24.0 | 38.6 | 8 | +13.0 [+10.4, +15.7] * | +15.6 [+11.2, +20.0] * | +4.4 [+0.2, +8.8] * |
| final + coverage gate | 475 | 22.1 | 13.5 | 53.7 | 40.0 | 24.4 | 38.6 | 8 | +13.0 [+10.4, +15.7] * | +15.4 [+10.9, +19.8] * | +4.8 [+0.8, +9.3] * |

### S-EMBER: effect of each fix (paired)

| Comparison | ΔmIoU | ΔR@0.5 | ΔAcc. |
|---|---|---|---|
| answer reserve: Random + PCD | +1.9 [+1.0, +2.8] * | +1.9 [+0.6, +3.2] * | +1.1 [+0.2, +2.1] * |
| answer reserve: Retrieve + Cover + PCD | +1.8 [+1.0, +2.8] * | +1.7 [+0.6, +2.9] * | +0.0 [+0.0, +0.0] |
| coverage gate: Retrieve + Cover | -0.3 [-1.0, +0.2] | -0.2 [-1.3, +0.6] | -0.8 [-1.9, +0.2] |
| coverage gate: final | +0.0 [-0.6, +0.7] | -0.2 [-1.1, +0.6] | +0.4 [-0.8, +1.7] |
| final vs SemVID + PCD | +7.5 [+4.7, +10.3] * | +8.4 [+4.2, +12.8] * | +3.8 [-0.2, +8.2] |
| final vs SemVID + PCD + reserve (same decoding) | +4.9 [+2.3, +7.7] * | +5.7 [+1.3, +9.9] * | – |
| answer reserve: SemVID + PCD | +2.5 [+1.5, +3.7] * | +2.7 [+1.5, +4.2] * | – |
| final vs Random + PCD + reserve | +4.2 [+1.6, +6.7] * | +6.9 [+2.5, +11.4] * | +1.3 [-2.7, +5.5] |

### S-EMBER token budget (phase 30 / 32); Acc. from session fresh30 (p32 rows: fresh32)

| Method | n | Acc. | GQ@0.5 | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed |
|---|---|---|---|---|---|---|---|---|
| Random 5% | 475 | 17.7 | 6.5 | 33.3 | 20.8 | 12.0 | 23.9 | 0 |
| SemVID 5% | 475 | 17.9 | 5.3 | 29.1 | 17.5 | 9.9 | 20.8 | 8 |
| Retrieve + Cover 5% | 475 | 23.2 | 12.8 | 49.1 | 35.6 | 21.9 | 34.6 | 6 |
| Random + PCD 5% | 475 | 17.3 | 10.7 | 46.3 | 32.4 | 18.9 | 32.0 | 30 |
| SemVID + PCD 5% | 475 | 14.1 | 6.7 | 39.2 | 25.7 | 15.4 | 27.2 | 73 |
| Retrieve + Cover + PCD 5% | 475 | 20.0 | 12.4 | 52.2 | 38.3 | 22.9 | 36.4 | 46 |
| Random + PCD + reserve 5% | 475 | 17.1 | 10.9 | 49.5 | 34.7 | 20.2 | 34.0 | 4 |
| SemVID + PCD + reserve 5% | 475 | – | – | 44.8 | 30.1 | 17.1 | 31.0 | 13 |
| final 5% | 475 | 19.4 | 12.2 | 56.0 | 40.6 | 23.8 | 38.8 | 7 |
| Random 10% | 475 | 17.3 | 7.4 | 36.2 | 24.6 | 13.3 | 25.6 | 4 |
| SemVID 10% | 475 | 19.8 | 7.6 | 32.6 | 20.4 | 12.4 | 23.0 | 6 |
| Retrieve + Cover 10% | 475 | 22.3 | 11.2 | 45.9 | 34.1 | 21.1 | 32.5 | 3 |
| Random + PCD 10% | 475 | 19.2 | 11.2 | 47.6 | 31.4 | 20.6 | 32.6 | 34 |
| SemVID + PCD 10% | 475 | 18.3 | 9.9 | 43.8 | 31.8 | 20.2 | 31.1 | 51 |
| Retrieve + Cover + PCD 10% | 475 | 21.9 | 13.5 | 51.4 | 38.5 | 23.2 | 36.8 | 39 |
| Random + PCD + reserve 10% | 475 | 20.0 | 11.8 | 50.3 | 33.3 | 21.7 | 34.4 | 8 |
| SemVID + PCD + reserve 10% | 475 | – | – | 47.6 | 34.5 | 21.5 | 33.7 | 13 |
| final 10% | 475 | 21.9 | 13.5 | 53.9 | 40.2 | 24.0 | 38.6 | 8 |
| Random 20% | 475 | 17.3 | 8.4 | 38.5 | 26.1 | 15.2 | 27.4 | 3 |
| SemVID 20% | 475 | 18.9 | 6.9 | 37.7 | 25.7 | 14.1 | 26.3 | 6 |
| Retrieve + Cover 20% | 475 | 21.9 | 11.4 | 44.0 | 32.6 | 19.8 | 31.0 | 3 |
| Random + PCD 20% | 475 | 19.6 | 13.3 | 47.4 | 34.9 | 21.9 | 34.3 | 36 |
| SemVID + PCD 20% | 475 | 17.5 | 10.5 | 45.7 | 34.3 | 20.4 | 32.8 | 44 |
| Retrieve + Cover + PCD 20% | 475 | 21.1 | 12.0 | 49.3 | 35.2 | 21.9 | 34.6 | 40 |
| Random + PCD + reserve 20% | 475 | 19.6 | 13.5 | 49.7 | 36.2 | 22.5 | 35.8 | 7 |
| SemVID + PCD + reserve 20% | 475 | – | – | 48.4 | 35.6 | 21.1 | 34.6 | 9 |
| final 20% | 475 | 20.8 | 12.0 | 51.8 | 36.4 | 21.9 | 36.1 | 9 |

### S-EMBER budget: method gaps at each budget (paired)

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| 5%: Retrieve + Cover vs SemVID | +13.8 [+11.4, +16.2] * | +18.1 [+14.3, +22.1] * |
| 20%: Retrieve + Cover vs SemVID | +4.7 [+2.2, +7.2] * | +6.9 [+2.9, +11.2] * |
| 5%: Retrieve + Cover vs Random | +10.7 [+8.0, +13.2] * | +14.7 [+10.7, +18.7] * |
| 20%: Retrieve + Cover vs Random | +3.7 [+1.3, +6.1] * | +6.5 [+2.7, +10.5] * |
| 5%: Retrieve + Cover + PCD vs SemVID + PCD | +9.2 [+6.2, +12.3] * | +12.6 [+8.2, +17.3] * |
| 20%: Retrieve + Cover + PCD vs SemVID + PCD | +1.8 [-0.8, +4.5] | +0.8 [-3.2, +5.1] |
| 5%: final vs Random + PCD + reserve | +4.7 [+2.2, +7.2] * | +5.9 [+1.7, +10.1] * |
| 20%: final vs Random + PCD + reserve | +0.3 [-2.3, +2.8] | +0.2 [-3.8, +4.4] |
| 5%: final vs SemVID + PCD + reserve | +7.8 [+5.2, +10.5] * | +10.5 [+5.9, +14.9] * |
| 10%: final vs SemVID + PCD + reserve | +4.9 [+2.3, +7.7] * | +5.7 [+1.3, +9.9] * |
| 20%: final vs SemVID + PCD + reserve | +1.5 [-1.0, +4.0] | +0.8 [-2.9, +5.1] |

### S-EMBER: per-layer vs shared token set (phase 32; Acc. from session fresh32)

| Method | n | Acc. | GQ@0.5 | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed |
|---|---|---|---|---|---|---|---|---|
| Random, per layer | 475 | 17.1 | 7.2 | 36.2 | 24.6 | 13.3 | 25.6 | 4 |
| Random, shared | 475 | 16.2 | 6.9 | 37.7 | 24.2 | 13.3 | 26.3 | 2 |
| Random + PCD + reserve, per layer | 475 | 19.8 | 11.8 | 50.3 | 33.3 | 21.7 | 34.4 | 8 |
| Random + PCD + reserve, shared | 475 | 17.3 | 10.7 | 51.6 | 33.9 | 22.3 | 35.3 | 6 |
| Retrieve + Cover, per layer | 475 | 22.3 | 11.2 | 45.9 | 34.1 | 21.1 | 32.5 | 3 |
| Retrieve + Cover, shared | 475 | 22.5 | 12.6 | 45.7 | 32.6 | 18.5 | 31.7 | 5 |
| final, per layer | 475 | 21.7 | 13.3 | 53.9 | 40.2 | 24.0 | 38.6 | 8 |
| final, shared | 475 | 19.6 | 12.0 | 53.1 | 38.7 | 23.2 | 37.0 | 8 |

### S-EMBER: shared minus per-layer (paired)

| Comparison | ΔmIoU | ΔR@0.5 | ΔAcc. |
|---|---|---|---|
| Random | +0.7 [-0.7, +2.2] | -0.4 [-2.9, +2.1] | -0.8 [-3.2, +1.5] |
| Random + PCD + reserve | +0.9 [-0.7, +2.5] | +0.6 [-2.1, +3.4] | -2.5 [-5.1, +0.2] |
| Retrieve + Cover | -0.8 [-2.4, +0.8] | -1.5 [-4.0, +1.1] | +0.2 [-2.3, +2.9] |
| final | -1.6 [-3.0, -0.1] * | -1.5 [-4.0, +1.1] | -2.1 [-4.6, +0.4] |

### S-EMBER: random selection seeds (phase 32)

| Method | n | Acc. | GQ@0.5 | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed |
|---|---|---|---|---|---|---|---|---|
| Random, seed 0 | 475 | 17.1 | 7.2 | 36.2 | 24.6 | 13.3 | 25.6 | 4 |
| Random, seed 1 | 475 | 17.1 | 7.4 | 37.1 | 24.8 | 13.9 | 26.7 | 4 |
| Random, seed 2 | 475 | 16.8 | 6.9 | 36.4 | 23.4 | 13.3 | 25.6 | 1 |

Random mIoU over 3 seeds: 26.0 ± 0.6 (sd)

### Charades-STA (500 test queries, ~30 s videos), 10% of visual tokens

| Method | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed | ΔmIoU vs Random | ΔR@0.5 vs Random |
|---|---|---|---|---|---|---|---|---|
| Unpruned (100%) | 500 | 24.8 | 14.2 | 6.6 | 16.6 | 2 | -0.8 [-2.7, +0.9] | +0.4 [-2.4, +3.0] |
| Random | 500 | 26.6 | 13.8 | 5.4 | 17.4 | 2 | – | – |
| FastVID | 500 | 20.6 | 11.6 | 4.4 | 13.3 | 0 | -4.1 [-6.1, -2.0] * | -2.2 [-5.4, +1.0] |
| VidCom² | 500 | 18.6 | 8.4 | 5.0 | 12.3 | 1 | -5.2 [-7.0, -3.3] * | -5.4 [-8.4, -2.4] * |
| FlashVID | 500 | 21.4 | 10.8 | 4.6 | 14.1 | 3 | -3.3 [-5.2, -1.4] * | -3.0 [-6.0, +0.0] |
| SemVID | 500 | 22.2 | 11.2 | 5.4 | 14.6 | 2 | -2.8 [-4.7, -0.7] * | -2.6 [-5.6, +0.6] |
| Cover (uniform frames) | 500 | 24.6 | 13.6 | 5.8 | 16.4 | 2 | -1.0 [-2.8, +0.7] | -0.2 [-3.0, +2.6] |
| Retrieve + Cover | 500 | 23.0 | 12.0 | 6.2 | 15.4 | 1 | -2.1 [-3.8, -0.4] * | -1.8 [-4.8, +0.8] |
| Random + PCD | 500 | 35.2 | 21.8 | 10.8 | 23.4 | 42 | +6.0 [+3.6, +8.2] * | +8.0 [+4.4, +11.6] * |
| SemVID + PCD | 500 | 23.2 | 13.8 | 5.8 | 16.0 | 68 | -1.4 [-3.8, +1.1] | +0.0 [-3.6, +3.6] |
| Retrieve + Cover + PCD | 500 | 32.0 | 18.6 | 7.8 | 20.6 | 28 | +3.2 [+0.9, +5.4] * | +4.8 [+1.2, +8.4] * |
| Random + PCD + reserve | 500 | 39.6 | 24.0 | 11.4 | 26.1 | 3 | +8.7 [+6.5, +11.0] * | +10.2 [+6.6, +14.0] * |
| SemVID + PCD + reserve | 500 | 32.2 | 19.2 | 7.0 | 21.7 | 0 | +4.2 [+2.0, +6.6] * | +5.4 [+1.8, +9.0] * |
| Retrieve + Cover + PCD + reserve | 500 | 36.0 | 21.4 | 8.4 | 23.1 | 0 | +5.6 [+3.5, +7.8] * | +7.6 [+4.0, +11.2] * |
| Retrieve(gated) + Cover | 500 | 24.6 | 13.6 | 5.8 | 16.4 | 2 | -1.0 [-2.8, +0.7] | -0.2 [-3.0, +2.6] |
| final + coverage gate | 500 | 34.8 | 21.2 | 9.2 | 23.3 | 0 | +5.9 [+3.7, +8.1] * | +7.4 [+4.0, +11.2] * |

### Charades: effect of each fix (paired)

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| answer reserve: Random + PCD | +2.7 [+1.7, +3.8] * | +2.2 [+0.8, +3.8] * |
| answer reserve: SemVID + PCD | +5.6 [+4.2, +7.2] * | +5.4 [+3.6, +7.6] * |
| answer reserve: Retrieve + Cover + PCD | +2.4 [+1.5, +3.6] * | +2.8 [+1.4, +4.4] * |
| coverage gate: Retrieve + Cover | +1.0 [-0.3, +2.3] | +1.6 [-0.4, +4.0] |
| final+gate vs Random + PCD + reserve | -2.8 [-4.6, -1.0] * | -2.8 [-6.0, +0.6] |

### QVHighlights (500 single-window val queries, ~150 s videos), 10% of visual tokens

| Method | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed | ΔmIoU vs Random | ΔR@0.5 vs Random |
|---|---|---|---|---|---|---|---|---|
| Unpruned (100%) | 500 | 35.4 | 27.2 | 17.8 | 27.3 | 35 | +1.6 [-1.0, +4.1] | +2.8 [-0.8, +6.4] |
| Random | 500 | 34.0 | 24.4 | 16.0 | 25.7 | 23 | – | – |
| FastVID | 500 | 25.4 | 15.0 | 8.6 | 18.4 | 11 | -7.3 [-10.5, -4.4] * | -9.4 [-13.8, -5.2] * |
| VidCom² | 500 | 26.4 | 17.6 | 8.4 | 18.9 | 14 | -6.8 [-9.7, -4.1] * | -6.8 [-11.0, -2.8] * |
| FlashVID | 500 | 31.2 | 20.0 | 10.2 | 22.1 | 16 | -3.6 [-6.2, -1.1] * | -4.4 [-8.0, -0.8] * |
| SemVID | 500 | 33.0 | 20.8 | 12.2 | 23.3 | 16 | -2.4 [-5.1, +0.5] | -3.6 [-7.6, +0.6] |
| Cover (uniform frames) | 500 | 42.8 | 31.8 | 20.0 | 32.0 | 24 | +6.3 [+3.8, +9.0] * | +7.4 [+3.8, +11.2] * |
| Retrieve + Cover | 500 | 45.4 | 34.2 | 19.4 | 33.2 | 15 | +7.5 [+4.9, +10.4] * | +9.8 [+5.8, +13.8] * |
| Retrieve + random | 500 | 37.6 | 26.4 | 14.4 | 27.3 | 17 | +1.6 [-0.7, +3.9] | +2.0 [-1.6, +5.6] |
| Random + PCD | 500 | 42.0 | 30.0 | 16.6 | 29.1 | 112 | +3.4 [+0.4, +6.2] * | +5.6 [+1.4, +9.8] * |
| Random + PCD + reserve | 500 | 44.4 | 31.0 | 16.8 | 30.6 | 88 | +4.9 [+1.9, +7.7] * | +6.6 [+2.4, +11.0] * |
| SemVID + PCD + reserve | 500 | 44.2 | 31.0 | 15.2 | 30.7 | 63 | +5.0 [+2.0, +8.2] * | +6.6 [+2.0, +11.2] * |
| Retrieve + Cover + PCD | 500 | 54.2 | 43.6 | 24.8 | 38.5 | 109 | +12.8 [+9.4, +16.2] * | +19.2 [+14.6, +24.4] * |
| final | 500 | 63.4 | 50.2 | 28.0 | 44.6 | 49 | +18.9 [+15.7, +22.0] * | +25.8 [+21.2, +30.8] * |
| final, 5% of tokens | 500 | 13.6 | 6.2 | 2.2 | 9.0 | 48 | -16.7 [-19.8, -14.0] * | -18.2 [-22.4, -14.4] * |
| final, 20% of tokens | 500 | 63.4 | 50.0 | 27.8 | 44.7 | 54 | +19.0 [+16.0, +22.1] * | +25.6 [+21.2, +30.4] * |

### QVHighlights: key comparisons (paired)

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| answer reserve: Random + PCD | +1.5 [+0.9, +2.2] * | +1.0 [+0.2, +2.0] * |
| answer reserve: Retrieve + Cover + PCD | +6.1 [+4.5, +7.8] * | +6.6 [+4.6, +9.0] * |
| final vs Unpruned | +17.3 [+14.0, +20.5] * | +23.0 [+18.2, +28.2] * |
| final vs Random + PCD + reserve | +14.0 [+11.3, +16.6] * | +19.2 [+14.6, +24.0] * |
| final vs SemVID + PCD + reserve | +13.9 [+10.9, +16.8] * | +19.2 [+14.6, +24.2] * |
| Retrieve + Cover vs Random | +7.5 [+4.9, +10.4] * | +9.8 [+5.8, +13.8] * |
| Retrieve + random vs Random | +1.6 [-0.7, +3.9] | +2.0 [-1.6, +5.6] |
| PCD + reserve on Random | +4.9 [+1.9, +7.7] * | +6.6 [+2.4, +11.0] * |

### ActivityNet Captions (500 val_1 queries, one per video; 224 px / 3 fps copy, so 10% = only 120-300 tokens: an extreme-budget stress test — see the 30% / 50% table), 10% of visual tokens

| Method | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed | ΔmIoU vs Random | ΔR@0.5 vs Random |
|---|---|---|---|---|---|---|---|---|
| Unpruned (100%) | 500 | 31.8 | 18.0 | 7.6 | 22.4 | 16 | +4.7 [+2.2, +7.1] * | +3.6 [-0.4, +7.4] |
| Random | 500 | 25.0 | 14.4 | 4.8 | 17.7 | 7 | – | – |
| FlashVID | 500 | 27.8 | 15.2 | 7.4 | 20.1 | 6 | +2.4 [+0.1, +4.7] * | +0.8 [-2.8, +4.4] |
| SemVID | 500 | 29.8 | 15.6 | 6.8 | 20.9 | 6 | +3.1 [+0.8, +5.4] * | +1.2 [-2.4, +4.8] |
| Cover (uniform frames) | 500 | 16.2 | 8.8 | 3.6 | 11.8 | 2 | -6.0 [-7.7, -4.2] * | -5.6 [-8.4, -2.6] * |
| Retrieve + Cover | 500 | 15.2 | 7.6 | 3.2 | 11.3 | 0 | -6.4 [-8.1, -4.5] * | -6.8 [-9.8, -3.6] * |
| Retrieve + random | 500 | 24.4 | 12.8 | 5.4 | 16.5 | 5 | -1.3 [-2.6, +0.1] | -1.6 [-4.0, +1.0] |
| Random + PCD + reserve | 500 | 28.2 | 16.6 | 7.4 | 19.3 | 82 | +1.6 [-0.2, +3.5] | +2.2 [-0.8, +5.4] |
| SemVID + PCD + reserve | 500 | 38.2 | 23.2 | 8.8 | 26.3 | 15 | +8.6 [+6.2, +11.2] * | +8.8 [+4.6, +13.2] * |
| final | 500 | 15.0 | 7.6 | 3.0 | 11.4 | 16 | -6.3 [-8.4, -4.3] * | -6.8 [-10.2, -3.4] * |

### ActivityNet Captions: key comparisons (paired)

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| final vs Random + PCD + reserve | -7.9 [-10.2, -5.8] * | -9.0 [-12.4, -5.6] * |
| final vs SemVID + PCD + reserve | -14.9 [-17.2, -12.5] * | -15.6 [-19.4, -11.6] * |
| final vs Unpruned | -11.0 [-13.5, -8.3] * | -10.4 [-14.0, -6.4] * |
| Retrieve + Cover vs Random | -6.4 [-8.1, -4.5] * | -6.8 [-9.8, -3.6] * |
| Cover vs Random | -6.0 [-7.7, -4.2] * | -5.6 [-8.4, -2.6] * |
| PCD + reserve on Random | +1.6 [-0.2, +3.5] | +2.2 [-0.8, +5.4] |

### ActivityNet Captions at 30% and 50% of the visual tokens (main ActivityNet comparison)

| Method | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed |
|---|---|---|---|---|---|---|
| Unpruned (100%) | 500 | 31.8 | 18.0 | 7.6 | 22.4 | 16 |
| Random 30% | 500 | 30.0 | 18.4 | 8.4 | 22.6 | 12 |
| SemVID 30% | 500 | 32.0 | 18.4 | 9.2 | 22.6 | 5 |
| Cover (uniform frames) 30% | 500 | 33.4 | 21.4 | 9.2 | 24.2 | 12 |
| Retrieve + Cover 30% | 500 | 35.4 | 21.0 | 9.4 | 24.9 | 10 |
| Random + PCD + reserve 30% | 500 | 40.6 | 27.6 | 14.4 | 29.5 | 40 |
| SemVID + PCD + reserve 30% | 500 | 43.0 | 26.8 | 12.4 | 29.2 | 23 |
| final 30% | 500 | 47.0 | 31.0 | 14.4 | 33.3 | 33 |
| Random 50% | 500 | 31.4 | 19.2 | 9.2 | 23.2 | 15 |
| SemVID 50% | 500 | 30.2 | 18.2 | 8.6 | 21.7 | 15 |
| Cover (uniform frames) 50% | 500 | 32.6 | 20.0 | 9.6 | 23.7 | 16 |
| Retrieve + Cover 50% | 500 | 34.6 | 20.6 | 8.4 | 24.2 | 18 |
| Random + PCD + reserve 50% | 500 | 44.0 | 29.8 | 15.2 | 30.8 | 35 |
| SemVID + PCD + reserve 50% | 500 | 42.0 | 29.2 | 12.8 | 29.5 | 39 |
| final 50% | 500 | 45.4 | 30.2 | 15.2 | 32.3 | 42 |

### ActivityNet at 30% / 50%: key comparisons (paired)

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| 30%: Retrieve + Cover vs Random | +2.3 [+0.2, +4.4] * | +2.6 [-0.6, +6.0] |
| 30%: Cover vs Random | +1.6 [-0.1, +3.3] | +3.0 [+0.2, +5.8] * |
| 30%: Retrieve + Cover vs SemVID | +2.3 [+0.1, +4.5] * | +2.6 [-0.6, +6.2] |
| 30%: final vs Random + PCD + reserve | +3.8 [+1.5, +6.1] * | +3.4 [-0.4, +7.4] |
| 30%: final vs SemVID + PCD + reserve | +4.1 [+2.0, +6.4] * | +4.2 [+0.2, +8.0] * |
| 30%: final vs Unpruned | +11.0 [+8.3, +13.8] * | +13.0 [+8.8, +17.2] * |
| 50%: Retrieve + Cover vs Random | +1.0 [-1.0, +2.9] | +1.4 [-1.8, +4.8] |
| 50%: Cover vs Random | +0.5 [-1.1, +2.0] | +0.8 [-1.8, +3.4] |
| 50%: Retrieve + Cover vs SemVID | +2.4 [+0.3, +4.4] * | +2.4 [-0.8, +5.8] |
| 50%: final vs Random + PCD + reserve | +1.4 [-0.6, +3.5] | +0.4 [-3.0, +3.8] |
| 50%: final vs SemVID + PCD + reserve | +2.7 [+0.7, +4.9] * | +1.0 [-2.8, +4.6] |
| 50%: final vs Unpruned | +9.9 [+7.4, +12.6] * | +12.2 [+8.2, +16.6] * |

### Charades-STA: 64 vs 32 frames at the same token budget (64 frames at 10% = 32 frames at 20%)

| Method | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed |
|---|---|---|---|---|---|---|
| Unpruned, 64 frames | 500 | 24.8 | 14.2 | 6.6 | 16.6 | 2 |
| Unpruned, 32 frames | 500 | 24.0 | 12.0 | 4.0 | 15.2 | 0 |
| Random, 64 | 500 | 26.6 | 13.8 | 5.4 | 17.4 | 2 |
| Random, 32 | 500 | 23.2 | 14.0 | 4.2 | 16.5 | 0 |
| SemVID, 64 | 500 | 22.2 | 11.2 | 5.4 | 14.6 | 2 |
| SemVID, 32 | 500 | 22.2 | 11.2 | 3.8 | 14.3 | 1 |
| Cover, 64 | 500 | 24.6 | 13.6 | 5.8 | 16.4 | 2 |
| Cover, 32 | 500 | 23.2 | 12.8 | 4.4 | 15.7 | 0 |
| Random + PCD + reserve, 64 | 500 | 39.6 | 24.0 | 11.4 | 26.1 | 3 |
| Random + PCD + reserve, 32 | 500 | 38.6 | 22.0 | 8.8 | 24.6 | 0 |
| SemVID + PCD + reserve, 64 | 500 | 32.2 | 19.2 | 7.0 | 21.7 | 0 |
| SemVID + PCD + reserve, 32 | 500 | 35.0 | 16.0 | 5.6 | 20.8 | 0 |
| Cover + PCD + reserve, 64 (= gated final) | 500 | 34.8 | 21.2 | 9.2 | 23.3 | 0 |
| Cover + PCD + reserve, 32 | 500 | 38.0 | 21.0 | 9.0 | 23.6 | 0 |

### Charades at 32 frames: paired comparisons

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| 32 minus 64 frames: Random + PCD + reserve | -1.6 [-3.5, +0.3] | -2.0 [-5.4, +1.4] |
| 32 minus 64 frames: Cover + PCD + reserve | +0.3 [-1.6, +2.2] | -0.2 [-3.4, +3.0] |
| 32 frames: Cover + PCD + reserve vs Random + PCD + reserve | -0.9 [-2.4, +0.5] | -1.0 [-3.2, +1.2] |
| 32 frames: Cover + PCD + reserve vs SemVID + PCD + reserve | +2.9 [+1.0, +4.8] * | +5.0 [+1.6, +8.4] * |
| 32 frames: Cover + PCD + reserve vs Unpruned | +8.4 [+6.3, +10.6] * | +9.0 [+5.6, +12.4] * |

### QVHighlights at ~260 tokens (64 frames at 5%): same token budget, 64 / 32 / 16 frames

| Method | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed |
|---|---|---|---|---|---|---|
| final, 64 frames (5%) | 500 | 13.6 | 6.2 | 2.2 | 9.0 | 48 |
| Random + PCD + reserve, 32 frames (10%) | 500 | 49.8 | 28.8 | 14.8 | 31.3 | 41 |
| SemVID + PCD + reserve, 32 frames (10%) | 500 | 54.6 | 33.0 | 15.6 | 34.7 | 12 |
| final, 32 frames (10%) | 500 | 61.4 | 45.2 | 24.8 | 41.9 | 18 |
| Random + PCD + reserve, 16 frames (20%) | 500 | 50.6 | 32.4 | 16.2 | 33.3 | 0 |
| SemVID + PCD + reserve, 16 frames (20%) | 500 | 44.8 | 28.0 | 14.0 | 30.6 | 0 |
| final, 16 frames (20%) | 500 | 62.6 | 41.0 | 18.6 | 40.4 | 0 |

### QVHighlights at ~260 tokens (64 frames at 5%): paired comparisons

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| 32 frames: final vs Random + PCD + reserve | +10.6 [+8.0, +13.3] * | +16.4 [+12.0, +21.0] * |
| 16 frames: final vs Random + PCD + reserve | +7.1 [+4.2, +9.9] * | +8.6 [+3.8, +13.6] * |
| 32 frames: final vs SemVID + PCD + reserve | +7.2 [+4.5, +10.0] * | +12.2 [+7.6, +17.0] * |
| 16 frames: final vs SemVID + PCD + reserve | +9.8 [+7.2, +12.5] * | +13.0 [+8.4, +17.6] * |
| final: 32 minus 64 frames | +32.9 [+29.9, +35.9] * | +39.0 [+34.4, +43.8] * |
| final: 16 minus 64 frames | +31.4 [+28.8, +34.0] * | +34.8 [+30.4, +39.4] * |

### ActivityNet at ~120-300 tokens (64 frames at 10%): same token budget, 64 / 32 / 16 frames

| Method | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed |
|---|---|---|---|---|---|---|
| Random + PCD + reserve, 64 frames (10%) | 500 | 28.2 | 16.6 | 7.4 | 19.3 | 82 |
| SemVID + PCD + reserve, 64 frames (10%) | 500 | 38.2 | 23.2 | 8.8 | 26.3 | 15 |
| final, 64 frames (10%) | 500 | 15.0 | 7.6 | 3.0 | 11.4 | 16 |
| Random + PCD + reserve, 32 frames (20%) | 500 | 42.2 | 26.4 | 12.2 | 29.0 | 11 |
| SemVID + PCD + reserve, 32 frames (20%) | 500 | 40.8 | 25.8 | 11.6 | 28.3 | 5 |
| final, 32 frames (20%) | 500 | 49.2 | 32.0 | 14.6 | 33.7 | 6 |
| Random + PCD + reserve, 16 frames (40%) | 500 | 46.2 | 26.8 | 13.6 | 30.5 | 1 |
| SemVID + PCD + reserve, 16 frames (40%) | 500 | 43.4 | 27.6 | 12.6 | 30.0 | 0 |
| final, 16 frames (40%) | 500 | 47.4 | 29.8 | 14.0 | 32.0 | 2 |

### ActivityNet at ~120-300 tokens (64 frames at 10%): paired comparisons

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| 32 frames: final vs Random + PCD + reserve | +4.7 [+2.3, +7.0] * | +5.6 [+1.6, +9.4] * |
| 16 frames: final vs Random + PCD + reserve | +1.5 [-0.7, +3.7] | +3.0 [-1.2, +7.0] |
| 32 frames: final vs SemVID + PCD + reserve | +5.4 [+3.3, +7.8] * | +6.2 [+2.0, +10.4] * |
| 16 frames: final vs SemVID + PCD + reserve | +2.1 [-0.1, +4.3] | +2.2 [-1.8, +6.2] |
| final: 32 minus 64 frames | +22.4 [+19.7, +25.0] * | +24.4 [+20.2, +28.6] * |
| final: 16 minus 64 frames | +20.7 [+18.1, +23.3] * | +22.2 [+18.0, +26.4] * |
