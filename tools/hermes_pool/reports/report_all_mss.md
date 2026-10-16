# Results, phases 28-34 (Qwen3-VL-8B, 64 frames) — m:ss times read as seconds

### S-EMBER grounded QA, 10% of visual tokens (475 questions; one judging session)

| Method | n | Acc. | GQ@0.5 | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed | ΔmIoU vs Random | ΔR@0.5 vs Random | ΔAcc. vs Random |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Unpruned (100%) | 475 | 19.8 | 8.8 | 38.7 | 26.7 | 15.8 | 28.0 | 6 | +2.5 [+0.6, +4.5] * | +2.1 [-0.8, +5.5] | +2.5 [-0.8, +6.1] |
| Random | 475 | 17.3 | 7.4 | 36.2 | 24.6 | 13.3 | 25.6 | 4 | – | – | – |
| FastVID | 475 | 15.2 | 4.8 | 23.8 | 15.8 | 9.5 | 17.9 | 0 | -7.6 [-10.2, -5.2] * | -8.8 [-12.6, -5.1] * | -2.1 [-5.7, +1.3] |
| VidCom² | 475 | 16.2 | 5.9 | 27.6 | 18.7 | 10.7 | 20.4 | 1 | -5.2 [-7.4, -2.8] * | -5.9 [-9.7, -2.3] * | -1.1 [-4.4, +2.5] |
| FlashVID | 475 | 17.3 | 6.1 | 32.8 | 21.5 | 10.9 | 22.7 | 0 | -2.9 [-4.9, -0.9] * | -3.2 [-6.3, +0.0] | +0.0 [-3.4, +3.2] |
| SemVID | 475 | 19.8 | 7.6 | 33.3 | 20.6 | 12.4 | 23.3 | 2 | -2.3 [-4.5, -0.1] * | -4.0 [-7.6, -0.6] * | +2.5 [-1.1, +6.1] |
| Cover (question top-k, uniform frames) | 475 | 22.5 | 11.2 | 44.2 | 32.4 | 19.4 | 31.7 | 4 | +6.1 [+4.1, +8.1] * | +7.8 [+4.4, +11.2] * | +5.3 [+2.1, +8.4] * |
| Retrieve + random | 475 | 21.1 | 10.9 | 42.5 | 29.5 | 15.6 | 29.4 | 3 | +3.8 [+1.7, +6.0] * | +4.8 [+1.3, +8.2] * | +3.8 [+0.2, +7.6] * |
| Retrieve + Cover | 475 | 22.3 | 11.2 | 45.9 | 34.1 | 21.1 | 32.5 | 3 | +6.9 [+4.6, +9.3] * | +9.5 [+5.7, +13.5] * | +5.1 [+1.5, +8.6] * |
| Random + PCD | 475 | 19.4 | 11.6 | 48.4 | 32.0 | 21.1 | 33.2 | 26 | +7.6 [+5.2, +10.2] * | +7.4 [+3.2, +11.8] * | +2.1 [-1.7, +5.9] |
| Random + PCD + reserve | 475 | 20.4 | 12.2 | 51.2 | 33.9 | 22.1 | 35.0 | 0 | +9.5 [+7.0, +12.0] * | +9.3 [+5.1, +13.5] * | +3.2 [-0.6, +6.9] |
| SemVID + PCD | 475 | 17.9 | 9.9 | 45.7 | 33.1 | 20.6 | 32.3 | 39 | +6.7 [+3.9, +9.5] * | +8.4 [+4.0, +12.8] * | +0.6 [-3.2, +4.6] |
| SemVID + PCD + reserve (not judged in this session) | 475 | – | – | 49.5 | 35.8 | 21.9 | 34.8 | 0 | +9.2 [+6.5, +12.1] * | +11.2 [+6.7, +15.6] * | – |
| Retrieve + Cover + PCD | 475 | 21.7 | 13.3 | 52.6 | 39.2 | 23.4 | 37.5 | 31 | +11.9 [+9.2, +14.9] * | +14.5 [+10.1, +18.9] * | +4.4 [+0.2, +8.8] * |
| Retrieve + Cover + PCD + reserve (final) | 475 | 21.7 | 13.3 | 55.2 | 40.8 | 24.2 | 39.3 | 0 | +13.7 [+11.1, +16.4] * | +16.2 [+11.6, +20.6] * | +4.4 [+0.2, +8.8] * |
| final + coverage gate | 475 | 22.1 | 13.5 | 54.9 | 40.6 | 24.6 | 39.3 | 0 | +13.7 [+11.0, +16.5] * | +16.0 [+11.6, +20.4] * | +4.8 [+0.8, +9.3] * |

### S-EMBER: effect of each fix (paired)

| Comparison | ΔmIoU | ΔR@0.5 | ΔAcc. |
|---|---|---|---|
| answer reserve: Random + PCD | +1.9 [+1.0, +2.8] * | +1.9 [+0.6, +3.2] * | +1.1 [+0.2, +2.1] * |
| answer reserve: Retrieve + Cover + PCD | +1.8 [+1.0, +2.8] * | +1.7 [+0.6, +2.9] * | +0.0 [+0.0, +0.0] |
| coverage gate: Retrieve + Cover | -0.3 [-1.0, +0.2] | -0.2 [-1.3, +0.6] | -0.8 [-1.9, +0.2] |
| coverage gate: final | +0.0 [-0.6, +0.7] | -0.2 [-1.1, +0.6] | +0.4 [-0.8, +1.7] |
| final vs SemVID + PCD | +7.0 [+4.2, +9.9] * | +7.8 [+3.4, +12.0] * | +3.8 [-0.2, +8.2] |
| final vs SemVID + PCD + reserve (same decoding) | +4.5 [+1.9, +7.2] * | +5.1 [+0.6, +9.3] * | – |
| answer reserve: SemVID + PCD | +2.5 [+1.5, +3.7] * | +2.7 [+1.5, +4.2] * | – |
| final vs Random + PCD + reserve | +4.2 [+1.7, +6.7] * | +6.9 [+2.5, +11.4] * | +1.3 [-2.7, +5.5] |

### S-EMBER token budget (phase 30 / 32); Acc. from session fresh30 (p32 rows: fresh32)

| Method | n | Acc. | GQ@0.5 | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed |
|---|---|---|---|---|---|---|---|---|
| Random 5% | 475 | 17.7 | 6.5 | 33.3 | 20.8 | 12.0 | 23.9 | 0 |
| SemVID 5% | 475 | 17.9 | 5.3 | 29.3 | 17.5 | 9.9 | 20.9 | 6 |
| Retrieve + Cover 5% | 475 | 23.2 | 12.8 | 49.1 | 35.6 | 21.9 | 34.6 | 6 |
| Random + PCD 5% | 475 | 17.3 | 10.7 | 47.2 | 32.8 | 19.2 | 32.5 | 26 |
| SemVID + PCD 5% | 475 | 14.1 | 7.2 | 40.4 | 26.7 | 16.0 | 28.2 | 60 |
| Retrieve + Cover + PCD 5% | 475 | 20.0 | 12.4 | 53.3 | 38.9 | 22.9 | 37.0 | 39 |
| Random + PCD + reserve 5% | 475 | 17.1 | 10.9 | 50.3 | 35.2 | 20.4 | 34.5 | 0 |
| SemVID + PCD + reserve 5% | 475 | – | – | 46.1 | 31.2 | 17.7 | 31.9 | 0 |
| final 5% | 475 | 19.4 | 12.2 | 57.1 | 41.3 | 23.8 | 39.3 | 0 |
| Random 10% | 475 | 17.3 | 7.4 | 36.2 | 24.6 | 13.3 | 25.6 | 4 |
| SemVID 10% | 475 | 19.8 | 7.6 | 33.3 | 20.6 | 12.4 | 23.3 | 2 |
| Retrieve + Cover 10% | 475 | 22.3 | 11.2 | 45.9 | 34.1 | 21.1 | 32.5 | 3 |
| Random + PCD 10% | 475 | 19.2 | 11.4 | 48.4 | 32.0 | 21.1 | 33.2 | 26 |
| SemVID + PCD 10% | 475 | 18.3 | 10.1 | 45.7 | 33.1 | 20.6 | 32.3 | 39 |
| Retrieve + Cover + PCD 10% | 475 | 21.9 | 13.5 | 52.6 | 39.2 | 23.4 | 37.5 | 31 |
| Random + PCD + reserve 10% | 475 | 20.0 | 12.0 | 51.2 | 33.9 | 22.1 | 35.0 | 0 |
| SemVID + PCD + reserve 10% | 475 | – | – | 49.5 | 35.8 | 21.9 | 34.8 | 0 |
| final 10% | 475 | 21.9 | 13.5 | 55.2 | 40.8 | 24.2 | 39.3 | 0 |
| Random 20% | 475 | 17.3 | 8.4 | 38.5 | 26.1 | 15.2 | 27.4 | 3 |
| SemVID 20% | 475 | 18.9 | 6.9 | 37.9 | 25.9 | 14.1 | 26.5 | 3 |
| Retrieve + Cover 20% | 475 | 21.9 | 11.4 | 44.0 | 32.6 | 19.8 | 31.0 | 3 |
| Random + PCD 20% | 475 | 19.6 | 13.5 | 48.2 | 35.6 | 22.3 | 34.9 | 29 |
| SemVID + PCD 20% | 475 | 17.5 | 10.5 | 46.7 | 34.9 | 20.8 | 33.5 | 35 |
| Retrieve + Cover + PCD 20% | 475 | 21.1 | 12.0 | 50.5 | 35.8 | 22.1 | 35.3 | 31 |
| Random + PCD + reserve 20% | 475 | 19.6 | 13.7 | 50.5 | 36.8 | 22.9 | 36.5 | 0 |
| SemVID + PCD + reserve 20% | 475 | – | – | 49.5 | 36.2 | 21.5 | 35.3 | 0 |
| final 20% | 475 | 20.8 | 12.0 | 53.1 | 37.1 | 22.1 | 36.8 | 0 |

### S-EMBER budget: method gaps at each budget (paired)

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| 5%: Retrieve + Cover vs SemVID | +13.7 [+11.2, +16.0] * | +18.1 [+14.3, +22.1] * |
| 20%: Retrieve + Cover vs SemVID | +4.5 [+2.0, +7.0] * | +6.7 [+2.7, +10.9] * |
| 5%: Retrieve + Cover vs Random | +10.7 [+8.0, +13.2] * | +14.7 [+10.7, +18.7] * |
| 20%: Retrieve + Cover vs Random | +3.7 [+1.3, +6.1] * | +6.5 [+2.7, +10.5] * |
| 5%: Retrieve + Cover + PCD vs SemVID + PCD | +8.8 [+5.8, +12.0] * | +12.2 [+7.6, +16.8] * |
| 20%: Retrieve + Cover + PCD vs SemVID + PCD | +1.8 [-0.9, +4.5] | +0.8 [-3.2, +5.1] |
| 5%: final vs Random + PCD + reserve | +4.8 [+2.2, +7.3] * | +6.1 [+1.9, +10.1] * |
| 20%: final vs Random + PCD + reserve | +0.3 [-2.3, +2.8] | +0.2 [-4.0, +4.2] |
| 5%: final vs SemVID + PCD + reserve | +7.4 [+4.7, +10.2] * | +10.1 [+5.5, +14.7] * |
| 10%: final vs SemVID + PCD + reserve | +4.5 [+1.9, +7.2] * | +5.1 [+0.6, +9.3] * |
| 20%: final vs SemVID + PCD + reserve | +1.5 [-1.0, +3.9] | +0.8 [-3.2, +5.1] |

### S-EMBER: per-layer vs shared token set (phase 32; Acc. from session fresh32)

| Method | n | Acc. | GQ@0.5 | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed |
|---|---|---|---|---|---|---|---|---|
| Random, per layer | 475 | 17.1 | 7.2 | 36.2 | 24.6 | 13.3 | 25.6 | 4 |
| Random, shared | 475 | 16.2 | 6.9 | 37.7 | 24.2 | 13.3 | 26.3 | 2 |
| Random + PCD + reserve, per layer | 475 | 19.8 | 12.0 | 51.2 | 33.9 | 22.1 | 35.0 | 0 |
| Random + PCD + reserve, shared | 475 | 17.3 | 10.7 | 52.2 | 34.3 | 22.5 | 35.8 | 0 |
| Retrieve + Cover, per layer | 475 | 22.3 | 11.2 | 45.9 | 34.1 | 21.1 | 32.5 | 3 |
| Retrieve + Cover, shared | 475 | 22.5 | 12.6 | 45.9 | 32.8 | 18.7 | 31.9 | 4 |
| final, per layer | 475 | 21.7 | 13.3 | 55.2 | 40.8 | 24.2 | 39.3 | 0 |
| final, shared | 475 | 19.6 | 12.0 | 54.5 | 39.6 | 23.4 | 37.8 | 0 |

### S-EMBER: shared minus per-layer (paired)

| Comparison | ΔmIoU | ΔR@0.5 | ΔAcc. |
|---|---|---|---|
| Random | +0.7 [-0.7, +2.2] | -0.4 [-2.9, +2.1] | -0.8 [-3.2, +1.5] |
| Random + PCD + reserve | +0.7 [-0.9, +2.3] | +0.4 [-2.3, +3.2] | -2.5 [-5.1, +0.2] |
| Retrieve + Cover | -0.6 [-2.3, +1.1] | -1.3 [-4.0, +1.5] | +0.2 [-2.3, +2.9] |
| final | -1.5 [-2.9, -0.0] * | -1.3 [-3.8, +1.3] | -2.1 [-4.6, +0.4] |

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
| Random + PCD | 500 | 35.2 | 21.8 | 10.8 | 23.5 | 39 | +6.1 [+3.7, +8.3] * | +8.0 [+4.4, +11.6] * |
| SemVID + PCD | 500 | 23.2 | 13.8 | 5.8 | 16.0 | 68 | -1.4 [-3.8, +1.1] | +0.0 [-3.6, +3.6] |
| Retrieve + Cover + PCD | 500 | 32.0 | 18.6 | 7.8 | 20.6 | 28 | +3.2 [+0.9, +5.4] * | +4.8 [+1.2, +8.4] * |
| Random + PCD + reserve | 500 | 39.6 | 24.0 | 11.4 | 26.2 | 0 | +8.8 [+6.6, +11.0] * | +10.2 [+6.6, +14.0] * |
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
| final+gate vs Random + PCD + reserve | -2.9 [-4.7, -1.1] * | -2.8 [-6.0, +0.6] |

### QVHighlights (500 single-window val queries, ~150 s videos), 10% of visual tokens

| Method | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed | ΔmIoU vs Random | ΔR@0.5 vs Random |
|---|---|---|---|---|---|---|---|---|
| Unpruned (100%) | 500 | 38.6 | 28.6 | 18.4 | 29.1 | 2 | +2.2 [-0.3, +4.9] | +3.6 [+0.0, +7.2] |
| Random | 500 | 36.0 | 25.0 | 16.2 | 26.9 | 0 | – | – |
| FastVID | 500 | 26.4 | 15.4 | 9.0 | 19.2 | 1 | -7.7 [-10.8, -4.8] * | -9.6 [-14.0, -5.4] * |
| VidCom² | 500 | 28.0 | 18.2 | 9.0 | 19.9 | 2 | -7.0 [-9.9, -4.5] * | -6.8 [-10.8, -2.8] * |
| FlashVID | 500 | 32.4 | 20.4 | 10.2 | 22.8 | 1 | -4.0 [-6.6, -1.5] * | -4.6 [-8.4, -1.0] * |
| SemVID | 500 | 34.8 | 21.2 | 12.4 | 24.5 | 0 | -2.4 [-5.0, +0.6] | -3.8 [-7.8, +0.4] |
| Cover (uniform frames) | 500 | 45.4 | 33.0 | 20.6 | 33.7 | 1 | +6.8 [+4.3, +9.4] * | +8.0 [+4.4, +11.6] * |
| Retrieve + Cover | 500 | 47.4 | 35.6 | 20.2 | 34.5 | 1 | +7.7 [+4.9, +10.5] * | +10.6 [+6.6, +14.8] * |
| Retrieve + random | 500 | 39.8 | 28.0 | 15.2 | 28.7 | 2 | +1.9 [-0.5, +4.2] | +3.0 [-0.8, +6.8] |
| Random + PCD | 500 | 52.8 | 38.2 | 20.4 | 36.3 | 25 | +9.5 [+6.4, +12.4] * | +13.2 [+8.8, +17.6] * |
| Random + PCD + reserve | 500 | 55.2 | 39.2 | 20.6 | 37.8 | 0 | +11.0 [+8.0, +13.9] * | +14.2 [+9.8, +18.8] * |
| SemVID + PCD + reserve | 500 | 51.2 | 34.8 | 16.4 | 34.8 | 0 | +7.9 [+4.8, +11.0] * | +9.8 [+5.0, +14.6] * |
| Retrieve + Cover + PCD | 500 | 61.0 | 48.2 | 27.0 | 42.8 | 60 | +15.9 [+12.5, +19.3] * | +23.2 [+18.0, +28.2] * |
| final | 500 | 70.2 | 54.8 | 30.2 | 48.8 | 0 | +22.0 [+18.8, +24.9] * | +29.8 [+25.0, +35.0] * |
| final, 5% of tokens | 500 | 14.6 | 6.6 | 2.2 | 9.7 | 0 | -17.2 [-20.2, -14.5] * | -18.4 [-22.6, -14.2] * |
| final, 20% of tokens | 500 | 70.6 | 56.0 | 30.2 | 49.4 | 0 | +22.6 [+19.5, +25.5] * | +31.0 [+26.4, +35.8] * |

### QVHighlights: key comparisons (paired)

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| answer reserve: Random + PCD | +1.5 [+0.9, +2.2] * | +1.0 [+0.2, +2.0] * |
| answer reserve: Retrieve + Cover + PCD | +6.1 [+4.5, +7.8] * | +6.6 [+4.6, +9.0] * |
| final vs Unpruned | +19.7 [+16.7, +22.8] * | +26.2 [+21.2, +31.4] * |
| final vs Random + PCD + reserve | +11.0 [+8.3, +13.5] * | +15.6 [+10.8, +20.4] * |
| final vs SemVID + PCD + reserve | +14.1 [+11.2, +17.1] * | +20.0 [+15.0, +25.2] * |
| Retrieve + Cover vs Random | +7.7 [+4.9, +10.5] * | +10.6 [+6.6, +14.8] * |
| Retrieve + random vs Random | +1.9 [-0.5, +4.2] | +3.0 [-0.8, +6.8] |
| PCD + reserve on Random | +11.0 [+8.0, +13.9] * | +14.2 [+9.8, +18.8] * |

### ActivityNet Captions (500 val_1 queries, one per video; 224 px / 3 fps copy, so 10% = only 120-300 tokens: an extreme-budget stress test — see the 30% / 50% table), 10% of visual tokens

| Method | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed | ΔmIoU vs Random | ΔR@0.5 vs Random |
|---|---|---|---|---|---|---|---|---|
| Unpruned (100%) | 500 | 32.6 | 18.2 | 7.6 | 22.9 | 3 | +5.1 [+2.5, +7.6] * | +3.8 [-0.2, +7.8] |
| Random | 500 | 25.2 | 14.4 | 4.8 | 17.8 | 0 | – | – |
| FlashVID | 500 | 28.4 | 15.4 | 7.4 | 20.5 | 0 | +2.6 [+0.3, +4.9] * | +1.0 [-2.4, +4.6] |
| SemVID | 500 | 30.2 | 16.0 | 7.0 | 21.2 | 1 | +3.4 [+1.1, +5.7] * | +1.6 [-2.0, +5.2] |
| Cover (uniform frames) | 500 | 16.2 | 8.8 | 3.6 | 11.8 | 0 | -6.1 [-7.8, -4.3] * | -5.6 [-8.4, -2.6] * |
| Retrieve + Cover | 500 | 15.2 | 7.6 | 3.2 | 11.3 | 0 | -6.5 [-8.2, -4.7] * | -6.8 [-9.8, -3.6] * |
| Retrieve + random | 500 | 24.4 | 12.8 | 5.4 | 16.5 | 1 | -1.4 [-2.7, -0.0] * | -1.6 [-4.0, +1.0] |
| Random + PCD + reserve | 500 | 31.0 | 18.2 | 8.0 | 21.5 | 0 | +3.7 [+1.8, +5.7] * | +3.8 [+0.6, +7.2] * |
| SemVID + PCD + reserve | 500 | 38.8 | 23.6 | 9.0 | 26.8 | 0 | +9.0 [+6.6, +11.6] * | +9.2 [+5.0, +13.8] * |
| final | 500 | 15.4 | 7.6 | 3.0 | 11.6 | 0 | -6.2 [-8.3, -4.2] * | -6.8 [-10.2, -3.4] * |

### ActivityNet Captions: key comparisons (paired)

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| final vs Random + PCD + reserve | -9.9 [-12.1, -7.9] * | -10.6 [-14.0, -7.2] * |
| final vs SemVID + PCD + reserve | -15.2 [-17.6, -12.7] * | -16.0 [-20.0, -12.0] * |
| final vs Unpruned | -11.3 [-13.8, -8.6] * | -10.6 [-14.4, -6.6] * |
| Retrieve + Cover vs Random | -6.5 [-8.2, -4.7] * | -6.8 [-9.8, -3.6] * |
| Cover vs Random | -6.1 [-7.8, -4.3] * | -5.6 [-8.4, -2.6] * |
| PCD + reserve on Random | +3.7 [+1.8, +5.7] * | +3.8 [+0.6, +7.2] * |

### ActivityNet Captions at 30% and 50% of the visual tokens (main ActivityNet comparison)

| Method | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed |
|---|---|---|---|---|---|---|
| Unpruned (100%) | 500 | 32.6 | 18.2 | 7.6 | 22.9 | 3 |
| Random 30% | 500 | 30.4 | 18.6 | 8.6 | 22.9 | 1 |
| SemVID 30% | 500 | 32.2 | 18.6 | 9.2 | 22.8 | 2 |
| Cover (uniform frames) 30% | 500 | 34.2 | 21.6 | 9.2 | 24.6 | 1 |
| Retrieve + Cover 30% | 500 | 36.4 | 21.2 | 9.4 | 25.4 | 0 |
| Random + PCD + reserve 30% | 500 | 43.6 | 29.4 | 15.0 | 31.4 | 0 |
| SemVID + PCD + reserve 30% | 500 | 44.4 | 27.8 | 12.8 | 30.2 | 0 |
| final 30% | 500 | 48.8 | 32.4 | 15.0 | 34.7 | 0 |
| Random 50% | 500 | 32.0 | 19.2 | 9.2 | 23.5 | 3 |
| SemVID 50% | 500 | 30.8 | 18.4 | 8.6 | 22.3 | 3 |
| Cover (uniform frames) 50% | 500 | 33.4 | 20.4 | 9.8 | 24.2 | 3 |
| Retrieve + Cover 50% | 500 | 36.0 | 21.0 | 8.6 | 25.0 | 0 |
| Random + PCD + reserve 50% | 500 | 46.0 | 30.8 | 15.4 | 32.2 | 0 |
| SemVID + PCD + reserve 50% | 500 | 44.6 | 31.2 | 13.2 | 31.4 | 0 |
| final 50% | 500 | 48.2 | 32.2 | 16.0 | 34.4 | 0 |

### ActivityNet at 30% / 50%: key comparisons (paired)

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| 30%: Retrieve + Cover vs Random | +2.5 [+0.4, +4.5] * | +2.6 [-0.8, +6.0] |
| 30%: Cover vs Random | +1.7 [-0.0, +3.4] | +3.0 [+0.2, +5.8] * |
| 30%: Retrieve + Cover vs SemVID | +2.6 [+0.5, +4.9] * | +2.6 [-0.6, +6.2] |
| 30%: final vs Random + PCD + reserve | +3.3 [+1.1, +5.5] * | +3.0 [-0.8, +7.0] |
| 30%: final vs SemVID + PCD + reserve | +4.5 [+2.3, +6.7] * | +4.6 [+0.6, +8.2] * |
| 30%: final vs Unpruned | +11.8 [+9.3, +14.6] * | +14.2 [+10.0, +18.8] * |
| 50%: Retrieve + Cover vs Random | +1.5 [-0.5, +3.4] | +1.8 [-1.4, +5.2] |
| 50%: Cover vs Random | +0.7 [-0.9, +2.3] | +1.2 [-1.4, +3.8] |
| 50%: Retrieve + Cover vs SemVID | +2.8 [+0.6, +4.9] * | +2.6 [-0.8, +6.0] |
| 50%: final vs Random + PCD + reserve | +2.2 [+0.1, +4.2] * | +1.4 [-2.2, +4.8] |
| 50%: final vs SemVID + PCD + reserve | +3.0 [+0.9, +5.1] * | +1.0 [-2.8, +4.8] |
| 50%: final vs Unpruned | +11.5 [+9.0, +14.3] * | +14.0 [+9.8, +18.4] * |

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
| Random + PCD + reserve, 64 | 500 | 39.6 | 24.0 | 11.4 | 26.2 | 0 |
| Random + PCD + reserve, 32 | 500 | 38.6 | 22.0 | 8.8 | 24.6 | 0 |
| SemVID + PCD + reserve, 64 | 500 | 32.2 | 19.2 | 7.0 | 21.7 | 0 |
| SemVID + PCD + reserve, 32 | 500 | 35.0 | 16.0 | 5.6 | 20.8 | 0 |
| Cover + PCD + reserve, 64 (= gated final) | 500 | 34.8 | 21.2 | 9.2 | 23.3 | 0 |
| Cover + PCD + reserve, 32 | 500 | 38.0 | 21.0 | 9.0 | 23.6 | 0 |

### Charades at 32 frames: paired comparisons

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| 32 minus 64 frames: Random + PCD + reserve | -1.7 [-3.5, +0.2] | -2.0 [-5.4, +1.4] |
| 32 minus 64 frames: Cover + PCD + reserve | +0.3 [-1.6, +2.2] | -0.2 [-3.4, +3.0] |
| 32 frames: Cover + PCD + reserve vs Random + PCD + reserve | -0.9 [-2.4, +0.5] | -1.0 [-3.2, +1.2] |
| 32 frames: Cover + PCD + reserve vs SemVID + PCD + reserve | +2.9 [+1.0, +4.8] * | +5.0 [+1.6, +8.4] * |
| 32 frames: Cover + PCD + reserve vs Unpruned | +8.4 [+6.3, +10.6] * | +9.0 [+5.6, +12.4] * |

### QVHighlights at ~260 tokens (64 frames at 5%): same token budget, 64 / 32 / 16 frames

| Method | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed |
|---|---|---|---|---|---|---|
| final, 64 frames (5%) | 500 | 14.6 | 6.6 | 2.2 | 9.7 | 0 |
| Random + PCD + reserve, 32 frames (10%) | 500 | 51.6 | 29.2 | 15.0 | 32.4 | 0 |
| SemVID + PCD + reserve, 32 frames (10%) | 500 | 55.8 | 33.6 | 16.0 | 35.5 | 0 |
| final, 32 frames (10%) | 500 | 63.0 | 46.6 | 25.2 | 43.1 | 0 |
| Random + PCD + reserve, 16 frames (20%) | 500 | 50.6 | 32.4 | 16.2 | 33.3 | 0 |
| SemVID + PCD + reserve, 16 frames (20%) | 500 | 44.8 | 28.0 | 14.0 | 30.6 | 0 |
| final, 16 frames (20%) | 500 | 62.6 | 41.0 | 18.6 | 40.4 | 0 |

### QVHighlights at ~260 tokens (64 frames at 5%): paired comparisons

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| 32 frames: final vs Random + PCD + reserve | +10.7 [+8.2, +13.4] * | +17.4 [+13.0, +22.2] * |
| 16 frames: final vs Random + PCD + reserve | +7.1 [+4.2, +9.9] * | +8.6 [+3.8, +13.6] * |
| 32 frames: final vs SemVID + PCD + reserve | +7.7 [+5.1, +10.5] * | +13.0 [+8.4, +17.8] * |
| 16 frames: final vs SemVID + PCD + reserve | +9.8 [+7.2, +12.5] * | +13.0 [+8.4, +17.6] * |
| final: 32 minus 64 frames | +33.4 [+30.6, +36.4] * | +40.0 [+35.4, +44.6] * |
| final: 16 minus 64 frames | +30.7 [+28.2, +33.3] * | +34.4 [+30.0, +39.0] * |

### ActivityNet at ~120-300 tokens (64 frames at 10%): same token budget, 64 / 32 / 16 frames

| Method | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed |
|---|---|---|---|---|---|---|
| Random + PCD + reserve, 64 frames (10%) | 500 | 31.0 | 18.2 | 8.0 | 21.5 | 0 |
| SemVID + PCD + reserve, 64 frames (10%) | 500 | 38.8 | 23.6 | 9.0 | 26.8 | 0 |
| final, 64 frames (10%) | 500 | 15.4 | 7.6 | 3.0 | 11.6 | 0 |
| Random + PCD + reserve, 32 frames (20%) | 500 | 43.0 | 26.8 | 12.2 | 29.5 | 0 |
| SemVID + PCD + reserve, 32 frames (20%) | 500 | 40.8 | 25.8 | 11.6 | 28.4 | 0 |
| final, 32 frames (20%) | 500 | 49.4 | 32.0 | 14.6 | 33.9 | 0 |
| Random + PCD + reserve, 16 frames (40%) | 500 | 46.2 | 26.8 | 13.6 | 30.5 | 0 |
| SemVID + PCD + reserve, 16 frames (40%) | 500 | 43.4 | 27.6 | 12.6 | 30.0 | 0 |
| final, 16 frames (40%) | 500 | 47.4 | 29.8 | 14.0 | 32.0 | 0 |

### ActivityNet at ~120-300 tokens (64 frames at 10%): paired comparisons

| Comparison | ΔmIoU | ΔR@0.5 |
|---|---|---|
| 32 frames: final vs Random + PCD + reserve | +4.4 [+2.0, +6.6] * | +5.2 [+1.2, +9.0] * |
| 16 frames: final vs Random + PCD + reserve | +1.5 [-0.7, +3.7] | +3.0 [-1.2, +7.0] |
| 32 frames: final vs SemVID + PCD + reserve | +5.5 [+3.4, +7.9] * | +6.2 [+2.0, +10.4] * |
| 16 frames: final vs SemVID + PCD + reserve | +2.1 [-0.1, +4.3] | +2.2 [-1.8, +6.2] |
| final: 32 minus 64 frames | +22.3 [+19.7, +24.8] * | +24.4 [+20.2, +28.6] * |
| final: 16 minus 64 frames | +20.4 [+17.9, +23.0] * | +22.2 [+18.0, +26.4] * |
