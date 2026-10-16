| Run | Acc. | mIoU | R@0.5 | GQ@0.5 | Unparsed |
|---|---|---|---|---|---|
| random | 17.1 | 25.6 | 24.6 | 7.2 | 4 |
| random+PCD | 19.4 | 32.6 | 31.4 | 11.2 | 34 |
| equal, uniform | 22.5 | 31.7 | 32.4 | 11.2 | 4 |
| equal, retrieval | 22.1 | 32.5 | 34.1 | 10.9 | 3 |
| equal, retrieval +PCD | 21.9 | 36.8 | 38.5 | 13.5 | 39 |
| SigLIP quota, uniform | 20.4 | 31.4 | 32.0 | 10.3 | 3 |
| SigLIP quota, retrieval | 22.3 | 33.0 | 33.7 | 11.6 | 4 |
| SigLIP quota, retrieval +PCD | 21.3 | 36.5 | 38.9 | 12.6 | 36 |
| hybrid quota, retrieval | 21.3 | 32.9 | 34.9 | 11.2 | 3 |
| hybrid quota, retrieval +PCD | 21.5 | 36.3 | 38.3 | 12.4 | 40 |
| SigLIP quota 25% floor, retrieval | 20.6 | 32.6 | 34.3 | 10.7 | 3 |

| Test | Comparison | ΔmIoU | ΔR@0.5 | ΔGQ@0.5 | ΔAcc. |
|---|---|---|---|---|---|
| PRIMARY | SigLIP quota, uniform vs equal, uniform | -0.2 [-1.2, +0.8] | -0.4 [-2.1, +1.3] | -0.8 [-2.3, +0.4] | -2.1 [-4.0, -0.4] * |
| exploratory | SigLIP quota, retrieval vs equal, retrieval | +0.5 [-0.5, +1.4] | -0.4 [-2.1, +1.3] | +0.6 [-0.6, +2.1] | +0.2 [-1.7, +2.1] |
| exploratory | SigLIP quota, retrieval +PCD vs equal, retrieval +PCD | -0.3 [-1.6, +1.1] | +0.4 [-1.7, +2.5] | -0.8 [-2.5, +0.8] | -0.6 [-2.7, +1.3] |
| exploratory | hybrid quota, retrieval vs equal, retrieval | +0.4 [-0.7, +1.5] | +0.8 [-1.1, +2.9] | +0.2 [-1.3, +1.9] | -0.8 [-2.7, +1.1] |
| exploratory | hybrid quota, retrieval +PCD vs equal, retrieval +PCD | -0.5 [-1.8, +0.8] | -0.2 [-2.1, +1.7] | -1.1 [-2.7, +0.6] | -0.4 [-2.5, +1.7] |
| exploratory | SigLIP quota 25% floor, retrieval vs equal, retrieval | +0.1 [-1.2, +1.3] | +0.2 [-1.9, +2.3] | -0.2 [-1.7, +1.3] | -1.5 [-3.4, +0.2] |
