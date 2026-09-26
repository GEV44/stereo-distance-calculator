## Real dataset — leave-one-out cross-validation (27 points)

| Model | MAE (m) | MAPE | 95 % CI | median APE | δ<1.05 | worst |
|---|---|---|---|---|---|---|
| v1 original (reproduced) | 0.319 | **10.08 %** | 5.7–16.3 % | 5.98 % | 37 % | 77.2 % |
| Pinhole + disparity offset (2 params) | 0.175 | **5.90 %** | 4.3–7.7 % | 5.09 % | 48 % | 15.8 % |
| Pinhole + rotation (Γ frozen) | 0.143 | **5.40 %** | 3.8–7.2 % | 4.01 % | 52 % | 18.0 % |
| Γ grid, no rotation | 0.226 | **8.22 %** | 5.4–11.2 % | 5.77 % | 41 % | 27.8 % |
| Γ grid + rotation (v2, default) | 0.151 | **5.14 %** | 3.3–7.2 % | 4.03 % | 59 % | 20.8 % |
| v2 + quadratic post-correction | 0.166 | **6.89 %** | 4.4–10.1 % | 5.52 % | 44 % | 36.7 % |

Paired bootstrap, v1 − v2 MAPE: **4.89 pp** (95 % CI 0.66 – 10.44), P(v2 not better) = 0.008

## Synthetic rig — held-out MAPE (%)

Click noise σ = 0.5 px

| Model | N=15 | N=30 | N=60 | N=120 |
|---|---|---|---|---|
| Pinhole + offset | 3.95 ± 0.21 | 4.09 ± 0.27 | 3.85 ± 0.21 | 3.90 ± 0.41 |
| v1 original | 10.73 ± 1.97 | 10.97 ± 1.74 | 8.22 ± 1.01 | 7.15 ± 0.46 |
| Pinhole + rotation | 3.09 ± 0.33 | 2.57 ± 0.47 | 2.43 ± 0.33 | 2.52 ± 0.13 |
| **Γ grid + rotation (v2)** | 0.98 ± 0.20 | 0.72 ± 0.07 | 0.62 ± 0.04 | 0.50 ± 0.02 |

Click noise σ = 2.0 px

| Model | N=15 | N=30 | N=60 | N=120 |
|---|---|---|---|---|
| Pinhole + offset | 3.95 ± 0.27 | 4.11 ± 0.28 | 3.89 ± 0.17 | 3.96 ± 0.43 |
| v1 original | 10.70 ± 2.00 | 11.09 ± 1.60 | 8.34 ± 1.02 | 7.17 ± 0.44 |
| Pinhole + rotation | 3.13 ± 0.24 | 2.66 ± 0.47 | 2.52 ± 0.30 | 2.58 ± 0.14 |
| **Γ grid + rotation (v2)** | 1.28 ± 0.21 | 0.98 ± 0.06 | 0.93 ± 0.04 | 0.83 ± 0.02 |

## Automatic matching on the ray-traced scene

- clicks: 200, accepted by ZNCC + left-right check: 80 %
- precision of accepted matches (depth error < 3 %): **92.5 %**
- median / mean depth error of accepted matches: 0.60 % / 1.36 %
- time per click: 17 ms
