# 19 — Grid-convergence study — results

The sweep ran on macOS/MoltenVK through `gpu_resident_batch`.

| Case | Metric | Coarse-to-fine errors | Order | Monotone |
| --- | --- | --- | ---: | --- |
| Stoker | L1 | 2.19e-2, 1.35e-2, 7.69e-3, 4.38e-3, 2.52e-3 | 0.79 | yes |
| Ritter | L1 | 1.90e-2, 1.35e-2, 8.62e-3, 5.31e-3, 3.17e-3 | 0.65 | yes |
| Radial | L1 | 8.64e-2, 5.11e-2, 2.84e-2, 1.47e-2 | 0.85 | yes |
| Flood propagation | Depth L1 | 1.06e-1, 5.59e-2, 2.60e-2, 1.03e-2 | 1.12 | yes |
| Flood propagation | Speed L1 | 8.49e-2, 3.52e-2, 1.26e-2, 3.67e-3 | 1.51 | yes |
| Flood propagation | Arrival | 2.08e-1, 1.07e-1, 6.26e-2, 2.52e-2 | 0.99 | yes |

Smooth propagation converges at roughly first order; discontinuous cases show
the expected lower L1 order. The Test 4 front over-speed is therefore spatial
discretization error, not a persistent bias.

## Mass budget

| dx | Cells | Final minus injected volume |
| ---: | ---: | ---: |
| 20 m | 5,000 | +5.8 m³ |
| 10 m | 20,000 | +43.4 m³ |
| 5 m | 80,000 | +83.4 m³ |
| 2.5 m | 320,000 | +120.6 m³ |

The sign also changes with stopping time: at two hours the 5 m case is
-38.98 m³, while at five hours it is +83.4 m³. At least two opposing terms are
present. The positivity clamp is the identified positive source; the sink
remains unidentified. Float32 output reduction error is at most 2.7e-7 and is
ruled out.

Next: instrument clamp-created volume directly, infer the sink by difference,
and only then decide whether a float64 state comparison is warranted.
