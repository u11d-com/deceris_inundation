# 14 — Tier 1 radial dam-break benchmark — results

Status: **passing** on macOS/MoltenVK with `gpu_resident_batch`.

| Backend | L1 rel | L2 rel | Front rel | Isotropy | Vol drift | Min h | Repro | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `gpu_resident_batch` | 0.051 | 0.089 | 0.017 | 0.000 | 3.8e-8 | 0.243 | yes | yes |

The profile tracks the axisymmetric reference through the bore and rarefaction.
Angular-sector fronts show no spread at the 0.4 m bin resolution. Median time
was about 3.6 s for 390 RK2 steps on the Apple GPU through MoltenVK.

As in effort 13, `fixed_dt_batch_barrier` is not the validated MoltenVK path;
`gpu_resident_batch` remains the local reference.
