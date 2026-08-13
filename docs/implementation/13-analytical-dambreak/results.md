# 13 — Tier 1 analytical dam-break benchmark — results

Status: **passing** on macOS/MoltenVK with `gpu_resident_batch`.

| Case | Backend | L1 rel | L2 rel | Front rel | Vol drift | Repro | Pass |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Stoker | `gpu_resident_batch` | 0.0077 | 0.0170 | — | 1.4e-7 | yes | yes |
| Ritter | `gpu_resident_batch` | 0.0086 | 0.0119 | 0.129 | 7.8e-8 | yes | yes |

The Ritter front lag is consistent with first-order dry-front diffusion and
remains within the 0.15 gate. Median execution was 1110 RK2 steps in roughly
1.5–1.8 s on the Apple GPU through MoltenVK.

`fixed_dt_batch_barrier` did not develop the wave under MoltenVK; its
host-driven CFL readback and redispatch path therefore remains unsuitable on
that platform. `gpu_resident_batch` is the validated local reference.
