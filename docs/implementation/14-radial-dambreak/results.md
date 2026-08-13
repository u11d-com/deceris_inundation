# 14 — Tier 1 radially symmetric dam-break benchmark — results

Status: **passing** on macOS/MoltenVK with `gpu_resident_batch`. CUDA
backends (`cuda_graphs`/`cuda_streams`) and the GPU production reference
`fixed_dt_batch_barrier` still to be recorded on the GPU node.

## macOS / MoltenVK (Apple GPU, patched kp 0.9.0)

Default setup (200 × 200 grid on 40 × 40 m, dam center (20, 20), radius
2.5 m, inner 2.5 m / outer 0.5 m, `t_end` 1.5 s, `dt_max` 0.05 s,
`cfl_interval` 5), `--repeats 3 --warmup 1`. Reference: fine-grid
axisymmetric FV, 2000 cells; bore-front radius 8.555 m at `t_end`.

| Backend              | L1 rel | L2 rel | Front rel | Isotropy | Vol drift | Min h | Repro | Pass |
| -------------------- | ------ | ------ | --------- | -------- | --------- | ----- | ----- | ---- |
| `gpu_resident_batch` | 0.051  | 0.089  | 0.017     | 0.000    | 3.8e-8    | 0.243 | ✓     | ✓    |

Perf (median wall, single final snapshot): 390 RK2 steps in ~3.6 s
(~108 steps/s, ~4.3 M cell-steps/s) on the Apple GPU via MoltenVK — not a
performance target; the GPU CUDA path is the production timing reference.

The numerical profile tracks the axisymmetric reference across the whole
domain (see `circular-gpu_resident_batch.gif`; `--gif-2d` also emits a
top-down `circular-gpu_resident_batch-2d.gif` heatmap showing the circular
spreading directly): outward bore, inward rarefaction, and a positive
residual center depth (`min h` 0.243 m > 0 — no dry-out, no clamp). The
front-radius isotropy across 12 angular sectors quantizes to 0.000 at the
0.4 m sector-bin resolution, i.e. the circular front is isotropic to well
within the gate — the primary 2D directional-correctness signal.

## Backend note

As with effort 13, `fixed_dt_batch_barrier` (the GPU production
reference) is not exercised here because its host-driven CFL-readback loop
does not develop the wave under MoltenVK (see
[`../13-analytical-dambreak/results.md`](../13-analytical-dambreak/results.md)).
`gpu_resident_batch` (device-resident CFL + time advance) is correct on
both MoltenVK and CUDA and is the default and the macOS reference.
Validate the CUDA and barrier backends on Linux/GPU.

## Environment note

The Vulkan backends run natively on the macOS host (Docker Desktop has no
GPU/Metal passthrough). This requires the patched `kp` built from source
per [`../../reference/kp-patches.md`](../../reference/kp-patches.md).
