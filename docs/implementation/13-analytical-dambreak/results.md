# 13 — Tier 1 analytical dam-break benchmark — results

Status: **passing** on macOS/MoltenVK with `gpu_resident_batch`. CUDA
backends (`cuda_graphs`/`cuda_streams`) and the GPU production reference
`fixed_dt_batch_barrier` still to be recorded on the GPU node.

## macOS / MoltenVK (Apple GPU, patched kp 0.9.0)

Default setup (500 × 5 grid, dam at 500 m, upstream 10 m, `t_end` 20 s,
`dt_max` 0.02 s, `cfl_interval` 5), `--repeats 3 --warmup 1`.

| Case               | Backend              | L1 rel | L2 rel | Front rel | Vol drift | Repro | Pass |
| ------------------ | -------------------- | ------ | ------ | --------- | --------- | ----- | ---- |
| Stoker (wet bed)   | `gpu_resident_batch` | 0.0077 | 0.0170 | —         | 1.4e-7    | ✓     | ✓    |
| Ritter (dry front) | `gpu_resident_batch` | 0.0086 | 0.0119 | 0.129     | 7.8e-8    | ✓     | ✓    |

Perf (median wall, single final snapshot): ~0.6 M cell-steps/s at 1110
RK2 steps in ~1.5–1.8 s (Apple GPU via MoltenVK; not a performance
target — the GPU CUDA path is the production timing reference).

The Ritter profile tracks the analytical rarefaction across the whole
domain (see `ritter-gpu_resident_batch.gif`); the ~13 % front-position
lag is the expected first-order dry-front diffusion, inside the 15 %
gate.

## Backend note: `fixed_dt_batch_barrier` under MoltenVK

`fixed_dt_batch_barrier` (the GPU production reference, Option #4b) does
**not** produce correct results under MoltenVK: the dam-break rarefaction
never develops — the state stays pinned near the initial step (L1 ≈ 0.23,
front never leaves the dam). Its individual flux/update kernels and
`OpComputeBarrier` sequences were verified correct in isolation, so the
failure is in the host-driven CFL-readback / re-dispatch interleaving
under MoltenVK, not the kernels. The fully GPU-resident
`gpu_resident_batch` (Option #5), which keeps CFL and time advance on the
device, is correct on both MoltenVK and CUDA and is therefore the default
and the macOS reference. Validate `fixed_dt_batch_barrier` on Linux/GPU
where it is the production path.

## Environment note

The Vulkan backends run natively on the macOS host (Docker Desktop has no
GPU/Metal passthrough, so the dev container cannot run Vulkan on macOS).
This requires the patched `kp` built from source per
[`../../reference/kp-patches.md`](../../reference/kp-patches.md).
