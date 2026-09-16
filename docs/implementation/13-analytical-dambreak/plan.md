# 13 — Tier 1 analytical dam-break benchmark

## Goal

Validate the solver independently with Stoker wet-bed and Ritter dry-bed
closed-form solutions. Both run on a generated flat rectangular channel with no
external data.

## Harness

Run `inundation/bench/dambreak.py` through `just benchmark-dambreak`.
The default is `gpu_resident_batch`; other retained Vulkan implementations are
selectable with `--backends`.

## Default setup

- 1000 m × 10 m channel; 500 × 5 quads.
- Dam at 500 m; upstream depth 10 m.
- Downstream depth 1 m for Stoker and dry for Ritter.
- End time 20 s; `dt_max=0.02`; `cfl_interval=5`.

## Gates

Depth L1 relative error must remain at most 0.05, Ritter front-position error at
most 0.15 of analytical travel, relative volume drift at most 1e-5, and depth
must remain finite and non-negative. Repeated runs must agree within 1e-5 of
peak depth.

Timed runs report wall time and throughput. Animation uses a separate untimed
run so rendering cannot affect timing.
