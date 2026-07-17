# SWE Solver: CUDA Validation Plan

Testing/benchmarking methodology for the CUDA migration
(`cuda-migration.md`). Describes *how* correctness and performance are
established and *why* the plan is two-tiered — not current results. For
current pass/fail status and measured numbers, see
`../implementation/benchmark.md` (the ledger) and the individual
`../implementation/NN-*/results.md` folders. For the Go/No-Go outcome and
what it justified, see `../planning/decisions.md`.

Motivation: `swe_gpu_async_sync_window.py` (Option #3, see
`cuda-migration.md` §1.1) shipped with wrong results despite the existing
cross-variant bitwise-parity checks in `benchmark_inundation_metal.py`.
That check alone was not sufficient to catch it. The validation plan below
is deliberately two-tiered as a direct response to that.

## 1. Correctness bar

**Tier 1 — independent analytical validation (new, not present in the
existing benchmark script).** Implement and pass standard closed-form SWE
test cases, self-contained (no dependency on an external reference solver
such as ANUGA/GeoClaw/HEC-RAS — none is available or needed):

- 1D dam-break (Ritter/Stoker analytical solution).
- Lake-at-rest / C-property check — **as a port-parity test, not an
  absolute physics gate.** Planning-stage review found the current flux
  kernel applies hydrostatic reconstruction (`z_face = max(zbL, zbR)`) but
  appears to **lack the Audusse interface source-term correction**
  (`0.5 * g * (hL^2 - hLs^2) * n * len` added back to each adjacent cell),
  and the update kernel has no bed-slope term — so today's solver is
  likely *not* exactly well-balanced, and a faithful port will inherit
  that. The test therefore is: construct a synthetic still-water case (no
  external data needed), measure the spurious currents the **Vulkan
  reference** (Option #4b, `fixed_dt_batch_barrier`) produces, and require
  the CUDA port to match that magnitude within a tolerance factor —
  "exactly at rest" is not required (and is unachievable in float32 even
  for truly well-balanced schemes). Step-by-step instructions:
  `../implementation/00-lake-at-rest/plan.md`.
- Radially symmetric dam-break.

Specific error tolerances (e.g. acceptable relative L2 error against the
Ritter solution) are left to the implementer to set using standard
literature values for this scheme's order of accuracy, and should be
recorded explicitly in the benchmark output/report rather than left
implicit in a pass/fail assertion.

**Tier 2 — cross-backend agreement (extends the existing approach).**
Compare the CUDA solver's output against the current Vulkan solver, using
**Option #4b (`fixed_dt_batch_barrier`, the production default)** and
**Option #5 (`gpu_resident_batch`, already independently confirmed to
agree with #4b)** as the two trusted references. Do **not** use Option #3
(`async_sync_window`) or unfixed Option #4 (`fixed_dt_batch` — races,
produces wrong results) as a reference for anything.

The comparison bar is **physical-invariant parity, not bitwise**: volume
conservation ratio, minimum wet-cell count, monotonic snapshot times —
reuse the invariant checks already implemented in
`benchmark_inundation_metal.py` (`_validate_run_invariants`) as the
starting point. Do not require a bit-exact hash match between CUDA and
Vulkan output — the two backends are expected to use different
reduction/atomic strategies and will not match bit-for-bit; this is
acceptable.

**Separately: run-to-run determinism within the CUDA solver itself is
required** (bit-identical output for identical inputs, every run) — see
`cuda-migration.md` §2.6 for why this is expected to hold by construction
given the chosen kernel design, and should be explicitly tested (run the
same case twice, compare hashes) rather than assumed.

## 2. Precision

`float32` for all state/geometry arrays, matching the current solver. Sole
exception: the scalar simulated-time (and cumulative scalar) bookkeeping,
which must be float64 or integer-based — see `cuda-migration.md` §2.3 for
why float32 time fails at the 72h target.

## 3. Explicitly out of scope for validation

To prevent scope creep during implementation:

- No native-Vulkan-on-Linux/GPU baseline run (considered and explicitly
  not worth the setup effort — today's macOS/MoltenVK numbers, or simply
  the Tier 1/Tier 2 checks above, are sufficient).
- No historical real-world flood event replay/validation (considered and
  explicitly not needed — Tier 1 + Tier 2 is the complete validation
  plan).
- No automated CI gate for any of the above (see `cuda-migration.md`
  §3.3).

## 4. Benchmark scope and scale

- Primary benchmark mesh: the real ~5M-cell production-representative
  mesh already available (not just the 280k local mesh, which remains
  useful for fast iteration but is not representative of production scale
  or of multi-GPU partitioning behavior).
- Single-GPU: benchmark the CUDA Graphs variant (§2.4 rung 2) against the
  persistent-cooperative-kernel variant (§2.4 rung 3, where the occupancy
  gate allows it) on the 5M mesh, and record both against the existing
  Vulkan production-default numbers as context (acknowledging the
  hardware-comparability caveat in §1 above).
- Multi-GPU (Phase 1): benchmark 2/4/8-GPU strong-scaling on the 5M mesh.
  Explicitly measure and report **wet/dry load imbalance across shards**
  (e.g. per-GPU wall time per step, or per-GPU wet-cell fraction over the
  run), not just aggregate wall-clock speedup — this measurement is what
  determines whether the source-aware-partitioning follow-up
  (`cuda-migration.md` §2.8) is actually needed.
- Snapshot cadence/fields for benchmark runs should match the real
  production target: **1 snapshot per simulated hour, 72-hour simulated
  duration** (72 snapshots total per run). Capture `h, hu, hv` per
  snapshot (not just `h`) — `hu`/`hv` are already full state arrays solved
  every step, so this is a free extension of the snapshot-capture step,
  needed for a near-term (separately scoped) downstream
  force/pressure-on-buildings calculation. This cadence is also
  deliberately favorable to the synchronization strategy in
  `cuda-migration.md` §2.4: 72 forced host syncs over an entire 72-hour
  simulated run leaves enormous room for batching many steps between
  them.
- Warm-start (`cuda-migration.md` §2.3): verify explicitly that resuming
  from a saved intermediate state reproduces the equivalent of a
  continuous from-rest run reaching the same simulated time, within the
  Tier 2 invariant bar.

## 5. Go/No-Go criteria

Phase 1 (single-node CUDA, up to 8x GPU) is considered done, and a
reasonable checkpoint to decide on further investment (Phase 2 multi-node,
any Hopper-feature stretch goals, source-aware partitioning, etc.), only
once **all** of the following hold on the real ~5M-cell mesh. **For
current status against each item, see `../implementation/benchmark.md`
and `../planning/roadmap.md` — this list is the specification, not a
live status board.**

1. All three Tier 1 analytical test cases pass: dam-break and radial
   dam-break within documented, literature-justified tolerances;
   lake-at-rest within the Vulkan-parity bar defined in §1 above /
   `../implementation/00-lake-at-rest/plan.md`.
2. Tier 2 invariant-based agreement holds against both Option #4b and
   Option #5 on both the 280k and 5M meshes.
3. Run-to-run determinism holds (bit-identical repeats) for the CUDA
   solver, single-GPU and multi-GPU.
4. Warm-start reproduces equivalent results to an uninterrupted run (§4).
5. No regression against the explicitly-unchanged scope: boundary
   conditions (reflective only), source terms (point-source only), no
   hydraulic structures — i.e. behavior matches today's solver exactly
   wherever `cuda-migration.md` says it should.
6. Multi-GPU (2/4/8-GPU) results meet the same invariant/determinism bar
   as single-GPU, on the 5M mesh.
7. Multi-GPU load-imbalance measurement (§4) is recorded and reviewed —
   informational for prioritizing follow-up work, not itself a pass/fail
   gate, but must not be skipped.

If any of 1-6 fail, treat it as a correctness bug to fix before
considering any further optimization work (including the
persistent-kernel variant, Phase 2, or any stretch-goal Hopper feature) —
do not proceed to optimization on top of an unconfirmed-correct solver.
