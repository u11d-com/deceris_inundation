<<<<<<< HEAD
# Lake-at-Rest Vulkan Baseline Results

The unbarriered fixed-step batch diverged because consecutive compute
dispatches lacked a write-to-read barrier. `fixed_dt_batch_barrier` fixes that
race and matches `gpu_resident_batch`.

| Variant | Friction | Max absolute momentum | Max absolute velocity | Volume drift |
| --- | --- | ---: | ---: | ---: |
| A | minimal | 2.696e-01 | 2.811e-01 | 8.611e-05 |
| A | production | 2.705e-01 | 2.822e-01 | 9.339e-05 |
| B | minimal | 2.265e-01 | 5.062e-01 | 9.910e-04 |
| B | production | 1.911e-01 | 4.230e-01 | 1.185e-03 |

All four 600-second runs completed without non-finite state. The GPU-resident
implementation agreed to four significant figures. Remaining currents belong
to the retained numerical scheme and require a new baseline if that scheme
changes.
=======
# Lake-at-Rest (C-Property) — Vulkan Baseline Results

Referenced from `lake-at-rest-test.md` §4 (Step 1 — Vulkan baseline). This
records the first execution of that baseline, the finding that stopped it
from producing usable reference constants (§1-§8), and a follow-up session
(§10-§14) that root-caused the defect, implemented a fix, and produced a
stable reference-constants table.

**Status: RESOLVED.** §10-§14 below root-cause the divergence to a missing
GPU compute barrier in `SWESolverFixedDtBatch`'s batched dispatch sequence
(not a scheme/Audusse/bed-slope issue, as §5 already ruled out), implement
the fix as a new class (`SWESolverFixedDtBatchBarrier` in
`swe_gpu_fixed_dt_batch_barrier.py`, `--backend fixed_dt_batch_barrier`),
and produce the first stable full-matrix reference-constants table.
`swe_gpu_fixed_dt_batch.py` (Option #4) itself is left unmodified.

## 1. Test script

Implemented `benchmark_lake_at_rest.py` (this directory), per §4.1/§6 of the
test doc: self-contained NumPy + solver imports, CLI flags for
`--backend`, `--variant`, `--friction`, plus `--t-end` / `--output-interval`
overrides (not in the original doc) added to allow fast iteration instead of
waiting out the full 600s simulated horizon on every run, and `--quiet` to
disable per-batch progress logs. Blow-ups are caught and recorded as report
rows (`blew_up=True`) instead of crashing the whole matrix.

Added for diagnosis (§4-§6 below, not in the original doc): `--dt-init`,
`--dt-max`, `--cfl-interval` overrides; `--trace` (downloads `hu`/`hv` every
output-interval chunk instead of only at the end, via resumed `run()` calls);
`--flat-bed` (zb=0 everywhere, a true trivial-rest control case); `--hash`
(sha256 of final h/hu/hv for cross-run determinism comparison).

## 2. Hardware / environment

- Platform: `macOS-15.7.7-arm64-arm-64bit`
- Processor: `arm`
- Python: `3.12.13`
- Backend: Vulkan via MoltenVK (macOS dev box — acceptable per doc §4.4,
  parity bar is backend-relative, not hardware-absolute)

- kp (Kompute) built via `deployment/gpu-worker/build-kp.sh`, run from the
  separate `.venv-gpu` environment (kp is not part of the main `uv` project
  env — see `requirements.txt` in this directory)

## 3. Result: baseline does not stabilize — solver diverges almost instantly

Ran all 4 variants (A/B × minimal/production friction) on **Option #4**
(`fixed_dt_batch`, the production reference), using the doc's exact
prescribed parameters (`g=9.81`, `dry_tol=1e-4`, `cfl=0.45`,
`dt_max=0.05`, `dt_init=1e-2`, `cfl_interval=10`).

At `t_end=600s` (doc default), all 4 variants hit the solver's own
`RuntimeError("CFL dt too small ...")` guard (`dt_cfl < 1e-10`):

```text
variant  friction     backend         status
A        minimal      fixed_dt_batch  blew up at step 12860, dt_cfl=3.669e-18
A        production   fixed_dt_batch  blew up at step  6700, dt_cfl=6.776e-18
B        minimal      fixed_dt_batch  blew up at step  3840, dt_cfl=1.536e-13
B        production   fixed_dt_batch  blew up at step  4230, dt_cfl=1.452e-16
```

To rule out a slow 600s-scale accumulation (which is what the doc's C-property
background in §2 anticipates — a gradual well-balance leak), the horizon was
bisected downward for variant A / minimal friction:

| `--t-end` | Outcome |
|---|---|
| 30s | blew up (step 1920, `dt_cfl=1.048e-17`) |
| 2s  | blew up (step 13790, `dt_cfl=2.725e-12`) |
| 0.3s | survived a snapshot at `t=0.05s`, then blew up before `t=0.3s` (step 2868, `dt_cfl=1.630e-16`) |

**Onset is under 0.3 simulated seconds** on a still-water case with
`hu0=hv0=0`. This is not the gradual, small-magnitude spurious-current drift
described in the test doc's background (§2) — it is a near-instant numerical
divergence, well beyond the doc's escalation trigger of `max_abs_u > 0.01 m/s`
(§4), since the solver never reaches any measurable steady-state magnitude at
all before diverging.

Cross-check attempt on **Option #5** (`gpu_resident_batch`) for variant A /
production friction: the run did not raise the same `RuntimeError`, but
progress logging showed simulated time advancing far slower than wall time
(`sim=7.80/600.00s` after 36,480 GPU-side batches, `sim=15.63/600.00s` after
72,960), consistent with the same underlying CFL collapse — `dt` being
driven to near-zero internally without the CPU-side early-exit guard that
Option #4 has. This run was aborted before completion (would have taken
hours) rather than left to grind. A full opt5 cross-check should be re-run
with a short `--t-end` once the Option #4 issue is understood, since at
present it cannot even reach a comparable state in reasonable wall time.

## 4. `dt_init` hypothesis — refuted

The original working hypothesis (§4, prior revision) was that
`dt_init=1e-2` (the doc's prescribed initial timestep) is well above the
CFL-stable timestep for this mesh (`dt_CFL ≈ cfl · edge_len / wave_speed ≈
0.45 · 0.0078 / 3.13 ≈ 0.0011s`, i.e. `dt_init` ~9x the limit), and that this
alone explained the immediate blow-up.

**This is refuted, both by code inspection and by experiment.**

Code (`swe_gpu_fixed_dt_batch.py:84-88`): every batch recomputes `dt_cfl`
from the GPU CFL-reduction kernel *before* dispatching flux/update, and
overwrites `dt` with it (`dt = min(dt_cfl * cfl_safety, dt_max)`). `dt_init`
only survives if `dt_cfl >= 1e10` (i.e. the CFL kernel reports "no
constraint"), which does not happen at step 0. So `dt_init` cannot be
driving the observed divergence — it is essentially unused after the first
CFL check.

Experiment: variant A / minimal friction, `t_end=0.3s`, `output_interval=0.02s`,
`--trace` (downloads `hu`/`hv` after each output chunk):

| `dt_init` | `max_abs_hu` @ t=0.02s | @ t=0.04s | @ t=0.06s | @ t=0.08s | blow-up step |
|---|---|---|---|---|---|
| 1e-2 | 3.091 | 2.834 | 6.799 | 1.448e+03 | 22150 (t_end=0.3 run) |
| 1e-3 | 2.078 | 4.968 | 5.564e+02 | — (blew up) | 1540 |
| 1e-4 | 1.910 | 1.909 | 3.298 | 8.757 | 160 (blew up at t≈0.10s) |

`max_abs_hu` is already O(1-3) at the very first samplable instant
(`t=0.02s`, within the first output chunk) **regardless of `dt_init`**. For
`h0=1m` still water this corresponds to velocities of 1-3 m/s from nothing —
not the "small spurious current" the doc's background (§2) describes, and
already three orders of magnitude past the doc's own escalation threshold
(`max_abs_u > 0.01 m/s`, §4) before any meaningful simulated time has
elapsed.

Also tested `cfl_interval=1` (CFL recomputed every single step, the
tightest possible — rules out "stale CFL over a 10-step batch" as the
cause): still blows up, at step 661.

## 5. Flat-bed control — bed-slope/well-balance ruled out

To separate "well-balance leak on a bumpy bed" (the mechanism the doc's §2
background anticipates) from "kernel defect independent of bathymetry", the
benchmark script gained a `--flat-bed` flag (`zb=0` everywhere — a true
trivial-rest case, not just "small bed slope").

**Flat bed still blows up.** Variant A / minimal friction, `t_end=30s`:
blew up at step 4520 (`dt_cfl=8.008e-18`) — same order of magnitude and
same failure mode as the Gaussian-bump case. Traced short-horizon run
(`t_end=0.3s`, `output_interval=0.02s`) shows the same immediate growth
pattern as the bumpy-bed case:

```text
t=0.02s  max_abs_hu=1.196e+00
t=0.04s  max_abs_hu=2.006e+00
t=0.06s  max_abs_hu=5.441e+00
t=0.08s  max_abs_hu=1.233e+02
```

**Conclusion: this is not a bed-slope/well-balance/Audusse-correction
issue at all.** With zero bed slope there is nothing for a missing
Audusse correction to be missing — the pressure-gradient sum over any
cell's edges is identically balanced on a flat bed regardless of the
flux kernel's hydrostatic-reconstruction treatment. The divergence must
originate in the flux/update/CFL kernel chain itself (shared by both
variants), independent of bathymetry.

## 6. Determinism — confirmed broken, from the first batch

Two identical-config runs (`fixed_dt_batch`, variant A, minimal friction,
flat bed, `t_end=0.01s` — a single ~10-step batch under `cfl_interval=10`,
before any large-scale divergence could plausibly accumulate) produced:

| run | `max_abs_hu` | `max_abs_u` | `volume_drift` | `state_hash` (sha256 of h\|hu\|hv) |
|---|---|---|---|---|
| 1 | 9.806e-01 | 9.303e-01 | 1.322e-02 | `1071a7ba...` |
| 2 | 7.421e-01 | 7.297e-01 | 5.219e-03 | `7a070605...` |

Different hashes, different magnitudes, after the *first* batch. This is
not float-summation-order noise being chaotically amplified over hundreds
of steps (there hasn't been time for that at t=0.01s) — the run-to-run
difference is present immediately. This points to a genuine race condition
in the GPU dispatch (e.g. an unsynchronized scatter/gather write in the
flux or CFL-reduction kernel), not merely a scheme deficiency.

Also notable: `volume_drift` is nonzero (1.3e-2 / 5.2e-3) with no sources
active — momentum is diverging *and* mass is not being conserved, from the
first batch, on a flat bed, at rest. This is inconsistent with "a well-known
class of scheme is imperfect" and consistent with an implementation bug
(indexing, atomics, or synchronization) in the flux/CFL/update kernel path.

## 7. What this means for the test plan

- **No reference constants could be recorded.** The doc's §6 deliverable
  table (magnitude comparison table for Step 2 CUDA parity) cannot be
  produced from this run — there is no stable baseline magnitude, and no
  stable magnitude is achievable until the underlying defect is fixed.

- The doc's own diagnosis (§2, missing Audusse correction / bed-slope term)
  does **not** explain the observed failure: the defect reproduces
  identically on a flat bed where no such correction is needed.

- Per `../../architecture/cuda-migration.md` §2.3, no scheme/kernel changes were
  attempted here — diagnosing/fixing the flux, update, or CFL kernels is
  out of scope for this port-parity test and requires separate,
  explicitly-approved sign-off.

- Cross-check on Option #5 (`gpu_resident_batch`) deliberately **not**
  re-attempted this round: both backends dispatch the same shared SPIR-V
  kernels (`swe_gpu.py`: `_pc_flux`, `_pc_update`, `_pc_cfl_accum`,
  `_pc_cfl_reduce`), differing only in host-side batching, so the defect
  is expected to reproduce there too. Opt5 should only be revisited as a
  final cross-check once Option #4 is stable.

- **Escalation: mandatory, before any further baselining or CUDA-porting
  work.** This is a kernel-level defect (likely a GPU race/synchronization
  bug, see §6) reproducing on trivial still water with a flat bed — not a
  known-imperfect-scheme magnitude question. It also raises a question
  outside this test's scope: has production ever produced trustworthy
  results with non-flat bathymetry, given the flux/update chain appears
  broken independent of bed slope?

## 8. Suggested next step (not yet executed)

1. Isolate which kernel is at fault: run flat-bed, still-water case with
   the *update* kernel dispatch removed/no-op'd (flux only) and vice versa,
   or inspect the CFL-reduction kernel for an unguarded atomic/shared-memory
   race — the `cfl_interval=1` result (§4) shows the problem exists even
   when CFL is recomputed every step, and the determinism failure (§6)
   isolates to a single ~10-step batch, so the fault is tractable to binary-
   search within one batch's dispatch sequence.

2. Once root-caused: maintainer decides whether the fix lands before the
   CUDA port (re-baselining after) or the port proceeds by faithfully
   reproducing the current (broken) behavior, per plan §2.3 — this is a
   scope/sign-off decision, not a test-script decision.

3. Only after a stable Vulkan baseline exists: re-run the full 4-variant
   matrix (doc §3.3) for the §6 reference-constant table, and re-attempt
   the Option #5 cross-check.

## 9. Script usage reference

```sh

# Fast iteration (30s horizon, no progress spam):

PYTHONPATH=src .venv-gpu/bin/python -m deceris.inundation.benchmark_lake_at_rest \
  --backend fixed_dt_batch --variant all --friction all \
  --t-end 30 --output-interval 10 --quiet

# Full doc-spec run (600s, single variant):

PYTHONPATH=src .venv-gpu/bin/python -m deceris.inundation.benchmark_lake_at_rest \
  --backend fixed_dt_batch --variant A --friction minimal

# Diagnostic: per-chunk hu/hv trace with a smaller dt_init (§4):

PYTHONPATH=src .venv-gpu/bin/python -m deceris.inundation.benchmark_lake_at_rest \
  --backend fixed_dt_batch --variant A --friction minimal \
  --t-end 0.3 --output-interval 0.02 --dt-init 1e-4 --quiet --trace

# Diagnostic: flat-bed control, true trivial rest (§5):

PYTHONPATH=src .venv-gpu/bin/python -m deceris.inundation.benchmark_lake_at_rest \
  --backend fixed_dt_batch --variant A --friction minimal \
  --t-end 30 --output-interval 60 --flat-bed --quiet --hash

# Diagnostic: determinism check, run twice and diff state_hash (§6):

PYTHONPATH=src .venv-gpu/bin/python -m deceris.inundation.benchmark_lake_at_rest \
  --backend fixed_dt_batch --variant A --friction minimal \
  --t-end 0.01 --output-interval 0.01 --flat-bed --quiet --hash

# kp (Kompute) is only available in .venv-gpu, not the main uv-managed venv —
# see requirements.txt in this directory for why it can't be a normal dependency.

```

## 10. Root cause identified — missing GPU compute barrier, not a scheme defect

Follow-up session, executed §8's suggested next step via a different, cheaper
route than "no-op one kernel and see": a faithful CPU/NumPy transliteration of
the exact `swe_shaders_flux.py`/`swe_shaders_update.py` math (same HLLC
branches, same Heun RK2 stages), run on the identical flat-bed/still-water
case for 50 steps.

**The CPU transliteration is perfectly stable: `max|hu|=0.0`, `max|h-1|=0.0`
at every checkpoint, zero drift.** This proves the scheme itself (HLLC flux,
Heun RK2, dry/wet handling) is correct and stable — the bug is specific to
GPU *execution*, not the numerical method. This also independently confirms
§5's conclusion (not a bed-slope/Audusse issue) from a different angle: a
mesh-closure check (`Σ edges (nx,ny)·len = 0` per cell, both pre- and
post-`hilbert_reorder`) found `max residual = 0.0` across all 16,384 cells,
ruling out a geometry/indexing bug too.

That leaves GPU dispatch/synchronization as the only remaining candidate,
and the fix was found by diffing `swe_gpu_fixed_dt_batch.py` against its
sibling `swe_gpu_gpu_resident_batch.py` (Option #5):

- **`SWESolver.run()`** (`swe_gpu.py`, the class this file's title calls
  "baseline"): every dispatch (`flux(0)→update(0)→flux(1)→update(1)`) gets
  its own `sequence()...eval()` call — a full GPU fence sync between every
  stage.

- **`SWESolverFixedDtBatch.run()`** (Option #4): chains up to `cfl_interval`
  (10) full RK2 steps — dozens of dispatches — into **one** sequence with a
  single `.eval()` at the end. No synchronization between `flux`'s atomic
  writes to `dh`/`dhu`/`dhv` and `update`'s immediate read of them, batch
  after batch.

- **`SWESolverGpuResidentBatch.run()`** (Option #5): batches similarly, but
  already inserts an explicit `kp.OpComputeBarrier(...)` after every single
  dispatch, covering exactly the buffers it wrote that the next dispatch
  reads (`_barrier_flux`, `_barrier_state`, `_barrier_cfl`, `_barrier_dt`,
  `_barrier_time`; see `swe_gpu_gpu_resident_batch.py:84-100`).

Option #4 is simply missing the barrier insertion that its sibling class
already does correctly — a missing-barrier / unsynchronized read-after-write
hazard on the `dh`/`dhu`/`dhv`/`h`/`hu`/`hv`/`h1`/`hu1`/`hv1` storage
buffers, exposed by batching many dispatches per submit without an
intervening host sync. This explains every symptom recorded in §3-§6:
instant divergence (race, not gradual leak), reproduction on flat bed
(hazard, not scheme), and non-determinism from the first batch (race, not
chaotic amplification — there wasn't time for chaos, because it isn't
chaos).

Confirmed empirically before any fix was written: running raw `SWESolver`
directly (bypassing `fixed_dt_batch`'s batching entirely) on the same
flat-bed/still-water case stayed at `max_abs_hu = 0.0` exactly at both
`t=0.05s` and `t=1.0s` (no blow-up), and on the Gaussian-bump case (variant
A, minimal friction, `t=2s`, two independent runs) gave consistent bounded
magnitudes (`max_abs_hu≈0.3858`, `max_abs_u≈0.4167` both runs, agreeing to
~4 significant figures) — versus Option #4's catastrophic run-to-run
divergence (§6: `0.98` vs `0.74` after a single batch). Raw `SWESolver` is,
however, impractically slow at this scale (per-dispatch fencing — 12s wall
for a 2s sim) due to the fine-grained sync, which is why Option #4's
batching optimization exists in the first place; it just needs the missing
barrier, not a full return to per-dispatch fencing.

## 11. Fix implemented — `SWESolverFixedDtBatchBarrier`

New file `swe_gpu_fixed_dt_batch_barrier.py`, new class
`SWESolverFixedDtBatchBarrier(SWESolverFixedDtBatch)` — subclasses Option #4
to inherit its host-driven CFL/dt bookkeeping and CPU-side early-exit
`RuntimeError` guard unchanged, and overrides only `_build_algorithms()` (to
precompute the barrier ops once) and `run()` (to insert them). Mirrors
Option #5's exact barrier placement:

- `kp.OpComputeBarrier([t_cfl_scratch, t_dtbuf])` between `cfl_accum` and
  `cfl_reduce`

- `kp.OpComputeBarrier([t_dh, t_dhu, t_dhv])` between each `flux` dispatch
  and the `update` dispatch that immediately follows it

- `kp.OpComputeBarrier([t_h, t_hu, t_hv, t_h1, t_hu1, t_hv1, t_dh, t_dhu,
  t_dhv])` between each `update` dispatch and the next `flux` dispatch (or
  the optional GPU source dispatch, or the next step/batch)

`swe_gpu_fixed_dt_batch.py` (Option #4) itself is **left unmodified** — this
is a new, additive implementation, not an in-place edit, per the decision to
keep the original (broken) class available for faithful-transliteration
purposes per `../../architecture/cuda-migration.md` §2.3 pending a scope/sign-off
decision. Wired into `benchmark_lake_at_rest.py` as
`--backend fixed_dt_batch_barrier`.

## 12. Validation — flat-bed control and determinism

Flat-bed, still-water, variant A, minimal friction, `t_end=30s`:

| backend | max_abs_hu | state_hash |
|---|---|---|
| `fixed_dt_batch` (unfixed) | blew up (§3/§5: step 4520-7120, `dt_cfl≈1e-16`) | — |
| `fixed_dt_batch_barrier` (fixed) | `0.000e+00` | `1bf10081c8bf6c4c7358efc900f539bac371c2bdd576bb9d1ea1b987de531198` |
| `gpu_resident_batch` (Option #5, unmodified) | `0.000e+00` | `1bf10081c8bf6c4c7358efc900f539bac371c2bdd576bb9d1ea1b987de531198` |

The two independently-barriered implementations produce a **bit-for-bit
identical hash** on the trivial case — strong cross-confirmation both are
now correct.

Gaussian bump, variant A, minimal friction, `t=2s`, two independent runs per
backend:

| backend | run 1 max_abs_hu / max_abs_u | run 2 max_abs_hu / max_abs_u | run-to-run agreement |
|---|---|---|---|
| `fixed_dt_batch` (unfixed) | — (catastrophic divergence) | — | order-of-magnitude different (§6: `0.98` vs `0.74`) |
| `fixed_dt_batch_barrier` (fixed) | 0.3858 / 0.4167 | 0.3858 / 0.4167 | agrees to ~4 sig figs |
| `gpu_resident_batch` (Option #5) | 0.3858 / 0.4167 | 0.3858 / 0.4167 | agrees to ~4 sig figs |

Small residual non-determinism remains in the `volume_drift` figure and the
sha256 hash (~1e-7 relative magnitude, e.g. `5.021e-07` vs `5.648e-07`) —
this is ordinary float32 atomic-add reordering noise across GPU threads
(non-associative floating point), present identically in raw `SWESolver`
too, and is **not** the catastrophic race that was fixed. Full bit-exact
determinism (`lake-at-rest-test.md` §5 gate 4) would require a separate,
further change (deterministic reduction order for the scatter-add), out of
scope for this fix.

`t_end=30s` on the Gaussian-bump case completed in ~41s wall
(`fixed_dt_batch_barrier`) and ~42s wall (`gpu_resident_batch`) — both far
faster than raw `SWESolver`'s per-dispatch-fenced ~12s-per-2-simulated-second
rate, confirming the barrier fix preserves Option #4's batching performance
advantage while restoring correctness.

## 13. Full doc-spec reference-constants table (§6 deliverable)

First stable execution of the full 4-variant matrix at the doc's exact
default parameters (`t_end=600s`, `output_interval_s=60s`,
`dt_max=0.05`, `dt_init=1e-2`, `cfl_interval=10`), on
`--backend fixed_dt_batch_barrier`:

```text
variant  friction    backend                 max_abs_hu  max_abs_u  volume_drift  pass
A        minimal     fixed_dt_batch_barrier  2.696e-01   2.811e-01  8.611e-05     ref
A        production  fixed_dt_batch_barrier  2.705e-01   2.822e-01  9.339e-05     ref (escalation, see below)
B        minimal     fixed_dt_batch_barrier  2.265e-01   5.062e-01  9.910e-04     ref
B        production  fixed_dt_batch_barrier  1.911e-01   4.230e-01  1.185e-03     ref (escalation, see below)
```

No blow-ups, no NaN/inf snapshots, all 4 variants ran to completion. Every
variant's magnitude stayed bounded across the full horizon (e.g. variant A
minimal actually settled from `0.296` at `t=30s` to `0.281` at `t=600s` — a
stable, non-runaway trend, not a slow leak toward divergence).

Every variant trips the doc's own escalation trigger (`max_abs_u >
0.01 m/s`, §4) at both friction levels — expected and explicitly anticipated
by the doc's §2 background (missing Audusse interface correction / no
bed-slope term in the update kernel), and **not** a new defect: variant B
(partial dry, wet/dry front) shows the largest magnitudes (`max_abs_u`
0.42-0.51 m/s) and volume drift (~1e-3), consistent with the doc's own
prediction that the wetting/drying path is "typically the worst leak"
(§3.3). Per the doc (§4 escalation trigger note) and `../../architecture/cuda-migration.md`
§2.3, whether to add the Audusse correction is a separate,
explicitly-approved scheme-change decision with its own re-baselining — not
addressed by this fix, which only restores GPU-execution correctness.

**`TOL_FACTOR`/floor constants for Step 2 CUDA parity (doc §5):** not yet
finalized — this table is the first candidate set of reference magnitudes;
recommend confirming `TOL_FACTOR=3.0`, `ABS_FLOOR_HU=1e-6`,
`ABS_FLOOR_U=1e-5` (the doc's starting constants) against these specific
values before use, since the magnitudes here are notably larger than the
doc's original floor-only expectation (production-friction `max_abs_u≈0.28-0.42
m/s`, well above `ABS_FLOOR_U=1e-5`) — the escalation-trigger note above
applies.

## 14. Option #5 (`gpu_resident_batch`) audit — no fix needed

Full read-through of `swe_gpu_gpu_resident_batch.py`'s dispatch chain
confirms every dispatch is already followed by a matching
`kp.OpComputeBarrier` covering exactly the buffers it wrote that the next
dispatch reads:

```text
dt_reset         → barrier_dt
cfl_accum        → barrier_cfl
cfl_reduce       → barrier_cfl
cfl_resolve_time → barrier_dt
[loop _BATCH_STEPS times]:
  flux(stage0)     → barrier_flux
  update(stage0)   → barrier_state
  flux(stage1)     → barrier_flux
  update(stage1)   → barrier_state
  [source]         → barrier_state   (if enabled)
  time_advance     → barrier_time
  cfl_resolve_time → barrier_dt
```

No missing-barrier gap exists. This is, in fact, the reference
implementation the §10-§11 fix was modeled on. §3's "far slower than wall
time" observation on Option #5 was not reproduced at `t_end=30s` on the
Gaussian-bump case (§12: ~42s wall, `max_abs_u=0.299`, bounded) — if that
symptom is real, it is a separate and much smaller question (possibly an
artifact of the aborted 600s cross-check attempted in §3, or a `dry_dt`/
`cfl_resolve_time` tuning issue at longer horizons), not a synchronization
bug, and does not require a new file or fix.

**Full 600s/4-variant cross-check against `fixed_dt_batch_barrier`
(completed):**

```text
variant  friction    fixed_dt_batch_barrier (hu/u)  gpu_resident_batch (hu/u)  agreement
A        minimal     0.2696 / 0.2811                0.2696 / 0.2811            identical (4 sig figs)
A        production  0.2705 / 0.2822                0.2705 / 0.2822            identical
B        minimal     0.2265 / 0.5062                0.2265 / 0.5062            identical
B        production  0.1911 / 0.4230                0.1911 / 0.4230            identical
```

No blow-ups on either backend at the full doc-spec horizon. Both
independently-implemented, correctly-barriered backends agree to 4
significant figures on every variant — this closes out the cross-check the
original §3 attempt could not complete, and confirms §13's reference table
is not an artifact of one particular implementation.

## 15. Updated next steps

1. **Done:** root cause identified (§10), fix implemented as an additive new
   class (§11), validated for stability and bounded-magnitude behavior
   (§12), full reference-constants table produced (§13), Option #5 audited
   and confirmed already correct with a full 600s/4-variant cross-check
   (§14).

2. **Done:** production default switched from `fixed_dt_batch` to
   `fixed_dt_batch_barrier` (`INUNDATION_SOLVER_IMPL` in `flow.py`,
   `solver_impl` in `solver_workflow.py`, `--backend` default in
   `benchmark_lake_at_rest.py`) — the raced class is no longer reachable via
   the default pipeline path. `swe_gpu_fixed_dt_batch.py` itself remains
   unmodified and importable (for faithful-transliteration/comparison
   purposes per `../../architecture/cuda-migration.md` §2.3) but is no longer the
   production default and must not be used as a reference (see
   `../../architecture/cuda-migration.md` §1.1, `../../architecture/validation-plan.md` §1 Tier 2).

3. **Open, separate scope:** whether to add the Audusse interface correction
   / bed-slope term to close the escalation gap (§13) — an explicit,
   maintainer-approved scheme change per `../../architecture/cuda-migration.md` §2.3, not
   part of this fix.

4. **Open, separate scope:** full bit-exact determinism (doc §5 gate 4)
   would need a deterministic reduction order for the atomic scatter-add;
   current residual non-determinism (§12) is small (~1e-7) and of a
   fundamentally different (benign) character than the race this fix
   resolved, but still fails a strict byte-hash equality gate.

5. Once 3-4 are resolved or explicitly deferred: proceed to
   `lake-at-rest-test.md` §5 (CUDA parity bar) using §13's table as the
   Vulkan reference constants.

## 16. Script usage reference (updated)

```sh

# Fast iteration (30s horizon, no progress spam), fixed backend:

PYTHONPATH=src .venv-gpu/bin/python -m deceris.inundation.benchmark_lake_at_rest \
  --backend fixed_dt_batch_barrier --variant all --friction all \
  --t-end 30 --output-interval 10 --quiet

# Full doc-spec run (600s, all variants), fixed backend:

PYTHONPATH=src .venv-gpu/bin/python -m deceris.inundation.benchmark_lake_at_rest \
  --backend fixed_dt_batch_barrier --variant all --friction all --quiet --hash

# Diagnostic: flat-bed control, true trivial rest, fixed backend (§12):

PYTHONPATH=src .venv-gpu/bin/python -m deceris.inundation.benchmark_lake_at_rest \
  --backend fixed_dt_batch_barrier --variant A --friction minimal \
  --t-end 30 --output-interval 60 --flat-bed --quiet --hash

# Diagnostic: determinism check, run twice and diff state_hash (§12):

PYTHONPATH=src .venv-gpu/bin/python -m deceris.inundation.benchmark_lake_at_rest \
  --backend fixed_dt_batch_barrier --variant A --friction minimal \
  --t-end 2 --output-interval 2 --quiet --hash

# kp (Kompute) is only available in .venv-gpu, not the main uv-managed venv —
# see requirements.txt in this directory for why it can't be a normal dependency.

```
>>>>>>> 2f36b6f (fix: Docs rework 2)
