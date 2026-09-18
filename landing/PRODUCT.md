# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Astro — user explicitly requested AstroJS for `landing/`. Deploy target undecided.

## Users

Primary: researchers benchmarking shallow-water / flood solvers. They compare numerics, correctness evidence, and performance before adopting, citing, or contributing.

Secondary (confirmed): open-source evaluators and potential contributors — the team is open-sourcing this solver with a landing page. They need to grasp what it does, verify the evidence, and reach the code.

Situation: desktop web evaluation of a technical solver — reading mechanism, checking benchmark proof, then moving to code or deceris.io.

## Product Purpose

deceris-inundation is a GPU-accelerated shallow-water flood-modelling solver on polygonal meshes: NumPy preprocesses the mesh, Vulkan/Kompute compute shaders run the physics. It models inundation with GPU + physics and powers inundation inside the deceris.io platform/application.

This landing page exists to open-source the solver: explain what it does, show real benchmark evidence, and route researchers to the code and to deceris.io. Success means a benchmarking researcher can understand the mechanism, trust the proof, and take the next step (repo / report / deceris.io) without encountering invented claims.

## Positioning

GPU speed on polygonal meshes: the solver runs HLLC flux, source, update, and CFL stages as Vulkan compute through Kompute, over polygon-mesh cells with edge/CSR adjacency (optionally Hilbert-reordered and cached) — not a raster-only model.

Validated pair of backends: `fixed_dt_batch_barrier` as the reference implementation, `gpu_resident_batch` reducing host synchronization. Reproducible benchmark harnesses plus a full benchmark report are the proof, not marketing copy.

## Operating Context

Evaluation workflow: load mesh cells with optional bed elevations → build edge + CSR adjacency (optional Hilbert reorder + cache) → compile GLSL to SPIR-V → run solver stages → download depth snapshots at cadence (optional momentum snapshots).

Mesh inputs: `.gpkg`, `.shp`, `.parquet`, `.geoparquet`, `.obj`. Polygon features need ≥ 3 vertices. GIS `z_mean` column supplies per-cell bed elevation; OBJ Z is ignored; bed defaults to zero without `z_mean`. Python API: `SWEWorkflow`, `WorkflowConfig`, `SimulationPhase`, `PointSource`, `WorkflowResult`; `prepare()` must run before `run()`; `geometry_cache_dir` reuses processed meshes.

Environments: Docker Compose dev container (Python 3.12, Vulkan tools, Kompute binding, bind-mounted source); production is Apptainer with NVIDIA `--nv` passthrough on Linux NVIDIA nodes; macOS/MoltenVK path is dev/correctness only.

## Capabilities and Constraints

Confirmed: polygonal-mesh shallow-water solving via Vulkan/Kompute; NumPy preprocessing with edge/CSR + Hilbert + cache; HLLC/source/update/CFL stages; depth + optional momentum snapshots; Python workflow API above; six benchmark harnesses (lake-at-rest, analytical dam-break, radial dam-break, floodplain depressions, momentum obstruction, flood propagation); standalone benchmark-report HTML.

Constraints: landing must keep solver truth exact (formats, bed rules, `prepare()`-before-`run()`, backend names, Docker/Apptainer/MoltenVK roles). Must link deceris.io as the platform home. Must not invent result numbers, plots, testimonials, customers, benchmarks, pricing, or licensing.

Undecided: deploy target/hosting for `landing/`; open-source license text to display; which figures from the benchmark report the landing may excerpt; docs-site relationship.

## Brand Commitments

Name: deceris-inundation. Factual technical voice matching the repo (mechanism + evidence, no hype). Must link to deceris.io. No logo, palette, typography, or other identity assets confirmed — none may be invented. Standing preference (canon): category-standard OSS landing, played straight at full fidelity; no comparator products named yet (user deferred comparisons).

## Evidence on Hand

Real: `../bench-report/deceris-inundation-benchmark_report.html` ("Deceris inundation — benchmark report"); benchmark harnesses beside their docs under `../inundation/bench/` (`lake_at_rest.md`, `dambreak.md`, `radial_dambreak.md`, `floodplain_depressions.md`, `momentum_obstruction.md`, `flood_propagation.md`); bundled EA test datasets under `../benchmark_assets/` (`Test2 dataset 2010/`, `Test3 dataset 2010/`, `Test4 dataset 2010/`); product mechanism per `../README.md` and `../inundation/workflow.py`.

Absent (must not be fabricated): testimonials, customers, case studies, press, result excerpts not yet selected, pricing/licensing claims.

## Product Principles

1. Evidence over claims: every performance/correctness statement traces to the report or a harness.
2. Mechanism made explicit: name the real pipeline (mesh → adjacency → shaders → snapshots), never a black box.
3. Reproducibility first: backends, mesh rules, and environment paths stay exact so results can be re-run.
4. Open-source honesty: mark pending content as pending; never fill gaps with invented proof or identity.
5. Platform link, not spill: deceris.io is the home link; the landing stays about the solver.
