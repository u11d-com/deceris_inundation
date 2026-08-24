# 19 — Grid-convergence study

## Goal

Measure whether reference errors shrink at the expected order and identify
metrics that hit a numerical floor.

## Scope

Sweep resolution-independent references only: Stoker and Ritter dam-break,
radial dam-break, and flood propagation. Terrain-bound cases are excluded
because resampling moves both terrain and derived bounds.

`bench/grid_convergence.py` runs each existing harness and fits pairwise plus
least-squares order for its recorded metrics. Spatial levels are 8 to 0.5 m for
1D dam-break, 0.4 to 0.05 m for radial dam-break, and 20 to 2.5 m for flood
propagation.

## Decisions

Monotone profile errors establish convergence. Non-monotone mass drift signals
an error budget not explained by truncation alone. A float64 state build is
considered only after cheaper instrumentation separates positive clamp mass
from the remaining drift.
