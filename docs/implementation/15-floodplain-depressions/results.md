# 15 — EA Test 2 floodplain depressions — results

Status: **passing** on macOS/MoltenVK with `gpu_resident_batch` using the
published May-2010 dataset.

| Backend | Vol drift | Min h | Ponded | Full | Above sill | Capacity ratio | Storage | Far column | Repro | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `gpu_resident_batch` | 9.4e-5 | 0.000 | 11/16 | 7 | 0.007 m | 0.835 | 0.999 | 0.000 m | yes | yes |

Settled depth and sill depth at points 1–16, in metres:

```text
depth 0.38 0.33 0.33 0.33 0.28 0.33 0.33 0.33 0.00 0.06 0.17 0.18 0.00 0.00 0.00 0.00
sill  0.37 0.32 0.32 0.32 0.37 0.32 0.32 0.32 0.37 0.32 0.32 0.32 0.62 0.32 0.32 0.32
```

The three western columns hold water while the eastern column remains dry.
Water above discrete sill elevation stays inside the 0.03 m allowance, 99.9%
of volume is ponded, and mass closes within 9.4e-5 over 215,600 steps. Median
execution was 73.7 s on the Apple GPU through MoltenVK.

A routed fill-and-spill prediction was rejected: the inflow head crosses
multiple similar saddles, so terrain-derived storage and sill bounds are the
valid reference rather than a single-path cascade.
