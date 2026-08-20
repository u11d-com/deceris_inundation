# 16 — EA Test 3 momentum obstruction — results

Status: **passing** on macOS/MoltenVK with `gpu_resident_batch` using the
published May-2010 dataset.

| Backend | Vol drift | Min h | Point 1 | Crest | Point 2 | Control Point 2 | Disconnected | Repro | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `gpu_resident_batch` | 0.000 | 0.000 | 0.241 m | 0.000 m | 0.047 m | 0.000 m | yes | yes | yes |

The 1310 m³ hydrograph nearly fills the first depression to its crest. The
release carries 0.047 m into Point 2 while the equal-volume still-water control
leaves Point 2 exactly dry. The crest also drains completely, so the two ponds
are disconnected and Point 2 is an earned momentum signature rather than a
well-balance artifact.

The 8710-step run completed in 4.96 s on the Apple GPU through MoltenVK. Volume
closes exactly and repeated runs agree.
