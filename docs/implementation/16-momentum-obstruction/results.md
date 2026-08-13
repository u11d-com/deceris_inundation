# 16 — EA Test 3 momentum over an obstruction — results

Status: **failing because the still-water control exposes a well-balance gap**.

| Backend | Vol drift | Min h | Valley depth | Ponds disconnected | Far depth | Control far depth | Pass |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `gpu_resident_batch` | 6.1e-6 | 0.011 | 0.120 m | yes | 0.054 m | 0.091 m | no |

The at-rest control crosses the sill and fills the far bowl more deeply than
the release run. Far-bowl water therefore cannot yet be credited to preserved
momentum. The control gate records this as `well_balance_leak` until the missing
hydrostatic interface correction is implemented.

The release and control remain finite, non-negative, and volume-conserving.
Both final ponds are hydraulically disconnected below the crest. Approximately
7000 RK2 steps completed in 3.0 s on the Apple GPU through MoltenVK.
