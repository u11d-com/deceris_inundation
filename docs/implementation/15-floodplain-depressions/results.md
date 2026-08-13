# 15 — EA Test 2 floodplain depressions — results

Status: **passing** on macOS/MoltenVK with `gpu_resident_batch`.

| Backend | Mass error | Min h | Ponded | 15 and 16 dry | Plateau wet | Repro | Pass |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| `gpu_resident_batch` | 1.8e-6 | 0.000 | 5 / 16 | yes | 0.104 | yes | yes |

Settled point depths in metres:

```text
0.00 0.16 0.23 0.24 0.00 0.00 0.17 0.23 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
```

Five disconnected depressions remain ponded while the northeast points remain
dry. Roughly 8950 RK2 steps completed in 4.5 s on the Apple GPU through
MoltenVK. `gpu_resident_batch` remains the validated local reference.
