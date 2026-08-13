# Lake-at-Rest Vulkan Baseline Results

Two independent defects were identified and corrected. Batched dependent
dispatches needed a write-to-read compute barrier, and hydrostatic
reconstruction needed the Audusse interface source correction.

| Variant | Friction | Max velocity after correction |
| --- | --- | ---: |
| A | minimal | 7.616e-05 |
| A | production | 9.001e-05 |
| B | minimal | 4.983e-05 |
| B | production | 4.720e-05 |

All four cases remain finite and sit below the 0.01 m/s escalation threshold.
`fixed_dt_batch_barrier` and `gpu_resident_batch` provide the retained barrier
implementations; MoltenVK analytical work uses `gpu_resident_batch`.
