# Lake-at-Rest Vulkan Baseline Results

The original unbarriered fixed-step batch diverged because consecutive compute
dispatches lacked a write-to-read barrier. `fixed_dt_batch_barrier` fixes that
execution race and matches the independently barriered `gpu_resident_batch`
implementation.

## Reference matrix

| Variant | Friction | Implementation | Max absolute momentum | Max absolute velocity | Volume drift |
| --- | --- | --- | ---: | ---: | ---: |
| A | minimal | `fixed_dt_batch_barrier` | 2.696e-01 | 2.811e-01 | 8.611e-05 |
| A | production | `fixed_dt_batch_barrier` | 2.705e-01 | 2.822e-01 | 9.339e-05 |
| B | minimal | `fixed_dt_batch_barrier` | 2.265e-01 | 5.062e-01 | 9.910e-04 |
| B | production | `fixed_dt_batch_barrier` | 1.911e-01 | 4.230e-01 | 1.185e-03 |

All four 600-second runs completed without non-finite state. The GPU-resident
implementation agreed to four significant figures. The remaining currents are
attributed to the retained numerical scheme rather than the synchronization
race; changing that scheme requires a new baseline.
