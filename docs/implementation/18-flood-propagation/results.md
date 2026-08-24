# 18 — EA Test 4 flood propagation — results

Status: **passing** on macOS/MoltenVK with `gpu_resident_batch`.

| Backend | Vol drift | Min h | Arrival err | Depth L1 | Speed L1 | Isotropy | Repro | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `gpu_resident_batch` | 2.9e-4 | 0.129 | 0.063 | 0.026 | 0.013 | -0.001 m | yes | yes |

Gauge values at the two-hour probe:

```text
radius [m]          50.0  100.0  200.0  300.0  400.0  424.3
arrival [min]       10.0   15.0   28.0   42.0   58.0   61.0
reference [min]      9.9   16.0   29.8   44.7   60.5   64.4
depth [m]           .314   .263   .212   .178   .149   .143
reference depth [m] .321   .269   .217   .183   .154   .148
speed [m/s]         .399   .232   .142   .110   .094   .091
reference speed     .390   .233   .144   .112   .095   .093
```

The front is systematically 4–6% early at the five-metre specification grid.
Effort 19 shows the arrival error falling 0.208, 0.107, 0.063, and 0.025 under
successive refinement, with observed order 0.99. The bias is spatial
discretization error.

Mass drift is the difference of opposing terms. The positivity clamp supplies
mass at wet/dry margins; the sink remains unidentified. Instrument the clamp
before attributing the remainder or extrapolating this configuration.
