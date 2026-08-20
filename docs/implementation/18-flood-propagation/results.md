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

The front is systematically 4–6% early, consistent with coarse wetting-front
diffusion. The off-axis gauge performs at least as well as aligned gauges.
127,520 steps completed in 65.9 s on the Apple GPU through MoltenVK.

Mass drift worsens under spatial and timestep refinement rather than shrinking,
consistent with float32 increment absorption. This remains a measured
hypothesis until a float64 state comparison is available; the 1e-3 gate applies
to this configuration only.
