# Audusse Well-Balance Correction — Results

Status: **done for both Vulkan GLSL flux variants**.

| Variant | Friction | Max velocity before | Max velocity after |
| --- | --- | ---: | ---: |
| A | minimal | 2.972e-01 | 7.616e-05 |
| A | production | similar | 9.001e-05 |
| B | minimal | similar | 4.983e-05 |
| B | production | similar | 4.720e-05 |

The correction reduces every lake-at-rest case to float32 noise. In the
momentum-obstruction control, far-bowl depth falls from 0.091 m to 0.000 m and
volume drift is zero. Repeated runs agree.

The release now remains in the valley instead of crossing the sill, so the
scenario's momentum gate still fails for an honest, separate reason: the
release is too weak to overtop after numerical leakage is removed. Retune that
scenario without weakening the still-water control.

The critical implementation detail is asymmetry: left and right cells use
their own reconstructed-depth correction along their own outward normals.
