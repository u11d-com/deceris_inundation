# Audusse Well-Balance Correction — Plan

## Trigger

Still water climbed the obstruction sill. The existing hydrostatic
reconstruction lacked the interface source-term correction needed to balance
pressure against bed elevation.

## Scope and fix

Apply the standard Audusse correction to both GLSL flux shader variants. Each
cell receives its own depth-versus-reconstructed-depth term along its outward
normal:

```text
SL = 0.5 * g * (hL^2 - hLs^2)
SR = 0.5 * g * (hR^2 - hRs^2)
left  += flux + SL * normal
right -= flux + SR * normal
```

## Verification

The four lake-at-rest variants must reduce maximum velocity below 0.01 m/s.
The momentum-obstruction still-water control must keep the far bowl below
0.005 m while volume and repeatability gates remain satisfied. Visual output
must still show a genuine release moving downhill.
