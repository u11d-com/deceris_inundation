# Lake-at-Rest Test

Run the Vulkan `fixed_dt_batch_barrier` solver on a non-flat bed initialized to
a constant free-surface elevation. With no source or friction forcing, verify:

- finite depth and momentum arrays;
- non-negative water depth;
- bounded free-surface and momentum drift;
- conserved total water volume within the configured tolerance.

Use `benchmark_lake_at_rest.py` for the reproducible command-line scenario.
