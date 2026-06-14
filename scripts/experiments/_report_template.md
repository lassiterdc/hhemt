# Synthetic compute-config sensitivity — engineering report

## Executive summary
{one-paragraph verdict: are hydraulic outputs byte-identical across compute config at fixed 3.5m res?}

## Methods
- Substrate: synthetic model, 3.5m native, 64x120, 30-min sim. Two experiments: clean (single-allocation) + resume (forced-kill).
- Read-only `ssh rivanna`, completed sims only (c_run flags); final MH = max-index `out_tritonswmm/bin/MH_*.out`; md5 grouping.

## Results
### H1 — within-config replicates byte-identical (no race)? {fill}
### H2 — OMP thread count byte-irrelevant (one group)? {fill}
### H3 — MPI rank count -> N distinct groups? {fill}
### H4 — a6000 1-GPU == a100 1-GPU? {fill}
### H5 — 2-GPU and 3-GPU each own group; a6000 vs a100? {fill}
### Clean-vs-resume comparison — resume divergence-onset per resumed sim {fill}

## Discussion
{toolkit-vs-TRITON isolation; resume is TRITON-internal restart-precision; implications}
