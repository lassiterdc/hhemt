# Release history

## v0.1.0 (2026-XX-XX)

First public release of the H&H Ensemble Modeling Toolkit (hhemt). Coupled
TRITON-SWMM flood-ensemble orchestration across local + HPC (NVIDIA/AMD), full
preprocessing→compile→run→process→consolidate→report lifecycle, interactive
analysis report, and a portable render-bundle. Docs at
https://hhemt.readthedocs.io. Cite via the Zenodo DOI (see README).

**Installation — pip works; conda is recommended for HPC.** `pip install hhemt`
installs the package and pins the SWMM engine (`swmm-toolkit` and `pyswmm`)
directly, so a pip environment resolves a stack that passes the toolkit's runtime
validation guard. That guard is a deliberate fail-closed contract: before running
SWMM the toolkit checks the installed engine versions and raises rather than run
against a build it cannot vouch for. Conda is the recommended path for running
simulations because `environment.yaml` pins the whole HPC stack, including the
Snakemake SLURM executor plugins, not because pip cannot run SWMM.
