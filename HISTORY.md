# Release history

## v0.1.0 (2026-07-14)

First public release of the H&H Ensemble Modeling Toolkit (hhemt). Coupled
TRITON-SWMM flood-ensemble orchestration across local + HPC (NVIDIA/AMD), full
preprocessing→compile→run→process→consolidate→report lifecycle, interactive
analysis report, and a portable render-bundle. Docs at
https://hhemt.readthedocs.io. Cite via the Zenodo DOI (see README).

<!-- hhemt:maturity-disclosure -->
> **A note on this entry, added after the fact.** This release was made to build and
> prove the machinery a public release requires. Its author did not consider the
> software ready for general outside use at the time, and the release intended for
> general use is planned alongside the paper introducing the software.
<!-- /hhemt:maturity-disclosure -->

**Installation — pip works; conda is recommended for HPC.** `pip install hhemt`
installs the package and pins the SWMM engine (`swmm-toolkit` and `pyswmm`)
directly, so a pip environment resolves a stack that passes the toolkit's runtime
validation guard. That guard is a deliberate fail-closed contract: before running
SWMM the toolkit checks the installed engine versions and raises rather than run
against a build it cannot vouch for. Conda is the recommended path for running
simulations because `environment.yaml` pins the whole HPC stack, including the
Snakemake SLURM executor plugins, not because pip cannot run SWMM.
