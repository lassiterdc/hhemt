# Release history

## v0.1.0 (2026-XX-XX)

First public release of the H&H Ensemble Modeling Toolkit (hhemt). Coupled
TRITON-SWMM flood-ensemble orchestration across local + HPC (NVIDIA/AMD), full
preprocessing→compile→run→process→consolidate→report lifecycle, interactive
analysis report, and a portable render-bundle. Docs at
https://hhemt.readthedocs.io. Cite via the Zenodo DOI (see README).

**Installation — pip installs the toolkit, conda runs SWMM.** `pip install hhemt`
installs the Python package and its pure-Python dependencies, but a plain pip
environment **cannot execute SWMM**: the pinned SWMM runoff engine is provisioned
only through the conda environment (`environment.yaml`). This is a deliberate
fail-closed dependency contract — a bare pip install raises a clear error at SWMM
execution time rather than silently running against an untested engine build.
Conda is therefore the supported path for running simulations.
