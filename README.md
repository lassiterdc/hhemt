# H&H Ensemble Modeling Toolkit (hhemt)

[![PyPI version](https://img.shields.io/pypi/v/hhemt.svg)](https://pypi.org/project/hhemt/)
[![Documentation Status](https://readthedocs.org/projects/hhemt/badge/?version=latest)](https://hhemt.readthedocs.io/en/latest/?version=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21359151.svg)](https://doi.org/10.5281/zenodo.21359151)

**hhemt** orchestrates coupled TRITON–SWMM flood-ensemble simulations from a
single Python interface, across both a local workstation and HPC (NVIDIA and
AMD GPUs).

- **Coupled TRITON–SWMM flood-ensemble orchestration** over the full
  preprocessing → compile → run → process → consolidate → report lifecycle.
- **Local + HPC execution** (NVIDIA/AMD), driven by a Snakemake workflow with a
  SLURM executor for cluster runs.
- **Interactive analysis report** plus a **portable render-bundle** so results
  travel without the source tree.

* PyPI: https://pypi.org/project/hhemt/
* Documentation: https://hhemt.readthedocs.io
* License: [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)

## Installation

The full toolkit — including SWMM hydrology execution — is validated only against
the conda environment shipped in this repo. This is the supported install:

```bash
conda env create -n hhemt --file environment.yaml
conda activate hhemt
pip install --no-deps "swmmio==0.8.2"
pip install -e . --no-deps
```

Both `--no-deps` steps are required *inside a conda environment*: they stop pip
from displacing conda-resolved packages such as `numpy` and `pandas`. They are
not a SWMM-engine requirement — `pyproject.toml` pins the validated engine
(`swmm-toolkit` 0.15.x + `pyswmm` 2.x) directly, so a plain `pip install hhemt`
resolves a stack that passes the runtime validation guard. Conda is recommended
because `environment.yaml` pins the whole HPC stack, including the Snakemake
SLURM executor plugins, not because pip cannot run SWMM. See
[`docs/how-to/installation.md`](docs/how-to/installation.md) for details.

## Usage

```bash
hhemt --help                                              # every command
hhemt run --system-config CFG_SYS --analysis-config CFG_ANA --dry-run
```

An analysis is driven by two YAML configs — a *system* config describing the
modelled area and an *analysis* config describing the events and how to run them.
`--dry-run` validates both and prints the workflow without executing anything,
which is the cheapest way to check a configuration.

Where to go next, depending on what you are doing:

| You want to | Start at |
|---|---|
| Get something running | [Quickstart](https://hhemt.readthedocs.io/en/latest/tutorials/quickstart/) |
| Follow a complete real example | [Norfolk end-to-end](https://hhemt.readthedocs.io/en/latest/tutorials/norfolk-end-to-end/) |
| Look up a config field | [Configuration schema](https://hhemt.readthedocs.io/en/latest/reference/config-schema/) |
| Look up a command | [CLI reference](https://hhemt.readthedocs.io/en/latest/reference/cli/) |
| Know what the outputs contain | [Output data model](https://hhemt.readthedocs.io/en/latest/reference/output-data-model/) |
| Work out why a run failed | [Diagnosing a failed run](https://hhemt.readthedocs.io/en/latest/how-to/diagnosing-a-failed-run/) |
| Know what it does *not* do | [Limitations](https://hhemt.readthedocs.io/en/latest/explanation/limitations/) |
| Contribute a change | [Contributing](https://hhemt.readthedocs.io/en/latest/contributing/) |

## How to cite

If you use this software, please cite it via its Zenodo DOI. Citation metadata is
maintained in [`CITATION.cff`](CITATION.cff), which GitHub's "Cite this
repository" resolves.

**Cite the version you actually ran, not the badge.** The DOI badge above is the
*concept* DOI — it always resolves to the newest release, so a reader following
it later may land on a version that behaves differently from the one that
produced your results. Zenodo also mints a *version* DOI for each release; that
is the one to put in a paper. Find it on the Zenodo record for your release, or
read it back from your analysis's own provenance: a consolidated output records
the toolkit version that produced it, so the run itself can tell you what to
cite.
