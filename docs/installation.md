# Installation

## Create environment

```bash
conda create -n TRITON_SWMM_toolkit python=3.11
conda activate TRITON_SWMM_toolkit
```

## Install

```bash
pip install -e ".[docs]"
```

!!! note
    The `[docs]` extra installs MkDocs and mkdocstrings for building documentation locally.
    Omit it for a minimal install: `pip install -e .`
