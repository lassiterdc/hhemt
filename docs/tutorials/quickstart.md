# Quickstart

Get from a fresh clone to a running coupled TRITON-SWMM simulation in a few minutes, entirely from an interactive Python session.

## Prerequisites

--8<-- "prerequisites-base.md"
- `conda` (Miniforge or Miniconda), plus `git` and `make` on your `PATH`: the
  solver build uses both. No HPC and no CLI: this tutorial drives everything from
  an interactive Python session.

## 1. Clone and set up the environment

```bash
git clone https://github.com/lassiterdc/hhemt.git
cd hhemt
```

--8<-- "install-commands.md"

Both `--no-deps` flags are required. [Installation](../how-to/installation.md)
carries the reason, the pip-only alternative, and when `PYTHONNOUSERSITE` matters.

## 2. Get the Norfolk example data

The example uses the Norfolk, VA coastal-flooding case study. You do not download it by hand: the data is fetched automatically the first time you call `NorfolkIreneExperiment.load()` in the next step. The public Norfolk case study downloads anonymously, so **no HydroShare account is needed** (the download tries anonymous retrieval first).

## 3. Run from an interactive Python session

```python
from hhemt.experiments import NorfolkIreneExperiment
norfolk = NorfolkIreneExperiment.load()       # anonymous Hydroshare download; builds system + analysis
norfolk.system.compile_TRITON_SWMM()          # once per machine: clones and builds the solver
result = norfolk.analysis.run(from_scratch=False, execution_mode="auto")
```

`NorfolkIreneExperiment.load()` downloads the case data (once), builds the system and analysis objects, and hands you back an experiment whose `.analysis` is the orchestrator. `compile_TRITON_SWMM()` builds the solver the first time and skips the build once it exists. A run checks for that build and never performs it, so this line comes before the first `run()` on every machine. [Compile the solver](../how-to/compiling-the-solver.md) has the cluster form and when to rebuild. `run(from_scratch=False)` resumes any completed work rather than rebuilding from scratch, and `execution_mode="auto"` detects whether you are in a SLURM allocation or on a local machine.

For user-authored configs instead of the canned example:

```python
from hhemt import Toolkit
tk = Toolkit.from_configs("system.yaml", "analysis.yaml")
tk.system.compile_TRITON_SWMM()                # once per machine
result = tk.analysis.run(from_scratch=False)   # tk.analysis.run(...), not tk.run(...)
```

`Toolkit.from_configs(...)` loads your own system and analysis YAML files; call `run()` on its `.analysis` the same way as the canned example.

## Next steps

- [Norfolk end-to-end tutorial](norfolk-end-to-end.md): a full case study across compute configs.
- [Capabilities overview](../explanation/capabilities.md): what the toolkit makes possible.
