# Quickstart

Get from a fresh clone to a running coupled TRITON-SWMM simulation in a few minutes, entirely from an interactive Python session.

## Prerequisites

--8<-- "platform-support.md"

- What [Installation](../how-to/installation.md) lists under *Before you start*. Step 1 below performs the install itself.
- `git` and `make` on your `PATH`: the solver build uses both. No HPC and no CLI:
  this tutorial drives everything from an interactive Python session.

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
# hhemt_sha is the 40-hex commit you are running; `git rev-parse HEAD` prints it.
norfolk = NorfolkIreneExperiment.load(hhemt_sha="REPLACE-WITH-THE-FULL-40-HEX-HHEMT-COMMIT")
norfolk.system.compile_TRITON_SWMM()          # once per machine: clones and builds the solver
result = norfolk.analysis.run(from_scratch=False, execution_mode="auto")
```

The Norfolk example ships its compute configuration, and this tutorial runs it unchanged: `run_mode: serial` (one process, one thread, no GPU), `multi_sim_run_method: local` (the ensemble is dispatched to a local thread pool, with no SLURM), and `local_cpu_cores_for_workflow: 4` (how many cores that pool may use). `execution_mode="auto"` reads `multi_sim_run_method`, so `auto` resolves to a local run here, which is why this tutorial needs no HPC.

What each of these calls does, and why the compile line comes before the first `run()` on every machine, is set out step by step in [the Norfolk tutorial's worked path](norfolk-end-to-end.md#the-worked-path-a-local-serial-run).

For user-authored configs instead of the canned example. Your `analysis.yaml` must
carry `hhemt_sha:`, the full 40-hex commit you are running. It is the same value the
snippet above passes, and it is required with no default:

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
