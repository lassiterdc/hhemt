# Quickstart

## Prerequisites

- Meet the requirements that [Installation](how-to/installation.md) lists under *Before you start*. Step 1 below performs the install itself.
- Put `git` and `make` on your `PATH`. The solver build uses both.

## 1. Clone and set up the environment

```bash
git clone https://github.com/lassiterdc/hhemt.git
cd hhemt
```

--8<-- "install-commands.md"

Both `--no-deps` flags are required. [Installation](how-to/installation.md)
carries the reason, the pip-only alternative, and when `PYTHONNOUSERSITE` matters.

## 2. Get the Norfolk example data

The example uses the Norfolk, VA coastal-flooding case study. The toolkit fetches the data automatically the first time you call `NorfolkIreneExperiment.load()` in the next step. The Norfolk case-study data is published on HydroShare and downloads anonymously, so **no HydroShare account is needed**.

## 3. Run from an interactive Python session

```python
from hhemt.experiments import NorfolkIreneExperiment
# hhemt_sha is the 40-hex commit you are running; `git rev-parse HEAD` prints it.
norfolk = NorfolkIreneExperiment.load(hhemt_sha="REPLACE-WITH-THE-FULL-40-HEX-HHEMT-COMMIT")
norfolk.system.compile_TRITON_SWMM()          # once per machine: clones and builds the solver
result = norfolk.analysis.run(from_scratch=False, execution_mode="auto")
```

The Norfolk example ships a compute configuration that this page runs unchanged.

- `run_mode: serial` runs one process with one thread and no GPU.
- `multi_sim_run_method: local` runs the simulations locally. The [glossary](reference/glossary.md#multi_sim_run_method) lists the values.
- `local_cpu_cores_for_workflow: 4` sets how many cores the local pool may use.

The argument `execution_mode="auto"` reads `multi_sim_run_method`, so `auto` resolves to local execution here and this page needs no HPC.

Written out, the same compute configuration in the analysis config is:

```yaml
run_mode: serial
multi_sim_run_method: local
local_cpu_cores_for_workflow: 4
```

The [interactive Python tutorial](tutorials/interactive-python.ipynb) explains each call, including why the compile line comes before the first `run()` on every machine.

For configs you author yourself, set `hhemt_sha` in `analysis.yaml` to the value that
the snippet passes. [Config filling](how-to/config-filling.md#4-point-the-analysis-config-at-your-events)
states the value.

```python
from hhemt import Toolkit
tk = Toolkit.from_configs("system.yaml", "analysis.yaml")
tk.system.compile_TRITON_SWMM()                # once per machine
result = tk.analysis.run(from_scratch=False)   # tk.analysis.run(...), not tk.run(...)
```

`Toolkit.from_configs(...)` loads your own system and analysis YAML files. Call `run()` on the `.analysis` attribute the same way as in the canned example.

## Next steps

- [Running a simulation interactively in Python](tutorials/interactive-python.ipynb) reads the results of an analysis into a Python session and plots them.
- [Capabilities](explanation/capabilities.md) describes the supported experiments and hardware.
