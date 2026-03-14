---
impact: Medium
urgency: High
loe: Low
risk: Low
priority: 2.55
priority-label: "Do now"
created: 2026-03-14
description: VS Code Remote-SSH for terminal access, globus-sdk Python transfer module with Analysis API methods, Pydantic-validated transfer configs, and two project skills for setup and runtime transfer triggering.
---

<!-- Written: 2026-03-14 | Last edited: 2026-03-14 — plan sync 2: branch corrected to debug_full_scale_testing; updated SE quick win line refs for this branch; added QW-5 through QW-8 from second SE review; added new RC candidates to SE Appendix; added DoD items for follow-on idea capture and new file corrections -->

# Streamlined HPC Workflow

## Task Understanding

### Requirements

1. Replace MobaXterm with VS Code Remote-SSH for SSH terminal access to UVA Rivanna/Afton and Frontier.
2. Replace Globus GUI transfers with a Python script driven by Pydantic-validated YAML transfer specs in `configs/transfers/`.
3. Add `TRITONSWMM_analysis.globus_to_local()` — transfers results from HPC to local Ubuntu desktop, excluding raw sim output by default, with configurable inclusion.
4. Add `TRITONSWMM_analysis.globus_to_hpc()` — uploads input files defined in configs to a structured HPC destination path.
5. Globus transfers must include hidden files (dotfiles, `.snakemake/`).
6. Endpoint UUIDs and site base paths live in `constants.py` (not hardcoded in YAML).
7. Two project-scoped skills: `/setup-hpc-integration` (onboarding) and `/fetch-hpc-results` (runtime transfer trigger).
8. Install Globus Connect Personal on the Ubuntu desktop (current local machine for analysis work).

### Assumptions

- User has `globus-sdk` installable into the `triton_swmm_toolkit` conda environment.
- User's Ubuntu desktop will have Globus Connect Personal installed during setup.
- UVA Globus collection: "UVA Standard Security Storage" (confirmed).
- Frontier Globus collection: exact UUID must be verified at app.globus.org by searching "OLCF" — do not hardcode without confirming.
- `globus login` OAuth browser flow stores tokens persistently; scripts do not require re-login.
- `TransferData.filter_rules` is the SDK mechanism for excluding subdirectories (verify exact schema in `globus-sdk` docs before implementing exclusion logic).

### Success Criteria

- VS Code connects to `uva-rivanna` and `frontier` via Remote-SSH with `ControlMaster` (DUO only on first connection per session).
- `analysis.globus_to_local()` submits a single Globus task containing all result paths and blocks until completion.
- `analysis.globus_to_hpc()` uploads all config-defined input files to the HPC destination.
- `/setup-hpc-integration` walks a new-machine setup start to finish and populates `configs/transfers/` YAML.
- `/fetch-hpc-results` triggers `globus_to_local()` with transfer preview and user confirmation.
- Transfer configs load without error when valid; raise `ConfigurationError` on invalid fields.

---

## Evidence from Codebase

- `src/TRITON_SWMM_toolkit/constants.py` — already holds `FRONTIER_DEFAULT_PLATFORM_CONFIG` and `UVA_DEFAULT_PLATFORM_CONFIG` via `PlatformConfig` from `platform_configs.py`. Globus endpoint constants fit naturally here alongside existing HPC constants. UVA scratch path pattern: `/scratch/<username>/`. Frontier scratch path: `/lustre/orion/<project>/scratch/<username>/`.
- `src/TRITON_SWMM_toolkit/platform_configs.py` — `PlatformConfig` dataclass holds per-system HPC settings. Globus endpoint UUIDs are site-level constants (not per-analysis) — they belong in `constants.py` directly, not in `PlatformConfig`.
- `src/TRITON_SWMM_toolkit/analysis.py` — `TRITONSWMM_analysis` class is the right public API surface for `globus_to_local()` / `globus_to_hpc()` as thin wrappers. Analysis already has `analysis_paths.analysis_dir` (HPC source) and `cfg_analysis` (config-defined paths).
- `src/TRITON_SWMM_toolkit/paths.py` — `AnalysisPaths.analysis_dir` and `AnalysisPaths.simulation_directory` are the key paths for transfer source. `simulation_directory` is the directory to exclude by default in `globus_to_local()`.
- `src/TRITON_SWMM_toolkit/config/` — Globus config models must use plain `pydantic.BaseModel` with `ConfigDict(extra="forbid")` — NOT `cfgBaseModel`. `cfgBaseModel` has a `@field_validator("*")` that calls `.exists()` on every `Path` field; HPC paths like `/scratch/***REMOVED***/...` do not exist locally and would cause `ValueError` at load time.
- `src/TRITON_SWMM_toolkit/exceptions.py` — `ConfigurationError(field, message, config_path)` is the correct exception for invalid transfer config values.
- No existing Globus code anywhere in `src/`.
- `configs/` directory does not yet exist in the repo root.

---

## Implementation Strategy

### Chosen approach

New `src/TRITON_SWMM_toolkit/globus_transfer.py` module holds all Globus SDK logic. `TRITONSWMM_analysis` gets two thin wrapper methods that call into it. Transfer specs are Pydantic-validated at load time. Endpoint constants live in `constants.py`.

Steps:
1. Add Globus endpoint constants to `constants.py`.
2. Add `GlobusTransferSpec` and `GlobusConfig` Pydantic models to `config/` (or a new `config/globus.py`).
3. Add transfer YAML loader to `config/loaders.py`.
4. Create `configs/transfers/` directory with a template YAML and a `.gitignore` to exclude populated UUIDs.
5. Create `src/TRITON_SWMM_toolkit/globus_transfer.py` with `GlobusTransferManager` class.
6. Add `globus_to_local()` and `globus_to_hpc()` thin wrappers to `TRITONSWMM_analysis`.
7. Add `globus-sdk` to `pyproject.toml` and `workflow/envs/triton_swmm.yaml`.
8. Create two project-scoped skills in `$AGENTIC_WORKSPACE/prompts/workspaces/projects/TRITON-SWMM_toolkit/skills/` and wire via `/create-skill`.
9. Add SSH config template to `configs/ssh/README.md`.

### Alternatives considered

- Logic directly in `analysis.py` — rejected. Globus SDK imports and transfer logic would bloat the analysis module and couple a network dependency to every `analysis.py` import.
- `globus-cli` shell commands instead of `globus-sdk` — rejected. N paths would require N separate Globus tasks or a batch file; no structured `task_wait()`; fragile stdout parsing for error handling.

### Trade-offs

- `globus-sdk` adds a new runtime dependency. It is pip-installable and widely used; risk is low.
- Storing endpoint UUIDs in `constants.py` means they are repo-tracked. These are not secrets (Globus endpoint UUIDs are public identifiers). Acceptable.
- Transfer YAML files in `configs/transfers/` contain user-specific HPC paths. A `.gitignore` entry prevents accidental commit of populated configs; the template is committed.

---

## File-by-File Change Plan

### New files

| File | Purpose |
|------|---------|
| `src/TRITON_SWMM_toolkit/config/globus.py` | `GlobusEndpoints` (site constants container), `GlobusTransferItem` (one source+dest pair), `GlobusTransferSpec` (full transfer YAML schema) — plain `BaseModel` with `ConfigDict(extra="forbid")`, not `cfgBaseModel` |
| `src/TRITON_SWMM_toolkit/globus_transfer.py` | `GlobusTransferManager` class: `transfer_to_local()`, `transfer_to_hpc()`, hidden-file inclusion, `filter_rules` for sim output exclusion, `task_wait()` polling |
| `configs/transfers/template_transfer.yaml` | Template transfer spec with placeholder UUIDs and example paths; committed to repo |
| `configs/transfers/.gitignore` | Ignores `*.yaml` except `template_transfer.yaml` to prevent committing populated configs with real paths |
| `$AGENTIC_WORKSPACE/prompts/workspaces/projects/TRITON-SWMM_toolkit/skills/setup-hpc-integration/SKILL.md` | Project-scoped skill: walks SSH config setup, Globus Connect Personal install on Ubuntu desktop, endpoint UUID discovery, YAML population |
| `$AGENTIC_WORKSPACE/prompts/workspaces/projects/TRITON-SWMM_toolkit/skills/fetch-hpc-results/SKILL.md` | Project-scoped skill: reads transfer config, previews pending transfers, calls `globus_to_local()` with confirmation |

### Modified files

| File | Change |
|------|--------|
| `src/TRITON_SWMM_toolkit/constants.py` | Add `UVA_GLOBUS_COLLECTION_NAME`, `UVA_GLOBUS_COLLECTION_UUID`, `UVA_GLOBUS_SCRATCH_BASE`, `FRONTIER_GLOBUS_COLLECTION_NAME`, `FRONTIER_GLOBUS_COLLECTION_UUID`, `FRONTIER_GLOBUS_SCRATCH_BASE`, `FRONTIER_GLOBUS_PROJECT_BASE` constants. UUIDs: UVA confirmed (`e6b338df-...`), Frontier left as `None` placeholder. **SE quick wins**: remove dead `# from pathlib import Path` (line 1), `# TESTING` (line 7), and `# NORFOLK_sensitivity_EXP_CONFIG` (line 11). |
| `src/TRITON_SWMM_toolkit/config/loaders.py` | Add `load_transfer_config(path: Path) -> GlobusTransferSpec` function. **SE quick win**: add return type annotations to the three existing loaders; annotate `cfg_dict: dict` param on `load_system_config_from_dict`. |
| `src/TRITON_SWMM_toolkit/config/__init__.py` | **SE quick win**: add `globus` to the `Submodules:` docstring list. (Full export wiring via `load_transfer_config` in loaders; models importable directly from `config.globus`.) |
| `src/TRITON_SWMM_toolkit/analysis.py` | Add `globus_to_local(transfer_yaml: Path) -> str` and `globus_to_hpc(transfer_yaml: Path) -> str` as lazy-import wrappers. **SE quick wins**: (1) fix `print_cfg` line 387: `if which == ["system", "both"]:` → `if which in ["system", "both"]:`, (2) add parens for operator precedence clarity at line 2169: `(self.cfg_analysis.n_gpus or 0) if self.cfg_analysis.run_mode == "gpu" else 0`, (3) update `run()` docstring to match actual signature (`from_scratch: bool`; remove references to `mode`/`phases`). |
| `pyproject.toml` | Add `globus-sdk>=3.0` to dependencies. **SE quick win**: remove hardcoded Windows `pythonCommand` path (line 62 — `C:\\Users\\Daniel\\...`). |
| `workflow/envs/triton_swmm.yaml` | Add `globus-sdk>=3.0` under `pip:` section. |
| `configs/ssh/README.md` | **Correction**: change `login.hpc.virginia.edu` to `login1.hpc.virginia.edu` to match `hpc_login_node` in `UVA_DEFAULT_PLATFORM_CONFIG` on this branch. |

### Import sites

- `analysis.py` imports `GlobusTransferManager` from `globus_transfer` — lazy import inside method body to avoid coupling Globus to the analysis import path when not needed.
- `config/__init__.py` exports the new Globus config models alongside existing exports.

---

## Risks and Edge Cases

- **Frontier Globus UUID unconfirmed**: The exact OLCF Globus collection UUID must be verified at app.globus.org before the transfer config can be populated. Leave as `None` in `constants.py` with a comment. The setup skill should guide this discovery step explicitly.
- **Local endpoint on Ubuntu desktop**: Globus Connect Personal must be installed on the Ubuntu desktop. The setup skill must cover this step. Without a local endpoint, `globus_to_local()` has no destination.
- **`filter_rules` schema**: The `globus-sdk` `TransferData.filter_rules` exact parameter schema is not confirmed from the specialist research. Read the SDK docs at implementation time before writing the exclusion logic. Fallback: pass a list of `--exclude` patterns via `TransferData`'s documented filter mechanism.
- **UVA `ControlPersist` idle timeout**: UVA HPC may disconnect idle `ControlMaster` sockets. Add `ServerAliveInterval 60` and `ServerAliveCountMax 3` to the SSH config block to keep connections alive during long VS Code sessions.
- **Hidden files confirmed included**: `globus transfer --recursive` includes dotfiles by default (confirmed). No special flag needed. Document this explicitly in `GlobusTransferManager` docstring.
- **Sim output exclusion**: `AnalysisPaths.simulation_directory` is the directory to exclude in `globus_to_local()` when `include_sim_output=False`. Pass its name as a `filter_rules` exclusion.
- **`_GLOBUS_CLIENT_ID` placeholder**: `globus_transfer.py` contains a placeholder client ID (`61338d24-...`). A real Globus native app must be registered at developers.globus.org. The `/setup-hpc-integration` skill Step 3 guides this. Until registered, `GlobusTransferManager()` will fail at the OAuth2 step. Replace the placeholder with the real client ID after registration.
- **`globus_sdk.tokenstorage` API**: `SimpleJSONFileAdapter` API may vary between globus-sdk versions. Verify against installed version at implementation time; update if needed.
- **`config/globus.py` unused `Path` import**: The `from pathlib import Path` import in `config/globus.py` is unused (all paths stored as `str`). Remove at implementation time.

---

## Validation Plan

All transfer operations require a live Globus connection and cannot be unit-tested without mocking. Validation is empirical.

### Local (non-HPC)

```bash
# 1. Install globus-sdk
conda run -n triton_swmm_toolkit pip install globus-sdk

# 2. Verify import
conda run -n triton_swmm_toolkit python -c "import globus_sdk; print(globus_sdk.__version__)"

# 3. Verify Pydantic models load correctly
conda run -n triton_swmm_toolkit python -c "
from TRITON_SWMM_toolkit.config.globus import GlobusTransferSpec
from TRITON_SWMM_toolkit.config.loaders import load_transfer_config
from pathlib import Path
spec = load_transfer_config(Path('configs/transfers/template_transfer.yaml'))
print(spec)
"

# 4. Verify analysis methods exist
conda run -n triton_swmm_toolkit python -c "
from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
print(hasattr(TRITONSWMM_analysis, 'globus_to_local'))
print(hasattr(TRITONSWMM_analysis, 'globus_to_hpc'))
"
```

### Empirical HPC testing (paste results back)

```bash
# 5. Confirm Globus CLI login (Ubuntu desktop)
globus whoami
# Expected: your Globus identity email
```

```
[paste output here]
```

```bash
# 6. Confirm local endpoint UUID
globus endpoint local-id
# Expected: UUID of Globus Connect Personal on Ubuntu desktop
```

```
[paste output here]
```

```bash
# 7. Test a small transfer (single file) to confirm auth and path resolution
# Fill in real UUIDs from app.globus.org before running
conda run -n triton_swmm_toolkit python -c "
import globus_sdk
client = globus_sdk.NativeAppAuthClient('<client_id>')
# ... (see GlobusTransferManager for full auth flow)
"
```

---

## Documentation and Tracker Updates

- `configs/ssh/README.md` — document the SSH config block (`ControlMaster`, `ControlPersist`, `ServerAliveInterval`).
- `architecture.md` (agentic workspace) — per SE specialist recommendations (see Appendix): add `globus_transfer.py` to Key Modules; add `cfgBaseModel` path-validation gotcha under Configuration System; add lazy-import guidance under Code Style; add Globus constants guidance under System-Specific Constants.
- `constants.py` — add inline comments explaining that Globus UUIDs must be populated manually after endpoint discovery.

---

## Decisions Needed from User

All resolved prior to writing this plan:
- `globus-sdk` over `globus-cli` shell commands — **resolved: SDK**
- Logic in `globus_transfer.py` module, thin wrappers on `analysis` — **resolved: separate module**
- Pydantic validation for transfer configs — **resolved: yes**
- Local machine is Ubuntu desktop (not laptop) — **resolved**
- `configs/transfers/` location — **resolved**
- Skills are project-scoped (not agentic workspace) — **resolved**

**One open item** (low risk, resolved at implementation time):
- Frontier Globus collection UUID — must be confirmed at app.globus.org. Leave as `None` placeholder in `constants.py`.

---

## Definition of Done

- [ ] `globus-sdk` added to `pyproject.toml` and `workflow/envs/triton_swmm.yaml`
- [ ] `src/TRITON_SWMM_toolkit/config/globus.py` created with `GlobusEndpoints`, `GlobusTransferItem`, `GlobusTransferSpec` Pydantic models
- [ ] `load_transfer_config()` added to `config/loaders.py`; new models exported from `config/__init__.py`
- [ ] Globus endpoint constants added to `constants.py` (UUIDs as `None` placeholder with comments)
- [ ] `src/TRITON_SWMM_toolkit/globus_transfer.py` created with `GlobusTransferManager`
- [ ] `TRITONSWMM_analysis.globus_to_local()` and `globus_to_hpc()` added as lazy-import wrappers
- [ ] `configs/transfers/template_transfer.yaml` and `configs/transfers/.gitignore` committed
- [ ] `skills/setup-hpc-integration/SKILL.md` and `skills/fetch-hpc-results/SKILL.md` created
- [ ] SSH config template documented in `configs/ssh/README.md` (ControlMaster, ServerAliveInterval)
- [ ] SE quick wins applied (from-main): `print_cfg` bug fixed (`analysis.py:387`), dead comments removed (`constants.py` lines 1 and 7), return type annotations added (`loaders.py`), `globus` added to `config/__init__` docstring
- [ ] SE quick wins applied (branch-new): dead comment removed (`constants.py:11`), operator precedence parens added (`analysis.py:2169`), `run()` docstring updated to match actual signature (`analysis.py:~1307`), Windows `pythonCommand` path removed (`pyproject.toml:62`)
- [ ] `config/globus.py`: unused `Path` import removed
- [ ] `configs/ssh/README.md`: UVA hostname corrected to `login1.hpc.virginia.edu`
- [ ] `globus_transfer.py`: `_GLOBUS_CLIENT_ID` documented as placeholder requiring registration at developers.globus.org
- [ ] Local smoke tests pass (import, Pydantic load, method existence checks)
- [ ] Empirical transfer test completed on Ubuntu desktop with real endpoint UUIDs
- [ ] `architecture.md` updated to include `globus_transfer.py` in Key Modules table
- [ ] Update the workspace's architecture instruction file if module structure changed
- [ ] If any performance or memory risks were surfaced, entries added to `docs/planning/tech_debt_known_risks.md`
- [ ] **Follow-on ideas evaluated and captured** — for each RC below, decide: create idea file, or skip (not relevant). Create idea files for those that are relevant:
  - RC-1: Extract `_update_log` → `LogStatusSnapshot` dataclass
  - RC-2: Extract `df_status` → `StatusTableBuilder`
  - RC-3: Generic `_load_config(path, model_cls)` helper in `loaders.py`
  - RC-4: Sensitivity-analysis delegation pattern → Strategy pattern
  - RC-5: `run()` method parameter drift cleanup
  - RC-6: `pyproject.toml` / `triton_swmm.yaml` dependency alignment
- [ ] When a plan moves to `completed/`, update any active plan `dependencies:` entries that reference the old path
- [ ] Before moving to `completed/`, run the pre-completion accuracy check from `prompts/instructions/protocols/plan-accuracy-gate.md`
- [ ] Set `completed: true` in this plan's YAML frontmatter, then move to `docs/planning/features/completed/` and run `scripts/generate_planning_tables.py --planning-dir docs/planning`
- [ ] Copy originating idea file verbatim into `## Appendix: Originating Idea`, then delete `docs/planning/ideas/streamlined_hpc_workflow.md` and re-run `scripts/generate_planning_tables.py`

---

## Appendix: SE Specialist Report (2026-03-14)

Full output from `software-engineering-specialist` invoked during preflight for this plan.

### In-scope quick wins (applied in this implementation)

**1. Bug — `print_cfg` comparison operator** (`analysis.py:387` on `debug_full_scale_testing`)
`if which == ["system", "both"]:` compares a string to a list — always `False`. Change to `if which in ["system", "both"]:`. Zero behavior change for callers passing `"both"` or `"analysis"`.

**2. Return type annotations on existing loaders** (`config/loaders.py:7-21`)
The three existing loader functions lack return annotations. Add `-> system_config` / `-> analysis_config` to match the new `load_transfer_config() -> GlobusTransferSpec` and make the module internally consistent.

**3. Dead comments in `constants.py`** (lines 1, 7)
Line 1: `# from pathlib import Path` (commented-out import). Line 7: `# TESTING` section comment is misleading — the constants below it are not test-only. Remove both.

**4. Docstring currency in `config/__init__.py`** (lines 1-8)
Add `globus` to the `Submodules:` list in the existing docstring when the new module is added.

### Out-of-scope refactor candidates (to be captured as follow-on ideas)

**RC-1. Extract `_update_log` status aggregation** (`analysis.py:~478-576`)
~100 lines of nested boolean accumulators across six status dimensions, duplicated between sensitivity and non-sensitivity branches. A `LogStatusSnapshot` dataclass returned by each branch would reduce the method to ~15 lines and eliminate the flag-accumulator pattern.

**RC-2. Extract `df_status` into a builder** (`analysis.py:~2100-2231` on debug branch)
130+ line property mixing data assembly, I/O (log file parsing), and validation across three model types and two execution paths. A `StatusTableBuilder` would make it independently testable and reduce the analysis class surface.

**RC-3. Generic loader pattern in `loaders.py`**
Four loaders follow an identical pattern: read YAML, call `model_validate`. A private `_load_config(path, model_cls)` helper with typed public wrappers would make future config types a one-liner and eliminate the repetition.

**RC-4. Sensitivity-analysis delegation pattern** (`analysis.py`)
~10 properties repeat `if toggle_sensitivity_analysis: return self.sensitivity.X` pattern. A Strategy pattern or delegator helper would eliminate the repeated conditional dispatch and make each provider independently testable.

**RC-5. `run()` method parameter drift** (`analysis.py:~1307`)
Commented-out parameters, hardcoded TODOs overriding user input (`translate_mode("resume")` ignoring `from_scratch`), and stale docstring. Needs a focused cleanup pass to finalize the public API, remove dead parameters, and rewrite the docstring.

**RC-6. `pyproject.toml` / `triton_swmm.yaml` dependency alignment**
Conda environment includes many runtime deps (`typer`, `pydantic`, `pyyaml`, `zarr`, `scipy`, etc.) not in `pyproject.toml`. `pip install` of the package will fail without these. Alignment pass needed.

### Architecture doc recommendations (applied post-implementation)

1. Add `globus_transfer.py` to Key Modules table.
2. Add `cfgBaseModel` path-validation note under Configuration System: *"Any config model containing remote/HPC paths must use plain `BaseModel` with `ConfigDict(extra='forbid')` — `cfgBaseModel` validates all `Path` fields exist on disk at instantiation time."*
3. Add lazy-import guidance under Code Style: *"Use lazy imports (inside method bodies) for optional dependencies not always needed, to avoid coupling them to the module's import path (e.g., `globus_sdk` imported inside `globus_to_local()`, not at top of `analysis.py`)."*
4. Add Globus constants guidance under System-Specific Constants: *"Globus endpoint UUIDs and site-level scratch base paths live in `constants.py`. Per-user paths (usernames, project codes) belong in `configs/transfers/` YAML files, not in `constants.py`."*

### SE specialist integration — design note

The SE specialist review should be an instructional surface, not just a code review. When codifying this into the `/plan-implementation` and `/proceed-with-implementation` workflows, the SE specialist output should include brief explanations of *why* each pattern matters — vocabulary, design philosophy, and best-practice rationale — so the developer learns through implementation rather than just receiving a list of changes. Keep explanations concise (2-4 sentences) to preserve flow.

---

## Appendix: Originating Idea

```
---
impact: Medium
urgency: Medium
loe: Low
risk: Low
priority: 2.02
priority-label: "Core priority"
created: 2026-03-14
description: Replace MobaXterm+Globus GUI context-switching with VS Code Remote-SSH and a Globus-integrated Analysis API (globus_to_local / globus_to_hpc), with project skills for setup and runtime transfer triggering.
---

# Streamlined HPC Workflow

## Problem

Running experiments on UVA and Frontier HPC systems requires constant context-switching
between three separate applications: MobaXterm for SSH terminal access, the Globus GUI for
transferring results locally, and VS Code for debugging. This friction slows the iteration
loop between experiment execution and local analysis, and makes it hard to manage transfers
across multiple concurrent experiments. In real-world (non-case-study) applications, users
also need a structured way to push local input data to HPC before runs.

## Approach Notes

### Part 0: Setup skill (`/setup-hpc-integration`)
Use `/create-skill` to build a `setup-hpc-integration` skill that walks through the full
integration setup interactively — SSH config, Globus CLI install, endpoint UUID discovery,
and `configs/transfers/` YAML population. The skill records each confirmed value as it goes,
so setup is reproducible on a new machine or when onboarding a new HPC system.

### Part 1: VS Code Remote-SSH (replaces MobaXterm)
SSH config lives in `~/.ssh/config`. Note: UVA may require DUO/MFA — verify ControlMaster.

### Part 2: Transfer config model + constants
Transfer specs live in `configs/transfers/` as durable YAMLs. Endpoint UUIDs stored in
constants.py — YAML configs reference constant names rather than raw UUID strings.

### Part 3: `analysis.globus_to_local()`
Default: transfers everything except raw simulation output folder. Configurable.
Must include hidden files. Derives source from analysis_config/system_config.

### Part 4: `analysis.globus_to_hpc()`
Uploads input files defined in configs to structured HPC destination.
Real-world use case: push locally-processed DEM/Manning's/rainfall to HPC before a run.

### Part 5: Skill wrapper (`/fetch-hpc-results`)
Both skills project-scoped — written to this repo, not the agentic workspace.

### Open questions (resolved during planning)
- globus-sdk over globus-cli: resolved — SDK
- GlobusTransferManager in separate module: resolved — yes
- Pydantic validation: resolved — yes
- Local endpoint: confirmed exists on laptop; needs install on Ubuntu desktop
```
