# Front-End Validation Checklist (Human-Readable)

## Goal

A practical list of checks the CLI/API should run **before launching expensive
simulation work**, so users catch mistakes early and avoid wasted compute time.

---

## 1) Command + Profile Selection Checks

1. **Profile is valid**
   - `--profile` must be one of: `production`, `testcase`, `case-study`.

2. **Profile-specific selection is present**
   - If `testcase`: a valid `--testcase` must be selected.
   - If `case-study`: a valid `--case-study` must be selected.

3. **No conflicting selector flags**
   - Don’t allow both testcase and case-study selectors together.

4. **No conflicting event selectors**
   - `--event-ilocs` and `--event-range` should be mutually exclusive unless
     explicit merge behavior is implemented.

---

## 2) Config File Integrity Checks

1. **Required config files exist and are readable**
   - system config path
   - analysis config path
   - optional profile catalog (`tests_and_case_studies.yaml`) if provided

2. **YAML structure is parseable**
   - fail fast on syntax errors.

3. **Unknown keys are rejected (strict mode)**
   - prevent typo’d or stale parameters from being silently ignored.

4. **Required fields are present**
   - report exact missing fields with actionable fix text.

---

## 3) System Config Checks (`cfg_system`)

## Core path checks

- `system_directory`, `watershed_gis_polygon`, `DEM_fullres`,
  `SWMM_hydraulics`, `TRITONSWMM_software_directory`,
  `triton_swmm_configuration_template` exist.

## Toggle dependency checks

1. **Manning’s selection dependency**
   - If `toggle_use_constant_mannings = True`: `constant_mannings` required.
   - If False: landuse lookup/raster columns required.

2. **Hydrology dependency**
   - If `toggle_use_swmm_for_hydrology = True`: `SWMM_hydrology`,
     `subcatchment_raingage_mapping`, and mapping column name required.

3. **Standalone SWMM dependency**
   - If `toggle_swmm_model = True`: `SWMM_full` required.

## Toggle exclusion checks (forbid incompatible inputs)

These should usually be **hard errors** (or at minimum strict warnings in a
temporary transition period):

1. If `toggle_use_constant_mannings = True`, disallow landuse-derived manning's
   configuration inputs that would be ignored (e.g., lookup/raster/lookup
   column fields).
2. If `toggle_use_swmm_for_hydrology = False`, disallow hydrology-specific
   inputs (`SWMM_hydrology`, subcatchment↔gage mapping fields) unless explicitly
   marked as retained metadata.
3. If `toggle_swmm_model = False`, disallow standalone SWMM model-specific
   execution inputs that are only meaningful for standalone SWMM runs.

## Model selection sanity

- At least one model should be enabled (`triton`, `tritonswmm`, or `swmm`) for run.

---

## 4) Analysis Config Checks (`cfg_analysis`)

## Core integrity

1. `analysis_id` format valid (safe naming).
2. `weather_timeseries` exists and can be opened.
3. `weather_event_summary_csv` exists when required by workflow.
4. `weather_event_indices` fields are valid and non-empty.

## Run-mode consistency checks

1. **serial**
   - MPI = 1/None, OMP = 1/None, GPUs = 0/None, nodes = 1/None.

2. **openmp**
   - MPI = 1/None, OMP > 1, GPUs = 0/None, nodes = 1/None.

3. **mpi**
   - MPI > 1, OMP = 1/None, GPUs = 0/None, and `n_mpi_procs >= n_nodes`.

4. **hybrid**
   - MPI > 1, OMP > 1, GPUs = 0/None, and `n_mpi_procs >= n_nodes`.

5. **gpu**
   - `n_gpus >= 1`; if MPI used with multi-node, enforce
     `n_mpi_procs >= n_nodes`.

## Analysis toggle dependencies

1. If `toggle_sensitivity_analysis = True`: `sensitivity_analysis` file required.
2. If `toggle_storm_tide_boundary = True`: boundary GIS + data variable + units required.
3. If `toggle_sensitivity_analysis = True`: disallow unsupported multi-model sensitivity combinations for now.
   - Treat this as a **hard error** during preflight when sensitivity mode includes model combinations known to share/overwrite run artifacts.
   - **# TODO (remove later):** remove this restriction once multi-model sensitivity analysis is fully supported and TRITON/TRITON-SWMM outputs are fully independent (no shared/overwritten log artifacts).

## Analysis toggle exclusion checks (forbid incompatible inputs)

1. If `toggle_sensitivity_analysis = False`, disallow sensitivity-design-only
   inputs that would never be consumed during execution.
2. If `toggle_storm_tide_boundary = False`, disallow storm-tide-only fields
   (`storm_tide_boundary_line_gis`, storm tide variable, storm tide units).
3. If `run_mode != gpu`, disallow GPU-only allocation fields where applicable.
4. If `multi_sim_run_method = local`, disallow scheduler-only settings that
   imply SLURM submission behavior (unless explicitly allowed as inert metadata).
5. If `multi_sim_run_method` is scheduler-based, disallow local-only knobs that
   would not affect execution.

---

## 5) Multi-Simulation + HPC Checks

## `multi_sim_run_method` compatibility

1. **local mode**
   - validate local core limits and avoid oversubscription.

2. **batch_job mode**
   - require account/partition/time settings needed for scheduler submission.

3. **1_job_many_srun_tasks mode**
   - require `hpc_total_nodes`, `hpc_total_job_duration_min`, and resource
     descriptors needed for planning task concurrency.

## HPC sanity checks

- all count fields must be positive integers.
- walltime/duration fields valid and non-zero.
- if GPU mode, ensure relevant GPU scheduler fields are set (where required).

---

## 6) Testcase / Case-Study Inheritance Checks

1. Selected testcase/case-study exists in catalog.
2. Profile entry resolves required config paths.
3. Merge precedence applies correctly:
   - CLI > selected profile > profile defaults > analysis > system > internal defaults.
4. Null/empty profile values do not erase valid lower-priority values.
5. Final resolved HPC settings are printed before execution.

---

## 7) Data Cross-Consistency Checks

1. Event identifiers align between weather time series and event summary/index files.
2. Storm tide variable exists in weather dataset when storm tide toggle is enabled.
3. Units are explicit and valid (`rainfall_units`, storm tide units).
4. Expected index columns exist in CSV inputs.

---

## 8) Environment + Tooling Checks

1. Required executables and software directories are present.
2. If redownload/update requested, verify success before continuing.
3. Validate run-mode compatibility with compiled backends (CPU/GPU backend expectations).
4. If HPC module loading is configured, validate module string format and non-empty tokens.

---

## 9) Output Safety + Resume Behavior Checks

1. If `--from-scratch`, clearly warn about deletions and scope.
2. If `--overwrite`, confirm output replacement policy.
3. If `--resume`, verify checkpoint/log state is coherent.
4. Verify output directories are writable before run.

---

## 10) Preflight UX Requirements

Before any compute launch, print a short **Resolved Run Plan**:

- profile + selected testcase/case-study (if any)
- resolved model mode and processing scope
- resolved HPC/runtime settings
- selected event subset
- overwrite/resume/from-scratch behavior

This acts as a user confirmation surface and catches accidental misconfiguration.

---

## Suggested Error Message Style

Use consistent, actionable messages:

- **What failed**: `analysis.run_mode`
- **Current value**: `serial`
- **Why invalid**: `n_omp_threads=8 is not allowed for serial mode`
- **How to fix**: `Set n_omp_threads to 1, or change run_mode to openmp`

For exclusion violations, use explicit wording:

- **What failed**: `system.toggle_use_constant_mannings`
- **Current value**: `True`
- **Invalid extra input**: `landuse_lookup_file=/path/...`
- **Why invalid**: `landuse-derived mannings inputs are not used when constant mannings is enabled`
- **How to fix**: `Remove landuse_lookup_file or set toggle_use_constant_mannings=False`

---

## Implementation Hint

Treat this checklist as two tiers:

1. **Hard errors** (must fail before run)
2. **Warnings** (allowed to continue, but prominently shown)

When possible, run all checks and return a single consolidated report so users
can fix everything in one pass.
