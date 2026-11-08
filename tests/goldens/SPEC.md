# Golden specification

Specifies the two committed golden trees: the layers they carry (section 1) and how far their dimensions shrink (section 2), the directory layout (3), the completion records (4), the configuration and input files and path relocation (5), the minimal software root (6), the raw layer (7), the summary layer and consolidated tiers (8), what a golden does not carry (9), and the open items and the rule for changing this file (10). It also records how the generator stays independent of hhemt (the paragraph below and `tests/test_golden_generator_independence.py`).

This file is the only specification the golden generator under `tests/goldens/generator/` follows, besides the published documents it names. The generator imports the Python standard library, `numpy`, `xarray` and `zarr`, and nothing from `hhemt` or from the rest of `tests/`.

## 0. How to read this file

Most statements carry one tag, so a later reader can tell a published rule from this file's own choice. A sentence with no tag records none. Only the heading of section 3 states a default (`[SRC]` unless tagged). The headings of sections 2 and 5.2 carry a bare `[CHOICE]`, which marks the choices the section makes and does not say that it covers the section's other sentences, each of which carries its own tag where it has one. The untagged sentences of sections 1, 4, 6, 7, 8 and 9 are covered by no default.

| Tag | Meaning |
|---|---|
| `[DOC]` | Stated by a published document, named with its path or URL. The generator follows it. |
| `[SRC]` | Read from a source file (hhemt or a solver) in the worktree at `b7e963b2`, named as `path::symbol`. It states what a consumer reads or what a writer writes; it is not a published specification. |
| `[OBS]` | Observed in a finished synthetic tree under `~/.cache/hhemt/synthetic_test_runs/docs-and-experiment-system/`. Those trees carry a layout stamp of 22 and the worktree was at layout version 26 when this file was written and is at 27 since the consolidated-metadata index was added (`src/hhemt/version_migration/constants.py::LAYOUT_VERSION`), so `[OBS]` is a baseline and never the final word. |
| `[CHOICE]` | A value or layout this file picks because no document fixes it. Free to change only under the rule in section 10. |
| `[OPEN]` | Not determined here. Section 10 names who decides it. |

The independence rule has two residual limits that apply to this whole file, repeated here so nobody prices them as closed. First, an import ban cannot see copying, and this file was written by a party that read hhemt's source, so every `[SRC]` statement is hhemt's current behaviour stated as a requirement. Second, a conformance failure is never cured by bending the golden to hhemt's reader: any generator or `SPEC.md` edit made after `test_golden_conformance.py` first reds is recorded with the published line it follows, in the change log at the end of section 10.

## 1. What a golden is

A golden is one committed zip file holding a finished analysis tree and the files hhemt needs to construct an analysis object from it. There are two:

| Golden | Zip | Contents |
|---|---|---|
| multi-sim | `tests/fixtures/goldens/multisim.zip` | one regular analysis with 3 events and the three model arms (`triton`, `tritonswmm`, `swmm`) |
| sensitivity | `tests/fixtures/goldens/sensitivity.zip` | one sensitivity experiment with 4 members and 1 event, and ONE model arm, the coupled `tritonswmm` |

The zip root is the system directory. Both goldens carry the raw layer, the summary layer, the completion records and the consolidated tiers, so that `tests/test_golden_conformance.py` can corrupt each layer in turn and check that the reader of that layer rejects it. The sensitivity golden carries the coupled `tritonswmm` arm only, and the multi-sim golden carries all three. Ground: hhemt refuses to plan a sensitivity analysis with more than one enabled model type (`src/hhemt/workflow.py::generate_master_snakefile_content`, "Sensitivity analysis does not support multi-model execution"), and the observed sensitivity tree carried `tritonswmm` only (`[OBS]` `.../synth_sensitivity/synth_sensitivity/members/member_0/`). Its system configuration sets `toggle_tritonswmm_model: true`, `toggle_triton_model: false` and `toggle_swmm_model: false`. Every arm-dependent statement below applies to the sensitivity golden with `tritonswmm` as the only arm: no `out_triton/`, `out_swmm/`, `log_triton.json`, `log_swmm.json`, `TRITON_only_*` or `SWMM_only_*` store, no `c_run_`/`d_process_` flag, `.done` file or analysis-level run log of the `triton` or `swmm` model, no `TRITON.cfg`, and in the consolidated trees only the `/tritonswmm` group.

## 2. Reduced dimensions `[CHOICE]`

Section 1 fixes the scenario structure (3 events with three arms for the multi-sim golden; 4 members by 1 event with one arm for the sensitivity golden); this section chooses the grid and the step count. Only the grid and the number of output time steps shrink.

| Quantity | Observed (`[OBS]`) | Golden | Unit |
|---|---|---|---|
| DEM rows by columns | 30 by 16 (`system_log.json` `dem_shape` is `[1, 30, 16]`) | **6 by 4** | cells |
| cell size | 10.0 | **10.0** | m |
| lower-left corner | 0.0, 0.0 | **0.0, 0.0** | m |
| TRITON reporting step `TRITON_reporting_timestep_s` | 10.0 | **10.0** | s |
| number of reporting steps | 1080 | **12** | steps |
| simulated duration `sim_duration` in the TRITON cfg | 10800 | **120** | s |
| SWMM report step | 10 s (`REPORT_STEP           00:00:10`, line 22 of the observed `sims/event_index.1/swmm/hydraulics.inp`) | **10** | s |

Ground for 12 steps at 10 s: `src/hhemt/constants.py::TEST_N_REPORTING_TSTEPS_PER_SIM` defines `TEST_N_REPORTING_TSTEPS_PER_SIM = 12` and `TEST_TRITON_REPORTING_TIMESTEP_S = 10` for `analysis.test()`, so the repository already runs its own minimal test at exactly this size `[SRC]`. Ground for 6 by 4: the reason for this choice is not recorded in this file. What the file does state: the DEM is 6 rows by 4 columns of 10 m cells (y from 0 to 60 m, x from 0 to 40 m), the network paragraph below places the 14 nodes at x = (i mod 4 + 0.5) x 10 m and y = (i div 4 + 0.5) x 10 m, so they fill 4 rows by 4 columns of it (the largest x and the largest y are both 35 m), and rows differ from columns, so a transposed array is detectable; none of these is stated to be the reason. Constraints a consumer imposes that this file has verified: the processed DEM has shape `(1, rows, cols)` (`src/hhemt/analysis_validation.py::check_system_setup`) and the DEM and Mannings grids share a shape (the same function). Constraints it has NOT verified are in section 10 (OI-D1).

Network size `[CHOICE]` is not reduced, because only the grid and the step count shrink: the SWMM network keeps the observed 14 nodes, 12 links and its subcatchments, laid out inside the DEM extent (x in 0 to 40 m, y in 0 to 60 m). Measured on the observed model `~/.cache/hhemt/synthetic_test_models/379f676fc1ef6a60/swmm_full.inp` (unit: non-comment, non-blank rows in each section; matcher: `awk '/^\[/{s=$1;next} /^[ \t]*(;|$)/{next} s=="[JUNCTIONS]"||s=="[OUTFALLS]"{n++} s=="[CONDUITS]"{c++} s=="[SUBCATCHMENTS]"{u++} END{print n,c,u}'`, which prints `14 12 11`): 14 nodes, 12 conduits, 11 subcatchments. Every node carries a `[COORDINATES]` row in the hydraulics, hydrology and full templates and in each prepared `.inp`, placed at the centre of its own TRITON cell: node `i` (the 13 junctions in file order, then the outfall as node 13) at x = (i mod 4 + 0.5) x 10 m, y = (i div 4 + 0.5) x 10 m, so all 14 lie inside x 0 to 40 m and y 0 to 60 m. hhemt refuses to plan a coupled system with no such junction (`src/hhemt/analysis.py::_refuse_coupled_system_with_no_coupling_candidate`, `_refuse_coupled_system_with_no_coupling_candidate`; `scenario.py::coupling_candidates`, `coupling_candidates`, `all_junctions` mode) and refuses one outside the grid (`analysis.py::_refuse_coupling_nodes_outside_triton_grid`). The outfall's row is for the system-overview report plot, which reads a coordinate for every outfall (`src/hhemt/report_renderers/system_overview.py::_draw_hydraulics_panel`); it does not make the outfall a coupling candidate, because `all_junctions` mode counts `JUNCTION` nodes only (`scenario.py::coupling_candidates`, `coupling_candidates`). The node and link counts match the observed store dimensions (`node_id: 14`, `link_id: 12`).

## 3. Directory layout `[SRC]` unless tagged

`{G}` is the zip root; it is the `system_directory`. Layout version of the `_version.json` stamps is the value of `LAYOUT_VERSION` at the commit that regenerates the golden (26 at `b7e963b2`, 27 from the commit that added the consolidated-metadata index, 28 from the commit that adds `V0028`, 29 from the commit that adds `V0029`, 30 from the commit that adds `V0030`; `src/hhemt/version_migration/constants.py::LAYOUT_VERSION`, `src/hhemt/version_migration/versions/V0027__consolidate_zarr_metadata.py`, `src/hhemt/version_migration/versions/V0028__flood_depth_and_unit_discharge_names.py` `src/hhemt/version_migration/versions/V0029__remove_unread_settings.py` and `src/hhemt/version_migration/versions/V0030__swmm_run_attributes_to_variables.py`). The goldens are at 30 and every zarr store in them carries the index (section 8).

Multi-sim golden:

```text
{G}/
  _version.json                         layout stamp for the system directory
  system_config.yaml
  analysis_config.yaml
  system_log.json
  weather_events_to_simulate.csv
  elevation_10.00m.dem                  processed DEM, ESRI ASCII grid
  mannings_10.00m.dem                   processed Mannings grid, same shape
  inputs/                               every input file a config names (section 5)
  _software/                            minimal software root (section 6)
  {analysis_id}/                        the analysis directory
    _version.json
    log.json                            analysis log
    analysis_datatree.zarr/             per-analysis tier
    _status/                            flags (section 4.5)
    logs/sims/model_{model}_evt-{event_id}.log     analysis-level run logs (4.7)
    sims/{event_id}/                    one directory per event, see below
```

Per event directory `sims/{event_id}/`, where `{event_id}` is `event_index.0`, `event_index.1`, `event_index.2` (`scenario.compute_event_id_slug` joins `{indexer}.{value}`, `src/hhemt/scenario.py::compute_event_id_slug`, and the weather event index column is `event_index` `[OBS]`):

```text
scenario_prep_log.json                  scenario log (4.3)
log_triton.json  log_tritonswmm.json  log_swmm.json      model logs (4.3)
sim_weather.nc                          per-event weather (4.8)
dats/  extbc/  strmflow/  swmm/         prepared inputs (4.8)
TRITON.cfg  TRITONSWMM.cfg              prepared solver configs (4.8)
out_triton/  out_tritonswmm/  out_swmm/ raw layer (section 7)
processed/                              summary layer (section 8)
```

Sensitivity golden: the same system-level files; the analysis directory is the experiment root and holds `members/member_{member_id}/` for `member_id` 0 to 3, each with its own `sims/event_index.0/`, `analysis_datatree.zarr/`, `log.json` and `_status/`; the experiment root holds `experiment_datatree.zarr/` and the flags in 4.5 `[DOC]` `docs/reference/output-data-model.md` lines 32-36 and 89-95, `[SRC]` `src/hhemt/sensitivity_analysis.py::TRITONSWMM_sensitivity_analysis.__init__` (`self.members_dir = ... / "members"`). The directory `subanalyses/sa_*` and the `sa-*` flags present in the observed sensitivity tree belong to a retired naming (V0021 and V0022 renamed the experiment tree) and must not be written; the current names are `member_{id}` and `member-{id}`.

## 4. Completion records

Hhemt decides "finished" from log fields, flag files and marker text, never from a directory listing (`[DOC]` `docs/reference/output-data-model.md` lines 61-71; `[SRC]` `src/hhemt/process_simulation.py::_already_written`, `_already_written`). Every record below is written by the generator as plain JSON or text; no hhemt writer is called.

### 4.1 Layout stamp

`_version.json` in the system directory and in the analysis directory (and in each member directory, if the sensitivity writer stamps members; OI-R1). Fields observed `[OBS]`: `created_at` (UTC, `Z` suffix), `layout_version` (integer; 30 in the goldens, section 3), `migration_history` (empty list), `toolkit_version` (string). `[CHOICE]` `created_at` fixed at `2000-01-01T00:00:00Z`, `toolkit_version` `0.0.0+golden`, so regeneration is deterministic.

### 4.2 System log `{G}/system_log.json`

A flat JSON object. The class is `TRITONSWMM_system_log` (`src/hhemt/log.py::TRITONSWMM_system_log`); each log field serializes as its bare value, `null` when unmeasured (`src/hhemt/log.py::_logfield_serializer`, `_logfield_serializer`). A golden whose compile state must read "successful" carries, at least:

| Field | Value | Reader, or the observed source when no reader is named |
|---|---|---|
| `logfile` | `@GOLDEN_ROOT@/system_log.json` | replaced on load with the real path (`src/hhemt/log.py::TRITONSWMM_log.from_json`) |
| `dem_processed` | `true` | `[OBS]` |
| `dem_shape` | `[1, 6, 4]` | `[OBS]`, `check_system_setup` requires `shape[0] == 1` (`analysis_validation.py::check_system_setup`) |
| `mannings_processed` | `true` | `[OBS]` |
| `mannings_shape` | `[1, 6, 4]` | `[OBS]` |
| `compilation_tritonswmm_cpu_successful` | `true` | `[OBS]`; re-derived from the compile log on read (4.9, `system.py::compilation_cpu_successful`) |
| `compilation_triton_cpu_successful` | `true` | as above (`system.py::compilation_triton_only_cpu_successful`) |
| `compilation_swmm_successful` | `true` | as above (`system.py::compilation_swmm_successful`) |
| `dem_crs_epsg` | `3857` | `[OBS]` |
| `vertical_crs_epsg` | `5703` | `[OBS]` |
| every other field | `null` | `[OBS]` |

The compile properties re-derive their verdict from the build trees, so the three compile fields are consistency records, not the authority (section 6). The remaining observed fields (`triton_head_sha`, `sif_sha256`, producing-sha fields) are `null` in a golden because there is no real build `[CHOICE]`; the observed `triton_head_sha` was a real commit and is not reproduced.

### 4.3 Per-event logs

Scenario log `sims/{event_id}/scenario_prep_log.json`, class `TRITONSWMM_scenario_log` (`log.py::TRITONSWMM_scenario_log`). Preparation is complete when `scenario_creation_complete` is `true` (`analysis.py::_scenarios_not_created`, `_scenarios_not_created`, reads exactly that field after `refresh()`). Fields written by a finished preparation `[OBS]`: `event_iloc`, `event_idx` (`{"event_index": N}`), `simulation_folder`, `swmm_rainfall_dat_files`, `storm_tide_for_swmm`, `scenario_creation_complete`, `inp_hydraulics_model_created_successfully`, `inp_full_model_created_successfully`, `inp_hydro_model_created_successfully`, `hydro_swmm_sim_completed`, `extbc_tseries_created`, `extbc_loc_created`, `hyg_timeseries_created`, `hyg_locs_created`, `inflow_nodes_in_hydraulic_inp_assigned` (observed `null`), `triton_swmm_cfg_created`, `triton_cfg_created`, `sim_tritonswmm_executable_copied`, `triton_backend_used` (`"cpu"`). Booleans are `true` for every step the golden's prepared inputs (4.8) represent.

Model logs `sims/{event_id}/log_{model}.json`, class `TRITONSWMM_model_log` (`log.py::TRITONSWMM_model_log`), the file name fixed at `src/hhemt/scenario.py::get_log` (`f"log_{model_type}.json"`). Fields `[OBS]`: `event_iloc`, `event_idx`, `simulation_folder`, `simulation_completed`, `sim_run_time_minutes`, `n_resumes`, `resume_reporting_tsteps`, `processing_log`, and the written-markers below; unmeasured optional fields are `null`.

| Model log | Written-markers set `true` | `processing_log.outputs` keys (each `success: true`) |
|---|---|---|
| `log_tritonswmm.json` | `performance_timeseries_written`, `performance_summary_written`, `TRITON_timeseries_written`, `TRITON_summary_written`, `SWMM_node_timeseries_written`, `SWMM_link_timeseries_written`, `SWMM_node_summary_written`, `SWMM_link_summary_written` | `TRITONSWMM_perf_tseries.zarr`, `TRITONSWMM_TRITON_tseries.zarr`, `TRITONSWMM_SWMM_node_tseries.zarr`, `TRITONSWMM_SWMM_link_tseries.zarr`, `TRITONSWMM_perf_summary.zarr`, `TRITONSWMM_TRITON_summary.zarr`, `TRITONSWMM_SWMM_node_summary.zarr`, `TRITONSWMM_SWMM_link_summary.zarr` |
| `log_triton.json` | `performance_timeseries_written`, `performance_summary_written`, `TRITON_timeseries_written`, `TRITON_summary_written` | `TRITON_only_perf_tseries.zarr`, `TRITON_only_tseries.zarr`, `TRITON_only_perf_summary.zarr`, `TRITON_only_summary.zarr` |
| `log_swmm.json` | `SWMM_node_timeseries_written`, `SWMM_link_timeseries_written`, `SWMM_node_summary_written`, `SWMM_link_summary_written` | `SWMM_only_node_tseries.zarr`, `SWMM_only_link_tseries.zarr`, `SWMM_only_node_summary.zarr`, `SWMM_only_link_summary.zarr` |

Each `processing_log.outputs` value has `filepath` (an absolute path, token-relocated by section 5), `size_MiB` (float), `time_elapsed_s` (float), `success`, `notes` (`""`), `warnings` (`""`). The key is the store's file name; a store counts as already written when its entry has `success is True` (`process_simulation.py::_already_written`). The store file names follow `src/hhemt/scenario.py::TRITONSWMM_scenario.__init__` (`.zarr` because the goldens use `target_processed_output_type: zarr`).

A simulation counts as complete only when `simulation_completed` is `true` AND the model's terminal artifact is finalized (`src/hhemt/run_simulation.py::model_run_completed` and `_coupled_swmm_report_finalized`). The terminal artifacts the golden must therefore carry are:

| Model | Terminal artifact the reader checks | What it must contain |
|---|---|---|
| `swmm` | `sims/{event_id}/out_swmm/full.rpt` | a line containing `Analysis ended on`, and no line matching `^\s*ERROR \d+` (`swmm_output_parser.py::rpt_status`, `rpt_status`) |
| `tritonswmm` | `sims/{event_id}/out_tritonswmm/swmm/hydraulics.rpt` | the same two conditions |
| `triton` | the analysis-level run log (4.7) and `sims/{event_id}/out_triton/performance.txt` | the run log contains `Simulation ends`; `performance.txt` exists (`run_simulation.py::_coupled_swmm_report_finalized`) |

### 4.4 Analysis log `{analysis_dir}/log.json`

Class `TRITONSWMM_analysis_log` (`log.py::TRITONSWMM_analysis_log`). `[OBS]` for a finished multi-sim: `datatree_consolidation_complete: true`, `consolidation_version: 1`, `consolidation_inputs_fingerprint` (a 40-hex string), `consolidation_build_stamp: null`, `cpu_backend_available`, `gpu_backend_available`, a `processing_log.outputs` entry keyed `analysis_datatree.zarr` with `success: true`, and every workflow-submission field `null`. Whether `consolidation_inputs_fingerprint` must equal a value hhemt recomputes from the inputs, so that a fresh extraction plans zero consolidation jobs, is `[OPEN]` (OI-F1): the generator cannot compute it without hhemt's rule, so the value is whatever the conformance test shows hhemt accepts, logged in the change log in section 10.

### 4.5 Status flags `{analysis_dir}/_status/`

A flag is a zero-byte file `NAME.flag` with a JSON sidecar `NAME.flag.json` (`src/hhemt/status_flags.py::write_status_flag`, `write_status_flag`). The sidecar holds `rule_name`, `written_at` (ISO-8601 with offset), `slurm_job_id` (`null`), and, when given, `model_type`, `sa_id`, `event_id`, `extra`. Names, with the module or symbol that mints them:

Multi-sim (`src/hhemt/workflow.py`): `a_setup_complete.flag`, `b_prepare_evt-{event_id}_complete.flag`, `c_run_{model}_evt-{event_id}_complete.flag`, `d_process_{model}_evt-{event_id}_complete.flag`, `e_consolidate_complete.flag`, `f_consolidate_scenario_evt-{event_id}_complete.flag`. `{model}` is one of `triton`, `tritonswmm`, `swmm`. All six families are present in the observed multi-sim tree (`[OBS]`).

Sensitivity (experiment root `_status/`): `a_setup_target_{target_id}_complete.flag`, `b_prepare_member-{member_id}_evt-{event_id}_complete.flag`, `c_run_{model}_member-{member_id}_evt-{event_id}_complete.flag` (`src/hhemt/constants.py::sim_run_flag_per_member`), `d_process_{model}_member-{member_id}_evt-{event_id}_complete.flag` (`constants.py::process_timeseries_flag_per_member`), `e_consolidate_member-{member_id}_complete.flag` (`constants.py::consolidate_analysis_flag`), `f_consolidate_experiment_complete.flag` (`constants.py::consolidate_experiment_flag`), plus the fingerprint record `member-{member_id}_inputs.json` (`constants.py::member_inputs_fingerprint_flag`; observed payload `{"__schema_version__":1,"fields":{"n_gpus":0,"n_mpi_procs":2,"n_nodes":1,"n_omp_threads":1,"run_mode":"mpi"}}` `[OBS]`). The `f_consolidate_master_complete.flag` and `*_sa-{id}_*` files in the observed tree are retired names and are not written. In the sensitivity golden `{model}` is `tritonswmm` only. `target_id` values: one target for the four-member golden (`[OBS]` `a_setup_target_0_complete.flag`). **Fingerprint byte form `[SRC]`**: each `member-{member_id}_inputs.json` is `json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"` (`src/hhemt/sensitivity_analysis.py::_write_member_id_fingerprint`, compare-and-write rewriting on any difference, in the same function), because any other byte form makes a plan rewrite the file and so makes every member stale.

A flag's modification time is not set by the generator: extraction discards stored times, and the fixture sets every time in dependency order after extraction (`tests/fixtures/golden_trees.py::set_times_in_dependency_order`).

### 4.6 Store completion flag

A per-scenario TRITON timeseries store is finished when its sibling file `{store name}.done` exists (`[DOC]` `docs/reference/output-data-model.md` lines 61-71; name from `src/hhemt/utils.py::unified_flag_for`, `unified_flag_for`: `root.with_name(root.name + ".done")`). Content is the constant text `ok`, UTF-8, no newline (`src/hhemt/utils.py::_publish_flag_crash_safe`, `_publish_flag_crash_safe`). The multi-sim golden carries `processed/TRITONSWMM_TRITON_tseries.zarr.done` and `processed/TRITON_only_tseries.zarr.done`; the sensitivity golden carries the first only. A `.chapters` directory is not carried (the doc calls a leftover one harmless; its absence is the normal finished state, lines 66-69).

### 4.7 Analysis-level run logs

`{analysis_dir}/logs/sims/model_{model}_evt-{event_id}.log` for a regular analysis (`src/hhemt/run_simulation.py::model_logfile_name`, `model_logfile_name`; the directory is `analysis_paths.simlog_directory`, `{analysis_dir}/logs/sims`). For a sensitivity member the name carries the member's analysis id: `model_{model}_{analysis_id}_evt-{event_id}.log` under the MASTER's `logs/sims`. `[CHOICE]` content: the solver's own stdout stand-in. (The sensitivity golden writes the `tritonswmm` logs only.) For `triton` and `tritonswmm` the file contains the line `Simulation ends` (marker at `run_simulation.py::model_run_completed`); for `swmm` it contains `EPA SWMM completed` and not `There are errors.` (`src/hhemt/swmm_markers.py::SWMM_COMPLETION_MARKER`).

### 4.8 Prepared per-event inputs

The raw-layer readers and the scenario-completion checks read these, so the golden carries them as small generator-written files: `sims/{event_id}/swmm/hydro.inp`, `swmm/hydraulics.inp`, `swmm/full.inp` (`scenario.py::TRITONSWMM_scenario.__init__`), `swmm/coupling_nodes.txt` (the `tritonswmm` arm only: the 13 junction names, one per line, in node order, which scenario preparation writes and `process_simulation.py::_attach_coupling_outputs` reads), `swmm/hydro.out` and `swmm/hydro.rpt`, `extbc/tseries.txt`, `extbc/loc.extbc`, `strmflow/tseries.hyg`, `strmflow/loc.txt`, `dats/grid-indRG_synth.dat`, `dats/waterlevel.dat`, `sim_weather.nc`, `TRITON.cfg` (multi-sim golden only), `TRITONSWMM.cfg`. Their contents are `[CHOICE]` and minimal; which of them any golden-backed test actually reads is `[OPEN]` (OI-D2). The per-scenario symlinks `build` and `build_triton` of the observed tree (absolute links into the software root) are NOT carried: their only readers are the run and prepare paths (`run_simulation.py::prepare_simulation_command`; `scenario.py::_copy_tritonswmm_build_folder_to_sim`), which a golden-backed test does not take.

### 4.9 How a consumer re-derives the compile state

See section 6. The compile fields in 4.2 are consistent with it and are not the authority.

## 5. Configuration and input files, and relocation

### 5.1 Files the golden carries

`system_config.yaml` and `analysis_config.yaml` in `{G}`, validated by the Pydantic models `system_config` and `analysis_config` (`src/hhemt/config/system.py`, `analysis.py`; `extra="forbid"`). Every path-typed field that is set must name a file that exists when the config loads (`src/hhemt/config/base.py::_check_paths_exist`), so the golden carries a file for each set path field. Required system fields (`system_config.model_fields`, measured with a one-line introspection): `system_directory`, `watershed_gis_polygon`, `DEM_fullres`, `SWMM_hydraulics`, `TRITONSWMM_git_URL`, `triton_swmm_configuration_template`, `toggle_use_swmm_for_hydrology`, `toggle_use_constant_mannings`, `toggle_triton_model`, `toggle_tritonswmm_model`, `toggle_swmm_model`, `target_dem_resolution`, `crs`. Required analysis fields: `analysis_id`, `weather_event_indices`, `weather_timeseries`, `weather_time_series_timestep_dimension_name`, `weather_time_series_spatial_mean_rainfall_datavar`, `rainfall_units`, `run_mode`, `toggle_sensitivity_analysis`, `toggle_storm_tide_boundary`, `weather_events_to_simulate`, `hhemt_sha`, `manhole_diameter`, `manhole_loss_coefficient`, `hydraulic_timestep_s`, `TRITON_reporting_timestep_s`, `open_boundaries`, `report`.

`[CHOICE]` values, taken from the observed configs (`[OBS]` `.../synth_multi_sim/system_config.yaml` and `analysis_config.yaml`) except where noted:

| Field | Golden value |
|---|---|
| `target_dem_resolution` | `10.0` |
| `crs` | `horizontal_epsg: 3857`, `vertical_epsg: 5703` |
| `toggle_tritonswmm_model`, `toggle_triton_model`, `toggle_swmm_model` | `true`, `true`, `true` for the multi-sim golden; `true`, `false`, `false` for the sensitivity golden |
| `toggle_use_swmm_for_hydrology` | `true` (requires `SWMM_hydrology`, `subcatchment_raingage_mapping`, and its gage colname, `config/system.py::system_config`) |
| `toggle_use_constant_mannings` | `true`, with `constant_mannings: 0.035` (this removes the landuse raster and lookup, `config/system.py::system_config`; the observed config used a raster and a lookup) |
| `TRITONSWMM_git_URL` | `https://github.com/lassiterdc/triton.git` (a string; no network is used) |
| `TRITON_reporting_timestep_s` | `10.0` |
| `hhemt_sha` | the 40-hex string `0000000000000000000000000000000000000000` |
| `weather_event_indices` | `[event_index]` |
| `weather_time_series_timestep_dimension_name` | `time` |
| `weather_time_series_spatial_mean_rainfall_datavar` | `RG_synth` |
| `weather_time_series_storm_tide_datavar` | `water_level` |
| `multi_sim_run_method` | `local` |
| `target_processed_output_type` | `zarr` |
| `run_mode` | `serial` for the multi-sim golden |
| `clear_raw`, `force_rerun` | `none`, `none` |
| `report` | `{}` for the multi-sim golden (as observed) |

Input files under `{G}/inputs/` `[CHOICE]`, one per set path field: `watershed.geojson`, `boundary.geojson`, `swmm_hydraulics.inp`, `swmm_hydrology.inp`, `swmm_full.inp`, `subcatchment_raingage_mapping.csv`, `tritonswmm.cfg`, `weather.nc`, and a raster for `DEM_fullres`. The three SWMM `.inp` templates must declare `FLOW_UNITS CMS` (`config/system.py`, `validate_swmm_templates_are_si`, reads the last `FLOW_UNITS` line by SWMM's prefix rules), so each carries an `[OPTIONS]` section with that line and the template placeholders hhemt substitutes at preparation (`${START_DATE}` and peers, `[OBS]` `.../synthetic_test_models/379f676fc1ef6a60/swmm_hydraulics.inp`). `DEM_fullres` is written as an ESRI ASCII grid with the processed DEM's values under the name `inputs/dem_fullres.asc`: hhemt only needs the file to exist when the config loads, and no golden-backed path re-reads it `[OPEN]` (OI-D3).

`weather.nc` `[OBS]`: dimensions `(event_index, time)`; variables `RG_synth` and `water_level`, both float32; coordinate `event_index` int32; coordinate `time` datetime64[ns]; attribute `title`. `[CHOICE]` for the golden: `event_index` 0 to 2 (multi-sim) or 0 (sensitivity), and `time` has 13 values at 10 s spacing starting `2000-01-01T00:00:00`, so one event spans the 12 reporting steps of section 2.

`weather_events_to_simulate.csv` `[OBS]`: header `event_index`, then one integer per row (`0`, `1`, `2`).

Sensitivity inputs: the analysis config sets `toggle_sensitivity_analysis: true` and `sensitivity_analysis:` to a CSV in `{G}/inputs/` `[OBS]` with header `member_id,analysis.run_mode,analysis.n_mpi_procs,analysis.n_omp_threads,analysis.n_gpus,analysis.n_nodes` and rows `0,mpi,2,1,0,1` / `1,openmp,1,2,0,1` / `2,hybrid,2,2,0,1` / `3,serial,1,1,0,1`. Each member directory holds `member_{member_id}.yaml`, a copy of the analysis config with `analysis_id: member_{member_id}` and the row's compute fields, plus `dataset_license: CC0-1.0` `[OBS]`.

### 5.2 Making absolute paths relocatable `[CHOICE]`

Configs, logs and cfg files hold absolute paths, and a golden is extracted into a different directory for every test. The rule is one token:

- The generator writes the literal string `@GOLDEN_ROOT@` wherever an absolute path to a location inside the golden would appear, followed by a `/`-separated path relative to the zip root. `system_directory` is exactly `@GOLDEN_ROOT@`. The analysis directory is `@GOLDEN_ROOT@/{analysis_id}`.
- No absolute path to any other location is written anywhere in a golden. The regeneration test (`tests/test_golden_regeneration.py::test_committed_zip_is_relocatable`) fails on a text member naming a path under `/home/`, `/tmp/` or `/Users/`.
- Extraction replaces every occurrence of `@GOLDEN_ROOT@` with the extraction directory's path in every file whose name ends `.yaml`, `.yml`, `.json`, `.cfg`, `.csv`, `.txt`, `.hyg`, `.extbc`, `.inp`, `.rpt`, `.dat`, `.log`, `.flag.json`, `.zattrs` or `zarr.json`, and copies every other file byte for byte. Binary solver files and arrays are never rewritten.
- The existing clone helper re-roots `system_directory` only (`tests/_failing_fixture_helpers.py`, `prepare_clone_dir`, line 99 of that file: `sys_cfg_dict["system_directory"] = str(dst_system_dir)`); the other absolute paths of the observed tree point into the shared `~/.cache/hhemt/synthetic_test_models/` cache. A golden has no such shared cache, so the token rule extends the helper's idea to every path.

A path that is stored only as a log field and is overwritten on load (`logfile`, `src/hhemt/log.py::TRITONSWMM_log.from_json`) still carries the token, so a reader never meets a foreign absolute path.

## 6. Minimal software root and compilation logs

`TRITONSWMM_software_directory` is `@GOLDEN_ROOT@/_software/triton` and `SWMM_software_directory` is `@GOLDEN_ROOT@/_software/swmm`. They hold only the build tools' compilation logs and no executable. Paths, from `src/hhemt/system.py::TRITONSWMM_system.__init__` and `compilation_swmm_successful`:

| File | Read by | Must contain |
|---|---|---|
| `_software/triton/build_tritonswmm_cpu/compilation.log` | `compilation_cpu_successful` (`system.py::compilation_cpu_successful`) | the line `Built target swmm5` AND the line `[100%] Built target triton.exe` |
| `_software/triton/build_triton_cpu/compilation.log` | `compilation_triton_only_cpu_successful` (`system.py::compilation_triton_only_cpu_successful`) | the line `[100%] Built target triton.exe` |
| `_software/swmm/compilation.log` | `compilation_swmm_successful` (`system.py::compilation_swmm_successful`) | either `Built target runswmm` or `Built target swmm5` (any one suffices; the markers `CMake Error`, `error:`, `FAILED` do not change the verdict, because the reader returns True when a success marker is present and False when none is) |

**Citation of the success lines.** The strings `Built target swmm5` and `[100%] Built target triton.exe` are the build driver's own output, printed by CMake's generated Makefile build when a target finishes (`[NN%] Built target {target}`); they are not hhemt strings, and hhemt's reader only greps for them (`system.py::compilation_cpu_successful`). Their authority is CMake: the Makefile generator emits `Built target {name}` for each target through `AppendEcho(commands, "Built target " + name, ...)` in `cmGlobalUnixMakefileGenerator3::WriteConvenienceRules2`, when the global property `TARGET_MESSAGES` is on (its default), (read in full 2026-10-31 from https://raw.githubusercontent.com/Kitware/CMake/v3.31.0/Source/cmGlobalUnixMakefileGenerator3.cxx, lines 710-711 and the `TARGET_MESSAGES` guard at 705; the tag, not `master`, is cited because `master` moves; matcher `grep -n 'AppendEcho(commands, "Built target "'` returns 710 on v3.31.0, 705 on v3.28.0 and 802 on master). The `[NN%] ` progress prefix is added by the Makefile generator's progress echo, which that fetch did not show, so the prefix rests on the observed log: `[OBS]` `~/.cache/hhemt/synthetic_test_runs/docs-and-experiment-system/_software/swmm/compilation.log` lines 33, 85 and 88 read `[ 11%] Built target swmm-output`, `[ 96%] Built target swmm5` and `[100%] Built target runswmm`. The generator writes a short log of that form for each build.

The software root lives INSIDE the zip (`@GOLDEN_ROOT@/_software/...`), because the zip is the whole golden (section 1) and a golden's software root is minimal, holding only the build tools' compilation logs; the observed arrangement, with software roots in a shared cache outside the system directory, is not followed, and both config fields must name directories that exist when the config loads (section 5.1).

Nothing under `_software/` is an executable, so `swmm_executable` resolves to `None` and any path that needs a binary is a real run, not a golden-backed test (`system.py::swmm_executable`).

## 7. Raw layer: the formats followed

The raw layer is `sims/{event_id}/out_triton/`, `out_tritonswmm/` and `out_swmm/`, written at the reduced dimensions of section 2; the sensitivity golden holds `out_tritonswmm/` only.

### 7.1 TRITON raw outputs

No prose document specifies the byte layout of TRITON's binary rasters. The checkout's `doc/Triton_UserGuide.pdf` (sections 6.7 and 8.2-8.3) and `doc/ref-manual/TritonReferenceManual.pdf` (`write_output_binary_sequential`) were also examined and neither gives the byte layout. The TRITON repository's configuration reference states that `output_format=BIN` selects "TRITON binary" and that `outfile_pattern="%s/%s/%s_%02d_%02d"` names the files (`doc/configuration_reference.rst` lines 157-170 and 207-218, in the TRITON checkout at commit `bb13ea28603d50879f8d85cd041c17f440fbdb2b` of https://github.com/lassiterdc/triton), and gives no layout. **The raw layer therefore follows the solver's source**, `src/output.h` of that checkout:

- One file per variable and reporting step, named `{PREFIX}_{step:02d}_{rank:02d}.out` in `{out_dir}/bin/`, with `rank` `00` (sequential output, `output_option=SEQ`) (`output.h`, `get_mat_path`). `PREFIX` is `H` (depth), `QX`, `QY` (unit discharge components) and `MH` (maximum depth); a `GR_*` ghost-ring side-file also appears in the observed tree and is not processable output (`[SRC]` `process_simulation.py::triton_raw_frame_or_raise`).
- File body: two `float64` (`sizeof(T)` for `T = double`) values, the row count then the column count of the interior (ghost cells removed, `GHOST_CELL_PADDING`), then the interior values row by row as `float64`, little-endian (`output.h`). `[OBS]` check: `H_01_00.out` is 3,856 bytes for a 30 by 16 grid, which is 16 + 30 x 16 x 8, with first two doubles `30.0` and `16.0`. For 6 by 4: 16 + 6 x 4 x 8 = 208 bytes.
- Step numbering: files exist for steps `1` to `12` (the observed tree's first file is `H_01_00.out` and its last `H_1080_00.out`, one per 10 s), with no step `0` (`[OBS]`). hhemt maps step `k` to `timestep = k x TRITON_reporting_timestep_s / 60` (`process_simulation.py::return_filelist_by_tstep`). The directory read is `{out_dir}/bin` (`TRITON_raw_output_type` default `"bin"`, `src/hhemt/config/analysis.py::analysis_config.TRITON_raw_output_type`; the join is at `process_simulation.py::_export_TRITON_only_outputs`).
- All four of `H`, `QX`, `QY`, `MH` must be present at every step, or the reader refuses the directory as ragged or missing a variable (`process_simulation.py::triton_raw_frame_or_raise`).
- `performance.txt` and `performance/performance{N}.txt` for `N` = `1` to `12`: a header line `%Rank, Compute, MPI, IO, Resize, SWMM, Other, Simulation, Init, Total`, one line per rank (`0, ...`), and an `Average, ...` line, comma-separated floats (`[OBS]` `out_triton/performance.txt` and `performance/performance100.txt`). Values are cumulative over the run (`process_simulation.py::_aggregate_perf_tseries`). `cfg/config_{N}.cfg` and `log.out` are also written by the solver; the golden carries `log.out`, whose `nTasks` and `OMP threads per task` lines state the scenario's MPI tasks and OpenMP threads per task, because hhemt reads them back as the run's actual resources and compares them with the configured ones (`src/hhemt/analysis.py::df_status`; `src/hhemt/consolidate_workflow.py::validate_resource_usage`): 1 and 1 for the multi-sim golden, and for each sensitivity member the `n_mpi_procs` and `n_omp_threads` of its row in section 5.1; their need by a golden-backed test is `[OPEN]` (OI-D2). `[OBS]` `log.out` header lines begin `---- TRITON RUN INFO ----`.

The same layout is used under `out_tritonswmm/`, which additionally holds `swmm/hydraulics.out` and `swmm/hydraulics.rpt` (`scenario.py::TRITONSWMM_scenario.__init__`).

### 7.2 SWMM raw outputs

EPA SWMM (tag `v5.2.4`, https://github.com/USEPA/Stormwater-Management-Model, the checkout at `[OBS]` `.../_software/swmm/swmm_source` is at that tag) writes `.out` (binary) and `.rpt` (text). The pinned checkout contains no prose document of either format (`doc/` holds only `Images` and `New SWMM 5.2 Features.md`; checked by listing; a web search for a published layout document found only third-party parsers (2026-10-31)). **The `.out` layout therefore follows the solver's source**: the writer `src/solver/output.c` (`output_open`, lines 120-200: magic number, version, flow units, counts, then ids, input data and per-period results, `BytesPerPeriod = sizeof(double) + numResults x sizeof(float)`) and the reader API declared in `src/outfile/include/swmm_output.h` (`SMO_open`, `SMO_getProjectSize`, `SMO_getNodeSeries`, ...). The `.rpt` is a text report with the sections hhemt parses (`Analysis ended on` trailer, flow-routing continuity table, node and link summary tables). The `.rpt` section list is `[OPEN]` (OI-R2) and is decided from EPA's `src/solver/report.c` and the `.rpt` the pinned solver writes, never from `swmm_output_parser.py`; the parser's requirements are checked by the conformance test (`tests/test_golden_conformance.py`), and a mismatch is resolved under the change rule in section 10.

Files: `out_swmm/full.out`, `out_swmm/full.rpt`, `out_tritonswmm/swmm/hydraulics.out`, `hydraulics.rpt`, `swmm/hydro.out`, `hydro.rpt` per event. Report step 10 s, 12 periods, 14 nodes, 12 links, 11 subcatchments (section 2). Flow units CMS, so the unit flag in the `.out` header is the code of the enumerator `SMO_CMS` in `src/outfile/include/swmm_output_enums.h` (line 24); its numeric value is `_FLOW_UNITS_CMS` in `tests/goldens/generator/raw.py`, which takes it from that enum, and is not stated here.

## 8. Summary layer and consolidated tiers: map to `docs/reference/output-data-model.md`

The document is the published specification for these tiers. Where it names a variable the generator writes it with the documented units and `standard_name`; where it is silent the generator follows the baseline inventory below and the conformance test decides, under the change rule in section 10. The baseline inventory records names, dimensions and dtypes and no per-variable units; the generator nevertheless writes a `units` attribute on EVERY numeric data variable, in SI spellings, because the document states that each stored numeric variable carries its unit (change log, row for the units attribute): depths, heads and lengths `m`; flows `m3 s-1`; volumes `10^6 L` for the SWMM report volumes and `m3` for the TRITON flood volume; velocities `m s-1`; unit discharges `m2 s-1`; times of maxima `min`; `hours_flooded` `h`; the two flow balance errors `percent` and `L`; the dimensionless ratios, capacities and the member counts of the `parameters` dataset `1`; the nine performance columns `s`. String-valued variables (`type`, `run_mode`, the producing-stamp coordinates) carry no unit.

| Golden file | Section of the document | What the document fixes |
|---|---|---|
| `sims/{event_id}/processed/*.zarr` (16 stores across the three arms, plus the two `.zarr.done` files, in the multi-sim golden; the 8 `TRITONSWMM_` stores and one `.zarr.done` in the sensitivity golden) | "The per-scenario tier" (lines 73-79) and "The output tiers" (11-55) | flat Zarr; the `.done` flag rule (lines 61-71); one directory per scenario |
| `processed/TRITONSWMM_TRITON_summary.zarr`, `TRITON_only_summary.zarr`, and the `*_tseries.zarr` TRITON stores | "2D surface results (TRITON)" (132-150) | `max_flood_depth`, `flood_depth`, `flood_depth_last_tstep`, `max_unit_discharge`, `unit_discharge_x`, `unit_discharge_y`, `unit_discharge_x_at_time_of_max_unit_discharge`, `unit_discharge_y_at_time_of_max_unit_discharge`, `time_of_max_unit_discharge`, `max_flood_velocity`, `time_of_max_flood_velocity`, `x_flood_velocity_at_time_of_max_flood_velocity`, `y_flood_velocity_at_time_of_max_flood_velocity`, `final_surface_flood_volume`, with the listed units and CF names |
| `processed/*_SWMM_node_*`, `*_SWMM_link_*`, `SWMM_only_*` | "Network results (SWMM)" (152-162) | `total_inflow_vol`, `max_flow`, `max_over_full_flow`, `max_over_full_depth`; the two ratio variables dimensionless, units `1` |
| `processed/*_perf_*` | "A caveat on performance columns" (209-219) | `performance.Total`, `Simulation`, `Init` carry wallclock semantics, the category columns are slowest-rank cost, values cumulative across allocations on a resumed run |
| `{analysis_dir}/analysis_datatree.zarr/` (sensitivity: only the `/tritonswmm` group) | "The per-analysis tier" (81-87) and "Opening the consolidated output" (106-125) | hierarchical `DataTree`, zarr, written WITH a consolidated-metadata index at layout 27 (see "Consolidated-metadata index" below; lines 119-120 of the document still describe the layout-26 form and are not updated by this file) |
| `{experiment root}/experiment_datatree.zarr/` | "The per-experiment tier" (89-95) and lines 122-125 | root `parameters` dataset describing every defined member, one `member_*` node per completed member |
| root attributes of each tree | "Global attributes" (199-207) | `Conventions: "CF-1.13"`, `analysis_id`, `system_id`; provenance in the root attribute `ro_crate_metadata` and a sidecar `ro-crate-metadata.json` |
| "Water-quality results" (164-197) | not carried | The golden's SWMM model declares no pollutant, so `Water Quality .......... YES` is absent from its report and "none of the names, coordinates or dimensions below exist" (lines 166-169). |
| `system_datatree.zarr` | "The per-system tier" (97-104) | not carried: "written only when that method is called explicitly. No run, command-line command or test calls it." |

Baseline inventory `[OBS]` (layout 22; known short of layout 27, see below), from a dump of `.../synth_multi_sim/synth_multi_sim/sims/event_index.1/processed/` with `xarray.open_zarr(..., consolidated=False)`:

- `TRITON*_tseries`: dims `timestep`, `y`, `x`; float32 variables `max_flood_depth`, `flood_depth`, `unit_discharge_x`, `unit_discharge_y`; coords `x`, `y` (float64), `timestep` (float64); attrs `sim_date`, `output_creation_date`.
- `TRITON*_summary`: dims `event_iloc` (1), `y`, `x`; variables `time_of_max_unit_discharge`, `unit_discharge_x_at_time_of_max_unit_discharge`, `unit_discharge_y_at_time_of_max_unit_discharge`, `max_flood_velocity`, `time_of_max_flood_velocity`, `x_flood_velocity_at_time_of_max_flood_velocity`, `y_flood_velocity_at_time_of_max_flood_velocity`, `time_of_max_flood_depth`, `max_unit_discharge` (float64); `final_surface_flood_volume`, `max_flood_depth`, `flood_depth_last_tstep` (float32); coords `event_iloc` (int64), `hhemt_producing_sha`, `hhemt_producing_version` (strings, dim `event_iloc`).
- `*_SWMM_node_tseries`, `SWMM_only_node_tseries`: dims `node_id`, `date_time`; 16 variables including `depth`, `head`, `inflow_flow`, `flooding` (time-varying) and per-node scalars such as `max_flow`, `hours_flooded`, `type`. `*_SWMM_node_summary`: dims `event_iloc`, `node_id`. Links: dims `link_id`, `date_time` (variables `velocity`, `link_depth`, `flow`, `capacity_setting` (coupled) or `capacity` (SWMM-only)). Dataset attrs on every SWMM store: `analysis_end_datetime`, `flow_units`, `runoff_continuity_error`, `flow_continuity_error`, `system_flooding`, each with a companion attribute `{key}_units` (`percent`, `percent`, `10^6 L`), `sim_date`, `output_creation_date`. The `timestep` coordinate of the TRITON and performance time series carries `units: min`.
- `*_perf_tseries`: dims `event_iloc`, `timestep`, `Rank`; float64 variables `Compute`, `MPI`, `IO`, `Resize`, `SWMM`, `Other`, `Simulation`, `Init`, `Total`. `*_perf_summary`: dim `event_iloc`; same nine variables; attrs `units: "seconds"` and `notes`.
- Consolidated `analysis_datatree.zarr`: root attrs `analysis_id`, `output_creation_date`, `consolidation_version`, `consolidation_inputs_fingerprint`, `Conventions` `CF-1.13`, `ro_crate_metadata`, `hhemt_producing_sha`, `hhemt_producing_version`; groups `/triton_only/{triton,performance}`, `/tritonswmm/{triton,swmm_node,swmm_link,performance}`, `/swmm_only/{swmm_node,swmm_link}`; each leaf has dim `event_iloc` (3) and the per-event string coordinates `event_id`, `hhemt_producing_sha`, `hhemt_producing_version`. Experiment tree: root attrs `Conventions`, `title`, `analysis_id`, `output_creation_date`, `ro_crate_metadata`; `/parameters` with dim `member_id` (4); nodes `/member_{id}/tritonswmm/{triton,swmm_node,swmm_link,performance}` (the sensitivity golden holds only the `/tritonswmm` group, as the note on the `analysis_datatree.zarr/` row of the mapping table above states for the per-analysis tree; the `experiment_datatree.zarr/` row does not repeat it).

**Known short of layout 26 (the inventory baseline is layout 22; layout 27 changed no variable, only the index above).** Three migrations since layout 22 changed the stores: V0023 adds the string coordinates `perf_column_set_across_allocations` (per-scenario performance stores) and `variable_set_across_members` (member join); V0024 adds rank-coherent performance reductions (the `*_coherent` columns, `src/hhemt/version_migration/versions/V0024__rank_coherent_perf_reductions.py`); V0025 moves stranded regular stores aside. The baseline inventory above does not include them. The writer for the performance stores (`tests/goldens/generator/summary.py::perf_tseries` and `perf_summary`) takes its variable list from the document's performance section and from the conformance run, not from this baseline alone (OI-P1).

**Consolidated-metadata index (layout 27) `[SRC]`.** From layout 27 the toolkit writes every zarr store with an index and opens every store through it: `src/hhemt/utils.py::consolidate_zarr_store` and `::zarr_store_has_index` (a format 3 store carries it as the `consolidated_metadata` key of its root `zarr.json`), the writers `write_zarr`, `write_datatree_zarr` and `append_zarr` of the same module, and the layout migration `src/hhemt/version_migration/versions/V0027__consolidate_zarr_metadata.py`, which gives every `*.zarr` store under a tree an index and changes nothing else. `[CHOICE]` Every `*.zarr` store of both goldens carries one: 49 stores in the multi-sim golden and 37 in the sensitivity golden, which is the full set `V0027` indexes (it walks every `*.zarr` directory outside `out_*`, `.snakemake` and `*.chapters`, and the goldens hold no store in a skipped place). Measured by running `python -m hhemt.version_migration migrate {dir} --apply` over an extraction of the layout-26 goldens: the migration changed exactly the root `zarr.json` of every store (the index key added) and the two `_version.json` stamps, and the generator now writes root `zarr.json` documents that are byte-identical to the migrated ones. The generator writes the index through xarray's `consolidated=True` write, which calls `zarr.consolidate_metadata`, the same call the toolkit's `consolidate_zarr_store` makes, and imports nothing from hhemt. A consequence outside the stores: `processing_log.outputs[...].size_MiB` in the analysis, member and scenario logs is measured on the finished store (`records.dir_size_mib`), so it grew with the index; no other field of any log changed.

## 9. What a golden does not carry

Not carried, because hhemt derives it when a test needs it or because no golden-backed path reads it; each is `[OPEN]` until the conformance test (`tests/test_golden_conformance.py`) or a golden-backed test of a timed operation such as `reprocess` (`tests/test_synth_07_reprocess.py`) shows a reader needing it (OI-D2): `Snakefile`, `Snakefile.prev`, `.snakemake/`, `workflow_summary.md`, `analysis_report.html`, `analysis_report.zip`, `plots/`, `report/`, `report_manifest.json`, `eda/`, `cid/`, `validation_report.json`, `scenario_status.csv`, `provenance_history.json`, `ro-crate-metadata.json` (the embedded `ro_crate_metadata` root attribute is carried), `_status/_du.json` and the per-scenario `_status/_du.json` disk-usage sentinels, `_status/_orchestrator/` (the clone helper deletes inherited orchestrator sentinels, `tests/_failing_fixture_helpers.py`, so none is carried), `processed/hydrology_inflow_summary.zarr` and `processed/hydrology_rpt_summary.zarr` (named by the document's per-scenario tier paragraph, `docs/reference/output-data-model.md` lines 24-30; read as capture stores at `src/hhemt/process_simulation.py::reclaim_scenario_scoped_classes`; absent from the observed layout-22 tree; whether any golden-backed reader opens them is `[OPEN]`, OI-D2), `_fixture_complete.json`, symlinks of any kind, and every executable.

A plan test (a dry run, as in `tests/test_golden_conformance.py`) needs a Snakefile; hhemt writes it when the workflow is built, so the fixture requests it from hhemt after extraction. That is a fixture duty (`tests/fixtures/golden_trees.py::dry_run_plan`) and not a generator duty.

## 10. Open items not decided here

In the last column, the conformance test is `tests/test_golden_conformance.py`, and a golden-backed test of a timed operation is one such as `tests/test_synth_07_reprocess.py`, which reprocesses an extracted golden.

| ID | Item | Decided by |
|---|---|---|
| OI-D1 | Minimum sizes a consumer needs (grid, steps, network) beyond section 2's verified shape constraints | the conformance test; a change is logged with the consumer line it follows |
| OI-D2 | Which of the section 4.8 prepared inputs, the `cfg/` and `log.out` solver files, and the section 9 omissions any golden-backed reader opens | the conformance test (construction and `validate_analysis`) and the golden-backed tests of timed operations |
| OI-D3 | Whether construction opens `DEM_fullres` or other set-but-unread inputs | the conformance test |
| OI-F1 | The value of `consolidation_inputs_fingerprint` and any other recomputed fingerprint hhemt accepts | the conformance test, logged in the change log below |
| OI-R1 | Whether member directories carry their own `_version.json` | the generator, from the sensitivity tree layout, confirmed by the conformance test |
| OI-R2 | The `.rpt` section list | the SWMM writer in `tests/goldens/generator/raw.py`, from EPA's `report.c`; the conformance test checks the result against `swmm_output_parser.py` |
| OI-P1 | The performance-store variable list at layout 26 | the summary writer in `tests/goldens/generator/summary.py`, from the document and the conformance test |
| OI-S1 | Model arms in the sensitivity golden | **CLOSED**: the coupled `tritonswmm` arm only (section 1; ground `src/hhemt/workflow.py::generate_master_snakefile_content`) |
| OI-S2 | `target_id` values and the member-fingerprint payload for a four-member experiment | **CLOSED**: one target, payload and byte form as in section 4.5 (ground `src/hhemt/sensitivity_analysis.py::_write_member_id_fingerprint`) |

**Rule for changing this file after the conformance test first reds.** A change to this file or the generator is made only with a recorded ground: the published document line, solver source line or measured constraint it follows. "hhemt's reader rejected the golden" is not a ground by itself; the ground is what the reader's rejection shows the documented behaviour to be. Every such change is recorded in the change log below with its ground.

**Change log (the generator repair made after the first conformance measurement).** Each entry names the ground, which is a source line or a measured constraint and never only "hhemt rejected it".

| Change | Ground |
|---|---|
| `[COORDINATES]` row for every junction, inside x 0 to 40 m, y 0 to 60 m (section 2) | `src/hhemt/analysis.py::_refuse_coupled_system_with_no_coupling_candidate` and `_refuse_coupling_nodes_outside_triton_grid`; `src/hhemt/scenario.py::coupling_candidates` |
| Sensitivity golden carries the `tritonswmm` arm only (sections 1, 2, 4.3 to 4.8, 5.1, 7, 8) | `src/hhemt/workflow.py::generate_master_snakefile_content` |
| `member-{m}_inputs.json` in hhemt's compare-and-write byte form (section 4.5) | `src/hhemt/sensitivity_analysis.py::_write_member_id_fingerprint` |
| `out_tritonswmm/log.out` states the scenario's `nTasks` and `OMP threads per task` (section 7.1; sensitivity members take their section 5.1 row) | `src/hhemt/analysis.py::df_status` and `src/hhemt/consolidate_workflow.py::validate_resource_usage`; measured by the conformance test: `Resource usage matches config` failed for 4 member scenarios with a constant `log.out` |
| The outfall carries a `[COORDINATES]` row too (section 2) | `src/hhemt/report_renderers/system_overview.py::_draw_hydraulics_panel` (`coords_df.at[n, ...]` for every outfall); measured on an extracted golden: `reprocess` failed in `plot_system_overview` with `KeyError: 'dummy_outfall'` |
| The TRITON discharge variables are named `unit_discharge_x`, `unit_discharge_y`, `max_unit_discharge`, `time_of_max_unit_discharge` and the two `unit_discharge_{x,y}_at_time_of_max_unit_discharge` (units `m2 s-1`, `min`; no `standard_name`), replacing the `velocity_*` names (sections 8) | `docs/reference/output-data-model.md` "2D surface results (TRITON)" as revised by the approved shape of run hhemt_11-01_1916_qx-velocity-label-fix, section 1.2: the solver's `QX` and `QY` are depth-integrated flow per meter of flow width, and the CF standard-name table has no name for it |
| The TRITON depth variables are named `max_flood_depth`, `flood_depth`, `flood_depth_last_tstep` and `time_of_max_flood_depth` (units `m`, `min`; `standard_name` `flood_water_thickness` on the three depth variables), replacing the retired water-level names (section 8) | `docs/reference/output-data-model.md` "2D surface results (TRITON)" as revised by the approved shape of run hhemt_11-01_1916_qx-velocity-label-fix, section 1.2: the solver's `H` and `MH` are depth above the bed, and the CF standard-name table (version 95) names that quantity `flood_water_thickness` |
| The TRITON summary carries `max_flood_velocity` (`flood_water_speed`, `m s-1`), `time_of_max_flood_velocity` (`min`) and the two components `x_flood_velocity_at_time_of_max_flood_velocity` and `y_flood_velocity_at_time_of_max_flood_velocity` (`m s-1`); the raw `QX` and `QY` files hold depth times a velocity field, so the golden flood velocity is the maximum over steps of the written discharge magnitude divided by the same step's depth, NaN at or below 0.001 m (section 7.1, section 8) | `docs/reference/output-data-model.md` "2D surface results (TRITON)" as revised by the approved shape of run hhemt_11-01_1916_qx-velocity-label-fix, sections 3.2 and 3.3: flood velocity is unit discharge over the same-print depth, undefined at or below the wet-depth threshold, and a maximum over output prints only |
| Every numeric data variable of every store carries a `units` attribute: the SWMM node and link variables (time series, `_last` and `_max` reductions, report columns), the nine performance columns (`s`) and the member counts of the experiment `parameters` dataset (`1`); the golden trees previously carried a unit on the TRITON and four SWMM variables only (section 8) | `docs/reference/output-data-model.md` as revised by the approved shape of run hhemt_11-01_1916_qx-velocity-label-fix, sections 0 item 3, 1.1 and 3.3: the unit moves out of the variable name and into a `units` attribute on every stored numeric variable, so a store that names no unit for a numeric variable is wrong |
| The time axis is `timestep` (was `timestep_min`) with `units: min` on the coordinate, and the three SWMM dataset attribute keys carry a `{key}_units` companion; the TRITON maximum variables' `cell_methods` is `time: maximum` and the last-step depth's `time: point`, the toolkit table's spelling, replacing `timestep_min: maximum` and `timestep_min: point` (sections 8) | `docs/reference/output-data-model.md` as revised by the approved shape of run hhemt_11-01_1916_qx-velocity-label-fix, sections 1.4 and 3.6: no unit token in a coordinate or attribute-key name, the unit in a `units` attribute or a `{key}_units` companion, time units spelled `min` so that no xarray version decodes them to a timedelta; `time` is the CF standard name class-3 spelling, `timestep` is now a dimension name and so cannot be a cell_methods name |
| The `_version.json` layout stamp is 28 (was 27), and the baseline sentences name layout 28 (section 3) | `src/hhemt/version_migration/constants.py::LAYOUT_VERSION` raised by the approved shape of run hhemt_11-01_1916_qx-velocity-label-fix, section 5.9: chunk 1a.9 adds migration V0028 and the golden trees are written at the layout the toolkit now stamps; no store content changes |
| The `_version.json` layout stamp is 29 (was 28), and the baseline sentences name layout 29 (sections 3 and 4.1) | `src/hhemt/version_migration/constants.py::LAYOUT_VERSION` raised by chunk 3c.10 of run hhemt_10-31_2206_gb-public-release-readiness, which adds migration `V0029__remove_unread_settings`; measured: `LayoutVersionError: current=28, target=29: refusing to relabel` raised by every analysis construction over the layout-28 goldens, 26 tests in the Rivanna run of job 22465193; ruled by the developer's `agreed and approved` of 2026-11-04 (erratum E-22); no store content changes |
| The `_version.json` layout stamp is 30 (was 29) (section 3) | `src/hhemt/version_migration/constants.py::LAYOUT_VERSION` raised by the migration `V0030__swmm_run_attributes_to_variables` (the three SWMM run-level attributes become scalar data variables); the golden trees are written at the layout the toolkit now stamps; the stamp is the only change: the SWMM stores of the goldens still carry the three figures as dataset attributes (section 8), the shape the layout-29 parser wrote, so a golden does not exercise the layout-30 variables |
| Each `tritonswmm` event carries `swmm/coupling_nodes.txt` listing the 13 junctions (section 4.8) | `src/hhemt/process_simulation.py::_attach_coupling_outputs` reads the scenario's coupling-node list when the coupled node store is written, and a reprocess of the golden failed with `FileNotFoundError` on `swmm/coupling_nodes.txt` (Rivanna job 22608592); `src/hhemt/scenario.py::coupling_nodes_file_for` and `scenario_inputs.py` name the file scenario preparation writes, under the system's default `all_junctions` mode every junction, each in its own cell (section 2) |
| The TRITON TIMESERIES variables (`max_flood_depth`, `flood_depth`, `unit_discharge_x`, `unit_discharge_y`) carry no `cell_methods`; the SUMMARY variables keep theirs (`time: maximum` on the maxima, `time: point` on the last-step depth) (section 8) | Decision D22 of the approved shape's re-open, ruled by the developer on 2026-11-02 ("agreed and approved on all fronts"): a reduction method describes a summary, not a series that keeps its time axis; the toolkit's `cf_conventions.timeseries_variable_attrs` writes `units`, `long_name` and `standard_name` and no `cell_methods`, and shape section 3.4 describes the series' attribute set without it; shape section 1.2's table cell that listed `time: maximum` for the series is the contradicted text |
| Goldens regenerated at layout 27 with a consolidated-metadata index in every store (sections 3, 4.1 and 8; generator `LAYOUT_VERSION = 27`, stores written with `consolidated=True`) | `src/hhemt/version_migration/constants.py::LAYOUT_VERSION` (27) and `src/hhemt/version_migration/versions/V0027__consolidate_zarr_metadata.py`; `src/hhemt/utils.py::zarr_store_has_index`; measured: `LayoutVersionError: current=26, target=27: refusing to relabel` raised by every analysis construction over the layout-26 goldens (Rivanna, `tests/test_synth_combine_pip2.py`, `tests/test_synth_06_report_target_predicate.py`, `tests/test_synth_08_sensitivity_reprocess.py`), and the migrated copy's index content equal to the regenerated one |
