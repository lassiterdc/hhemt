# Release history

## v0.3.0 (2026-11-27)

hhemt now runs a regular, runnable SWMM model as a coupled TRITON-SWMM model. Set
`SWMM_model` in the system config to a plain SWMM 5.2 `.inp` file. The toolkit derives the
coupled hydraulics model, the hydrology model and the standalone SWMM model from it, and
for each event rewrites only the simulation window, the rain gauges and the file each
rain series points to.

Every outfall of the coupled model is now a stated decision. List an outfall in
`swmm_outfalls_to_couple` to turn it into a node that exchanges water with the TRITON
surface, or in `swmm_outfalls_to_keep` to keep it an outfall. A variable-water-level
outfall can instead be named by a water-level record of the external-boundary layer: the
toolkit converts it and drives the TRITON boundary along the record's line from the
weather variable the config names for it, in the units and datum the config states. The
layer may also hold free-outflow, normal-slope and Froude-number segments. SWMM models the
toolkit writes hold no time-series data: every series points to a file written once. A
converted node's SWMM depth comes from the DEM the config names. SWMM's thread count is
its own setting, `swmm_threads`, default 1. Boundary names carry no coastal terms:
`toggle_storm_tide_boundary` and `storm_tide_boundary_line_gis` are now
`toggle_external_boundaries` and `external_boundary_gis`; a config with an old key is
refused with the new name to write, and stored analyses migrate to layout version 34.

The fillable SWMM templates and the TRITON configuration template are retired: a config
that sets `SWMM_hydraulics`, `SWMM_hydrology`, `SWMM_full` or
`triton_swmm_configuration_template` is refused with a message that says what replaces it.
The TRITON settings the template fixed are now config settings, so an analysis
config sets `TRITON_courant_number` and `TRITON_hextra_m`.

Two further TRITON inputs are now config settings. List observation points in the
`observation_points` file of the system config, and TRITON writes depth and unit
discharge at those points; the series are consolidated into the analysis zarr store, and
no extra store is written. Set `TRITON_initial_depth_raster`, `TRITON_initial_qx_raster`
and `TRITON_initial_qy_raster` to start a run from initial-condition rasters.

## v0.2.1 (2026-11-14)

The home page and the README no longer carry the statement about outside use of the release. Nothing else changed.

## v0.2.0 (2026-11-14)

Relicensed under the GNU Affero General Public License v3.0 or later
(AGPL-3.0-or-later). Release 0.1.0, on PyPI and Zenodo, was published under the
PolyForm Noncommercial License 1.0.0.

A report whose figure files total fewer than 15,000,000 bytes is now a single HTML file
instead of a zip. The `report_html_mode` setting, `auto` by default, chooses the
container on every render that is given no explicit format, and the `report_formats`
argument of `run()` and `submit_workflow()` is now `override_report_formats`.

`hhemt run` now takes `--mode {resume,fresh}`, as `hhemt run-experiment` does, in place of `--from-scratch` and `--resume`. `--mode fresh` replaces `--from-scratch`, and `--mode resume`, the default, replaces `--resume`. `--overwrite` is unchanged.

The `compile_TRITON_SWMM` keyword of `submit_workflow()` is now `run_setup_phase`, because the setup rule it controls asserts the solver tier and compiles nothing; a call that passes the old keyword raises `TypeError`. The `System.compile_TRITON_SWMM()` method keeps its name.

A bundle now carries the report settings its report was rendered with, and a run or re-process whose report settings differ from the existing report's re-renders the figures and the report.

A bundle written before layout 29 that sets a report, EDA or land-use column setting that no module read fails to load, because the models refuse an unknown key; write the bundle again from its migrated analysis with the current toolkit.

## v0.1.0 (2026-07-14)

First public release of the H&H Ensemble Modeling Toolkit (hhemt). Coupled
TRITON-SWMM flood-ensemble orchestration across local + HPC (NVIDIA/AMD), full
preprocessing→compile→run→process→consolidate→report lifecycle, interactive
analysis report, and a portable render-bundle. Docs at
https://hhemt.readthedocs.io. Cite via the Zenodo DOI (see README).

**Installation — pip works; conda is recommended for HPC.** `pip install hhemt`
installs the package and pins the SWMM engine (`swmm-toolkit` and `pyswmm`)
directly, so a pip environment resolves a stack that passes the toolkit's runtime
validation guard. That guard is a deliberate fail-closed contract: before running
SWMM the toolkit checks the installed engine versions and raises rather than run
against a build it cannot vouch for. Conda is the recommended path for running
simulations because `environment.yaml` pins the whole HPC stack, including the
Snakemake SLURM executor plugins, not because pip cannot run SWMM.
