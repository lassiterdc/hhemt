# Release history

## v0.2.0 (2026-11-12)

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

<!-- hhemt:maturity-disclosure -->
> **A note on this entry, added after the fact.** This release was made to build and
> prove the machinery a public release requires. Its author did not consider the
> software ready for general outside use at the time, and the release intended for
> general use is planned alongside the paper introducing the software.
<!-- /hhemt:maturity-disclosure -->

**Installation — pip works; conda is recommended for HPC.** `pip install hhemt`
installs the package and pins the SWMM engine (`swmm-toolkit` and `pyswmm`)
directly, so a pip environment resolves a stack that passes the toolkit's runtime
validation guard. That guard is a deliberate fail-closed contract: before running
SWMM the toolkit checks the installed engine versions and raises rather than run
against a build it cannot vouch for. Conda is the recommended path for running
simulations because `environment.yaml` pins the whole HPC stack, including the
Snakemake SLURM executor plugins, not because pip cannot run SWMM.
