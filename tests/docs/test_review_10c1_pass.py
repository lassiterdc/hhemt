"""Reviewer tests for chunk 10c.1: the interactive Python tutorial notebook and its committed example result.

Two properties that `tests/docs/test_retrofit_interactive_pass.py` does not read, each with a control that turns it
red on a planted defect.

1. The variable names the notebook reads, and the variable names the committed store holds, are names the toolkit
   writes at the current layout. The current names are `max_flood_depth` for TRITON depth, `max_flood_rate` for the
   node tier and `max_flow` for the link tier. A notebook that reads a name the toolkit does not write fails with a
   KeyError on the output of a reader's own run, and the committed store alone cannot show it.
   Unit: variable names. Matcher: set difference against the keys of `hhemt.cf_conventions._CF_VARIABLE_MAP` and the
   per-mode overrides, the maps the toolkit stamps stored variables from.
2. The compute-configuration table of the notebook names `run_mode` values, dispatch values and `n_*` fields that the
   analysis config accepts. No other test reads the table, so a changed value goes unseen without this one.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
import pytest
import xarray as xr

pytestmark = pytest.mark.docs

_REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = _REPO / "docs/tutorials/interactive-python.ipynb"
STORE = _REPO / "docs/tutorials/assets/irene-example-result.zarr"
_READ_NAME = re.compile(r'\.to_dataset\(\)\["([^"]+)"\]')


def current_variable_names() -> set[str]:
    from hhemt import cf_conventions

    names = set(cf_conventions._CF_VARIABLE_MAP)
    for per_mode in cf_conventions._CF_VARIABLE_OVERRIDES_BY_MODE.values():
        names |= set(per_mode)
    return names


def stale(names: set[str]) -> list[str]:
    """The names in `names` that the toolkit no longer writes."""
    return sorted(names - current_variable_names())


def notebook_read_names(nb) -> set[str]:
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    return set(_READ_NAME.findall(code))


def store_variable_names(store: Path) -> set[str]:
    tree = xr.open_datatree(store, engine="zarr", consolidated=False)
    return {name for node in tree.subtree for name in node.to_dataset().data_vars}


def test_notebook_reads_only_names_the_toolkit_writes_now():
    assert stale(notebook_read_names(nbformat.read(NOTEBOOK, as_version=4))) == []


def test_committed_store_holds_only_names_the_toolkit_writes_now():
    assert stale(store_variable_names(STORE)) == []


def test_control_the_retired_names_are_found_and_the_current_names_are_not():
    no_longer_written = {
        "max_wlevel" + "_m",
        "max_flow" + "_cms",
    }  # joined, so no tracked line spells a name the census forbids
    assert stale(no_longer_written) == sorted(no_longer_written)
    assert stale({"max_flood_depth", "max_flow"}) == []


def table_rows(nb) -> list[list[str]]:
    cell = next(c for c in nb.cells if c.cell_type == "markdown" and "| Config |" in c.source)
    rows = [line.strip().strip("|").split("|") for line in cell.source.splitlines() if line.strip().startswith("|")]
    return [[c.strip() for c in row] for row in rows[2:]]


def table_problems(rows: list[list[str]]) -> list[str]:
    from hhemt.config.analysis import analysis_config

    run_modes = set(analysis_config.model_fields["run_mode"].annotation.__args__)
    dispatches = set(analysis_config.model_fields["multi_sim_run_method"].annotation.__args__)
    fields = set(analysis_config.model_fields)
    problems: list[str] = []
    seen = set()
    for config, run_mode, key_fields, dispatch in rows:
        seen.add(run_mode)
        if run_mode not in run_modes:
            problems.append(f"{config}: run_mode {run_mode} is not accepted")
        problems += [
            f"{config}: dispatch {d.strip()} is not accepted"
            for d in re.split(r"/", dispatch)
            if d.strip() not in dispatches
        ]
        problems += [
            f"{config}: field {f} is not an analysis config field"
            for f in re.findall(r"\bn_[a-z]+(?:_[a-z]+)*", key_fields)
            if f not in fields
        ]
    problems += [f"run_mode {m} has no row" for m in sorted(run_modes - seen)]
    return problems


def test_compute_table_names_values_and_fields_the_config_accepts():
    assert table_problems(table_rows(nbformat.read(NOTEBOOK, as_version=4))) == []


def test_control_a_renamed_run_mode_dispatch_or_field_is_found():
    rows = table_rows(nbformat.read(NOTEBOOK, as_version=4))
    assert table_problems(rows) == []
    assert any(
        "run_mode omp is not accepted" in p
        for p in table_problems([[*r[:1], "omp", *r[2:]] if r[1] == "openmp" else r for r in rows])
    )
    assert any(
        "dispatch sbatch is not accepted" in p
        for p in table_problems([[*r[:3], "sbatch"] if r[3] == "batch_job" else r for r in rows])
    )
    assert any(
        "n_threads" in p
        for p in table_problems([[*r[:2], r[2].replace("n_omp_threads", "n_threads"), r[3]] for r in rows])
    )
    assert any("has no row" in p for p in table_problems([r for r in rows if r[1] != "hybrid"]))
