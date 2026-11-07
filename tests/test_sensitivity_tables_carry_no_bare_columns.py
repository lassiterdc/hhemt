"""Every golden zip and every tracked sensitivity table builds through `_retrieve_df_setup` without a bare column.

`_retrieve_df_setup` refuses a sensitivity-table header cell that names an `analysis_config` field without the
`analysis.` prefix. The refusal fires when a table is read, so a golden zip or a tracked table that keeps a bare
header cell breaks every test that constructs an analysis from it, and no conflict marker or import error shows
the defect beforehand. A merge that brings in a table written before the refusal landed is the case this test
covers: the table merges cleanly and fails at construction.

The test calls `_retrieve_df_setup` on a stand-in that holds only the path of the table, so it constructs no
analysis, compiles no solver and runs no simulation, and it runs on the laptop.

Two sets are read:

* every `inputs/sensitivity.csv` that a zip under `tests/fixtures/goldens/` carries (a golden that carries no table,
  as the multi-simulation golden, is skipped);
* every tracked `.csv` or `.xlsx` file whose header holds a `member_id` column, except the `scenario_status.csv`
  report tables, whose `member_id` column repeats by design and are outputs rather than definitions.

A table that fails for a reason other than a bare column (the older benchmarking workbooks under `test_data/` carry
columns that the current schema lacks) is outside this test's property and stays unreported here.
"""

from __future__ import annotations

import subprocess
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from hhemt.exceptions import ConfigurationError
from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis

_REPO = Path(__file__).resolve().parents[1]
_GOLDEN_DIR = _REPO / "tests" / "fixtures" / "goldens"
_GOLDEN_TABLE = "inputs/sensitivity.csv"
# The sentence `_retrieve_df_setup` raises with when a header cell is a bare `analysis_config` field name.
_BARE_MARKER = "are bare analysis_config field names"


def _bare_column_refusal(table: Path) -> str | None:
    """The refusal text when `table` carries a bare analysis column, else None.

    Any other failure of `_retrieve_df_setup` returns None, because the property here is the bare column only."""
    holder = types.SimpleNamespace(
        experiment=types.SimpleNamespace(cfg_analysis=types.SimpleNamespace(sensitivity_analysis=table))
    )
    try:
        TRITONSWMM_sensitivity_analysis._retrieve_df_setup(holder)
    except ConfigurationError as exc:
        return str(exc) if _BARE_MARKER in str(exc) else None
    except (ValueError, KeyError):
        return None
    return None


def _golden_tables(tmp_path: Path) -> dict[str, Path]:
    """Extract each golden zip's sensitivity table into `tmp_path`; map `{zip name}` to the extracted file."""
    out: dict[str, Path] = {}
    for zip_path in sorted(_GOLDEN_DIR.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            if _GOLDEN_TABLE not in zf.namelist():
                continue
            dest = tmp_path / zip_path.stem
            dest.mkdir()
            out[zip_path.name] = Path(zf.extract(_GOLDEN_TABLE, dest))
    return out


def _tracked_tables() -> list[Path]:
    """Tracked `.csv` and `.xlsx` files whose header holds `member_id`, minus the `scenario_status.csv` outputs."""
    listing = subprocess.run(
        ["git", "ls-files", "-z"], cwd=_REPO, capture_output=True, text=True, check=True
    ).stdout.split("\0")
    tables: list[Path] = []
    for rel in listing:
        path = _REPO / rel
        if path.name == "scenario_status.csv" or path.suffix not in (".csv", ".xlsx") or not path.is_file():
            continue
        try:
            header = (
                list(pd.read_csv(path, nrows=0).columns)
                if path.suffix == ".csv"
                else list(pd.read_excel(path, nrows=0).columns)
            )
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if "member_id" in header:
            tables.append(path)
    return tables


def test_every_golden_zip_table_builds_without_a_bare_column(tmp_path):
    """Property: no golden zip carries a bare analysis column in its sensitivity table.

    Class: REGRESSION for the merge of `develop`, whose sensitivity golden was written with bare header cells and
    merged without a conflict marker. Must-fire control: `test_the_probe_flags_a_bare_column` feeds the same
    function a table with a bare header cell and requires the refusal."""
    tables = _golden_tables(tmp_path)
    assert tables, "no golden zip carries a sensitivity table, so the probe read nothing"
    flagged = {name: _bare_column_refusal(path) for name, path in tables.items()}
    assert {name: text for name, text in flagged.items() if text} == {}


def test_every_tracked_sensitivity_table_builds_without_a_bare_column():
    """Property: no tracked sensitivity table carries a bare analysis column."""
    tables = _tracked_tables()
    assert tables, "no tracked file has a member_id column, so the probe read nothing"
    flagged = {str(path.relative_to(_REPO)): _bare_column_refusal(path) for path in tables}
    assert {name: text for name, text in flagged.items() if text} == {}


@pytest.mark.parametrize("header", ["member_id,run_mode,analysis.n_gpus", "member_id,analysis.run_mode,n_gpus"])
def test_the_probe_flags_a_bare_column(tmp_path, header):
    """Must-fire control: a table with one bare header cell, among prefixed ones, is refused with the bare marker.

    The two headers place the bare cell first and last, so the control varies the property under test (a bare
    cell present) and its position, and a matcher that read only the first cell would pass one of them."""
    table = tmp_path / "bare.csv"
    table.write_text(f"{header}\n0,serial,0\n")
    refusal = _bare_column_refusal(table)
    assert refusal is not None
    assert "analysis.run_mode" in refusal or "analysis.n_gpus" in refusal


def test_the_probe_accepts_a_prefixed_table(tmp_path):
    """Counter-control: the same shape with every cell prefixed is not flagged."""
    table = tmp_path / "prefixed.csv"
    table.write_text("member_id,analysis.run_mode,analysis.n_gpus\n0,serial,0\n")
    assert _bare_column_refusal(table) is None
