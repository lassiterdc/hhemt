"""The DEM-resolution route as the pages state it, against the matrix and the config the code builds.

The how-to section `## Running a DEM-resolution sweep instead` lists the member ids of the
matrix that `hhemt synth-experiment` builds for a set `dem_resolution_ladder`. The reference row
of the command and the docstring of `Toolkit.synthetic_experiment` name that field. These tests
compare the page's member ids with the ids `build_experiment_matrix` returns for the ladder the
page's own table holds, and compare each named field with `synthetic_experiment_config`. No test
writes a count: a set is compared with a set. Each control feeds a check a mutated text and expects
the refusal. No test here runs a solver.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.docs

_REPO = Path(__file__).resolve().parents[2]
_HOWTO = _REPO / "docs" / "how-to" / "synthetic-compute-sensitivity-experiment.md"
_CLI_PAGE = _REPO / "docs" / "reference" / "cli.md"
_HEADING = "## Running a DEM-resolution sweep instead"
_FIELD = "dem_resolution_ladder"


def _section(text: str) -> str:
    """The text from the DEM-resolution heading to the next heading of the same level."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == _HEADING)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


def _table(section: str) -> list[tuple[str, float]]:
    """(member id, rung) for each data row of the section's table whose header cell is `member_id`."""
    rows: list[tuple[str, float]] = []
    in_table = False
    for ln in section.splitlines():
        if not ln.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells[0] == "`member_id`":
            in_table = True
            continue
        if in_table and cells[0].startswith("`"):
            rows.append((cells[0].strip("`"), float(cells[-1])))
    return rows


def _matrix_ids(rungs: list[float], tmp_path: Path) -> set[str]:
    from hhemt.config.synthetic_experiment import synthetic_experiment_config
    from hhemt.synthetic_experiment import build_experiment_matrix

    hpc = tmp_path / "hpc.yaml"
    hpc.write_text("hpc_name: docs_check\npartitions:\n  standard:\n    max_runtime: 4320\n    cpus_per_node: 40\n")
    cfg = synthetic_experiment_config(
        hpc_system_config_yaml=hpc,
        ensemble_partition="standard",
        setup_partition="standard",
        dem_resolution_ladder=tuple(sorted(rungs)),
    )
    return set(build_experiment_matrix(cfg)["member_id"])


def check_member_ids(howto_text: str, tmp_path: Path) -> list[str]:
    """Problems: the section's member ids differ from the ids the matrix builder returns for its rungs."""
    rows = _table(_section(howto_text))
    if not rows:
        return ["the section holds no member_id table"]
    page_ids = {member for member, _ in rows}
    code_ids = _matrix_ids(sorted({rung for _, rung in rows}), tmp_path)
    problems = []
    if page_ids - code_ids:
        problems.append(f"only on the page: {sorted(page_ids - code_ids)}")
    if code_ids - page_ids:
        problems.append(f"only in the matrix: {sorted(code_ids - page_ids)}")
    return problems


def check_names_the_field(text: str) -> list[str]:
    """Problems: the text lacks a code span for the ladder field, or the config has no such field."""
    from hhemt.config.synthetic_experiment import synthetic_experiment_config

    problems = []
    if _FIELD not in synthetic_experiment_config.model_fields:
        problems.append(f"synthetic_experiment_config has no field {_FIELD}")
    if not re.search(rf"`{_FIELD}`", text):
        problems.append(f"the text names no `{_FIELD}` code span")
    return problems


_AXIS_CLAIM = re.compile(r"partition[- ]as[- ](an[- ])?axis")


def check_axis_claim(text: str) -> list[str]:
    """Problems: the text claims a partition axis without naming the ladder field that removes it."""
    if _AXIS_CLAIM.search(text) and f"`{_FIELD}`" not in text:
        return [f"the text claims a partition axis and does not qualify it by `{_FIELD}`"]
    return []


def _synth_row(cli_text: str) -> str:
    return next(ln for ln in cli_text.splitlines() if ln.startswith("| `synth-experiment` |"))


def _build_matrix_docstring() -> str:
    from hhemt.synthetic_experiment import build_experiment_matrix

    return build_experiment_matrix.__doc__ or ""


def _docstring() -> str:
    from hhemt.toolkit import Toolkit

    return Toolkit.synthetic_experiment.__doc__ or ""


def test_howto_member_ids_equal_the_matrix_ids(tmp_path):
    """Property: the how-to table names the member ids the matrix builder returns for the table's rungs.

    Class: any ladder the table holds. Why required: a reader copies the ids into a report filter.
    Killed by: a change of the id format in `dem_resolution_matrix_rows`, or a page that keeps ids the
    toolkit does not emit. A second correct implementation builds the ids from `experiment_matrix_rows`.
    """
    assert check_member_ids(_HOWTO.read_text(encoding="utf-8"), tmp_path) == []


def test_control_a_renamed_or_dropped_member_id_is_refused(tmp_path):
    text = _HOWTO.read_text(encoding="utf-8")
    first = _table(_section(text))[0][0]
    assert check_member_ids(text.replace(f"`{first}`", "`res_x`", 1), tmp_path)
    assert check_member_ids("\n".join(ln for ln in text.splitlines() if f"`{first}`" not in ln), tmp_path)


@pytest.mark.parametrize(
    "text_of",
    [
        lambda: _section(_HOWTO.read_text(encoding="utf-8")),
        lambda: _synth_row(_CLI_PAGE.read_text(encoding="utf-8")),
        _docstring,
    ],
    ids=["howto-section", "cli-reference-row", "toolkit-docstring"],
)
def test_pages_and_docstring_name_the_ladder_field(text_of):
    """Property: each text that states the ladder route names the field the config defines.

    Class: the three texts that state it. Killed by: a rename of the field without the texts.
    """
    assert check_names_the_field(text_of()) == []


def test_control_a_renamed_field_is_refused():
    assert check_names_the_field(f"`{_FIELD}s` selects the rows")


@pytest.mark.parametrize(
    "text_of",
    [
        lambda: _synth_row(_CLI_PAGE.read_text(encoding="utf-8")),
        _docstring,
        _build_matrix_docstring,
    ],
    ids=["cli-reference-row", "toolkit-docstring", "build-experiment-matrix-docstring"],
)
def test_a_partition_axis_claim_is_qualified_by_the_ladder_field(text_of):
    """Property: a text that says the matrix has the partition as an axis names the field that removes it.

    Class: the three texts that describe the matrix. Why required: a ladder config builds rows that
    all hold one partition. Killed by: an unqualified partition-axis sentence returning to a text.
    """
    assert check_axis_claim(text_of()) == []


def test_ladder_rows_hold_the_fixed_partition_and_the_compute_rows_more(tmp_path):
    """Code fact behind the claim above: the ladder rows share the fixed configuration's partition.

    Class: a ladder config against the empty-ladder config of one hpc file. Killed by: ladder rows
    that spread over partitions, or compute rows that collapse to one.
    """
    from hhemt.config.synthetic_experiment import synthetic_experiment_config
    from hhemt.synthetic_experiment import _DEM_FIXED_CONFIG, build_experiment_matrix

    hpc = tmp_path / "hpc.yaml"
    gpu = "    max_runtime: 4320\n    gpus_per_node: 8\n    cpus_per_node: 40\n    max_gpu: 8\n"
    hpc.write_text(
        "hpc_name: docs_check\npartitions:\n"
        f"  gpu-a6000:\n{gpu}    gpu_hardware: a6000\n    gpu_compilation_backend: CUDA\n"
        f"  gpu-a100-80:\n{gpu}    gpu_hardware: a100\n    gpu_compilation_backend: CUDA\n"
        "  standard:\n    max_runtime: 4320\n    cpus_per_node: 40\n"
    )

    def partitions(ladder):
        cfg = synthetic_experiment_config(
            hpc_system_config_yaml=hpc,
            ensemble_partition="gpu-a6000",
            setup_partition="standard",
            dem_resolution_ladder=ladder,
        )
        return set(build_experiment_matrix(cfg)["hpc.partition"])

    assert partitions((3.5, 7.0, 14.0)) == {_DEM_FIXED_CONFIG[5]}
    assert partitions(()) > partitions((3.5, 7.0, 14.0))
