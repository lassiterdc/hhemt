"""Two statements of the benchmarking tutorial, pinned to the code that decides them.

The review of the tutorials pages found two sentences of `docs/tutorials/benchmarking-study.md`
that the code of this branch had made false: the row that says a bare analysis-config column is
still read (the loader now refuses it), and the setup 4 instruction to unzip each bundle before
`hhemt combine` (the command takes the zips as they are). Each test locates its statement in the
page by a stable anchor, derives the matching fact from the code, and compares the two as sets.
No test runs a solver, compares a count or compares a sentence of prose. A control per test feeds
the page-side check the statement as it stood before the correction (expected to be refused) and
feeds the code-side check an input that must not be refused (expected to pass), so a comparison
that stops comparing turns a control red.
"""

from __future__ import annotations

import re
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from hhemt.bundle._combine import _BundleSource, _open_bundle_source
from hhemt.config.analysis import analysis_config
from hhemt.exceptions import ConfigurationError
from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis

_REPO = Path(__file__).resolve().parents[2]
_PAGE = "docs/tutorials/norfolk-compute-config-sweep.md"
_BARE_ROW_ANCHOR = "| A bare analysis-config field name |"
_SETUP_4_HEADING = "## The same experiment on a second machine"

# The row and the setup 4 blocks as the page stated them before the correction, kept as literals so
# the controls feed the page-side checks the statement the checks exist to refuse.
_ROW_BEFORE = (
    "| A bare analysis-config field name | `run_mode` | Deprecated: still read, but the loader warns "
    "on every such cell and names `analysis.{field}` as the replacement, so the form is not one to "
    "author a new sheet in. |"
)
_SETUP_4_BLOCKS_BEFORE = [
    "unzip benchmarking_second_frontier/render_bundle/*.zip -d bundles/frontier/\n"
    "unzip benchmarking_second_uva/render_bundle/*.zip -d bundles/uva/\n"
    "\n"
    "hhemt combine bundles/frontier/ bundles/uva/ -o combined/",
]


def _page_text() -> str:
    """The text of the benchmarking tutorial in the tree under test."""
    return (_REPO / _PAGE).read_text(encoding="utf-8")


def bare_column_cell(row: str) -> str:
    """The third cell (the description) of a sheet-forms table row, stripped of surrounding space."""
    return row.split("|")[3].strip()


def page_bare_column_row(text: str) -> str:
    """The table row whose first cell is `A bare analysis-config field name`, found by that anchor."""
    return next(ln for ln in text.splitlines() if ln.startswith(_BARE_ROW_ANCHOR))


def row_states_refusal(row: str) -> bool:
    """True when the row's description opens with `Refused` and names neither `still read` nor `Deprecated`."""
    cell = bare_column_cell(row)
    return cell.startswith("Refused") and "still read" not in cell and "Deprecated" not in cell


def _call_retrieve_df_setup(tmp_path: Path, columns: list[str]):
    """Call `_retrieve_df_setup` unbound on a sheet whose columns are `columns` (plus `member_id`).

    The method reads only `experiment.cfg_analysis.sensitivity_analysis` from `self`, so a
    SimpleNamespace chain stands in for the analysis object and no solver or system is built.
    """
    sheet = tmp_path / "sheet.csv"
    pd.DataFrame({"member_id": ["a", "b"], **{c: ["1", "2"] for c in columns}}).to_csv(sheet, index=False)
    stand_in = types.SimpleNamespace(
        experiment=types.SimpleNamespace(cfg_analysis=types.SimpleNamespace(sensitivity_analysis=sheet))
    )
    return TRITONSWMM_sensitivity_analysis._retrieve_df_setup(stand_in)


def bare_names_refused_naming_prefixed_form(tmp_path: Path) -> set[str]:
    """The set of analysis-config field names the loader refuses as a bare column, naming `analysis.{name}`."""
    refused: set[str] = set()
    for name in sorted(analysis_config.model_fields):
        try:
            _call_retrieve_df_setup(tmp_path, [name])
        except ConfigurationError as err:
            if f"analysis.{name}" in str(err):
                refused.add(name)
    return refused


def test_the_bare_column_row_states_the_refusal_the_loader_raises(tmp_path):
    """Property: the page row for a bare analysis-config column says `Refused`, as the loader does for every field.

    Class: REGRESSION on the page (the row said `Deprecated: still read` on the pre-change page).
    Wrong implementation: restoring the deprecation wording, or stating a refusal for a subset of
    the fields. Reachability: the row is found by its first cell, and the loader is called once per
    field of `analysis_config`. Both-states anchor: the control below feeds the pre-change row and
    expects the refusal. Second correct implementation: a row that opens `Refused` and lists the
    fields in a different order.
    """
    assert row_states_refusal(page_bare_column_row(_page_text()))
    assert bare_names_refused_naming_prefixed_form(tmp_path) == set(analysis_config.model_fields)


def test_the_bare_column_checks_refuse_the_pre_change_row_and_accept_a_prefixed_column(tmp_path):
    """Control: the pre-change row is refused by the page check, and a prefixed column raises nothing.

    The second assertion varies the property under suspicion (prefixed versus bare form) and not the
    token, so a loader that refused every column would turn it red.
    """
    assert not row_states_refusal(_ROW_BEFORE)
    assert "analysis.run_mode" in _call_retrieve_df_setup(tmp_path, ["analysis.run_mode"]).columns


def setup_4_blocks(text: str) -> list[str]:
    """The fenced `bash` blocks of the `## Setup 4:` section, each as its inner text."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(_SETUP_4_HEADING))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    section = "\n".join(lines[start:end])
    return re.findall(r"^```bash\n(.*?)^```$", section, flags=re.S | re.M)


def combine_positionals(block: str) -> list[str]:
    """The positional arguments of each `hhemt combine` command of a block, backslash continuations joined."""
    joined = re.sub(r"\\\n\s*", " ", block)
    args: list[str] = []
    for ln in joined.splitlines():
        if not ln.startswith("hhemt combine"):
            continue
        tokens = ln.split()[2:]
        skip = False
        for tok in tokens:
            if skip:
                skip = False
            elif tok in ("-o", "--output"):
                skip = True
            elif not tok.startswith("-"):
                args.append(tok)
    return args


def input_kind(arg: str) -> str:
    """`zip` for an argument ending `.zip`, `dir` for one ending `/`, `other` for anything else."""
    if arg.endswith(".zip"):
        return "zip"
    return "dir" if arg.endswith("/") else "other"


def kinds_the_code_accepts(tmp_path: Path) -> set[str]:
    """The input kinds `_open_bundle_source` returns a source for: a zip file and a directory, from real paths."""
    folder = tmp_path / "bundle_dir"
    folder.mkdir()
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("manifest.json", "{}")
    accepted: set[str] = set()
    for kind, path in (("zip", archive), ("dir", folder)):
        if isinstance(_open_bundle_source(path), _BundleSource):
            accepted.add(kind)
    return accepted


def setup_4_combine_is_consistent_with_the_code(blocks: list[str], accepted: set[str]) -> bool:
    """True when the blocks carry a combine command, no `unzip` line, and every argument kind is accepted."""
    args = [a for block in blocks for a in combine_positionals(block)]
    no_unzip = not any(ln.lstrip().startswith("unzip") for block in blocks for ln in block.splitlines())
    return (
        bool(args)
        and no_unzip
        and {input_kind(a) for a in args} <= accepted
        and "other" not in {input_kind(a) for a in args}
    )


def test_setup_4_combines_the_bundle_zips_without_an_unzip_step(tmp_path):
    """Property: setup 4 passes bundle zips to `hhemt combine` with no `unzip` line, and the loader takes zips.

    Class: REGRESSION on the page (the pre-change blocks unzipped each bundle first). Wrong
    implementation: restoring the `unzip` lines, or passing an argument that is neither a zip nor a
    directory. Reachability: the blocks are read from the `## Setup 4:` section by heading, and the
    code side opens a real zip and a real directory. Both-states anchor: the control below feeds
    the pre-change blocks and expects the refusal. Second correct implementation: the command
    written on one line instead of two.
    """
    accepted = kinds_the_code_accepts(tmp_path)
    assert accepted == {"zip", "dir"}
    assert setup_4_combine_is_consistent_with_the_code(setup_4_blocks(_page_text()), accepted)
    assert "zip" in {input_kind(a) for b in setup_4_blocks(_page_text()) for a in combine_positionals(b)}


def test_the_combine_checks_refuse_the_pre_change_blocks_and_the_loader_refuses_a_plain_file(tmp_path):
    """Control: the pre-change blocks are refused by the page check, and a file that is no zip is refused by the code.

    The second assertion varies the property under suspicion (a path that is neither a bundle
    directory nor a zip file) and not the token, so a loader that accepted every path would turn it red.
    """
    assert not setup_4_combine_is_consistent_with_the_code(_SETUP_4_BLOCKS_BEFORE, {"zip", "dir"})
    plain = tmp_path / "plain.txt"
    plain.write_text("not a zip", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="neither a bundle directory nor a bundle zip file"):
        _open_bundle_source(plain)
