"""Tests of the two tutorials, the tutorials index and the prerequisites snippet, each with a must-fire control.

Each tutorial carries its ruled heading and navigation title, shows one image under a link to its full report, and links
the installation guide for prerequisites. The page-versus-code statements of the benchmarking tutorial are pinned to the
code that decides them: the loader refuses a bare analysis-config column, and `hhemt combine` takes bundle zips as they
are.

No test builds an analysis, compiles or runs a solver, or compares a count or a sentence of prose. Each page-side
check is one callable that the primary test and its control both call, so a check that stops checking turns a
control red.
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
_TUTORIALS = _REPO / "docs" / "tutorials"
_ENSEMBLE = "docs/tutorials/norfolk-observed-events.md"
_BENCHMARKING = "docs/tutorials/norfolk-compute-config-sweep.md"
_REMOVED = "docs/tutorials/benchmarking-study.md"
_TITLES = {
    _ENSEMBLE: ("# Simulating an event ensemble", "Simulating an event ensemble"),
    _BENCHMARKING: ("# Running a benchmarking experiment", "Running a benchmarking experiment"),
}
_REPORTS = {
    _ENSEMBLE: "reference/bundles/norfolk-observed-events-report.html",
    _BENCHMARKING: "reference/bundles/norfolk-compute-config-sweep-report.html",
}
_BARE_ROW_ANCHOR = "| A bare analysis-config field name |"
_SECOND_MACHINE_HEADING = "## The same experiment on a second machine"

_IMAGE = re.compile(r"^!\[[^\]]*\]\(([^)\s]+)\)\s*$")
_LINK_LINE = re.compile(r"^\[[^\]]+\]\(([^)\s]+\.html)\)")
_EMBED = re.compile(r"<iframe|--8<--[^\n]*report\.html")


def _text(page: str) -> str:
    return (_REPO / page).read_text(encoding="utf-8")


def first_heading(text: str) -> str:
    """The first line of `text` that opens with `# `."""
    return next(ln for ln in text.splitlines() if ln.startswith("# "))


def nav_titles(mkdocs_text: str) -> dict[str, str]:
    """Navigation entries of `mkdocs.yml` that carry a title, as path -> title."""
    pairs = re.findall(r"^\s*-\s+([^:\n]+):\s+(\S+\.md)\s*$", mkdocs_text, flags=re.M)
    return {path: title for title, path in pairs}


def title_problems(page: str, text: str, mkdocs_text: str) -> list[str]:
    """A heading or navigation title of `page` that differs from the ruled name."""
    heading, title = _TITLES[page]
    problems = []
    if first_heading(text) != heading:
        problems.append(f"{page}: first heading {first_heading(text)!r}, ruled {heading!r}")
    if nav_titles(mkdocs_text).get(page.removeprefix("docs/")) != title:
        problems.append(f"mkdocs.yml: navigation title of {page} is not {title!r}")
    return problems


def test_each_tutorial_carries_its_ruled_heading_and_navigation_title():
    """Property: each tutorial carries the ruled heading and navigation title, and no benchmarking-study page exists.

    Class: REGRESSION on the rename. Wrong implementation: renaming the heading and not the navigation entry, or the
    reverse. Reachability: the heading is the first `# ` line and the title is the `mkdocs.yml` entry of the page path.
    """
    mkdocs = (_REPO / "mkdocs.yml").read_text(encoding="utf-8")
    for page in _TITLES:
        assert title_problems(page, _text(page), mkdocs) == []
    assert _REMOVED.removeprefix("docs/") not in mkdocs
    assert not (_REPO / _REMOVED).exists()


def test_the_title_check_refuses_the_pre_change_headings_and_navigation():
    """Control: the pre-change heading and navigation title of each tutorial are refused."""
    before_heading = "# Observed events: an ensemble over observed storms\n"
    before_nav = "  - Observed events: tutorials/norfolk-observed-events.md\n"
    assert title_problems(_ENSEMBLE, before_heading, before_nav) != []
    assert title_problems(_BENCHMARKING, "# Compute configurations: cores per simulation\n", before_nav) != []


def figure_problems(text: str, page_dir: Path, report: str) -> list[str]:
    """Problems of the report figure of a tutorial: an embed, the image count, or the line above the image.

    Exactly one image must stand on its own line, the nearest non-blank line above it must be a link to the
    full report, and the report file and the image file must exist.
    """
    problems = []
    if _EMBED.search(text):
        problems.append("a report is embedded")
    lines = text.splitlines()
    images = [i for i, ln in enumerate(lines) if _IMAGE.match(ln)]
    if len(images) != 1:
        return problems + [f"{len(images)} images on their own line, ruled 1"]
    i = images[0]
    if not (page_dir / _IMAGE.match(lines[i]).group(1)).is_file():
        problems.append("the image file is missing")
    above = next((ln for ln in reversed(lines[:i]) if ln.strip()), "")
    link = _LINK_LINE.match(above)
    if link is None:
        problems.append("the line above the image is not a link to a report")
    elif not (page_dir / link.group(1)).resolve().is_file() or not link.group(1).endswith(report):
        problems.append(f"the link above the image does not reach {report}")
    return problems


@pytest.mark.parametrize("page", [_ENSEMBLE, _BENCHMARKING])
def test_each_tutorial_links_the_full_report_above_one_image(page):
    """Property: a tutorial shows one image with a link to its full report directly above, and embeds no report.

    Class: REGRESSION on the embed. Wrong implementation: restoring the frame, a second image, or the link below
    the image. Reachability: the page text is scanned line by line and the files are resolved from the page directory.
    """
    assert figure_problems(_text(page), (_REPO / page).parent, _REPORTS[page]) == []


def test_the_figure_check_refuses_each_mutation_of_a_tutorial():
    """Control: an embed, an image above the link, two images and a link to another report are each refused."""
    page = _BENCHMARKING
    text, here = _text(page), (_REPO / page).parent
    report = _REPORTS[page]
    image_line = next(ln for ln in text.splitlines() if _IMAGE.match(ln))
    link_line = next(ln for ln in text.splitlines() if _LINK_LINE.match(ln))
    swapped = text.replace(link_line, "\x00").replace(image_line, link_line).replace("\x00", image_line)
    mutations = [
        text + '\n<iframe src="../../reference/bundles/x.html"></iframe>\n',
        swapped,
        text + "\n" + image_line + "\n",
        text.replace(link_line, link_line.replace("compute-config-sweep", "observed-events")),
    ]
    for mutated in mutations:
        assert figure_problems(mutated, here, report) != []


def prerequisites_problems(snippet: str, tutorials: dict[str, str]) -> list[str]:
    """A tutorial that includes the prerequisites snippet, or a snippet that includes the platform statement."""
    problems = [
        f"{name} includes the prerequisites snippet" for name, text in tutorials.items() if "prerequisites-base" in text
    ]
    if "platform-support" in snippet:
        problems.append("the prerequisites snippet includes the platform statement")
    return problems


def test_tutorials_link_the_installation_guide_and_carry_no_prerequisites_snippet():
    """Property: no tutorial includes the prerequisites snippet, and the snippet states no platform.

    Class: REGRESSION on the move. Wrong implementation: restoring the include in a tutorial, or restoring the
    platform include in the snippet. Reachability: the tutorials are the pages of `docs/tutorials/` and the snippet
    is read by path.
    """
    tutorials = {p.name: p.read_text(encoding="utf-8") for p in sorted(_TUTORIALS.glob("*.md"))}
    snippet = (_REPO / "docs/_snippets/prerequisites-base.md").read_text(encoding="utf-8")
    assert prerequisites_problems(snippet, tutorials) == []
    assert "how-to/installation.md" in tutorials["index.md"]


def test_the_prerequisites_check_refuses_the_pre_change_pages():
    """Control: the pre-change snippet and a tutorial that includes the snippet are each refused."""
    before = '--8<-- "platform-support.md"\n\n- A working install. See [Installation](../how-to/installation.md).\n'
    assert prerequisites_problems(before, {}) != []
    assert prerequisites_problems("- A working install.\n", {"a.md": '--8<-- "prerequisites-base.md"\n'}) != []


def _call_retrieve_df_setup(tmp_path: Path, columns: list[str]):
    """Call `_retrieve_df_setup` unbound on a sheet whose columns are `columns` plus `member_id`."""
    sheet = tmp_path / "sheet.csv"
    pd.DataFrame({"member_id": ["a", "b"], **{c: ["1", "2"] for c in columns}}).to_csv(sheet, index=False)
    stand_in = types.SimpleNamespace(
        experiment=types.SimpleNamespace(cfg_analysis=types.SimpleNamespace(sensitivity_analysis=sheet))
    )
    return TRITONSWMM_sensitivity_analysis._retrieve_df_setup(stand_in)


def bare_column_row_states_refusal(text: str) -> bool:
    """True when the table row for a bare analysis-config column opens its description with `Refused`."""
    row = next(ln for ln in text.splitlines() if ln.startswith(_BARE_ROW_ANCHOR))
    cell = row.split("|")[3].strip()
    return cell.startswith("Refused") and "still read" not in cell and "Deprecated" not in cell


def test_the_bare_column_row_states_the_refusal_the_loader_raises(tmp_path):
    """Property: the row for a bare analysis-config column says `Refused`, and the loader refuses every field.

    Class: REGRESSION on the page. Wrong implementation: a deprecation wording, or a refusal for a subset of the
    fields. Reachability: the row is found by its first cell and the loader is called once per field.
    """
    assert bare_column_row_states_refusal(_text(_BENCHMARKING))
    refused = set()
    for name in sorted(analysis_config.model_fields):
        try:
            _call_retrieve_df_setup(tmp_path, [name])
        except ConfigurationError as err:
            if f"analysis.{name}" in str(err):
                refused.add(name)
    assert refused == set(analysis_config.model_fields)


def test_the_bare_column_check_refuses_the_deprecation_wording_and_the_loader_accepts_a_prefixed_column(tmp_path):
    """Control: a deprecation wording is refused, and a prefixed column raises nothing."""
    before = f"{_BARE_ROW_ANCHOR} `run_mode` | Deprecated: still read, but the loader warns on every such cell. |\n"
    assert not bare_column_row_states_refusal(before)
    assert "analysis.run_mode" in _call_retrieve_df_setup(tmp_path, ["analysis.run_mode"]).columns


def second_machine_blocks(text: str) -> list[str]:
    """The fenced `bash` blocks of the second-machine section, each as its inner text."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(_SECOND_MACHINE_HEADING))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return re.findall(r"^```bash\n(.*?)^```$", "\n".join(lines[start:end]), flags=re.S | re.M)


def combine_inputs(block: str) -> list[str]:
    """The positional arguments of each `hhemt combine` command of a block, continuations joined."""
    joined = re.sub(r"\\\n\s*", " ", block)
    args: list[str] = []
    for ln in joined.splitlines():
        if not ln.startswith("hhemt combine"):
            continue
        skip = False
        for tok in ln.split()[2:]:
            if skip:
                skip = False
            elif tok in ("-o", "--output"):
                skip = True
            elif not tok.startswith("-"):
                args.append(tok)
    return args


def combine_is_consistent_with_the_code(blocks: list[str], accepted: set[str]) -> bool:
    """True when the blocks carry a combine command, no `unzip` line, and every argument is a zip or directory."""
    args = [a for block in blocks for a in combine_inputs(block)]
    kinds = {"zip" if a.endswith(".zip") else "dir" if a.endswith("/") else "other" for a in args}
    no_unzip = not any(ln.lstrip().startswith("unzip") for block in blocks for ln in block.splitlines())
    return bool(args) and no_unzip and kinds <= accepted and "other" not in kinds


def _kinds_the_code_accepts(tmp_path: Path) -> set[str]:
    folder = tmp_path / "bundle_dir"
    folder.mkdir()
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("manifest.json", "{}")
    return {
        kind
        for kind, path in (("zip", archive), ("dir", folder))
        if isinstance(_open_bundle_source(path), _BundleSource)
    }


def test_the_second_machine_section_combines_the_bundle_zips_without_an_unzip_step(tmp_path):
    """Property: the section passes bundle zips to `hhemt combine` with no `unzip` line, and the loader takes zips.

    Class: REGRESSION on the page. Wrong implementation: restoring the `unzip` lines, or an argument that is neither
    a zip nor a directory. Reachability: the blocks are read from the section by heading and the code side opens a
    real zip and a real directory.
    """
    accepted = _kinds_the_code_accepts(tmp_path)
    assert accepted == {"zip", "dir"}
    blocks = second_machine_blocks(_text(_BENCHMARKING))
    assert combine_is_consistent_with_the_code(blocks, accepted)
    assert any(a.endswith(".zip") for b in blocks for a in combine_inputs(b))


def test_the_combine_check_refuses_an_unzip_step_and_the_loader_refuses_a_plain_file(tmp_path):
    """Control: an unzip step is refused by the page check, and a file that is no zip is refused by the code."""
    before = ["unzip a/render_bundle/*.zip -d bundles/a/\n\nhhemt combine bundles/a/ bundles/b/ -o combined/"]
    assert not combine_is_consistent_with_the_code(before, {"zip", "dir"})
    plain = tmp_path / "plain.txt"
    plain.write_text("not a zip", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="neither a bundle directory nor a bundle zip file"):
        _open_bundle_source(plain)
