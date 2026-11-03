"""A caption may bold only page elements that every page kind it is attached to prints.

A caption's bold spans name page elements as the page prints them (a column, row, panel,
section or note heading). A caption sentence about an element that exists under one page
kind only is false on the others. This test renders each page under every kind listed in
``PAGE_KINDS`` from the stub inputs in ``tests/report_style/page_kinds/{page}/{kind}/``
and asserts that every bold name of the page's caption appears in the rendering's text.

Page text is what a reader sees: markup and script bodies are removed, entities are
unescaped, and the table titles and cell values that a data grid carries in its options
object are added back.

Pages whose renderers need rasters, consolidated trees or combine inputs are not rendered
here, because a stub that reproduced them would be a second renderer. Their captions are
checked by the reviewer against each page kind.

To add a page, add its kinds to ``PAGE_KINDS`` with a stub-input directory and a function
that builds the renderer's stub analysis; ``render_page``, ``bold_names`` and
``missing_names`` are the reusable harness.
"""

from __future__ import annotations

import html as html_lib
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from hhemt.config.report import report_config

TESTS = Path(__file__).resolve().parent
KINDS_DIR = TESTS / "report_style" / "page_kinds"
CAPTIONS = TESTS.parent / "src" / "hhemt" / "report_templates" / "captions"


# ---- Reusable harness ------------------------------------------------------------------

_JINJA = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.S)
_BOLD = re.compile(r"\*\*([^*\n]+?)\*\*")
_SCRIPT_STYLE = re.compile(r"(?is)<(script|style)\b.*?</\1\s*>")
_TAG = re.compile(r"<[^>]*>")
_OPTIONS = 'const tableOptions = {"data": '
_TITLE = re.compile(r'"title": "([^"]*)"')


def bold_names(caption_text: str) -> list[str]:
    """Bold spans of a caption, minus ``Sources:`` and Jinja expressions."""
    masked = _JINJA.sub(" ", caption_text)
    names = [m.group(1).strip() for m in _BOLD.finditer(masked)]
    return [n for n in names if n and n.rstrip(":") != "Sources"]


def page_text(html: str) -> str:
    """The text a reader sees on a rendered page, grid titles and cells included."""
    extra: list[str] = []
    starts = [m.start() for m in re.finditer(re.escape(_OPTIONS), html)]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(html)
        data, _end = json.JSONDecoder().raw_decode(html[start + len(_OPTIONS) : end])
        for row in data:
            extra.extend(str(v) for v in row.values())
        extra.extend(_TITLE.findall(html[start:end]))
    visible = _TAG.sub(" ", _SCRIPT_STYLE.sub(" ", html))
    return html_lib.unescape(" ".join([visible, *extra]))


def missing_names(caption_text: str, rendered_text: str) -> list[str]:
    """Bold names of the caption that the rendering does not print."""
    return [n for n in bold_names(caption_text) if n not in rendered_text]


@dataclass(frozen=True)
class PageKind:
    page: str
    kind: str
    renderer_module: str
    caption: str
    build_analysis: Callable[[Path, dict], SimpleNamespace]

    @property
    def stub_dir(self) -> Path:
        return KINDS_DIR / self.page / self.kind

    @property
    def id(self) -> str:
        return f"{self.page}-{self.kind}"


def render_page(pk: PageKind, tmp_path: Path) -> str:
    """Copy the stub inputs into a fresh analysis directory, render, return the page text."""
    meta = json.loads((pk.stub_dir / "kind.json").read_text())
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)
    for src in pk.stub_dir.rglob("*"):
        if src.is_file() and src.name != "kind.json":
            dest = analysis_dir / src.relative_to(pk.stub_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
    analysis = pk.build_analysis(analysis_dir, meta)
    out = tmp_path / f"{pk.page}.html"
    renderer = import_module(f"hhemt.report_renderers.{pk.renderer_module}")
    renderer.render(analysis, report_config(), out)
    return page_text(out.read_text(encoding="utf-8"))


# ---- Stub analyses (inputs only; the renderers under test are the real ones) -----------


# Stub fidelity: the scenario_status.csv stubs under report_style/page_kinds/P06 carry a few columns where
# the exporter writes many more. Extend the columns when a P06 caption bolds a column name.
def _sub(n_events: int) -> SimpleNamespace:
    return SimpleNamespace(df_sims=pd.DataFrame({"event": range(n_events)}))


def _analysis(analysis_dir: Path, meta: dict) -> SimpleNamespace:
    members = meta.get("members")
    sensitivity = (
        SimpleNamespace(
            members={m: _sub(meta["n_events"]) for m in members},
            df_setup=pd.DataFrame({"member": members}),
        )
        if members
        else None
    )
    return SimpleNamespace(
        analysis_id="stub_analysis",
        analysis_paths=SimpleNamespace(
            analysis_dir=analysis_dir,
            simlog_directory=analysis_dir / "logs" / "sims",
        ),
        cfg_analysis=SimpleNamespace(
            analysis_id="stub_analysis",
            toggle_sensitivity_analysis=bool(members),
            weather_event_indices=["event_id"],
        ),
        sensitivity=sensitivity,
        df_sims=pd.DataFrame({"event": range(meta["n_events"])}),
        _get_enabled_model_types=lambda: meta["model_types"],
    )


def _kinds(page: str, renderer_module: str, caption: str, kinds: list[str]) -> list[PageKind]:
    return [PageKind(page, k, renderer_module, caption, _analysis) for k in kinds]


# P09 and P10 are added by the package that finishes their renderers, using this harness.
PAGE_KINDS: list[PageKind] = [
    *_kinds("P05", "per_analysis_summary", "per_analysis_summary_table.rst", ["plain", "sweep"]),
    *_kinds("P06", "scenario_status_appendix", "scenario_status_appendix.rst", ["plain", "sweep", "csv_missing"]),
    *_kinds(
        "P07",
        "errors_and_warnings",
        "errors_and_warnings.rst",
        ["failures", "all_not_applicable", "record_missing"],
    ),
    *_kinds("P08", "disk_utilization", "disk_utilization.rst", ["live", "record_missing"]),
]
IDS = [pk.id for pk in PAGE_KINDS]


def _by_id(page_kind_id: str) -> PageKind:
    return next(pk for pk in PAGE_KINDS if pk.id == page_kind_id)


# ---- The checks ------------------------------------------------------------------------


def test_every_kind_has_stub_inputs() -> None:
    assert [pk.id for pk in PAGE_KINDS if not (pk.stub_dir / "kind.json").is_file()] == []


def test_every_stub_directory_is_a_listed_kind() -> None:
    on_disk = {f"{p.parent.parent.name}-{p.parent.name}" for p in KINDS_DIR.glob("*/*/kind.json")}
    assert sorted(on_disk - set(IDS)) == []


@pytest.mark.parametrize("pk", PAGE_KINDS, ids=IDS)
def test_caption_bold_names_appear_on_every_page_kind(pk: PageKind, tmp_path: Path) -> None:
    caption = (CAPTIONS / pk.caption).read_text(encoding="utf-8")
    assert missing_names(caption, render_page(pk, tmp_path)) == []


# Marker per kind: an element the rendering must print. Without it, a caption with no bold
# names would pass against an empty or wrong page, so these show each stub reaches its page.
_MARKERS = {
    "P05-plain": "Model runs",
    "P05-sweep": "successful",
    "P06-plain": "model_type",
    "P06-sweep": "member_id",
    "P06-csv_missing": "The status table is not shown",
    "P07-failures": "check_scenarios_run",
    "P07-all_not_applicable": "they do not apply to this analysis",
    "P07-record_missing": "Check results are not shown",
    "P08-live": "plots",
    "P08-record_missing": "The disk use table is not shown",
}


def test_every_kind_has_a_marker() -> None:
    assert sorted(set(IDS) - set(_MARKERS)) == []


@pytest.mark.parametrize("pk", PAGE_KINDS, ids=IDS)
def test_rendering_reaches_its_page_kind(pk: PageKind, tmp_path: Path) -> None:
    assert _MARKERS[pk.id] in render_page(pk, tmp_path)


def test_a_page_with_no_applicable_check_prints_no_banner_and_states_the_omission(tmp_path: Path) -> None:
    """P07-all_not_applicable: no check applies, so the page says so once and prints no pass or fail tally.

    Input class: the `all_not_applicable` page kind (two checks, none applicable). Required because "0 of 0 applicable
    checks passed." and "0 of 0 applicable checks failed." state nothing. Killed by restoring either banner branch.
    A second correct implementation words the omission note differently while printing no tally.
    Must-fire control: before the banner returned the empty string for an empty `applicable` list, this page rendered
    "0 of 0 applicable checks" and this assertion failed.
    """
    html = render_page(_by_id("P07-all_not_applicable"), tmp_path)
    assert "applicable checks" not in html
    assert "they do not apply to this analysis" in html


# ---- Must-fire control: a bold element that exists on one page kind only -----------------

_R_PS1 = (
    "Aggregate scenario status for **{{ snakemake.config['analysis_id'] }}**. The table counts scenarios "
    "(one per weather event in that row's analysis) and, separately, **model runs** "
    "(each scenario is run once per enabled model type).\n\n**Sources:**\n\n{{ snakemake.params.source_paths_rst }}\n"
)


def test_bold_names_exclude_sources_and_jinja() -> None:
    assert bold_names(_R_PS1) == ["model runs"]


def test_bold_element_present_on_the_plain_page_only_fails_on_the_sweep_page(tmp_path: Path) -> None:
    caption = _R_PS1.replace("**model runs**", "**Model runs**")
    plain = render_page(_by_id("P05-plain"), tmp_path / "plain")
    sweep = render_page(_by_id("P05-sweep"), tmp_path / "sweep")
    assert missing_names(caption, plain) == []
    assert missing_names(caption, sweep) == ["Model runs"]


def test_the_original_sentence_fails_on_the_sweep_page(tmp_path: Path) -> None:
    sweep = render_page(_by_id("P05-sweep"), tmp_path)
    assert missing_names(_R_PS1, sweep) == ["model runs"]
