"""Metadata report renderer (ADR-14 / C10).

Renders ONE mostly-static HTML page under the "Metadata" ReportingSet category,
with three sub-sections. It is NOT fully self-contained: per [Q148] the Reproduction
Guide tables load Tabulator from a CDN. That trade is recorded in full below; it is
stated here too so the first sentence does not have to be walked back by the reader
who only gets that far ([Q149]).

  (1) Provenance summary  -- projected from the persisted RO-Crate sidecar
      {analysis_dir}/ro-crate-metadata.json (the read-model persisted at
      consolidation; Decision D1). Excludes the volatile startTime/agent
      (producer hostname + wall-clock) graph fields.
  (2) Reproduction guide  -- every config field grouped USER=Supply /
      HPC=Amend / EXPERIMENT=Keep via reprex_taxonomy.all_field_bucket
      (pure config-SCHEMA introspection; placeholders only, zero-user-info).
  (3) SLURM efficiency    -- the UNION of every globbed
      slurm_efficiency_report_*.csv (Decision D2), enriched with the
      human-readable rule purpose from _status/*.flag.json and the hardware /
      concurrency columns from scenario_status.csv. EMPTY on the producing run
      (each CSV is written at Snakemake teardown, AFTER render_report);
      populates on a later re-render / reprocess. This is inherent, not a
      defect. The union is load-bearing: each CSV covers exactly ONE Snakemake
      invocation (the plugin builds it from `sacct --name={run_uuid}`), so the
      former latest-by-mtime selection showed only the most recent
      invocation's jobs -- on a re-render, the render jobs and nothing else.

Mostly-static inline-CSS HTML, with ONE deliberate exception ([Q148]). The
provenance, data-availability and SLURM sections are self-contained and render
with no network. The three Reproduction Guide tables load Tabulator from a CDN
(~420 KB at view time) so their columns are user-resizable -- a trade made
explicitly, accepting that THOSE tables do not render in an archived, emailed,
or bundle-local copy while the rest of the page still does. Inline-Tabulator
bundling would remove the dependency but is unimplemented and silently falls
back to CDN (Gotcha 51). Mirrors errors_and_warnings.py elsewhere.

Renderer-IO audit (Gotcha 53): the ONLY files opened during render() are the
declared sources -- the RO-Crate sidecar and (when present) the one globbed
SLURM efficiency CSV. Globbing fires os.scandir, not open, so it is
audit-invisible. The reprex taxonomy is pure in-memory introspection.
"""

from __future__ import annotations

import csv
import html as _html
import io
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import pandas as pd

    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.config.report import report_config

# R3 zero-user-info guarantee. The provenance projection is allow-list-BY-
# CONSTRUCTION: each _build_provenance_html sub-block reaches ONLY for the named
# safe fields it enumerates. This frozenset is a fail-closed defense-in-depth
# BACKSTOP over that named-field projection, NOT the primary filter -- a
# deny-list alone would be FAIL-OPEN, because a future crate-schema field
# carrying a hostname / username / absolute producer path under a NEW key would
# leak into the bundle-shippable render. `metadata._VOLATILE_GRAPH_KEYS` is
# empty, so the sidecar DOES carry these volatile keys; the projection simply
# never reaches for them, and `_prop` raises if a maintainer tries.
from hhemt.report_renderers._tabulator_defaults import (
    TableFragment,
    build_columns_spec,
    build_options_dict,
    build_table_fragment,
    tabulator_head_assets,
    tabulator_shared_js,
)

_VOLATILE_EXCLUDED_KEYS: frozenset[str] = frozenset({"startTime", "endTime", "agent"})

_SIDECAR_FILENAME = "ro-crate-metadata.json"
_SLURM_EFF_RELDIR = ("logs", "slurm_efficiency_report")
_SLURM_EFF_GLOB = "slurm_efficiency_report_*.csv"
_SLURM_EFF_INNER_GLOB = "efficiency_report_*.csv"  # plugin nests the real CSV inside the .csv-named dir

# Toolkit-owned enrichment sources for the SLURM table. Both are already carried in every
# render bundle (`_status/` via _copy_supporting_files' copytree, scenario_status.csv via
# its named-root-file list), so the join works bundle-locally with no cluster access.
_STATUS_RELDIR = "_status"
_STATUS_FLAG_JSON_GLOB = "*.flag.json"
_SCENARIO_STATUS_FILENAME = "scenario_status.csv"

# rule_name prefix -> human-readable purpose. Deterministic, not heuristic: these are the
# rule-name stems `workflow.py` emits, and `status_flags.write_status_flag` records the stem
# verbatim. Ordered longest-prefix-first so `master_consolidation` is not eaten by
# `consolidate`. An unmatched rule_name renders its raw stem rather than a guessed verb.
_RULE_PREFIX_TO_PURPOSE: tuple[tuple[str, str], ...] = (
    ("master_consolidation", "consolidate (master)"),
    ("consolidate", "consolidate"),
    ("simulation", "simulate"),
    ("run_", "simulate"),
    ("prepare", "prepare scenario"),
    ("process", "process outputs"),
    ("setup_target", "compile / setup"),
    ("render_report", "render report"),
    ("plot_", "render plot"),
    ("bundle", "bundle"),
    ("combine", "combine"),
    ("delete", "delete / cleanup"),
    ("reprocess", "reprocess"),
)

#: Recover sa_id / event_id from a rule NAME, for jobs that survive only in the executor's
#: log tree (Tier 2 of `_job_purpose_map`) and therefore carry no sidecar record. Lossless
#: because `workflow.py` mints per-sub-analysis rule names as `{phase}_sa_{sa_id}_evt_{id}`.
#: The sa_id alternation tolerates the evt-less forms (`consolidate_sa_{sa_id}`), and the
#: capture is non-greedy so an sa_id containing underscores (`gpu_0_r1`) stops at `_evt_`
#: rather than swallowing it.
_RULE_SA_ID_RE = re.compile(r"_sa_(.+?)(?:_evt_|$)")
_RULE_EVENT_ID_RE = re.compile(r"_evt_(.+)$")

_ROOT_ID = "./"
_APP_ID = "#hhemt-app"
_TOOLKIT_SRC_ID = "#hhemt-toolkit-src"

# Okabe-Ito CVD-safe qualitative palette for the USER/HPC/EXPERIMENT bucket
# badges. This is a CATEGORICAL DATA encoding, not brand chrome -- the
# brand_theme stipulation explicitly holds the Okabe-Ito categorical palette
# CODE-FROZEN and exempt from theming. Redundant-coded (Wilke 2019): every
# badge carries both a color AND the instruction verb, so the grouping survives
# grayscale printing and CVD.
_BUCKET_ORDER: tuple[str, ...] = ("user", "hpc", "experiment")
_BUCKET_VERB: dict[str, str] = {"user": "Supply", "hpc": "Amend", "experiment": "Keep"}
_BUCKET_COLOR: dict[str, str] = {
    "user": "#D55E00",  # vermillion
    "hpc": "#0072B2",  # blue
    "experiment": "#009E73",  # bluish green
}
_BUCKET_HEADING: dict[str, str] = {
    "user": "you provide these (USER)",
    "hpc": "bundled, but revise for your machine (HPC)",
    "experiment": "these define the experiment (EXPERIMENT)",
}
_BUCKET_INSTRUCTION: dict[str, str] = {
    "user": (
        "These are host-local and are never carried in the bundle. Supply each one "
        "for your own account and filesystem before running."
    ),
    "hpc": (
        "These ride along in the bundle but are specific to the producing cluster. "
        "Revise them for your target system — EXCEPT any row marked as varied by the "
        "sensitivity analysis, which is a measured axis of this experiment and must be "
        "reproduced as swept, not revised. Bundle.reprex(reprex_config, "
        "target_hpc_profile) emits the concrete per-(sa_id, column) problem pairs and "
        "validated-vs-advisory amendments for your target."
    ),
    "experiment": (
        "Do NOT change these. They define the scientific experiment; changing one "
        "changes what is being measured, not merely where it runs."
    ),
}
_BUCKET_PLACEHOLDER: dict[str, str] = {
    "hpc": "{amend for your target system}",
    "experiment": "{inherit — carried by the bundle}",
}

# `reprex_config` mixes two buckets, per its own docstring: four host-local USER
# fields, plus the HPC-revisable partition SELECTORS named here. These field
# names are NOT keys of `reprex_taxonomy.all_field_bucket` (that classifier is
# total over system_config | analysis_config only), so the split is declared here.
_REPREX_SELECTOR_FIELDS: frozenset[str] = frozenset(
    {"target_ensemble_partition", "target_setup_and_analysis_processing_partition"}
)

# Non-brand supplemental CSS. Brand chrome (h2/h3/table/.banner) is sourced from
# the brand_theme-driven report_cfg.errors_and_warnings.render_inline_css(), per
# the "brand_theme is the single config source of report brand colors"
# stipulation -- no brand hex literal appears here. The rules below are either
# layout-only or the sanctioned Okabe-Ito categorical data palette.
_SUPPLEMENTAL_CSS = """
h4 { margin-top: 18px; margin-bottom: 6px; }
/* Long identifiers -- git SHAs, SIF paths, @id URIs, DOIs -- must wrap rather
   than force horizontal scroll inside the report engine's iframe. The mirrored
   errors_and_warnings CSS lacks this because its cell values are short. */
td, td code { word-break: break-all; overflow-wrap: anywhere; }
/* ...but `break-all` sets a cell's min-content width to ONE character, so under
   auto table-layout the first column is treated as infinitely compressible and
   collapses to a one-character-per-line ribbon once the table gains columns.
   Harmless at 3 columns; the reproduction guide's 5-column form made it
   unreadable. A min-width floor keeps the identifier column legible while
   leaving `break-all` in force for the sha256 / URI columns that need it.
   The floor is STILL the fix after Iter-11 item 24 gave every table a
   `div.table-scroll` wrapper: the wrapper lets an over-wide table scroll instead
   of overflowing the iframe, but it does nothing about a column that COLLAPSES,
   which is what `break-all` causes and what this rule prevents. */
table td:first-child { min-width: 18ch; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
nav.jump-nav { margin: 8px 0 16px; font-size: 13px; }
nav.jump-nav a { text-decoration: none; }
p.instruction { font-weight: 600; margin: 4px 0 8px; }
p.note { font-size: 12px; color: #555; margin: 4px 0 10px; }
span.badge { display: inline-block; padding: 1px 7px; border-radius: 8px;
             color: white; font-size: 11px; font-weight: 700; }
/* Sortable / per-column-filter table affordances. Layout-only -- no brand hex literal
   (brand_theme stipulation); the sort indicator is a glyph, not a color, and NOTHING
   here restates the `th` background/color, which stays sourced from
   render_inline_css() (white on dark blue). The sort target is the label SPAN, not the
   whole `th`: the header cell now also contains that column's filter input, and binding
   sort to the cell would sort the table on every click into the input. */
table.sortable th { user-select: none; white-space: nowrap; }
table.sortable span.th-label { cursor: pointer; display: block; }
table.sortable span.th-label::after { content: " \\2195"; opacity: 0.35; font-size: 10px; }
table.sortable th.sorted-asc span.th-label::after { content: " \\2191"; opacity: 1; }
table.sortable th.sorted-desc span.th-label::after { content: " \\2193"; opacity: 1; }
input.col-filter { width: 100%; box-sizing: border-box; margin-top: 3px;
                   padding: 2px 4px; font-size: 11px; font-weight: 400; }
/* Iter-11 item 24. EVERY table on this page is wrapped in div.table-scroll, so every
   table is height-bounded and scrolls in BOTH axes rather than pushing the page down.
   max-height engages only when the content exceeds it, so a two-row identity table
   renders exactly as it did. The sticky header keeps the column names -- and, on the
   interactive tables, their filter inputs -- on screen while the body scrolls. */
div.table-scroll { overflow: auto; max-height: min(70vh, 640px);
                   overscroll-behavior: contain; }
div.table-scroll thead th { position: sticky; top: 0; z-index: 2; }
/* Iter-12 item {32}. Two changes, two distinct causes.
   (a) `overscroll-behavior: contain` stops scroll CHAINING. Without it, a wheel event
       that reaches this container's top or bottom is handed to the ancestor scroller,
       then to the iframe document, then to the shell's `overflow-auto` content wrapper
       -- four nested scrollers reacting to one gesture, which is what reads as extra
       scrollbars appearing as the reader scrolls around.
   (b) `min(70vh, 640px)` removes the RESIZE. `vh` inside an iframe resolves against the
       IFRAME's height, and the report stylesheet lets that height float
       (`.result iframe { height: auto }`, report.css.j2). So a scroll that reflowed the
       shell changed the iframe height, which re-resolved `70vh`, which resized this box
       -- the "scrolling at the extremes made the table taller or shorter" report. The
       fixed ceiling means a re-resolved `vh` cannot move the box past it. */
aside.col-panel { overscroll-behavior: contain; }
/* Iter-11 item 11. Column-selector panel at the LEFT of the table, not above it, so it
   does not consume vertical space the table needs. Emitted only where the call site
   asks for one -- the reproduction-guide tables explicitly do not (item 16). */
div.table-tools { display: flex; gap: 10px; align-items: flex-start; }
div.table-tools > div.table-scroll { flex: 1 1 auto; min-width: 0; }
aside.col-panel { flex: 0 0 190px; max-height: 70vh; overflow-y: auto;
                  border: 1px solid #DADADA; padding: 6px; font-size: 11px; }
aside.col-panel strong { display: block; margin-bottom: 4px; }
aside.col-panel label { display: block; padding: 1px 0; cursor: pointer;
                        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
/* Typeset units and symbolic operations (Iter-11, item 15). `line-height: 0` is not
   cosmetic: without it every superscripted exponent inflates its table row's height, so
   the data-dictionary rows stop aligning with the rest of the page. No math engine --
   see `_expr_html` for why a bundled KaTeX/MathJax was rejected. */
sub, sup { line-height: 0; font-size: 0.75em; }
span.units { white-space: nowrap; }
span.expr { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
"""

# Inline vanilla-JS sort + filter shim, used by the Run-timeline and SLURM-efficiency
# tables. Still NOT Tabulator -- but [Q148] retired the reason this comment used to give,
# and the correction belongs here rather than behind a later one ([Q149]). The page is NO
# LONGER network-free: the three Reproduction Guide tables adopted CDN Tabulator, so
# "a CDN dependency would contradict the page's own thesis" is FALSE as of that adoption.
# What survives is narrower and still true: these two tables keep the ~40-line shim so
# they render in an archived, emailed, or bundle-local copy where the CDN is unreachable
# -- the guide tables, by that same trade, do not. Inline-Tabulator bundling would remove
# the split but is unimplemented and silently falls back to CDN (Gotcha 51).
#
# Numeric-aware compare: cells that parse as floats sort numerically, everything else
# case-insensitively as text. Empty and em-dash cells sort last in both directions so a
# missing measurement never masquerades as the smallest value.
_SORT_FILTER_JS = """
(function () {
  function cellText(row, i) { return (row.cells[i] ? row.cells[i].innerText : "").trim(); }
  function cmp(a, b) {
    var blankA = (a === "" || a === "\\u2014"), blankB = (b === "" || b === "\\u2014");
    if (blankA || blankB) { return blankA && blankB ? 0 : (blankA ? 1 : -1); }
    var na = parseFloat(a), nb = parseFloat(b);
    if (!isNaN(na) && !isNaN(nb) && /^[-+0-9.eE]+$/.test(a) && /^[-+0-9.eE]+$/.test(b)) {
      return na - nb;
    }
    return a.toLowerCase().localeCompare(b.toLowerCase());
  }
  // Multi-key sort. The key LIST comes from the table's own data-default-sort attribute
  // or from a single clicked column; no table's order is named in this shim.
  function sortBy(table, body, keys, desc) {
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (r1, r2) {
      for (var k = 0; k < keys.length; k++) {
        var d = cmp(cellText(r1, keys[k]), cellText(r2, keys[k]));
        if (d !== 0) { return desc ? -d : d; }
      }
      return 0;
    });
    rows.forEach(function (r) { body.appendChild(r); });
  }
  // Iter-11 item 11: PER-COLUMN filters, ANDed. The retired single filter bar tested the
  // whole row's text, so it could not express a conjunction ("this partition AND that
  // model") and it matched text in columns the reader had not asked about -- at 23
  // columns that hid rows for reasons invisible on the page.
  function applyFilters(table, body) {
    var inputs = table.querySelectorAll("input.col-filter");
    Array.prototype.slice.call(body.rows).forEach(function (r) {
      var show = true;
      Array.prototype.forEach.call(inputs, function (inp) {
        if (!show) { return; }
        var needle = inp.value.toLowerCase();
        if (!needle) { return; }
        var i = parseInt(inp.getAttribute("data-col"), 10);
        if (cellText(r, i).toLowerCase().indexOf(needle) === -1) { show = false; }
      });
      r.style.display = show ? "" : "none";
    });
  }
  function setColumnVisible(table, idx, visible) {
    var v = visible ? "" : "none";
    Array.prototype.forEach.call(table.querySelectorAll("tr"), function (r) {
      var c = r.cells[idx];
      if (c) { c.style.display = v; }
    });
  }
  document.querySelectorAll("table.sortable").forEach(function (table) {
    var body = table.tBodies[0];
    if (!body) { return; }
    var heads = table.querySelectorAll("thead th");
    heads.forEach(function (th, idx) {
      var label = th.querySelector("span.th-label");
      if (label) {
        label.addEventListener("click", function () {
          var desc = th.classList.contains("sorted-asc");
          heads.forEach(function (o) { o.classList.remove("sorted-asc", "sorted-desc"); });
          th.classList.add(desc ? "sorted-desc" : "sorted-asc");
          sortBy(table, body, [idx], desc);
        });
      }
      var inp = th.querySelector("input.col-filter");
      if (inp) {
        // Typing in a header input must not also sort the column it sits in.
        inp.addEventListener("click", function (e) { e.stopPropagation(); });
        inp.addEventListener("input", function () { applyFilters(table, body); });
      }
    });
    // Iter-11 item 16: default sort, applied once on load. A comma-separated column-index
    // list, so a table can declare a stable secondary key; read off the DOM rather than
    // hard-coded here, which is what keeps this shim table-agnostic.
    var ds = (table.getAttribute("data-default-sort") || "").trim();
    if (ds) {
      var keys = ds.split(",").map(function (s) { return parseInt(s, 10); })
                   .filter(function (n) { return !isNaN(n); });
      if (keys.length) {
        if (heads[keys[0]]) { heads[keys[0]].classList.add("sorted-asc"); }
        sortBy(table, body, keys, false);
      }
    }
    // Iter-11 item 11: left column-selector panel, present ONLY when the emitting call
    // site asked for one. The reproduction-guide tables deliberately have none (item 16).
    var panel = document.querySelector("aside.col-panel[data-table='" + table.id + "']");
    if (panel) {
      heads.forEach(function (th, idx) {
        var lbl = th.querySelector("span.th-label");
        var row = document.createElement("label");
        var cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = true;
        cb.addEventListener("change", function () { setColumnVisible(table, idx, cb.checked); });
        row.appendChild(cb);
        row.appendChild(document.createTextNode(" " + (lbl ? lbl.textContent : "")));
        row.title = lbl ? lbl.textContent : "";
        panel.appendChild(row);
      });
    }
  });
})();
"""


def _esc(value: Any) -> str:
    """HTML-escape any dynamic value before interpolation.

    Every dynamic value on this page -- RO-Crate ``@id`` / ``downloadUrl`` URIs,
    sha256 / git-SHA tokens, Pydantic ``Field(description=...)`` prose, and every
    SLURM CSV cell -- flows through here. Unlike the mirrored
    ``errors_and_warnings`` template (whose content is toolkit-controlled
    validation strings), this renderer projects config-derived and CSV content
    that can legitimately contain ``<`` / ``>`` / ``&`` / ``"``. Raw
    interpolation would silently drop or corrupt content in the iframe render
    and is a self-XSS vector.
    """
    return _html.escape("" if value is None else str(value), quote=True)


def _code(value: Any) -> str:
    return f"<code>{_esc(value)}</code>"


def _strip_code(cell: str) -> str:
    """Recover the plain label from a ``_code(...)``-wrapped cell.

    Used only to key the config-value lookup off a row whose first cell was already
    rendered. Field labels are dotted ASCII identifiers, so `_esc` is a no-op on them
    and unwrapping the tags is lossless.
    """
    return cell.removeprefix("<code>").removesuffix("</code>")


def _prop(entity: dict, key: str) -> Any:
    """Read one allow-listed property off a graph entity; refuse volatile keys.

    Fail-closed backstop for R3. Reaching for ``startTime`` / ``endTime`` /
    ``agent`` is a programming error, not a runtime condition -- a bundle-
    shippable page must never carry the producer's wall-clock or hostname.
    """
    if key in _VOLATILE_EXCLUDED_KEYS:
        raise ValueError(
            f"metadata renderer refused to project volatile RO-Crate key {key!r}: "
            "the Metadata page is bundle-shippable and must carry zero producer "
            "hostname / wall-clock information (R3, C-ZERO-USER-INFO)."
        )
    return entity.get(key)


def _anchor(title: str) -> str:
    return title.lower().replace(" ", "-")


def _heading(title: str) -> str:
    return f'<h3 id="{_anchor(title)}">{_esc(title)}</h3>'


def _banner(message: str) -> str:
    return f'<div class="banner info">{_esc(message)}</div>'


def _absent_banner(section_title: str, message: str) -> str:
    """Always-present <h3 id> heading + a .banner.info placeholder body (R7)."""
    return f"{_heading(section_title)}\n{_banner(message)}"


def _kv_table(rows: list[tuple[str, str]]) -> str:
    """Static 2-column Field/Value table. Values are PRE-ESCAPED HTML fragments.

    Iter-11 item 24: wrapped in `div.table-scroll`, so this table is height-bounded and
    scrolls in both axes like every other table on the page. `max-height` engages only
    when the content exceeds it, so a short identity table renders as it did -- the one
    that actually needed this is the Outputs table, whose hasPart cell carries a
    29-path `<pre>` folder tree.
    """
    if not rows:
        return ""
    body = "\n    ".join(f"<tr><td>{_esc(k)}</td><td>{v}</td></tr>" for k, v in rows)
    return (
        '<div class="table-scroll">\n<table>\n'
        "  <thead><tr><th>Field</th><th>Value</th></tr></thead>\n"
        "  <tbody>\n    " + body + "\n  </tbody>\n</table>\n</div>"
    )


def _grid_table(headers: list[str], rows: list[list[str]]) -> str:
    """Static n-column table. Row cells are PRE-ESCAPED HTML fragments.

    Iter-11 item 24: wrapped in `div.table-scroll` -- height-bounded, scrolling in both
    axes, sticky header. The 7-column CF data dictionary and the Inputs digest table are
    the two that reach the bound today.
    """
    if not rows:
        return ""
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "\n    ".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (
        '<div class="table-scroll">\n<table>\n'
        f"  <thead><tr>{head}</tr></thead>\n"
        "  <tbody>\n    " + body + "\n  </tbody>\n</table>\n</div>"
    )


def _sortable_grid_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    table_id: str,
    column_panel: bool = False,
    default_sort: tuple[int, ...] = (),
    header_tooltips: tuple[str, ...] = (),
) -> str:
    """`_grid_table` plus click-to-sort headers, PER-COLUMN filters, and an optional
    left column-selector panel.

    Iter-11 item 11. The single whole-row filter box is GONE, and `filter_label` with it
    -- a clean cut, since nothing outside this module referenced either. That box tested
    the row's whole text, so it could not express a conjunction and it matched in columns
    the reader had not asked about; at 23 columns it hid rows for reasons the page did not
    show. Each column now carries its own input, inside its own `th`, ANDed across columns.

    Iter-11 item 16. `column_panel` is opt-IN. Two of three call-site families want a
    panel; the reproduction-guide tables explicitly do not, and an opt-OUT default would
    put a suppression flag on the tables that asked for less. `default_sort` is a column-
    index tuple rendered into `data-default-sort` and applied once on load by the shim, so
    no table's order is hard-coded in the JS.

    Iter-11 item 24. The table is wrapped in `div.table-scroll`: height-bounded, scrolling
    in both axes, with a sticky header so the column names and their filters stay on
    screen. The header's white-on-dark-blue colouring is UNTOUCHED -- it comes from
    `render_inline_css()` per the brand_theme stipulation, and nothing here restates it.

    Static HTML + the inline `_SORT_FILTER_JS` shim only -- no CDN, no Tabulator (see
    `_SORT_FILTER_JS` for why). Degrades to a plain readable table when JavaScript is
    unavailable, which is the state the page must survive in an archived/emailed copy.
    Row cells are PRE-ESCAPED HTML fragments.
    """
    if not rows:
        return ""
    # [Q153] header tooltip. Header TEXT goes through `_esc`, so a per-column statement needs
    # its own attribute rather than being smuggled into the label. One optional parameter,
    # deliberately not widened into a mechanism: the value is supplied by the caller's column
    # declaration, so this function states no rule of its own.
    head = "".join(
        '<th{tip}><span class="th-label">{label}</span>'
        '<input class="col-filter" type="text" data-col="{i}" placeholder="filter" '
        'aria-label="Filter by {label}"></th>'.format(
            i=i,
            label=_esc(h),
            tip=f' title="{_esc(header_tooltips[i])}"' if i < len(header_tooltips) and header_tooltips[i] else "",
        )
        for i, h in enumerate(headers)
    )
    body = "\n    ".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    sort_attr = (
        f' data-default-sort="{",".join(str(i) for i in default_sort)}"' if default_sort else ""
    )
    scroll = (
        '<div class="table-scroll">\n'
        f'<table class="sortable" id="{_esc(table_id)}"{sort_attr}>\n'
        f"  <thead><tr>{head}</tr></thead>\n"
        "  <tbody>\n    " + body + "\n  </tbody>\n</table>\n</div>"
    )
    if not column_panel:
        return scroll
    return (
        '<div class="table-tools">\n'
        f'<aside class="col-panel" data-table="{_esc(table_id)}" role="region" '
        'aria-label="Column visibility"><strong>Columns</strong></aside>\n'
        f"{scroll}\n</div>"
    )


# --- RO-Crate @graph navigation helpers --------------------------------------


def _graph(doc: dict) -> list[dict]:
    graph = doc.get("@graph") or []
    return [e for e in graph if isinstance(e, dict)]


def _types(entity: dict) -> set[str]:
    """Return the entity's @type set. RO-Crate permits a str OR a list."""
    raw = entity.get("@type")
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {raw}
    return {str(t) for t in raw}


def _by_id(graph: list[dict], entity_id: str) -> dict | None:
    return next((e for e in graph if e.get("@id") == entity_id), None)


def _of_type(graph: list[dict], type_name: str) -> list[dict]:
    return [e for e in graph if type_name in _types(e)]


def _ref_ids(value: Any) -> list[str]:
    """Normalize a property that may be a ref, a list of refs, or a bare string."""
    if value is None:
        return []
    if isinstance(value, dict):
        rid = value.get("@id")
        return [str(rid)] if rid else []
    if isinstance(value, str):
        return [value]
    out: list[str] = []
    for item in value:
        out.extend(_ref_ids(item))
    return out


def _find_sif(graph: list[dict]) -> dict | None:
    """The container image entity: a SoftwareApplication that is not #hhemt-app.

    Identified structurally (carries sha256/downloadUrl), so a native run --
    which emits no SIF entity at all (`sif_spec=None`) -- yields None.
    """
    for entity in _of_type(graph, "SoftwareApplication"):
        if entity.get("@id") == _APP_ID:
            continue
        if _prop(entity, "sha256") or _prop(entity, "downloadUrl"):
            return entity
    return None


def _find_input_files(graph: list[dict]) -> list[dict]:
    """By-reference input `File` parts (each carries a sha256 digest).

    Excludes the crate's own metadata descriptor and any workflow entity added
    by the bundle-side Workflow-Run-Crate upgrade (BUNDLE_SCHEMA_VERSION 3).
    """
    out = []
    for entity in _of_type(graph, "File"):
        if entity.get("@id") == _SIDECAR_FILENAME:
            continue
        if "ComputationalWorkflow" in _types(entity):
            continue
        if _prop(entity, "sha256"):
            out.append(entity)
    return out


def _find_consolidated_dataset(graph: list[dict]) -> dict | None:
    for entity in _of_type(graph, "Dataset"):
        if entity.get("@id") == _ROOT_ID:
            continue
        if _prop(entity, "encodingFormat") == "application/x-zarr":
            return entity
    return next((e for e in _of_type(graph, "Dataset") if e.get("@id") != _ROOT_ID), None)


# --- (1) Provenance ----------------------------------------------------------


def _provenance_bluf(app: dict, sif: dict | None, inputs: list[dict]) -> str:
    """The verifiability anchors, up top: what makes this run CHECKABLE, not merely disclosed."""
    git_sha = _prop(app, "softwareVersion")
    parts: list[str] = []
    parts.append(f"code git-SHA {_code(git_sha)}" if git_sha else "code git-SHA <em>not captured</em>")
    if sif is not None and _prop(sif, "sha256"):
        parts.append(f"environment SIF sha256 {_code(_prop(sif, 'sha256'))}")
    else:
        parts.append("no container digest (native run)")
    n_digests = sum(1 for e in inputs if _prop(e, "sha256"))
    parts.append(f"{n_digests} input digest(s)")
    return (
        '<div class="banner info"><strong>Verifiability anchors:</strong> '
        + " &middot; ".join(parts)
        + ". These let a reproducer confirm exact code and exact input bytes, "
        "rather than merely reading a description.</div>"
    )


def _provenance_identity(graph: list[dict], root: dict) -> str:
    lic_id = next(iter(_ref_ids(_prop(root, "license"))), None)
    lic_entity = _by_id(graph, lic_id) if lic_id else None
    lic_name = _prop(lic_entity, "name") if lic_entity else None
    rows: list[tuple[str, str]] = []
    for label, key in (
        ("Case name", "name"),
        ("Description", "description"),
        ("Analysis ID", "analysis_id"),
        ("System ID", "system_id"),
        ("Layout version (schemaVersion)", "schemaVersion"),
    ):
        value = _prop(root, key)
        if value:
            rows.append((label, _esc(value)))
    if lic_name or lic_id:
        lic_html = _esc(lic_name or "")
        if lic_id:
            lic_html = f"{lic_html} ({_code(lic_id)})" if lic_name else _code(lic_id)
        rows.append(("Dataset license", lic_html))
    if not rows:
        return "<h4>1. Identity</h4>\n" + _banner("Root dataset carries no identity fields.")
    return "<h4>1. Identity</h4>\n" + _kv_table(rows)


def _provenance_software(app: dict, src: dict) -> str:
    rows: list[tuple[str, str]] = []
    if _prop(src, "name"):
        # Label is the product IDENTIFIER, value is its expansion -- the pairing the
        # RO-Crate already models (`#hhemt-app` name "hhemt" alongside
        # `#hhemt-toolkit-src` name "H&H Ensemble Modeling Toolkit"). Was labelled
        # "Toolkit", which rendered a word and its own expansion side by side and
        # named the product nowhere.
        rows.append(("hhemt", _esc(_prop(src, "name"))))
    if _prop(src, "codeRepository"):
        rows.append(("Code repository", _code(_prop(src, "codeRepository"))))
    git_sha = _prop(app, "softwareVersion") or _prop(src, "version")
    if git_sha:
        # This SHA is the CONSOLIDATION-time toolkit version, not the version that ran
        # the simulations and not the version rendering this page. Measured on one real
        # bundle: crate c74f46412e2d, tree hhemt_producing_sha 01655abb60c2, render
        # manifest e229023ab83d -- three real and different values. The old caption
        # ("exact code that produced this") asserted the first was the third.
        rows.append(("Git SHA (consolidation)", _code(git_sha)))
    if not rows:
        return "<h4>2. Software</h4>\n" + _banner("No software provenance captured in this crate.")
    return "<h4>2. Software</h4>\n" + _kv_table(rows)


def _resolve_consolidated_tree(analysis_dir: Path, analysis: TRITONSWMM_analysis | None) -> Path | None:
    """Locate the consolidated DataTree store, by existence, HPC- and bundle-alike.

    Prefers the paths the analysis declares; falls back to the two canonical
    filenames directly under ``analysis_dir`` so a reconstituted bundle (whose
    ``analysis_paths`` may be repointed) still resolves. Existence-resolved over
    the sensitivity-master name FIRST, mirroring ``_combine_merge._resolve_root_tree``.
    """
    candidates: list[Path] = []
    paths = getattr(analysis, "analysis_paths", None) if analysis is not None else None
    for attr in ("sensitivity_datatree_zarr", "analysis_datatree_zarr"):
        declared = getattr(paths, attr, None) if paths is not None else None
        if declared is not None:
            candidates.append(Path(declared))
    candidates.extend(analysis_dir / name for name in ("sensitivity_datatree.zarr", "analysis_datatree.zarr"))
    return next((c for c in candidates if c.exists()), None)


def _read_producing_shas(tree_path: Path | None) -> tuple[str | None, str | None]:
    """Return (producing_sha, producing_version) off the consolidated tree ROOT attrs.

    Reads only the tree's metadata (no chunk is materialized). Returns the
    ``*_divergent`` JSON breadcrumb when producers differ across events, because
    `apply_producing_stamp` leaves the scalar ABSENT in that case and the
    breadcrumb is then the only honest answer -- collapsing it to one value would
    reintroduce exactly the single-SHA overclaim this section exists to repair.
    Never raises: a legacy or unstamped tree yields (None, None).
    """
    if tree_path is None or not tree_path.exists():
        return (None, None)
    try:
        import xarray as xr

        tree = xr.open_datatree(tree_path, engine="zarr", chunks=None, consolidated=False)
    except Exception:  # noqa: BLE001 -- provenance display must never break the render
        return (None, None)
    attrs = dict(tree.attrs)
    out: list[str | None] = []
    for key in ("hhemt_producing_sha", "hhemt_producing_version"):
        value = attrs.get(key)
        if value is None:
            divergent = attrs.get(f"{key}_divergent")
            value = f"divergent across events: {divergent}" if divergent else None
        out.append(str(value) if value is not None else None)
    return (out[0], out[1])


def _provenance_chain(analysis_dir: Path, analysis: TRITONSWMM_analysis | None, crate_sha: str | None) -> str:
    """Three-row recreation chain: who SIMULATED, who CONSOLIDATED, who RENDERED.

    ADR-14 D1 compliant: no second read-model is introduced. The three values are
    projected from records that already exist and already ride in every bundle --
    the ADR-15 per-event producing-stamp coordinates on the consolidated tree, the
    RO-Crate sidecar, and the running build. They are disambiguated rather than
    merged, because collapsing them is what produced a caption asserting the
    consolidation SHA was the code that generated the data.

    Idempotence, and it is falsifiable: the only row that varies between two renders
    of unchanged data is "Rendered by" -- precisely the entry that SHOULD change.
    """
    tree_path = _resolve_consolidated_tree(analysis_dir, analysis)
    produced_sha, produced_version = _read_producing_shas(tree_path)

    from hhemt import provenance as _prov

    try:
        render_stamp = _prov.producing_stamp()
    except Exception:  # noqa: BLE001 -- a detached/dirty checkout must not break the render
        render_stamp = {}

    rows: list[tuple[str, str]] = []
    if produced_sha:
        label = _code(produced_sha)
        if produced_version:
            label = f"{label} ({_esc(produced_version)})"
        rows.append(("Simulated / processed by", label))
    else:
        rows.append(
            (
                "Simulated / processed by",
                "<em>not stamped</em> — this analysis was processed by a toolkit build "
                "that predates the per-event producing stamp (ADR-15).",
            )
        )
    rows.append(("Consolidated by", _code(crate_sha) if crate_sha else "<em>not captured</em>"))
    render_sha = render_stamp.get("hhemt_sha") or render_stamp.get("sha")
    render_version = render_stamp.get("hhemt_version") or render_stamp.get("version")
    rendered = _code(render_sha) if render_sha else "<em>not captured</em>"
    if render_version:
        rendered = f"{rendered} ({_esc(render_version)})"
    rows.append(("Rendered by", rendered))

    note = (
        "<p class='note'>These three are legitimately different, and the difference is the "
        "point: the code that produced the data, the code that assembled the consolidated "
        "store, and the code that drew this page are separately recorded so none of them can "
        "be mistaken for the others. Only the last row changes when this report is "
        "regenerated from unchanged data.</p>"
    )
    return "<h4>2b. Provenance chain</h4>\n" + note + _kv_table(rows)


def _read_status_flag_payloads(analysis_dir: Path) -> tuple[list[dict], list[Path]]:
    """Read every `_status/*.flag.json` sidecar; return (payloads, files actually opened).

    The file list is returned so `render()` can DECLARE every sidecar it opened
    (ADR-6 Gate-A / the declared-subset-of-actual invariant). Globbing is
    audit-invisible (os.scandir), but `read_text()` is not.

    `_status/` is already copytree'd into every bundle by `_copy_supporting_files`,
    so declaring these adds manifest rows, never payload bytes.
    """
    status_dir = analysis_dir / _STATUS_RELDIR
    if not status_dir.is_dir():
        return ([], [])
    payloads: list[dict] = []
    opened: list[Path] = []
    for sidecar in sorted(status_dir.glob(_STATUS_FLAG_JSON_GLOB)):
        try:
            payload = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        opened.append(sidecar)
        if isinstance(payload, dict):
            payloads.append(payload)
    return (payloads, opened)


def _purpose_for_rule(rule_name: str) -> str:
    """Deterministic rule_name -> human-readable purpose. Never guesses."""
    for prefix, purpose in _RULE_PREFIX_TO_PURPOSE:
        if rule_name.startswith(prefix):
            return purpose
    return rule_name


def _job_purpose_map(
    payloads: list[dict],
    job_index: dict[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """`{slurm_job_id: {purpose, rule_name, sa_id, event_id, model_type, written_at}}`.

    TWO sources in stated precedence, because neither alone covers the table.

    Tier 1 -- `status_flags.write_status_flag` records `slurm_job_id` from the
    ``SLURM_JOB_ID`` environment variable, which is the PARENT job id, exactly the
    efficiency CSV's `MainJobID`, so the join is direct. It is the RICHER source (it
    alone carries sa_id/event_id/model_type as RECORDED values rather than parsed
    ones) and therefore wins any collision. But it is keyed on the flag PATH and
    rewritten every run, so it retains only the LAST job per rule: measured 60 distinct
    ids for 116 rule instances against 570 allocations.

    Tier 2 -- `_status/_job_index.json`, harvested by
    `status_flags.harvest_slurm_job_index` from the executor's own per-job log tree:
    `{jobid: rule_name}` for every job ever submitted, including the ones whose sidecar
    a later submission overwrote. It carries no sa_id/event_id directly; those are
    parsed back out of the rule name, which is lossless because `workflow.py` mints
    rule names as `{phase}_sa_{sa_id}_evt_{event_id}`.

    Tier 2 does NOT raise the join's match rate -- that is already 99% of the keys the
    sidecars retain. It raises the KEY CEILING, by keeping log files the executor would
    otherwise delete. It is therefore INERT until the generated profile sets
    `slurm-keep-successful-logs`, and even then it only covers runs submitted AFTER
    that lands; an absent or empty index means "not retained", never "did not run", and
    degrades to exactly the Tier-1 result.

    Locally-executed rules carry a null job id and are skipped rather than keyed on the
    empty string.
    """
    out: dict[str, dict[str, str]] = {}
    for payload in payloads:
        job_id = payload.get("slurm_job_id")
        if not job_id:
            continue
        rule_name = str(payload.get("rule_name") or "")
        out[str(job_id)] = {
            "purpose": _purpose_for_rule(rule_name),
            "rule_name": rule_name,
            "sa_id": str(payload.get("sa_id") or ""),
            "event_id": str(payload.get("event_id") or ""),
            "model_type": str(payload.get("model_type") or ""),
            "written_at": str(payload.get("written_at") or ""),
        }
    # Tier 2. `not in out` is what preserves Tier-1 precedence: a job the sidecar still
    # retains keeps its RECORDED sa_id/event_id/model_type rather than the parsed ones.
    for job_id, rule_name in (job_index or {}).items():
        job_id = str(job_id)
        if not job_id or job_id in out:
            continue
        rule_name = str(rule_name or "")
        sa_match = _RULE_SA_ID_RE.search(rule_name)
        evt_match = _RULE_EVENT_ID_RE.search(rule_name)
        out[job_id] = {
            "purpose": _purpose_for_rule(rule_name),
            "rule_name": rule_name,
            "sa_id": sa_match.group(1) if sa_match else "",
            "event_id": evt_match.group(1) if evt_match else "",
            # Not parseable from a rule name: the model type is a per-scenario property,
            # not a naming component. Left empty rather than guessed -- an em-dash here
            # means "not recovered", which is true, whereas a guess would be false.
            "model_type": "",
            "written_at": "",
        }
    return out


def _provenance_environment(sif: dict | None) -> str:
    if sif is None:
        return "<h4>3. Environment</h4>\n" + _banner(
            "Native run — no container environment captured. The software environment "
            "is not pinned, so recreation has reduced verifiability: a reproducer must "
            "reconstruct the toolchain from the module stack and conda environment."
        )
    rows: list[tuple[str, str]] = []
    if _prop(sif, "name"):
        rows.append(("Container", _esc(_prop(sif, "name"))))
    if _prop(sif, "softwareVersion"):
        rows.append(("Container version", _code(_prop(sif, "softwareVersion"))))
    if _prop(sif, "sha256"):
        rows.append(("SIF sha256 (exact-bytes anchor)", _code(_prop(sif, "sha256"))))
    if _prop(sif, "downloadUrl"):
        rows.append(("Download URL", _code(_prop(sif, "downloadUrl"))))
    return "<h4>3. Environment</h4>\n" + _kv_table(rows)


def _provenance_inputs(inputs: list[dict]) -> str:
    if not inputs:
        return "<h4>4. Inputs</h4>\n" + _banner(
            "Input digests not captured for this analysis. (This does not mean there "
            "were no inputs — the case manifest was not wired at consolidation, so no "
            "by-reference File parts were recorded.)"
        )
    rows = [
        [
            _code(entity.get("@id")),
            _code(_prop(entity, "sha256")),
            _esc(_prop(entity, "contentSize") or "—"),
            _esc(_prop(entity, "encodingFormat") or "—"),
        ]
        for entity in inputs
    ]
    return "<h4>4. Inputs</h4>\n" + _grid_table(["File", "sha256", "Size (bytes)", "Format"], rows)


def _provenance_process(runs: list[dict]) -> str:
    """Run units: COUNT + instrument->result edges ONLY.

    NEVER startTime / agent (_VOLATILE_EXCLUDED_KEYS): the producer's wall-clock
    and hostname must not reach a bundle-shippable page. A sensitivity-master
    crate is emitted with `with_run_units=False`, so it carries no CreateAction
    nodes at all -- render the hasPart sub-dataset story instead.
    """
    if not runs:
        return "<h4>5. Process</h4>\n" + _banner(
            "Run units not captured — this is a consolidation-level crate "
            "(a sensitivity master aggregates its sub-analyses, each of which "
            "carries its own per-run provenance)."
        )
    rows: list[list[str]] = []
    for action in runs:
        instruments = _ref_ids(_prop(action, "instrument"))
        results = _ref_ids(_prop(action, "result"))
        objects = _ref_ids(_prop(action, "object"))
        rows.append(
            [
                _esc(_prop(action, "name") or action.get("@id")),
                " ".join(_code(i) for i in instruments) or "—",
                _esc(len(objects)),
                " ".join(_code(r) for r in results) or "—",
            ]
        )
    summary = f"<p class='note'>{_esc(len(runs))} run unit(s) recorded.</p>"
    return "<h4>5. Process</h4>\n" + summary + _grid_table(["Run", "Instrument(s)", "Inputs", "Result(s)"], rows)


# Iter-11, item 14. A run of TWO identical siblings collapses to a sentinel line plus a
# shared subtree -- three lines where two stood, which is worse. Three is the smallest
# run where the collapse pays for itself.
_TREE_SENTINEL_MIN_SIBLINGS = 3


def _common_prefix(names: list[str]) -> str:
    """Longest shared leading substring across `names` (may be "")."""
    if not names:
        return ""
    head = names[0]
    for other in names[1:]:
        i = 0
        while i < len(head) and i < len(other) and head[i] == other[i]:
            i += 1
        head = head[:i]
        if not head:
            break
    return head


def _sentinel_label(names: list[str]) -> str:
    """`{stem…} × N` -- the parameterised stand-in for a run of N identical siblings.

    BOTH halves are DERIVED: the placeholder stem is the longest common prefix of the
    run's actual names, and N is the run length. Nothing here is authored per experiment,
    so the label cannot drift from the data.

    The count sits OUTSIDE the braces on purpose (Iter-11, item 14a). `{subanalysis id
    (28)}` nests a parenthesis inside a brace, which reads as though the count were part
    of the parameter's NAME; a trailing multiplier reads as a quantity applied to the
    placeholder, which is what it is. The user ruled for this form on 2026-08-17.
    """
    stem = _common_prefix(names)
    return f"{{{stem}…}} × {len(names)}" if stem else f"{{…}} × {len(names)}"


def _sibling_runs(node: dict) -> list[tuple[list[str], dict]]:
    """Split `node`'s sorted children into CONTIGUOUS runs of identical subtrees.

    Identity is DECIDED, never assumed. The writer builds every sub-analysis path from
    one f-string (`sensitivity_analysis.py`, `_sub_relpaths`), so structural identity is
    guaranteed at WRITE time -- but this function's input is a persisted crate that
    outlives the writer, and the RO-Crate contract is that the on-disk JSON-LD IS the
    object. So the subtrees are compared by canonical sorted-key serialization, and THAT
    COMPARISON IS THE DIVERGENCE DETECTOR: a sub-analysis whose structure legitimately
    differs breaks the run and renders under its own name. The collapse can never hide a
    real structural difference.

    Contiguous runs rather than global equivalence classes: global grouping would REORDER
    the listing whenever two identical siblings straddle a third that differs, and a
    reordered directory listing is a new readability problem in place of the old one.
    """
    runs: list[tuple[list[str], dict]] = []
    last_sig: str | None = None
    for name, child in sorted(node.items()):
        sig = json.dumps(child, sort_keys=True)
        if runs and sig == last_sig:
            runs[-1][0].append(name)
        else:
            runs.append(([name], child))
            last_sig = sig
    return runs


def _path_tree_html(paths: list[str]) -> str:
    """Render `paths` as a folder tree, the way a directory listing is normally shown.

    Iter-10 H. The user: 'the presentation of sub datasets is unreadable; it's just a
    massive list of filepaths which is impossible to read in this block of text; i think a
    branch structure like people use to display folder structure could be good for this.'

    Measured on the delivered bundle: 29 paths in ONE table cell, space-separated, over two
    distinct prefixes (`sensitivity_datatree.zarr/` and `subanalyses/`).

    Single-child chains are COLLAPSED onto one line (`sa_gpu_0_r1/analysis_datatree.zarr`
    rather than a directory line plus an indented leaf). Without that, 28 identically
    shaped sub-analyses render as 56 lines whose every other line is the same filename —
    which trades one unreadable shape for another rather than fixing the readability the
    user asked about.
    """
    tree: dict = {}
    for p in paths:
        segs = [s for s in p.split("/") if s]
        if not segs:
            continue
        node = tree
        for s in segs:
            node = node.setdefault(s, {})

    def _collapse(name: str, node: dict) -> tuple[str, dict]:
        # Fold a chain of single children into one label; stop at a branch or a leaf.
        while len(node) == 1:
            (child,) = node
            if not node[child]:
                name, node = f"{name}/{child}", {}
                break
            name, node = f"{name}/{child}", node[child]
        return name, node

    lines: list[str] = []
    collapsed: list[tuple[str, list[str]]] = []  # (sentinel label, member names)

    def _walk(node: dict, prefix: str) -> None:
        runs = _sibling_runs(node)
        for i, (names, child) in enumerate(runs):
            last = i == len(runs) - 1
            # Collapse a run only when there is shared structure to show ONCE (`child`
            # non-empty) and enough members to pay for the extra sentinel line. A run of
            # leaves has no subtree, so collapsing it would replace N names with a
            # sentinel and nothing beneath -- pure information loss.
            if child and len(names) >= _TREE_SENTINEL_MIN_SIBLINGS:
                label = _sentinel_label(names)
                collapsed.append((label, names))
                lines.append(f"{prefix}{'└── ' if last else '├── '}{_esc(label)}")
                _walk(child, prefix + ("    " if last else "│   "))
                continue
            for j, name in enumerate(names):
                sub_last = last and j == len(names) - 1
                label, sub = _collapse(name, child)
                lines.append(f"{prefix}{'└── ' if sub_last else '├── '}{_esc(label)}")
                if sub:
                    _walk(sub, prefix + ("    " if sub_last else "│   "))

    _walk(tree, "")
    _style = "margin:0;font-size:11px;line-height:1.45"
    tree_html = f"<pre class='hhemt-path-tree' style='{_style}'>" + "\n".join(lines) + "</pre>"
    if not collapsed:
        return tree_html
    members = "; ".join(
        f"<strong>{_esc(label)}</strong> = " + ", ".join(_code(n) for n in names)
        for label, names in collapsed
    )
    total = sum(len(names) for _, names in collapsed)
    return (
        tree_html
        + f"<details><summary class='note'>Collapsed runs — show the {_esc(total)} member name(s)"
        "</summary><p class='note'>Membership is decided by comparing each sibling's subtree, "
        "never inferred from its name. " + members + "</p></details>"
    )


def _consolidated_group_paths(
    analysis_dir: Path | None,
    analysis: TRITONSWMM_analysis | None,
) -> tuple[list[str], bool]:
    """Return (group paths, is_realized) for the consolidated store's node hierarchy.

    The Outputs section's other tree is the crate's `hasPart` FILE-PATH list, which stops
    AT `analysis_datatree.zarr` and never descends into it. This supplies the missing half:
    the DataTree GROUP hierarchy INSIDE the store.

    `is_realized` is True when the paths were read off the store on disk, False when they
    are the code-declared vocabulary from `processing_analysis`. The caller MUST label the
    two differently -- a provenance page presenting the toolkit's schema as this dataset's
    structure asserts nodes the run may not have produced.

    No new declared source: `_resolve_consolidated_tree` is already called in `render()`
    and the store it resolves is already in `source_paths`, so this opens nothing the
    renderer-IO provenance audit (Gotcha 53) has not already been told about. Only the
    tree's metadata is read; no chunk is materialized. Both arguments are forwarded because
    the resolver PREFERS the paths the analysis declares -- passing None for `analysis`
    would silently degrade resolution to the filename fallback on the sensitivity-master
    path, which is the shape this section is most often rendered for.
    """
    tree_path = _resolve_consolidated_tree(analysis_dir, analysis) if analysis_dir else None
    if tree_path is not None and tree_path.exists():
        try:
            import xarray as xr

            tree = xr.open_datatree(tree_path, engine="zarr", chunks=None, consolidated=False)
        except Exception:
            tree = None
        if tree is not None:
            groups = sorted(g.lstrip("/") for g in tree.groups if g.strip("/"))
            if groups:
                return [f"{tree_path.name}/{g}" for g in groups], True
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as _pp

    declared = sorted(
        set(_pp._MODE_TO_TREE_PATH.values()) | set(_pp._TIMESERIES_MODE_TO_TREE_PATH.values())
    )
    return [f"analysis_datatree.zarr/{g}" for g in declared], False


_MINUS = "−"  # U+2212 MINUS SIGN -- a hyphen is not a minus in a typeset exponent.


def _sup(text: str) -> str:
    return f"<sup>{_esc(text).replace('-', _MINUS)}</sup>"


def _unit_factor_html(tok: str) -> str:
    """One whitespace-delimited UDUNITS factor -> HTML with a real superscript.

    Two branches, both load-bearing against real entries in `_CF_VARIABLE_MAP`:

    * The caret branch exists for `10^6 L` (`total_inflow_vol_10e6_ltr`). Without it, a
      trailing-digit rule renders `1` raised to `0`.
    * The alphabetic-base guard exists for `1`, CF's dimensionless unit. Without it, the
      same rule produces an empty base carrying a superscript.
    """
    if "^" in tok:
        base, _, exp = tok.partition("^")
        return _esc(base) + _sup(exp) if base and exp else _esc(tok)
    i = len(tok)
    while i > 0 and tok[i - 1].isdigit():
        i -= 1
    if i > 1 and tok[i - 1] == "-":
        i -= 1
    base, exp = tok[:i], tok[i:]
    if base and exp and base[-1].isalpha():
        return _esc(base) + _sup(exp)
    return _esc(tok)


def _units_html(units: Any) -> str:
    """Render a UDUNITS string with true superscripts WITHOUT altering it at rest.

    `m s-1` renders as `m s` with a superscript minus-one; the CF `units` attribute on the
    data and the crate's `unitText` stay BYTE-IDENTICAL UDUNITS, which is what a CF-aware
    reader parses. This is a presentation transform in the renderer only -- it never calls
    into `cf_conventions`, so the `cf conventions canonical source` stipulation (which
    governs attribute APPLICATION, not rendering) is untouched.

    Do NOT "simplify" this by adding a `display_units` key to `_CF_VARIABLE_MAP`:
    `cf_conventions._set_attrs` stamps EVERY non-None key of an entry onto the DataArray,
    so a presentation key added there is published as a non-CF attribute on the data.

    The canonical string is preserved verbatim in `title` because superscripts and U+2212
    are not copy-pasteable back into a UDUNITS parser -- the machine form has to remain
    recoverable from the page.
    """
    text = str(units or "").strip()
    if not text:
        return "—"
    body = " ".join(_unit_factor_html(t) for t in text.split())
    return f'<span class="units" title="{_esc(text)}">{body}</span>'


def _expr_html(expr: str) -> str:
    """Render a compact symbolic expression with real sub/superscripts.

    Grammar is deliberately two characters wide -- `_` for a subscript and `^` for a
    superscript, each binding to the alphanumeric run that follows, or to a `{...}` group
    -- so the STORED value stays a short ASCII-ish string a human can read and grep.
    Operators are literal Unicode (√ Σ ∫ · Δ), not markup.

    No math engine. This page carries none. ([Q148]/[Q149]: this passage used to argue
    from "the module already refuses a CDN dependency", which stopped being true when the
    Reproduction Guide tables adopted CDN Tabulator. Corrected in place; the argument
    below never depended on it.) A BUNDLED engine would clear the self-containment bar but would
    vendor a third-party asset with its own license and provenance onto a page whose whole
    thesis is that it is self-describing, and would render math that is neither selectable
    nor greppable in an archived copy. Eight of the eleven descriptors are prose with no
    equation in them, so the engine would buy nothing for most rows.
    """
    out: list[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch in "_^" and i + 1 < len(expr):
            j = i + 1
            if expr[j] == "{":
                k = expr.find("}", j)
                if k == -1:
                    out.append(_esc(ch))
                    i += 1
                    continue
                run, i = expr[j + 1 : k], k + 1
            else:
                j2 = j
                while j2 < len(expr) and (expr[j2].isalnum() or expr[j2] == "-"):
                    j2 += 1
                run, i = expr[j:j2], j2
            tag = "sub" if ch == "_" else "sup"
            out.append(f"<{tag}>{_esc(run).replace('-', _MINUS)}</{tag}>")
            continue
        out.append(_esc(ch))
        i += 1
    return "".join(out)


def _operation_html(descriptor: dict) -> str:
    """Prose operation, with a typeset symbolic form appended when one is authored.

    The prose stays PRIMARY and unchanged. Most descriptors are genuinely prose ("value
    selected at the final reported timestep") with no equation to typeset, so a math-first
    rendering would leave the majority of rows worse. `operation_expr` is optional by
    design: an entry that has not authored one renders exactly what it renders today,
    which is what makes this safe to land row by row.
    """
    prose = descriptor.get("operation") or "—"
    expr = descriptor.get("operation_expr")
    if not expr:
        return _esc(prose)
    return f'{_esc(prose)}<br><span class="expr">{_expr_html(str(expr))}</span>'


def _provenance_outputs(
    graph: list[dict],
    root: dict,
    analysis_dir: Path | None = None,
    analysis: TRITONSWMM_analysis | None = None,
) -> str:
    dataset = _find_consolidated_dataset(graph)
    parts: list[str] = ["<h4>6. Outputs &amp; CF data dictionary</h4>"]
    if dataset is None:
        parts.append(_banner("No consolidated output dataset recorded in this crate."))
        return "\n".join(parts)

    rows: list[tuple[str, str]] = [("Output", _code(dataset.get("@id")))]
    if _prop(dataset, "name"):
        rows.append(("Name", _esc(_prop(dataset, "name"))))
    if _prop(dataset, "encodingFormat"):
        rows.append(("Format", _esc(_prop(dataset, "encodingFormat"))))
    conforms = next(iter(_ref_ids(_prop(dataset, "conformsTo"))), None)
    if conforms:
        rows.append(("Conforms to", _code(conforms)))
    sub_parts = _ref_ids(_prop(root, "hasPart"))
    if sub_parts:
        # Iter-10 H: a folder tree, not a space-joined run of <code> spans.
        rows.append((f"Sub-datasets (hasPart) — {len(sub_parts)}", _path_tree_html(list(sub_parts))))
    # Iter-13 bullet 9. The hasPart tree above is a FILE-PATH listing: it stops at
    # `analysis_datatree.zarr` and never descends into it, so a reader learns the store
    # exists and nothing about what is in it. This row supplies the store's own DataTree
    # GROUP hierarchy, rendered INLINE through the same branch-tree renderer rather than
    # behind a disclosure widget -- the user asked for the structure shown "rather than
    # forcing the user to hit the dropdown". The sub-analysis membership dropdown above is
    # deliberately left alone; the user called it good and desirable.
    _group_paths, _realized = _consolidated_group_paths(analysis_dir, analysis)
    if _group_paths:
        rows.append(
            (
                "Store hierarchy — this run"
                if _realized
                else "Store hierarchy — typical (store not carried in this bundle)",
                _path_tree_html(_group_paths),
            )
        )
    parts.append(_kv_table(rows))

    # Descriptor columns come from MODULE introspection (cf_conventions._QUANTITY_PROVENANCE),
    # intersected with the crate's variableMeasured names. Two reasons, both load-bearing:
    # the crate is written only at consolidation, so sourcing the descriptors from it would
    # require a re-consolidation to change them; and the crate's name list is what makes the
    # table describe THIS dataset rather than this toolkit version. A variable the module
    # does not cover renders an explicit em-dash, never a guess.
    from hhemt.cf_conventions import quantity_provenance

    var_rows: list[list[str]] = []
    for ref in _ref_ids(_prop(dataset, "variableMeasured")):
        pv = _by_id(graph, ref)
        if pv is None:
            continue
        var_name = _prop(pv, "name")
        descriptor = quantity_provenance(str(var_name)) or {}
        var_rows.append(
            [
                _code(var_name),
                _esc(_prop(pv, "description") or "—"),
                _units_html(_prop(pv, "unitText")),
                _code(_prop(pv, "propertyID")) if _prop(pv, "propertyID") else "—",
                _esc(descriptor.get("spatial_representation") or "—"),
                _esc(descriptor.get("source_variables") or "—"),
                _operation_html(descriptor),
            ]
        )
    if var_rows:
        parts.append(
            "<p class='note'><strong>Data dictionary.</strong> One row per variable in the "
            "consolidated store. <em>Spatial representation</em> is the geometry each value "
            "describes (grid cell, point/node, line/conduit, or a whole-domain scalar); "
            "<em>Source variables</em> are the raw model outputs the value was computed from; "
            "<em>Operation</em> is the computation applied to them. These three are derived "
            "from the toolkit's own quantity-provenance table rather than restated by hand, so "
            "they cannot drift from the variable set without failing a test. The CF "
            "<code>cell_methods</code> string remains stamped on the data itself, where a "
            "CF-aware reader will find it.</p>"
        )
        parts.append(
            _grid_table(
                [
                    "Variable",
                    "Long name",
                    "Units",
                    "CF standard_name",
                    "Spatial representation",
                    "Source variables",
                    "Operation",
                ],
                var_rows,
            )
        )
    return "\n".join(parts)


def _build_provenance_html(
    doc: dict,
    analysis_dir: Path | None = None,
    analysis: TRITONSWMM_analysis | None = None,
    status_payloads: list[dict] | None = None,
) -> str:
    """Project the RO-Crate JSON-LD @graph into a static provenance recreation chain.

    Ordered disclosed -> verifiable: BLUF verifiability anchors, then
    Identity -> Software -> Provenance chain -> Environment -> Inputs -> Process ->
    Run timeline -> Outputs. Allow-list BY CONSTRUCTION: each sub-block reaches only
    for the named safe fields it enumerates; `_prop` refuses the volatile keys as a
    backstop.

    The provenance chain and run timeline are appended only when the caller supplies
    the analysis context, so the crate-only call path (tests, a bare sidecar read)
    keeps working unchanged.
    """
    graph = _graph(doc)
    root = _by_id(graph, _ROOT_ID) or {}
    app = _by_id(graph, _APP_ID) or {}
    src = _by_id(graph, _TOOLKIT_SRC_ID) or {}
    sif = _find_sif(graph)
    inputs = _find_input_files(graph)
    runs = _of_type(graph, "CreateAction")

    crate_sha = _prop(app, "softwareVersion") or _prop(src, "version")
    chain_html = (
        _provenance_chain(analysis_dir, analysis, str(crate_sha) if crate_sha else None)
        if analysis_dir is not None
        else ""
    )
    timeline_html = _provenance_timeline(status_payloads or []) if status_payloads else ""

    return "\n".join(
        s
        for s in [
            _heading("Provenance"),
            _provenance_bluf(app, sif, inputs),
            _provenance_identity(graph, root),
            _provenance_software(app, src),
            chain_html,
            _provenance_environment(sif),
            _provenance_inputs(inputs),
            _provenance_process(runs),
            timeline_html,
            _provenance_outputs(graph, root, analysis_dir=analysis_dir, analysis=analysis),
        ]
        if s
    )


def _provenance_timeline(payloads: list[dict]) -> str:
    """Whole-experiment run timeline, projected from `_status/*.flag.json`.

    This is the record the Process section cannot supply on a sensitivity master
    (whose crate is emitted with ``with_run_units=False`` and therefore carries no
    CreateAction nodes at all). The sidecars are append-only per completed rule, so
    this table covers setup -> prepare -> simulate -> process -> consolidate for the
    WHOLE experiment and does not shrink when the report is regenerated.
    """
    rows = [
        [
            _esc(str(p.get("written_at") or "")),
            _esc(_purpose_for_rule(str(p.get("rule_name") or ""))),
            _code(p.get("rule_name") or "—"),
            _esc(str(p.get("sa_id") or "—")),
            _esc(str(p.get("event_id") or "—")),
            _esc(str(p.get("model_type") or "—")),
            _code(p.get("slurm_job_id")) if p.get("slurm_job_id") else "—",
        ]
        for p in payloads
    ]
    if not rows:
        return ""
    rows.sort(key=lambda r: r[0])
    note = (
        f"<p class='note'><strong>Run timeline.</strong> {_esc(len(rows))} completed "
        "rule(s), recorded one file at a time as each finished. This is the whole experiment "
        "from setup through consolidation, not the most recent run — regenerating this report "
        f"cannot shorten it. {_TABLE_INTERACTION_NOTE}</p>"
    )
    table = _sortable_grid_table(
        ["Completed at", "Job desc", "Rule", "Sub-analysis", "Event", "Model", "SLURM job"],
        rows,
        table_id="run-timeline",
        column_panel=True,
    )
    return "<h4>5b. Run timeline</h4>\n" + note + table


# --- (2) Reproduction guide --------------------------------------------------


def _config_field_rows() -> tuple[dict[str, list[list[str]]], list[str]]:
    """Bucket every config field into USER / HPC / EXPERIMENT.

    Takes NO analysis argument BY DESIGN: the reproduction guide must render
    zero producer values (C-ZERO-USER-INFO). Deriving the rows purely from the
    config *schema* -- `model_fields` plus `reprex_taxonomy.all_field_bucket` --
    makes that property true by construction rather than by discipline: this
    function cannot leak a value it never sees. Pure introspection, no file read.

    Two sources are unioned, because neither alone answers the reproducer's
    question:

      (a) every ``system_config`` / ``analysis_config`` field, bucketed by
          ``all_field_bucket`` (R4). Over this domain the USER bucket contains
          only the two software-directory paths -- ``all_field_bucket`` is total
          over the two configs, and HPC identity does not live there.
      (b) every ``reprex_config`` field -- the minimal set a TARGET user actually
          supplies to run a reprex bundle (account, login node, SIF path, scratch
          dir) plus the two partition SELECTORS. These are NOT fields of the two
          configs, so (a) alone would render a "Supply" block that omits
          everything a reproducer must in fact supply.

    Per ``reprex_config``'s own structure, its four host-local fields are USER
    (Supply) and its two ``target_*`` partition selectors are HPC-revisable
    (Amend).

    Returns (rows_by_bucket, unclassified_field_labels).
    """
    # Function-local import: `hhemt.config.reprex_taxonomy` imports
    # `hhemt.bundle._path_policy`, which executes `hhemt.bundle.__init__`. A
    # module-top import here would widen this renderer's import graph for no
    # benefit; the taxonomy module's own docstring mandates function-local
    # imports for any call reachable from `hhemt.bundle`.
    from hhemt.config import reprex_taxonomy
    from hhemt.config.analysis import analysis_config
    from hhemt.config.reprex_config import reprex_config
    from hhemt.config.system import system_config

    rows_by_bucket: dict[str, list[list[str]]] = {b: [] for b in _BUCKET_ORDER}
    unclassified: list[str] = []

    def _row(label: str, field_info: Any, bucket: str, field_name: str) -> list[str]:
        placeholder = _BUCKET_PLACEHOLDER.get(bucket, f"{{your-{field_name}}}")
        return [
            _code(label),
            _description_cell(field_info),
            _requiredness_cell(field_info),
            _code(placeholder),
        ]

    # (b) the target user's supply set. Listed FIRST inside each bucket so the
    # Supply block opens with what the reproducer literally types.
    for field_name, field_info in reprex_config.model_fields.items():
        bucket = "hpc" if field_name in _REPREX_SELECTOR_FIELDS else "user"
        rows_by_bucket[bucket].append(_row(f"reprex_config.{field_name}", field_info, bucket, field_name))

    # (a) every field of the two experiment configs.
    for config_label, model in (("system_config", system_config), ("analysis_config", analysis_config)):
        for field_name, field_info in model.model_fields.items():
            try:
                bucket = reprex_taxonomy.all_field_bucket(field_name)
            except KeyError:
                # Totality is test-enforced (test_field_bucket_is_total); a miss
                # means schema drift against an older bundle. Surface it rather
                # than raising -- an unbucketed field must never silently vanish
                # from a reproduction guide.
                unclassified.append(f"{config_label}.{field_name}")
                continue
            rows_by_bucket[bucket].append(_row(f"{config_label}.{field_name}", field_info, bucket, field_name))
    return rows_by_bucket, unclassified


def _is_toolkit_owned(field_info: Any) -> bool:
    """True when the field is marked `json_schema_extra={"toolkit_owned_output": True}`.

    Not a judgement call: this is the SAME marker `validation.py` and `config/base.py`
    already honour to skip existence checks on a path the toolkit creates for itself.
    Today it marks the two software-directory fields. NOTE WHAT THE SENTINEL DOES AND
    DOES NOT MEAN: it exempts the path from the load-time existence check because the
    toolkit CREATES that directory. It is NOT a statement that the user does not supply
    the path -- the user names the directory the clone/build gate builds into, and
    `system.py` raises ConfigurationError when it is None. These fields are schema-
    Optional only so a portability-scrubbed bundle's cfg_system.yaml loads for
    bundle-local EDA; they are required for any real run.
    """
    extra = getattr(field_info, "json_schema_extra", None)
    return isinstance(extra, dict) and bool(extra.get("toolkit_owned_output"))


def _options_tooltip(field_info: Any) -> str:
    """`key — definition` lines for a field's option glossary, or "" when undeclared.

    The SAME `declared(field_info, "options")` read `_description_cell` uses; the
    glossary is declared once on the field and rendered from that one declaration in
    both places. Plain text, newline-separated: Tabulator's tooltip renders a text
    node, and the cell keeps only the affordance count.
    """
    from hhemt.config.base import declared

    options = declared(field_info, "options")
    if not options:
        return ""
    return "\n".join(f"{k} — {v}" for k, v in options.items())


def _config_field_tooltips() -> dict[str, str]:
    """`{"{config_label}.{field_name}": option-glossary tooltip}` for every field.

    A SEPARATE walk from `_config_field_rows`, deliberately, for the same reason
    `_config_field_values` is separate: `_config_field_rows` returns row CELLS and its
    arity is depended on by six tests, so the tooltip payload rides beside it rather
    than widening it. Both walk `model_fields` + `declared(...)`, which IS the single
    declaration -- reading one source twice is not a second source.

    Takes NO analysis argument, preserving `_config_field_rows`'s zero-user-info
    property: a glossary is schema, never a producer value.
    """
    from hhemt.config.analysis import analysis_config
    from hhemt.config.reprex_config import reprex_config
    from hhemt.config.system import system_config

    out: dict[str, str] = {}
    for label, model in (
        ("reprex_config", reprex_config),
        ("system_config", system_config),
        ("analysis_config", analysis_config),
    ):
        for name, fi in model.model_fields.items():
            tip = _options_tooltip(fi)
            if tip:
                out[f"{label}.{name}"] = tip
    return out


def _df_for(
    headers: list[str], rows: list[list[str]], tips: dict[str, list[str]]
) -> pd.DataFrame:
    """Reshape `_sortable_grid_table`'s (headers, rows) into Tabulator's row-dict model.

    Cells are PRE-ESCAPED HTML fragments (badges, `<code>`, `<strong>`), so every
    column is rendered with `formatter: "html"` at the call site.

    `tips` maps a header to a per-row list of tooltip strings. Each becomes a
    `"{header}__tip"` COMPANION column in the returned frame. The companion is
    deliberately excluded from the columns SPEC by the caller (which builds the spec
    from `df[headers]`): Tabulator renders only declared columns while
    `row.getData()` carries every key, so the tip is reachable without occupying a
    column, a width, or a sidebar toggle.
    """
    import pandas as pd

    data: dict[str, list[str]] = {h: [r[i] for r in rows] for i, h in enumerate(headers)}
    for header, column in tips.items():
        if header in data:
            data[f"{header}__tip"] = column
    return pd.DataFrame(data)


def _description_cell(field_info: Any) -> str:
    """Description prose, plus the applicability note and option glossary when DECLARED.

    GRACEFUL DEGRADATION IS STRUCTURAL. For a field with no declaration this returns
    exactly `_esc(field_info.description or "—")` -- byte-identical to the pre-
    declaration renderer -- because both appends are guarded on the presence of a
    reserved key. 104 of the 107 rendered fields take that path today.
    """
    from hhemt.config.base import declared, render_clauses

    parts = [_esc(field_info.description or "—")]
    applies = declared(field_info, "applies_when")
    if applies:
        parts.append(f"<div class='instruction'>Applies when {_esc(render_clauses(applies))}.</div>")
    options = declared(field_info, "options")
    if options:
        # Iter-12 item 18. The glossary moves OUT of the cell and into the column's
        # Tabulator tooltip ([Q148]), UNIFORMLY -- there is deliberately no length
        # threshold. A threshold would put one field's definitions in a tooltip and
        # another's inline, so a reader could not learn one rule for where definitions
        # live, and two presentations of one concept is what the single-source mandate
        # forbids. If a glossary reads badly in a tooltip it is too long, and the fix is
        # to shorten it at its single declared source. The cell keeps only the
        # affordance; `_options_tooltip` supplies the text from the SAME declaration.
        parts.append(f" ({len(options)} options)")
    return "".join(parts)


def _requiredness_cell(field_info: Any) -> str:
    """Derive Required / Conditional / Optional (+default) from the Pydantic FieldInfo.

    Never hand-written. `is_required()` and `default` are the schema's own answer, and
    the CONDITIONAL branch reads the same `required_when` declaration that
    `cfgBaseModel._enforce_required_when` enforces -- there is no second source, so this
    cell cannot state a requirement the validator does not impose, or omit one it does.

    Branch order is load-bearing. Toolkit-owned is checked FIRST because such a field is
    nominally Optional-with-None but is not something a reproducer supplies at all.
    Unconditional `is_required()` is checked BEFORE the conditional branch so a field
    that is always required never renders as merely conditional.
    """
    from hhemt.config.base import declared, render_clauses

    if _is_toolkit_owned(field_info):
        return "<strong>Required</strong> — you choose the location; the toolkit clones/builds into it"
    try:
        required = field_info.is_required()
    except Exception:  # noqa: BLE001 -- schema introspection must not break the render
        return "—"
    if required:
        return "<strong>Required</strong>"
    clauses = declared(field_info, "required_when")
    if clauses:
        cell = f"<strong>Conditional</strong> — required when {_esc(render_clauses(clauses))}"
        # [Q151]: a conditionally-required field can ALSO carry a meaningful default --
        # the trigger says when it must be supplied, the default says what applies when
        # it is not. Today none of the 15 declared fields is both (all default to None),
        # so this branch is inert; it exists so the first one that IS does not silently
        # lose its default. `None` is deliberately NOT shown -- it is an absence, and
        # `_render_config_value` already rules that one absence gets one spelling per row.
        _default = getattr(field_info, "default", None)
        if _default is not None and type(_default).__name__ != "PydanticUndefined":
            cell += f" (default: {_esc(repr(_default))})"
        return cell
    default = getattr(field_info, "default", None)
    if getattr(default, "__class__", type(None)).__name__ == "PydanticUndefined":
        return "Optional"
    if callable(getattr(field_info, "default_factory", None)) and default is None:
        return "Optional (default: computed)"
    return f"Optional (default: {_esc(repr(default))})"


def _config_field_values(analysis: TRITONSWMM_analysis) -> dict[str, str]:
    """`{"{config_label}.{field_name}": rendered value}` for the HPC/EXPERIMENT buckets.

    A SEPARATE function from `_config_field_rows` on purpose. `_config_field_rows`
    takes no analysis argument by design: its zero-user-info guarantee (R3,
    C-ZERO-USER-INFO) is true BY CONSTRUCTION because it cannot leak a value it never
    sees. Adding an analysis parameter there would downgrade a structural guarantee to
    a discipline. So the value lookup lives here, and the caller applies it only to the
    two buckets whose values the bundle ALREADY ships in cfg_system.yaml /
    cfg_analysis.yaml after `_scrub_user_bucket_fields` has nulled the USER-bucket
    entries. Showing those discloses nothing the bundle does not already carry;
    USER-bucket values are never passed through here at all.

    Paths render relative to the analysis directory (with `..` segments when they live
    outside it), which is what makes them meaningful to a reader holding the bundle
    rather than the producer's filesystem.

    GRACEFUL-ABSENT by contract. A caller may hand over a config object that is not a
    Pydantic model at all (a lightweight stand-in, a partially reconstituted bundle
    config); such an object is SKIPPED and its bucket simply renders without a value
    column. A value column is an enhancement to the reproduction guide -- it must never
    be the reason a metadata page fails to render.
    """
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    out: dict[str, str] = {}
    # `cfg_system` hangs off the SYSTEM, never off the analysis -- `analysis._system`
    # is the accessor every other renderer uses (per_sim_peak_flood_depth.py:186,
    # system_overview.py:111). Reading `analysis.cfg_system` directly always returned
    # None, and the graceful-absent `continue` below turned that into 33 silently
    # blank EXPERIMENT rows rather than an error.
    for config_label, cfg in (
        ("system_config", getattr(getattr(analysis, "_system", None), "cfg_system", None)),
        ("analysis_config", getattr(analysis, "cfg_analysis", None)),
    ):
        if cfg is None:
            continue
        model_fields = getattr(type(cfg), "model_fields", None)
        if not model_fields:
            continue
        for field_name in model_fields:
            try:
                value = getattr(cfg, field_name)
            except Exception:  # noqa: BLE001
                continue
            out[f"{config_label}.{field_name}"] = _render_config_value(value, analysis_dir)
    return out


def _render_config_value(value: Any, analysis_dir: Path) -> str:
    """Render one config value for display: paths analysis-dir-relative, others repr-ish.

    Iter-11 item 20. An absent value renders `None`, in code voice, NOT italic `null`.
    `null` is the YAML/JSON spelling of the serialised file; the reader of this table is
    holding a Python config object whose value IS `None`, and the Required column
    immediately to the left already says so -- `_requiredness_cell` renders
    `Optional (default: {repr(default)})`, and `repr(None)` is `None`. Two spellings of
    one absence, two cells apart in one row, is the inconsistency the item names.
    """
    if value is None:
        return _code("None")
    if isinstance(value, Path):
        try:
            return _code(os.path.relpath(value, analysis_dir))
        except ValueError:
            # Different drive/root -- relpath is undefined, so show the absolute path.
            return _code(value)
    if isinstance(value, list | tuple):
        if not value:
            return "<em>empty</em>"
        return ", ".join(_render_config_value(v, analysis_dir) for v in value)
    if isinstance(value, bool | int | float | str):
        return _esc(value)
    return _esc(str(value))


def _bucket_badge(bucket: str) -> str:
    color = _BUCKET_COLOR[bucket]
    return f'<span class="badge" style="background-color:{color}">{_esc(_BUCKET_VERB[bucket])}</span>'


#: How many distinct swept values a `Value used` cell shows before eliding. A
#: continuous sweep over 40 rows must not render 40 values inside one table cell.
_VARIED_VALUE_PREVIEW: int = 6


def _sensitivity_varied_values(analysis: TRITONSWMM_analysis) -> dict[str, tuple[str, str]]:
    """`{"{config_label}.{field_name}": (cell, tooltip)}` for every parameter in the sweep table.

    The value is ALWAYS a 2-tuple, for a varied parameter and a constant one alike.
    A constant column has no hover payload, so its tooltip is the empty string -- the
    same `(cell, "")` pair the consumer already builds for a non-swept field. The two
    shapes this function used to return (a bare string here, a tuple there) crashed
    `_build_reprex_guide_html`'s unconditional `_cell, _tip = _v` unpack on the first
    real sweep table carrying a constant column, and the declared `dict[str, str]`
    described the crashing branch while contradicting the working one -- which is why
    neither a reader nor a type-checker caught it. Keep this return uniform.

    DERIVED, never hand-listed. The sensitivity CSV's own column names ARE the
    declaration of what varies, and `sensitivity_analysis` already owns that
    grammar (`system.{field}` / `analysis.{field}` / `hpc.{alias}` / the deprecated
    bare analysis-field name). Reading the frame's columns therefore yields exactly
    this experiment's swept set, and adding a sweep axis to the CSV changes this map
    with no renderer edit.

    Why the cell is REPLACED and not annotated: for a varied parameter the master's
    config value is one arm's setting. Rendering it as "Value used" presents one
    sub-analysis's configuration as though it were the experiment's, which is
    affirmatively misleading rather than merely incomplete. An appended "(varied)"
    tag does not undo a number the reader has already read as the answer.

    GRACEFUL-ABSENT, mirroring `_config_field_values`: a non-sensitivity analysis, a
    scrubbed bundle whose sensitivity CSV is not on disk, or any frame-access failure
    yields `{}` and the guide renders exactly as it did before.
    """
    sensitivity = getattr(analysis, "sensitivity", None)
    if sensitivity is None:
        return {}
    try:
        from hhemt.config.analysis import analysis_config
        from hhemt.sensitivity_analysis import (
            _is_analysis_overlay_column,
            _is_hpc_overlay_column,
            _is_system_overlay_column,
            _resolve_hpc_alias_to_analysis_field,
            _strip_analysis_prefix,
            _strip_system_prefix,
        )

        columns = list(sensitivity._df_setup_full.columns)
    except Exception:  # noqa: BLE001 -- a display column must never break the render
        return {}

    out: dict[str, tuple[str, str]] = {}
    for col in columns:
        if col == "system_config_yaml":
            continue
        if _is_system_overlay_column(col):
            label = f"system_config.{_strip_system_prefix(col)}"
        elif _is_hpc_overlay_column(col):
            label = f"analysis_config.{_resolve_hpc_alias_to_analysis_field(col)}"
        elif _is_analysis_overlay_column(col):
            label = f"analysis_config.{_strip_analysis_prefix(col)}"
        elif col in analysis_config.model_fields:
            label = f"analysis_config.{col}"  # deprecated bare-name form
        else:
            continue
        try:
            distinct = sorted({str(v) for v in sensitivity._df_setup_full[col].dropna().tolist()})
        except Exception:  # noqa: BLE001
            distinct = []
        # Iter-12 item 21: a column that takes exactly ONE value across every row is not
        # a varied parameter -- it is a constant that happens to be spelled in the sweep
        # table. Rendering the varied marker there withholds a value the reader can and
        # should be given, and misdescribes the experiment's axes.
        if len(distinct) == 1:
            # The 2-tuple is UNIFORM with the varied branch below: cell, then hover
            # payload. A constant column has nothing to hover, so the tooltip is empty
            # -- matching what the consumer already builds for a non-swept field.
            out[label] = (_code(distinct[0]), "")
            continue
        # Iter-12 item 17. The marker and its value list are returned SEPARATELY: the
        # marker is the cell, the list is the tooltip payload. Listing values inline
        # competed with the marker and blew out the column on a continuous sweep -- the
        # marker answers "what value was used", the value set is the follow-up question.
        # Splitting them here is also what lets the tooltip ENGINE change without
        # touching this function again.
        shown = ", ".join(str(v) for v in distinct[:_VARIED_VALUE_PREVIEW])
        if len(distinct) > _VARIED_VALUE_PREVIEW:
            shown += f", … ({len(distinct)} values)"
        out[label] = (
            '<strong class="tip-affordance">Varied by the sensitivity analysis</strong>',
            shown,
        )
    return out


class ReprexGuide(NamedTuple):
    """The guide's markup plus the Tabulator fragments its three tables need.

    Two-part return because the tables are Tabulator fragments now ([Q148]): the
    markup carries only mount points, and the caller emits the fragments' styles and
    scripts at DOCUMENT level so the ~40 KB shared filter blob lands once for three
    tables rather than three times.
    """

    html: str
    fragments: list[tuple[str, TableFragment]]


def _build_reprex_guide_html(
    values_by_field: dict[str, str] | None = None,
    varied_by_field: dict[str, tuple[str, str]] | None = None,
) -> ReprexGuide:
    """Static grouped table: every config field -> USER=Supply / HPC=Amend / EXPERIMENT=Keep.

    Grouped (not a flat sortable grid) because the primary task a reproducer
    performs is "what do I DO with this field?" -- answered pre-attentively by
    the bucket. Each group is redundant-coded with an Okabe-Ito badge AND the
    instruction verb, so it survives grayscale and CVD.

    Every column is DERIVED from the Pydantic schema -- description from
    `Field(description=...)`, requiredness and default from `FieldInfo` -- so none of
    it is restated by hand and none of it can drift from the model.

    ``values_by_field`` (from `_config_field_values`) and ``varied_by_field`` (from
    `_sensitivity_varied_values`) are applied to the HPC and EXPERIMENT buckets only.
    The USER bucket never receives either, so its cells stay placeholder-only and the
    zero-user-info property of the shipped page is preserved.
    """
    rows_by_bucket, unclassified = _config_field_rows()
    _field_tooltips = _config_field_tooltips()
    # (bucket, fragment) pairs, NOT a bare list: a bucket with no rows `continue`s
    # without appending a fragment, so zipping a bare list against _BUCKET_ORDER would
    # pair the wrong bucket's mount with the wrong table the moment any bucket is empty.
    fragments: list[tuple[str, TableFragment]] = []

    parts: list[str] = [_heading("Reproduction Guide")]
    parts.append(
        "<p class='note'>Every configuration field below is grouped by what a reproducer must "
        "do with it. <strong>Description</strong>, <strong>Required</strong>, the permissible "
        "options listed beneath a description, and the default shown beside <em>Optional</em> "
        "are read directly off the configuration schema, not maintained separately. A "
        "<strong>Conditional</strong> requirement names the setting that triggers it, and is "
        "read from the same declaration the configuration validator enforces. A field the "
        "sensitivity analysis VARIED shows that instead of a value: it has no single value "
        "used, and naming one arm's setting would misreport the experiment. Values are shown "
        "only for the bundled HPC and experiment "
        "settings — which the bundle already carries in <code>cfg_system.yaml</code> / "
        "<code>cfg_analysis.yaml</code>. The <em>Supply</em> block is placeholders only: this "
        "page never carries the producing user's own account, paths, or host details, so it is "
        "safe to ship inside a bundle. File paths are shown relative to the analysis "
        "directory. The <code>report</code> block may be supplied as a standalone YAML file "
        "instead of inline, by passing <code>report_config=&lt;path&gt;</code> to "
        "<code>analysis.run()</code>; an explicit path takes precedence over the inline "
        "block, and the inline block is used when no path is given.</p>"
    )

    if unclassified:
        parts.append(
            _banner(
                "Some configuration fields could not be classified against this toolkit "
                "version's reprex taxonomy (schema drift): " + ", ".join(unclassified)
            )
        )

    for bucket in _BUCKET_ORDER:
        rows = rows_by_bucket[bucket]
        parts.append(f"<h4>{_bucket_badge(bucket)} {_esc(_BUCKET_HEADING[bucket])}</h4>")
        parts.append(f"<p class='instruction'>{_esc(_BUCKET_INSTRUCTION[bucket])}</p>")
        if not rows:
            parts.append(_banner("No configuration fields fall in this bucket."))
            continue
        # Iter-12 item 20. `Placeholder` survives in the SUPPLY bucket only. There it is
        # the block's one per-row instruction and the block has no `Value used` column to
        # carry one (zero-user-info: `_config_field_rows` takes no analysis argument by
        # design). In `hpc` and `experiment` it carried a single bucket-CONSTANT across 27
        # and 74 rows -- a string already printed once in _BUCKET_HEADING and again in
        # _BUCKET_INSTRUCTION directly above the table -- and on any row with no producer
        # value the `Value used` fallback rendered that SAME constant a second time, two
        # cells apart. `_BUCKET_PLACEHOLDER` is RETAINED: it is still that fallback, which
        # is the cell that survives.
        _values = values_by_field or {}
        _varied = varied_by_field or {}
        value_tips: list[str] = []
        # DEVIATION FROM THE APPROVED item-20 SPEC, surfaced at apply time. The spec
        # dropped `Placeholder` from `hpc`/`experiment` unconditionally. That is correct
        # only when the `Value used` column REPLACES it -- and `Value used` is added just
        # below ONLY when producer values exist. On the no-values path (a scrubbed bundle,
        # or any render without an analysis) those two blocks would then carry NO per-row
        # instruction at all, and `_BUCKET_PLACEHOLDER`'s text would never appear. The
        # item-20 rationale was DUPLICATION, so the column is dropped exactly where a
        # duplicate exists and kept where none does.
        _has_value_col = bucket in ("hpc", "experiment") and bool(_values or _varied)
        headers = ["Field", "Description", "Required"]
        if _has_value_col:
            rows = [row[:3] for row in rows]
        else:
            headers.append("Placeholder")
        if _has_value_col:
            # Value disclosure is bucket-scoped. USER-bucket values are host-local and are
            # already nulled out of the bundle's cfg_*.yaml by _scrub_user_bucket_fields, so
            # rendering them here would disclose what the bundle deliberately withholds.
            # HPC/EXPERIMENT values ARE carried in the bundle already, so showing them
            # discloses nothing new.
            #
            # Precedence: a SWEPT parameter's cell is replaced outright, because the
            # master's stored value is one arm's setting and rendering it here would
            # present one sub-analysis's configuration as the experiment's. A field with
            # neither a swept marker nor a producer value falls back to the bucket
            # placeholder rather than a bare em-dash -- the reprex_config.target_*
            # rows have no producer value BY DESIGN (they describe the reproducer's
            # target system), and a dash there reads as "the value was empty".
            headers.append("Value used")
            _fallback = _code(_BUCKET_PLACEHOLDER.get(bucket, "—"))
            _new_rows = []
            for row in rows:
                _label = _strip_code(row[0])
                _v = _varied.get(_label)
                if _v is None:
                    # Not a swept parameter: the producer value, else the bucket
                    # placeholder. No tooltip.
                    _cell, _tip = _values.get(_label, _fallback), ""
                else:
                    # `_sensitivity_varied_values` returns a uniform (cell, tooltip)
                    # 2-tuple for BOTH branches: a varied parameter's cell is the
                    # marker and its tooltip is the value list; a constant one's cell
                    # is the value and its tooltip is empty.
                    _cell, _tip = _v
                _new_rows.append(row[:3] + [_cell])
                value_tips.append(_tip)
            rows = _new_rows
        # {16}: default the guide's row order -- required fields first, then
        # alphabetical within each tier. DATA-side deliberately: Tabulator's
        # `initialSort` sorts the RENDERED `Required` cell as TEXT, where ascending
        # puts `<strong>Required</strong>` LAST and descending happens to work only
        # by the accident that `R` > `O` > `C` -- so it would invert the moment a
        # fourth requiredness label is added. Sorting the data states the intent
        # directly. Placed at 8-space indent so it covers BOTH the value-column and
        # placeholder-column branches, in each of which `row[0]` is the `_code`-wrapped
        # field label and `row[2]` is the `_requiredness_cell` output. Both Required
        # forms (`_requiredness_cell` early-returns at two sites) begin with the same
        # `<strong>Required</strong>` literal, so the prefix test catches each.
        # SCOPE GUARD: this is a per-call-site sort and must NOT migrate into
        # `build_options_dict` / `_tabulator_defaults`, which are shared with the
        # 39-column `scenario_status_appendix`.
        rows = sorted(
            rows,
            key=lambda _r: (
                0 if _r[2].startswith("<strong>Required</strong>") else 1,
                _strip_code(_r[0]).lower(),
            ),
        )
        # [Q148]: these three tables render through Tabulator so their columns are
        # user-resizable, accepting the CDN dependency at view time. Scoped to THIS call
        # family -- the provenance, data-availability and SLURM sections keep the
        # self-contained `_sortable_grid_table` path, so only this section goes blank
        # without a network.
        #
        # `resizableColumns: True` is an EXPLICIT per-call override of the shared
        # default, which is False (`build_options_dict`, iter 9.1). WITHOUT it, adoption
        # buys the network dependency and NOT the resizing the ruling asked for. The
        # iter-9.1 rationale is keyed on 39 columns at `fitDataStretch`, where the 3px
        # resize handle sits adjacent to the next column's left edge; these tables carry
        # 3-4 columns, so the handles are far apart. Do NOT promote this to the shared
        # default -- that regresses the 39-column scenario_status appendix.
        #
        # `column_panel=False` preserves Iter-11 item 16: these tables get no
        # column-selector, because one would offer a way to hide the very columns the
        # block exists to deliver.
        tip_columns: dict[str, list[str]] = {}
        _desc_tips = [_field_tooltips.get(_strip_code(r[0]), "") for r in rows]
        if any(_desc_tips):
            tip_columns["Description"] = _desc_tips
        if "Value used" in headers:
            _vi = headers.index("Value used")
            tip_columns["Value used"] = [value_tips[i] for i in range(len(rows))]
        df_full = _df_for(headers, rows, tip_columns)
        columns_spec = build_columns_spec(
            df_full[headers], visible_columns_default=None, header_filter=True
        )
        for col_spec in columns_spec:
            # Cells are pre-escaped HTML fragments, not plain text.
            col_spec["formatter"] = "html"
            if col_spec["title"] in tip_columns:
                col_spec["tooltip"] = "__TRF_TOOLTIP__"
                # File 4's `.tabulator-cell.trf-has-tip` rule. Tabulator's `tooltip`
                # option renders its own popup and sets no DOM `title`, so the
                # attribute selectors cannot reach these cells; the class can.
                col_spec["cssClass"] = "trf-has-tip"
        fragments.append(
            (bucket, build_table_fragment(
                container_id=f"reprex-{bucket}",
                options=build_options_dict(
                    df_full,
                    columns_spec=columns_spec,
                    # Iter-13. Bounded exactly as `div.table-scroll` is bounded by the
                    # `max-height: min(70vh, 640px)` rule above, and for the same reason
                    # recorded there: `vh` resolves against the IFRAME's height, and the
                    # combined report sets that height IMPERATIVELY ONCE, from the arm
                    # frame's own onload handler (`scrollHeight + 24`). Tabulator reads
                    # options.height ONCE at construction (config/report.py, table_height
                    # field description), so a post-construction frame grow re-resolves the
                    # CSS box while the render window stays fixed at its pre-grow size --
                    # two one-shot measurements that cannot agree. A fixed ceiling makes
                    # that divergence unreachable, because a re-resolved `vh` cannot move
                    # the box past it. The Iter-11 item-24 height bound is preserved: 70vh
                    # still governs wherever it is the smaller of the two.
                    table_height="min(70vh, 640px)",
                    pagination_size=0,
                    persistence_id=f"reprex-{bucket}",
                    extra_options={"resizableColumns": True},
                ),
                js_mode="cdn",  # inline bundling is unimplemented (Gotcha 51)
                renderer_name="metadata",
                column_panel=False,
            ))
        )
        parts.append(f'<div id="reprex-{bucket}-mount"></div>')

    return ReprexGuide("\n".join(parts), fragments)


# --- (3) SLURM efficiency ----------------------------------------------------


#: Columns carried through from the plugin's CSV, in display order. The plugin's unnamed
#: index column and its intermediate unit-conversion columns are dropped -- they are
#: restatements of neighbours (MaxRSS vs MaxRSS_MB) and add width without adding meaning.
@dataclass(frozen=True)
class _Reduction:
    """ONE declaration of a reduction: its symbol, its sentence, and the code that DOES it.

    [Q153]'s single-source condition is met structurally rather than by discipline. `rule` is
    the ONLY place the reduction is stated in prose -- the header tooltip renders it and the
    caption renders it, which is two renderings of one declaration rather than two statements.
    `tag` is a symbol, not a restatement: it is meaningless without the tooltip that expands it,
    so it cannot drift into a second definition. And `apply` is the reduction, so a right label
    over a wrong sum is unreachable -- swapping the callable moves the number AND the sentence.
    """

    #: The caption's TERM for this reduction. REQUIRED (enforced below) and unique across
    #: `_EFF_COLUMNS` (enforced at import, beside the tuple). Split from `tag` because the two
    #: answer different questions, and conflating them produced the defect this closes: `tag`
    #: is a header SUFFIX and is legitimately empty for a value that was looked up rather than
    #: reduced, so the caption fell back to `r.tag or "joined"` and manufactured ONE term for
    #: THREE different operations -- the only term on the page with no declaration, inside the
    #: system whose premise is that every term has exactly one.
    #:
    #: Defaulted to "" rather than left required-positional so that adding the field does not
    #: have to be one atomic rewrite of all six construction sites. The default is NOT a
    #: permitted value: `__post_init__` rejects it, so the module still fails at IMPORT until
    #: every site declares one. Smaller diff, identical loudness.
    name: str = ""
    tag: str = ""
    #: WHY this reduction was chosen, as its own sentence. Separate from `rule` because `rule`
    #: had been carrying both what-the-reduction-is and why-it-was-picked in one run of prose,
    #: which is why the rationale could not be found as its own thing.
    why: str = ""
    rule: str = ""
    apply: Callable[[list[dict], dict], tuple[str, str]] | None = None

    def __post_init__(self) -> None:
        """Reject a reduction that cannot be rendered. Construction-time, because a frozen
        dataclass makes this a construction-time fact and there is no later moment at which
        it becomes false.

        Only PER-INSTANCE properties are checked here. Uniqueness is deliberately NOT: it is a
        property of a SET, not of an instance, so enforcing it via a module-level registry in
        `__post_init__` would be wrong-scoped -- it would false-trip on a test constructing a
        standalone reduction that reuses a name, and it would make import order significant.
        The set-level check lives beside `_EFF_COLUMNS`, which is the only place the set exists.
        """
        for field_name in ("name", "why", "rule"):
            if not getattr(self, field_name).strip():
                raise ValueError(
                    f"_Reduction declared with an empty {field_name!r}. Every reduction must "
                    f"name itself, say why it was chosen, and state its rule -- the caption, "
                    f"the header tooltip and the per-value provenance are all renderings of "
                    f"these, and a blank one is what let the term 'joined' be invented at "
                    f"render time for three different operations."
                )


def _slurm_seconds(text: str) -> float:
    """Parse a SLURM duration (`[DD-]HH:MM:SS[.mmm]`, `MM:SS.mmm`) to seconds; 0.0 if unparsable."""
    raw = (text or "").strip()
    if not raw:
        return 0.0
    days = 0.0
    if "-" in raw:
        d, _, raw = raw.partition("-")
        try:
            days = float(d)
        except ValueError:
            return 0.0
    parts = raw.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return 0.0
    while len(nums) < 3:
        nums.insert(0, 0.0)
    return days * 86400 + nums[0] * 3600 + nums[1] * 60 + nums[2]


def _slurm_rss_mb(text: str) -> float | None:
    """Parse a SLURM MaxRSS (`491288K`, `1.2G`, `900M`) to MB; None when absent/unparsable."""
    raw = (text or "").strip()
    if not raw:
        return None
    unit, digits = raw[-1].upper(), raw[:-1]
    scale = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024}.get(unit)
    if scale is None:
        digits, scale = raw, 1 / 1024  # bare number: sacct's default unit is K
    try:
        return float(digits) * scale
    except ValueError:
        return None


def _step_suffix(job_id: str) -> str:
    return job_id.split(".", 1)[1] if "." in job_id else "job"


def _attempt_label(n: int) -> str:
    """An attempt's reader-facing name — the SINGLE declaration of it ([Q153]).

    The purpose column's suffix and the per-attempt roster are two RENDERINGS of this one
    function, never two hand-authored statements of the same rule. Attempt 0 is the initial
    run rather than "resume 0", which would imply a resume that did not happen.
    """
    return "initial run" if n == 0 else f"resume {n}"


def _tres_cpu_seconds(tres: str) -> float | None:
    """Seconds from the `cpu=` key of a `TRESUsageIn*` field, or None if it carries none.

    The field is a flat comma-separated `key=value` list, e.g.
    ``cpu=00:00:13,energy=0,fs/disk=0,gres/gpumem=0,gres/gpuutil=0,mem=49524K``.
    None (not 0.0) on an absent key, so a missing measurement stays distinguishable from a
    measured zero -- the distinction the whole efficiency column turns on.
    """
    for item in (tres or "").split(","):
        key, sep, value = item.strip().partition("=")
        if sep and key.strip() == "cpu":
            value = value.strip()
            return _slurm_seconds(value) if value else None
    return None


def _reduce_cpu_sum(steps: list[dict], job_row: dict) -> tuple[str, str]:
    """CPU seconds SUMMED over every step including `.batch` -- `seff`'s own reduction.

    Sourced from `TRESUsageInTot`'s `cpu=` key, NOT from `TotalCPU`. That is the recovery
    module's own stated contract (`slurm_job_recovery.py`): `TotalCPU` "MUST NOT be the
    source of a CPU-efficiency figure" because it reads `00:00:00` for any work performed
    in an `srun` step, so on a simulation job it carries the batch wrapper's CPU only.
    Reading it here produced a FALSE SMALL number rather than an obviously-broken one --
    on the captured job 18396137 it summed to 3.744 s against 120 s elapsed (~3%) while
    the steps actually consumed 3 + 17 + 13 s (~27%). A visibly wrong 0.0 invites a second
    look; a plausible 3% does not.

    A step with no `cpu=` key contributes nothing and is NOT counted as a measured zero;
    if no step carries one, the cell is blank rather than 0, so an unpopulated recovery CSV
    renders an em-dash instead of asserting the job used no processor.
    """
    per_step = [(_step_suffix(s.get("JobID", "")), _tres_cpu_seconds(s.get("TRESUsageInTot", ""))) for s in steps]
    measured = [(n, v) for n, v in per_step if v is not None]
    if not measured:
        return ("", f"no step reported CPU time across {len(per_step)} step(s)")
    # FAIL SAFE, not fail plausible. When `.batch` is the only step that reported CPU but
    # the job HAS solver steps, the recovery CSV is stale or un-rerun and a sum over what
    # is present measures the Python wrapper alone -- 1.6% against a true 78.8% on job
    # 18396671. That number is worse than a blank, because a small plausible figure reads
    # as a wasted allocation and gets believed, while an em-dash invites a second look.
    # Blank it and say why, so the shortfall is legible rather than rendered.
    if [n for n, _v in measured] == ["batch"] and any(n.isdigit() for n, _v in per_step):
        return ("", f"only .batch reported CPU; {len(per_step) - 1} solver step(s) unmeasured")
    total = sum(v for _n, v in measured)
    nonzero = [n for n, v in measured if v > 0]
    detail = f"summed over {len(measured)} of {len(per_step)} step(s)"
    return (f"{total:.3f}", f"{detail}; nonzero on {', '.join(nonzero)}" if nonzero else f"{detail}; all zero")


def _reduce_rss_max(steps: list[dict], job_row: dict) -> tuple[str, str]:
    """Peak memory MAXED over steps -- `seff`'s reduction, and correct for request-sizing.

    Named per cell because the peak step is NOT constant: on a resumed sim it is the `python`
    wrapper, not the solver, and a reader comparing two rows needs to know that.
    """
    best: tuple[float, str] | None = None
    for s in steps:
        value = _slurm_rss_mb(s.get("MaxRSS", ""))
        if value is not None and (best is None or value > best[0]):
            best = (value, _step_suffix(s.get("JobID", "")))
    if best is None:
        return ("", f"no step reported memory across {len(steps)} step(s)")
    return (f"{best[0]:.1f}", f"peak of {len(steps)} step(s), from step {best[1]}")


def _reduce_job_field(field: str):
    """Read `field` off the JOB row. Not a step reduction -- the value has no step to name."""

    def _apply(steps: list[dict], job_row: dict) -> tuple[str, str]:
        value = (job_row or {}).get(field, "")
        if not value:
            return ("", "the job's accounting record was not recovered for this job")
        return (value, "read from the job's own accounting record")

    return _apply


def _reduce_joined(field: str):
    """Carry a value joined from hhemt's own records; no SLURM reduction is involved."""

    def _apply(steps: list[dict], job_row: dict) -> tuple[str, str]:
        return ((job_row or {}).get(field, ""), "")

    return _apply


_SUM_STEPS = _Reduction(
    name="Summed across steps",
    why=(
        "CPU time is additive across a job's steps, so the job's total is their sum. This is "
        "`seff`'s own reduction, chosen so this number agrees with what the cluster's standard "
        "tool reports rather than offering a second opinion a reader would have to reconcile."
    ),
    tag="\u03a3 steps",
    rule=(
        "Summed across every step of the job, including the batch step — `seff`'s own "
        "reduction, read from its source. The per-step CPU time is taken from each step's "
        "recorded resource usage rather than from its `TotalCPU` field, which reads zero "
        "for work done inside an `srun` step and so would report only the batch wrapper's "
        "own CPU. A step that recorded no CPU time is left out of the sum rather than "
        "counted as zero, so a job with no usable measurement reads blank, not 0%. The "
        "denominator is the CPU-time the allocation was BILLED — its CPU count times its "
        "wall time — so this is the share of what you were charged for that was consumed, "
        "not the share of its cores the solver kept busy. The two differ here and the gap "
        "is the point: a job's solver steps run one after another with the allocation idle "
        "between them, so on this campaign the figure tracks how much of the allocation had "
        "a solver running in it rather than how hard that solver worked."
    ),
    apply=_reduce_cpu_sum,
)
_MAX_STEPS = _Reduction(
    name="Maximum across steps",
    why=(
        "A memory request has to be sized for the largest step, not the average one, so the "
        "peak is the number that decides whether the next run fits. The peak step is often the "
        "runner's own wrapper rather than the solver -- measured 817884K against 33164K on one "
        "job -- which is why each cell names the step it came from."
    ),
    tag="max step",
    rule=(
        "The largest value across the job's steps, which is `seff`'s reduction and the correct "
        "one for sizing a request. Each cell names the step that produced it, because the peak "
        "is often the runner's own wrapper rather than the solver."
    ),
    apply=_reduce_rss_max,
)
_JOB_RECORD = _Reduction(
    name="Read from the job record",
    why=(
        "A job's wall time is neither the sum nor the maximum of its steps' -- measured on one "
        "job as 296 s against a 235 s sum and a 67 s max, three different numbers of which only "
        "one is the job's elapsed time. It is read from the allocation's own record because it "
        "is not reconstructible from the steps at all."
    ),
    tag="job record",
    rule=(
        "Read from the job's own accounting record rather than reduced from its steps -- `seff` "
        "does the same, because a job's wall time is neither the sum nor the maximum of its "
        "steps' wall times."
    ),
    apply=None,
)
#: `Queue, this job (s)` needs its OWN reduction rather than sharing `_TOOLKIT_JOIN`, because
#: it has TWO sources and `_TOOLKIT_JOIN`'s rule -- "not from SLURM accounting" -- is false for
#: the fallback. Sharing it would make the header tooltip lie on exactly the rows the fallback
#: fills, and the header rule is one of `[Q153]`'s three trackability surfaces.
#:
#: The toolkit map stays PREFERRED because it is per-job and exact. The fallback exists because
#: that map is routinely EMPTY: it is derived from `_walltime` ledgers that a render bundle does
#: not carry, so an off-cluster re-render has no access to it. Measured on this campaign
#: (2026-08-20, `synth_cc_clean_tritonswmm`): `queue_seconds_by_jobid` non-empty on **0 of 28**
#: scenarios and the toolkit's own `queue_seconds_coverage` reading `0/1` on all 28, against
#: SLURM's `Planned` populated on **963 of 963** job rows in the recovery CSV this module
#: already reads. So the column rendered an em-dash on every row while an authoritative value
#: sat in an already-open file. The fallback is ADDITIVE -- it never overwrites a toolkit value.
_QUEUE_JOIN = _Reduction(
    name="Queue-wait lookup",
    why=(
        "The wait happened to the allocation, not to any step, so there is nothing to reduce. "
        "hhemt's own per-job map is preferred because it is exact; the SLURM fallback exists "
        "because that map is routinely empty -- it derives from `_walltime` ledgers a render "
        "bundle does not carry, so an off-cluster re-render has no access to it."
    ),
    # EMPTY tag, matching the joined-column family. The tag slot renders as a header SUFFIX
    # describing a REDUCTION over steps -- `(job record)`, `(Σ steps)`, `(max step)` -- and a
    # queue wait is neither reduced nor summarised, it is looked up. A non-empty tag here
    # rendered `Queue, this job (s) (queue)`, which restates the column name and claims a
    # reduction that does not happen. The two-source disclosure belongs in `rule` (the header
    # tooltip) and in the per-value provenance, not in the header text.
    tag="",
    rule=(
        "How long this job waited before it started. Preferred source is hhemt's own per-job "
        "queue map; where that is absent the value falls back to SLURM's `Planned` on the job "
        "accounting record. Each cell's tooltip names which of the two produced that value."
    ),
    apply=None,
)
#: `Attempts` is a join like the others, but it carries a SCOPE LIMIT the other joined
#: columns do not, so it gets its own declaration rather than sharing `_TOOLKIT_JOIN`.
#: Authored once here and rendered three ways -- header symbol, header/value tooltip, and
#: the caption -- which is what keeps it from being restated in prose somewhere.
_ATTEMPTS_JOIN = _Reduction(
    name="Attempt count",
    why=(
        "How many times a simulation ran is a property of hhemt's own per-attempt records, not "
        "of SLURM accounting, which cannot distinguish an attempt from a concurrent simulation "
        "-- the accounting rows are structurally identical in the two cases. Where the run "
        "method cannot support the count, it is reported as not measured rather than given a "
        "number a reader could not tell from a real one."
    ),
    tag="",
    rule=(
        "How many times this simulation ran, joined from hhemt's own per-attempt records. "
        "The count is only recoverable when each attempt is its own SLURM job, which is how "
        "`batch_job` submits them. Under `1_job_many_srun_tasks` every attempt is a step of "
        "one shared allocation and a step cannot be attributed to a particular simulation, "
        "so the count is not recovered there and reads 1 for a run that may have resumed "
        "several times. A 1 in this column therefore means 'one attempt, or not recoverable' "
        "-- check the run method before reading it as a resume-free run."
    ),
    apply=None,
)

_TOOLKIT_JOIN = _Reduction(
    name="Carried from toolkit records",
    why=(
        "These describe what hhemt ASKED FOR rather than what SLURM measured, so no reduction "
        "applies and none is claimed -- the value is carried from the record that made the "
        "request. This is the family whose three members previously shared the invented term "
        "'joined' despite one of them reading from SLURM accounting after all."
    ),
    tag="",
    rule="Joined from hhemt's own per-rule records for this job, not from SLURM accounting.",
    apply=None,
)


@dataclass(frozen=True)
class _EffColumn:
    """One column: its data key, its reader-facing noun, and the reduction that produced it."""

    key: str
    label: str
    reduction: _Reduction

    @property
    def header(self) -> str:
        """Label plus the reduction's SYMBOL. The symbol's expansion lives only in `rule`."""
        return f"{self.label} ({self.reduction.tag})" if self.reduction.tag else self.label


#: The table's columns, in display order. [Q144]: `Req mem` sits ADJACENT to `Mem used`, and
#: both are retained on TRUTHFULNESS rather than universality -- a column true on the rows
#: where it applies is kept even where a whole job class leaves it empty.
_EFF_COLUMNS: tuple[_EffColumn, ...] = (
    _EffColumn("JobID", "Job ID", _JOB_RECORD),
    _EffColumn("Elapsed", "Elapsed", _JOB_RECORD),
    _EffColumn("NNodes", "Nodes", _JOB_RECORD),
    _EffColumn("NCPUS", "CPUs", _JOB_RECORD),
    # The header names the DENOMINATOR because every available misreading of this cell is a
    # misreading of the denominator. Measured over 56 multi-attempt campaign jobs: this
    # column's median is 78.9%, the solver's own per-step CPU utilization is 95.2%, and the
    # solver's occupancy of the job's wall time is 78.5% -- so the column tracks OCCUPANCY,
    # not solver efficiency, and "CPU eff" invited exactly the wrong reading.
    _EffColumn("cpu_eff_pct", "Billed CPU used (%)", _SUM_STEPS),
    _EffColumn("max_rss_mb", "Max RSS (MB)", _MAX_STEPS),
    _EffColumn("RequestedMem_MB", "Req mem (MB)", _JOB_RECORD),
    _EffColumn("mem_used_pct", "Mem used (%)", _MAX_STEPS),
    _EffColumn("attempts", "Attempts", _ATTEMPTS_JOIN),
    _EffColumn("record", "Record", _TOOLKIT_JOIN),
    _EffColumn("purpose", "Job desc", _TOOLKIT_JOIN),
    _EffColumn("sa_id", "Sub-analysis", _TOOLKIT_JOIN),
    _EffColumn("event_id", "Event", _TOOLKIT_JOIN),
    _EffColumn("model_type", "Model", _TOOLKIT_JOIN),
    _EffColumn("partition", "Partition", _TOOLKIT_JOIN),
    _EffColumn("gpu_hardware", "GPU hardware", _TOOLKIT_JOIN),
    _EffColumn("n_gpus", "GPUs", _TOOLKIT_JOIN),
    _EffColumn("n_mpi_procs", "MPI ranks", _TOOLKIT_JOIN),
    _EffColumn("n_omp_threads", "OMP threads", _TOOLKIT_JOIN),
    _EffColumn("run_mode", "Run mode", _TOOLKIT_JOIN),
    _EffColumn("backend_used", "Backend", _TOOLKIT_JOIN),
    # [Q31]: "Queue, this job (s) and Queue, sim total (s) are too similar. Drop the latter."
    # The state this arrived in was INVERTED -- `queue_seconds_this_job` was computed and
    # populated but carried no column, while the sim-total column the user named for dropping
    # was the one displayed. Keeping the per-job figure is also the truthful one at THIS grain:
    # one row is one JOB, so the queue this job waited is a property of the row, while the
    # simulation's accumulated total is a property of a set the row does not span.
    _EffColumn("queue_seconds_this_job", "Queue, this job (s)", _QUEUE_JOIN),
)


def _assert_reduction_names_unique(columns: tuple[_EffColumn, ...]) -> None:
    """Two reductions may not share a caption term. Import-time, over the set.

    A REQUIRED `name` closes the empty case but not the COLLISION case, and a collision is the
    identical defect wearing a different field: two distinct operations rendered under one
    heading, discovered by reading a shipped report rather than by running the code. That is
    precisely how the `joined` collapse surfaced -- three operations, one `<dt>`, noticed in a
    screenshot.

    Keyed on OBJECT IDENTITY, not equality: two separately-declared reductions that happen to
    be field-identical are still two declarations and still a collision. `_Reduction` is
    frozen, so `id()` is stable for the lifetime of the module.
    """
    by_name: dict[str, int] = {}
    for col in columns:
        red = col.reduction
        prior = by_name.setdefault(red.name, id(red))
        if prior != id(red):
            raise ValueError(
                f"Two distinct reductions both declare the caption term {red.name!r}. "
                f"One term names one operation: the caption renders each term once, so a "
                f"shared term silently merges two rules under one heading and the reader "
                f"cannot tell which column got which. Give each its own term."
            )


_assert_reduction_names_unique(_EFF_COLUMNS)


def _eff_label(key: str) -> str:
    """A live column's reader-facing LABEL, read from its declaration.

    Prose that names a column must not hand-author its label: that makes a second copy of a
    string the `_EffColumn` table already owns, and the two drift silently at the next rename.
    They already did -- the disclosure note below called this table's CPU column "CPU
    efficiency" after it was renamed to "Billed CPU used", so the note named a column the page
    no longer had.

    Fails LOUD on an unknown key, at import, because the alternative is a note rendering
    `<em></em>` -- a silently empty reference is the failure this accessor exists to prevent,
    and returning "" would reintroduce it in a new costume.

    Deliberately NOT used for every emphasised name in that note. Two of them -- `CPU model`
    and `Queue time` -- name a MEASUREMENT rather than a column, and have no declaration to
    read. `Queue time` is the clearest case: the live column's label is `Queue, this job (s)`,
    which does not fit the sentence it appears in, and that misfit is the diagnostic. Derive
    where the prose names a COLUMN; write plainly where it names a CONCEPT.
    """
    for col in _EFF_COLUMNS:
        if col.key == key:
            return col.label
    raise KeyError(
        f"No _EffColumn declares key {key!r}, so no label can be derived for prose that "
        f"names it. Either the column was removed and the prose referencing it is now stale, "
        f"or the key is misspelled; both are defects this lookup refuses to paper over."
    )


#: The sort/filter/hide affordance sentence, authored ONCE and rendered by every table that
#: offers those controls. It was previously written out at two call sites and had already
#: diverged -- one carried the "criteria across columns are combined" clause and the other
#: did not, so a reader met two different descriptions of one behaviour.
_TABLE_INTERACTION_NOTE = (
    "Click a column heading to sort; type in a column's own box to filter on that column "
    "(criteria across columns are combined); use the panel at left to hide columns."
)


def _reduction_caption() -> str:
    """The caption. Renders each DISTINCT `rule` exactly once, from the same declaration the
    headers and tooltips render -- so the page states each reduction in one place, not three."""
    # Every DISTINCT reduction a displayed column uses, with no filter on how the value was
    # obtained. The prior filter (`apply is not None or is _JOB_RECORD`) excluded the
    # join-class reductions, so a caveat carried on one of them reached the reader only on
    # hover -- which is how a scope limit becomes a true-looking number whose scope is not
    # stated. A rule worth authoring is worth rendering.
    seen: list[_Reduction] = []
    for col in _EFF_COLUMNS:
        if col.reduction not in seen:
            seen.append(col.reduction)
    # One BLOCK per reduction -- subheader, then why, then the method. The method line is
    # DERIVED from the callable rather than authored, so a reducer that is edited or swapped
    # changes this page. That is deterministic linkage: there is no hand-written sentence here
    # that can silently fall out of date, because there is no hand-written sentence here at all.
    blocks: list[str] = []
    for r in seen:
        if r.apply is None:
            method = (
                "No reduction is applied. The value is carried from its source as-is, so "
                "there is no function to link."
            )
        else:
            doc = (r.apply.__doc__ or "").strip().splitlines()
            summary = doc[0].strip() if doc else "(the function carries no docstring)"
            method = f"{r.apply.__name__}() — {summary}"
        blocks.append(
            f"<h5>{_esc(r.name)}</h5>"
            f"<p>{_esc(r.why)}</p>"
            f"<p><strong>Method.</strong> <code>{_esc(method)}</code></p>"
            f"<p>{_esc(r.rule)}</p>"
        )
    return (
        "<h4>How each column was reduced</h4>"
        "<p class='note'>Every column header carries its reduction's symbol, and hovering "
        "a header or a value repeats nothing — both read this same declaration.</p>"
        + "".join(blocks)
    )


#: Columns the user asked for that SLURM accounting does not capture. Rendered as an
#: explicit disclosure rather than omitted: an absent measurement that is silently left
#: out reads as "not applicable", and an empty column reads as zero. Neither is true.
_EFF_UNCAPTURED_NOTE = (
    "<p class='note'><strong>Not shown, and why.</strong> "
    "<em>CPU model</em> is not recorded by the toolkit today; it is "
    "recoverable on the cluster from <code>sacct -o NodeList</code> plus "
    "<code>scontrol show node</code> while the accounting database still holds the job. "
    "<em>Queue time</em> is captured only for runs submitted one job per simulation "
    "(<code>multi_sim_run_method: batch_job</code>). Under "
    "<code>1_job_many_srun_tasks</code> the whole ensemble shares a single allocation, so "
    "there is no per-simulation queue to measure and the column is left blank rather than "
    "filled with the allocation's own wait — a repeated number there would be a per-sim "
    "figure that no simulation actually experienced. A blank cell here means not "
    "measured; it never means the job did not wait. The column reports the queue THIS JOB "
    "waited, which is a property of the row; the simulation's accumulated queue across all "
    "of its allocations is a property of a set one row does not span, and is not shown here. "
    f"<em>{_eff_label('cpu_eff_pct').removesuffix(' (%)')}</em> is shown as not-measured "
    "wherever SLURM reported no CPU time "
    "for the job step; a zero there would claim the job used no processor, which is not "
    "what an absent measurement means.</p>"
    "<p class='note'><strong>Why most rows carry no purpose, sub-analysis or compute "
    "configuration.</strong> The toolkit columns in this table are joined in from "
    "<code>_status/*.flag.json</code>, which records ONE SLURM job id per rule — the most "
    "recent one. A simulation that resumed across several allocations, and any rule that "
    "re-ran in a later submission, therefore matches on its LAST allocation only; every "
    "earlier allocation appears here as a measured SLURM job with no toolkit label. On this "
    "report that is the majority of rows. An em-dash in these columns means the row could "
    "not be matched to a toolkit record — never that the job ran without a partition, "
    "without MPI ranks, or outside a sub-analysis. Every one of those values existed; this "
    f"table cannot currently reach them, and the <em>{_eff_label('record')}</em> column says "
    "which rows those "
    "are. This is a MEASURED CEILING rather than a capture failure: the join already matches "
    "87 of the 88 job ids the sidecars still retain, or 99% of its available keys. The "
    "shortfall is structural — this table is a cumulative history across every workflow "
    "submission, while <code>_status/*.flag.json</code> is a last-wins SNAPSHOT, so most rows "
    "predate the records that survive. The values are not recoverable elsewhere: "
    "<code>sacct</code> Comment is empty on this cluster, AdminComment carries node telemetry "
    "rather than rule identity, and the engine's log tree retains 5 of 771 job ids. Re-running "
    "does not recover them either — a fresh run overwrites the same per-rule slots.</p>"
)


def _parse_efficiency_csvs(csv_texts: list[tuple[str, str]]) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Merge every efficiency CSV into `{JobID: row}`; return (union header, rows).

    ``csv_texts`` is [(label, text)] in OLDEST-FIRST order, so a later file wins a
    duplicate ``JobID``. That ordering IS the idempotency guarantee, and it is
    cheaper than it looks: SLURM never reissues a job id within a cluster's id epoch,
    so a re-run contributes entirely NEW rows and cannot rewrite the rows of jobs that
    did not re-run. The user's requirement -- only genuinely-refreshed rows change --
    therefore holds by construction rather than by comparison.

    Parsed from in-memory strings (the caller already read and declared the files),
    so this adds no file-open audit surface.
    """
    merged: dict[str, dict[str, str]] = {}
    header: list[str] = []
    for _label, text in csv_texts:
        rows = [r for r in csv.reader(io.StringIO(text)) if r]
        if len(rows) < 2:
            continue
        file_header, body = rows[0], rows[1:]
        for column in file_header:
            if column and column not in header:
                header.append(column)
        for raw in body:
            record = dict(zip(file_header, raw, strict=False))
            job_id = (record.get("JobID") or "").strip()
            if not job_id:
                continue
            merged[job_id] = record
    return header, merged


def _enrich_efficiency_rows(
    merged: dict[str, dict[str, str]],
    purpose_map: dict[str, dict[str, str]],
    scenario_map: dict[tuple[str, str, str], dict[str, str]],
    gpu_hardware_for_partition,
) -> list[dict[str, str]]:
    """Left-join the merged CSV rows onto the toolkit's own per-job records.

    The join key is ``MainJobID`` (the parent job) against
    ``_status/*.flag.json``'s ``slurm_job_id``, which `status_flags.write_status_flag`
    takes from ``SLURM_JOB_ID`` -- the same number. A row with no match keeps every
    measured column and simply carries no purpose; it is never dropped, because the
    SLURM record is the authority on what ran and the toolkit record is the
    authority on why.
    """
    out: list[dict[str, str]] = []
    for job_id, record in merged.items():
        main_job_id = (record.get("MainJobID") or job_id.split(".")[0]).strip()
        enriched: dict[str, str] = dict(record)
        meta = purpose_map.get(main_job_id, {})
        # Item 25. Derived purely from the join above -- no new source, no new read, and
        # grain-independent (the key is main_job_id either way). `bool(meta)` rather than
        # `bool(purpose)`: a purpose recovered from the plugin's RuleName column is not a
        # surviving toolkit record, and on this cluster RuleName is absent anyway.
        enriched["record"] = "this run" if meta else "historical"
        # A rule name the plugin recovered from SLURM --comment beats nothing, but the
        # toolkit's own sidecar beats both. `RuleName` is present only on clusters that
        # store job comments; on UVA Rivanna it is absent, which is why JobName reads
        # `python` for every row and why the sidecar join is the load-bearing path.
        enriched["purpose"] = meta.get("purpose") or record.get("RuleName") or ""
        for key in ("sa_id", "event_id", "model_type"):
            enriched[key] = meta.get(key, "")
        scen = scenario_map.get((enriched["sa_id"], enriched["event_id"], enriched["model_type"]), {})
        # Resume labelling. A resumed simulation occupies several rows here -- they are the
        # srun STEPS of its one allocation -- and until now nothing on the page said which
        # row was which attempt, so filtering to a simulation gave an unordered set of
        # indistinguishable rows. The attempt integer is joined from the same already-read,
        # already-declared scenario_status.csv (no new read surface), keyed on this row's own
        # full `JobID`. Attempt 0 is the initial run and is labelled as such rather than as
        # "resume 0", which would imply a resume that did not happen. A row with no recorded
        # attempt keeps its bare purpose -- silence is correct where the record is absent.
        _att_raw = scen.get("attempt_by_jobstep", "")
        if _att_raw and enriched.get("purpose"):
            try:
                _att_map = json.loads(_att_raw)
            except (ValueError, TypeError):
                _att_map = {}
            if isinstance(_att_map, dict):
                _att = _att_map.get(job_id)
                if _att is not None:
                    _n = int(_att)
                    # Resolved ONCE, here, from the ledger. Carried on the row so the
                    # per-attempt roster renders the same resolution rather than deriving
                    # its own from the step suffix -- which would fabricate a number for
                    # any step the ledger does not record.
                    enriched["_attempt_n"] = _n
                    enriched["purpose"] += f" ({_attempt_label(_n)})"
        # Join-rate marker (develop `34b2624`). A total join FAILURE and legitimately
        # uncaptured data both render as an em-dash, so the rate is disclosed below and a
        # zero raises a banner. Carried onto the aggregated JOB row so the disclosure's
        # denominator matches the grain the table actually displays ([Q145]).
        enriched["_scen_joined"] = "1" if scen else ""
        # Three spellings, not one: `hpc.partition` is a sensitivity-CSV overlay column a
        # non-sensitivity export never emits, and older sensitivity trees carry
        # `hpc_ensemble_partition`. Falls back to in-memory config -- no new file read, so
        # the Gotcha-53 renderer-IO audit is unaffected.
        partition = _first_nonempty(scen, _PARTITION_COLUMN_SPELLINGS)
        enriched["partition"] = partition
        enriched["gpu_hardware"] = gpu_hardware_for_partition(partition) if partition else ""
        for src, dst in (
            ("n_gpus", "n_gpus"),
            ("n_mpi_procs", "n_mpi_procs"),
            ("n_omp_threads", "n_omp_threads"),
            ("n_nodes", "n_nodes_cfg"),
            ("run_mode", "run_mode"),
            ("backend_used", "backend_used"),
        ):
            enriched[dst] = scen.get(src, "")
        # O21 queue time. Sourced from scenario_status.csv -- the SAME already-declared
        # file this join already reads -- so no new render-time read surface is added and
        # the off-cluster bundle re-render works unchanged (the CSV is bundle-carried by
        # _copy_supporting_files; the _walltime ledgers are NOT, which is why they are not
        # read here). An absent value stays "" and renders as an em-dash, never as 0 --
        # a 0 in this column would assert the job did not wait.
        _q_total_raw = scen.get("queue_seconds_total", "")
        enriched["queue_seconds_total"] = _q_total_raw
        enriched["queue_seconds_coverage"] = scen.get("queue_seconds_coverage", "")
        enriched["queue_seconds_this_job"] = ""
        _q_map_raw = scen.get("queue_seconds_by_jobid", "")
        if _q_map_raw:
            try:
                _q_map = json.loads(_q_map_raw)
            except (ValueError, TypeError):
                _q_map = {}
            if isinstance(_q_map, dict) and main_job_id in _q_map:
                enriched["queue_seconds_this_job"] = _q_map[main_job_id]
        out.append(enriched)

    # Chronological by job id where it parses -- job ids increase monotonically on a
    # cluster, so this puts setup first and the most recent render last, which is the
    # order a reader scanning an experiment's history expects.
    def _sort_key(row: dict[str, str]):
        raw = (row.get("MainJobID") or row.get("JobID") or "").split(".")[0]
        return (0, int(raw)) if raw.isdigit() else (1, 0)

    return sorted(out, key=_sort_key)


#: CSV columns deliberately not displayed. Each is a RESTATEMENT of a neighbour the table
#: does show (raw `MaxRSS`/`ReqMem` strings vs their parsed `_MB` forms; `TotalCPU` vs the
#: CPU-efficiency percentage derived from it), or plumbing (`MainJobID` is the join key,
#: shown as `Job ID`; `JobName` is the executable name and is NOT always `python` -- the
#: campaign census recorded at `_is_attempt` returns python 4771 / apptainer 216 /
#: triton.exe 103 at step `.0` alone, which is exactly why that classifier keys ON it; it is
#: undisplayed because the reader-facing purpose is joined in, not because it is constant;
#: `RuleName` appears only on clusters that store job comments and is superseded by the
#: toolkit's own purpose).
#: Listing them here is what lets an UNRECOGNISED column be reported instead of dropped.
_EFF_KNOWN_UNDISPLAYED: frozenset[str] = frozenset(
    {
        "",  # the plugin's unnamed DataFrame index column
        "JobName",  # the executable, NOT always `python`; classified on, purpose joined in
        "TotalCPU",  # shown as the derived CPU-efficiency percentage
        "Elapsed_sec",  # parsed restatement of Elapsed
        "TotalCPU_sec",  # parsed restatement of TotalCPU
        "MaxRSS",  # raw string form of MaxRSS_MB
        "MaxRSS_MB",  # the plugin's PER-STEP parse; superseded by the max-over-steps reduction
        "ReqMem",  # raw string form of RequestedMem_MB
        # The plugin's own precomputed percentages, both superseded by this table's reductions.
        # `CPU Efficiency (%)` is the column measured at exactly 0.0 on 664 of 664 delivered
        # rows: it divides by a numerator the plugin never carries, because `TotalCPU` lives on
        # the allocation and batch rows it discards. Recomputing it is the point of Stage A.
        "CPU Efficiency (%)",
        "Memory Usage (%)",
        "MainJobID",  # the join key; the step id is shown as Job ID
        "RuleName",  # present only where SLURM stores job comments; superseded
        "Comment",
        "State",
    }
)

#: Per-step fields that exist ONLY in the Stage-A recovery CSV, never in the Snakemake
#: plugin's efficiency CSV, and are therefore joined onto the plugin's step rows in
#: `_aggregate_jobs`. Declared once because two consumers depend on the same join and a
#: second inline list would be a second place this knowledge lives.
#:
#: `TRESUsageInTot` feeds `_reduce_cpu_sum` -- a solver step reports `TotalCPU=00:00:00`
#: while its `cpu=` key carries the real time (`slurm_job_recovery.py` states this at its
#: field-set declaration), so the reduction is wrong without it.
#:
#: `State` feeds the per-attempt roster ONLY (`_attempt_details_html`) and is read by no
#: reducer -- verified by census: `State` appears exactly twice in this module, here and at
#: that one display site. It is what makes the CANCELLED/COMPLETED breakdown legible, which
#: `_build_efficiency_rows` names as this campaign's subject: the solver steps ARE the resume
#: attempts. Without the join the roster renders every attempt as "state not recorded", which
#: is a disclosure that discloses nothing -- the failure `[Q153]`'s trackability condition
#: exists to prevent.
_RECOVERED_ONLY_STEP_FIELDS: tuple[str, ...] = ("TRESUsageInTot", "State")


def _load_job_recovery(analysis_dir: Path) -> tuple[dict[str, dict[str, dict[str, str]]], Path | None]:
    """Read `slurm_job_recovery`'s Stage-A artifact; return the map and the file actually read.

    Graceful-absent by contract: an analysis whose back-fill has not run yields `({}, None)`
    and the table renders with em-dashes in the job-record columns rather than failing. The
    path is returned so `render()` can DECLARE it (ADR-6 Gate-A / Gotcha 53).
    """
    from hhemt.slurm_job_recovery import RECOVERY_FILENAME

    path = analysis_dir / "logs" / "slurm_efficiency_report" / RECOVERY_FILENAME
    if not path.is_file():
        return ({}, None)
    try:
        text = path.read_text()
    except OSError:
        return ({}, None)
    out: dict[str, dict[str, dict[str, str]]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        main = (row.get("MainJobID") or "").strip()
        kind = (row.get("StepKind") or "").strip()
        if main and kind:
            out.setdefault(main, {})[kind] = row
    return (out, path)


def _aggregate_jobs(
    merged: dict[str, dict[str, str]],
    recovery: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    """Collapse step rows to ONE row per job ([Q145]), applying each column's declared reducer.

    `merged` is the plugin's step-grained population keyed on step `JobID`; `recovery` is the
    Stage-A `slurm_job_recovery` output, `{main_job_id: {"job": row, "batch": row}}`, supplying
    the two row classes the plugin's parsing drops. The `.batch` row is folded into the step
    list because `seff` reduces over it and SLURM records CPU time nowhere else.

    Each row carries `_provenance`, a per-column map of where its value came from. That is the
    per-value half of [Q153]'s trackability, and it is PRODUCED BY the reducer rather than
    written beside it, so it cannot describe a reduction that did not happen.
    """
    by_job: dict[str, list[dict[str, str]]] = {}
    for record in merged.values():
        main = (record.get("MainJobID") or record.get("JobID", "").split(".", 1)[0]).strip()
        if main:
            by_job.setdefault(main, []).append(record)

    out: list[dict[str, Any]] = []
    for main_job_id, steps in by_job.items():
        recovered = recovery.get(main_job_id, {})
        job_row = dict(recovered.get("job", {}))
        batch_row = recovered.get("batch")
        # The plugin's step rows carry MaxRSS but NEITHER of the fields in
        # `_RECOVERED_ONLY_STEP_FIELDS`, so a solver step's CPU time and its terminal state
        # are both absent from them. Measured on this campaign's own plugin output: the
        # emitted columns are JobID/JobName/Elapsed/TotalCPU/NNodes/NCPUS/MaxRSS/ReqMem plus
        # the parsed `_sec`/`_MB` restatements, `MainJobID` and two precomputed percentages
        # -- no `TRESUsageInTot` and no `State` at any step kind. Where the recovery CSV holds
        # the same step, those fields are copied ONTO the plugin row rather than added as a
        # second row: `all_steps` keeps its membership, so the max-shaped reducer is untouched
        # and the sum-shaped one cannot count a step twice.
        for _s in steps:
            _suffix = _s.get("JobID", "").split(".", 1)[1] if "." in _s.get("JobID", "") else ""
            _rec = recovered.get(_suffix)
            if _rec:
                for _field in _RECOVERED_ONLY_STEP_FIELDS:
                    if not _s.get(_field):
                        _s[_field] = _rec.get(_field, "")
        # seff reduces over the batch step; excluding it is what zeroes the CPU column.
        all_steps = [*steps, batch_row] if batch_row else list(steps)

        row: dict[str, Any] = {"JobID": main_job_id}
        prov: dict[str, str] = {}
        job_row.setdefault("JobID", main_job_id)
        # RequestedMem_MB is inherited onto the steps by the plugin, so it survives there.
        job_row.setdefault("RequestedMem_MB", steps[0].get("RequestedMem_MB", "") if steps else "")

        cpu_value, cpu_prov = _reduce_cpu_sum(all_steps, job_row)
        row["cpu_seconds"], prov["cpu_seconds"] = cpu_value, cpu_prov
        rss_value, rss_prov = _reduce_rss_max(all_steps, job_row)
        row["max_rss_mb"], prov["max_rss_mb"] = rss_value, rss_prov

        for field in ("Elapsed", "NNodes", "NCPUS", "RequestedMem_MB"):
            value, why = _reduce_job_field(field)(all_steps, job_row)
            row[field], prov[field] = value, why

        # Queue-wait FALLBACK, staged here because `job_row` is in scope only inside this
        # function; the caller applies it after the toolkit join so the toolkit value wins.
        # Stored under a private key so it can never be mistaken for the column's own value
        # by anything that iterates `_EFF_COLUMNS`.
        # `00:00:00` is KEPT, and excluding it was a real defect measured on the artifact:
        # exactly 1 of `synth_cc_clean_triton`'s 672 job rows carries it, and it rendered an
        # em-dash where the truth is a queue wait under one second. SLURM reports an
        # unavailable field as empty or `Unknown`, never as a zero DURATION, so `00:00:00`
        # here is a MEASUREMENT rather than an absence -- and `[Q130]` is explicit that an
        # examined-zero and a not-measured must not render alike. The sibling comment about
        # never rendering 0 governs an ABSENT toolkit value, which is a different case.
        _planned_raw = (job_row.get("Planned") or "").strip()
        if _planned_raw:
            row["_planned_seconds"] = f"{_slurm_seconds(_planned_raw):.0f}"

        # CPU efficiency is `seff`'s own headline number, and [Q144] names it by name. It is
        # DERIVED from values already reduced above -- the summed CPU-time numerator and the
        # job record's own CPU count and wall time -- so it inherits their provenance instead
        # of asserting a fourth reduction. The numerator stays visible in that provenance
        # string, which is what lets the column be dropped without losing the CPU-seconds
        # figure: the number a reader would check against `seff` is still on the page.
        # An absent numerator yields an ABSENT cell, never 0.0 -- a zero here would assert the
        # job used no processor, and [Q130] rules that an examined-zero and a not-measured
        # must not render alike.
        elapsed_seconds = _slurm_seconds(row.get("Elapsed", ""))
        try:
            n_cpus = float(row.get("NCPUS") or 0)
        except (TypeError, ValueError):
            n_cpus = 0.0
        cpu_denominator = n_cpus * elapsed_seconds
        if cpu_value and cpu_denominator:
            row["cpu_eff_pct"] = f"{float(cpu_value) / cpu_denominator * 100:.2f}"
            prov["cpu_eff_pct"] = (
                f"{cpu_value} CPU-seconds ({cpu_prov}), against {n_cpus:g} CPU(s) x "
                f"{elapsed_seconds:g}s elapsed read from the job's own accounting record"
            )
        else:
            row["cpu_eff_pct"] = ""
            prov["cpu_eff_pct"] = (
                cpu_prov
                if not cpu_value
                else "the job's CPU count or elapsed time was not recovered, so the denominator is unknown"
            )

        # Memory-used % is derived from the two columns beside it, so it inherits their
        # provenance rather than asserting one of its own ([Q144] keeps it on truthfulness).
        try:
            req = float(row.get("RequestedMem_MB") or 0)
            row["mem_used_pct"] = f"{float(rss_value) / req * 100:.1f}" if (rss_value and req) else ""
        except (TypeError, ValueError):
            row["mem_used_pct"] = ""
        prov["mem_used_pct"] = f"{rss_prov}, against the requested memory beside it" if rss_value else ""

        # Per-attempt disclosure: the solver steps ARE the resume attempts, and their
        # CANCELLED/COMPLETED breakdown is what [Q143] calls this campaign's subject.
        def _is_attempt(step: dict[str, str]) -> bool:
            # Solver steps only, classified by NAME. Step INDEX is not step IDENTITY: a
            # campaign-window census of `JobName` at `.0` returns python 4771, apptainer
            # 216, triton.exe 103 -- so `.0` is a real solver step on 319 of 5198 jobs,
            # 283 of which have NO other numeric step and rendered an em-dash for a job
            # that ran. `_NON_SOLVER_STEP_NAMES` is the toolkit's single declaration of
            # which names are never the solver; imported locally to keep the renderer
            # free of a module-level dependency on a runner.
            from hhemt.run_simulation_runner import _NON_SOLVER_STEP_NAMES

            suffix = _step_suffix(step.get("JobID", ""))
            if not suffix.isdigit():
                return False
            name = (step.get("JobName", "") or "").strip().lower()
            # A NAMED step is classified by name -- that is what admits a solver sitting at
            # `.0` and excludes a wrapper wherever it sits. An UNNAMED step cannot be
            # classified that way at all, and reading "unknown" as "not the solver" silently
            # DROPS it from the roster, which is the disclosure-that-discloses-nothing this
            # roster exists to prevent. Absence is not a measurement, so fall back to the
            # index rule -- the best available discriminator when no name is present.
            # `_parse_step_ids` keeps the plain `name and ...` conjunct because it reads
            # `sacct -o JobID,JobName`, where the field is present by construction.
            if not name:
                return suffix != "0"
            return name not in _NON_SOLVER_STEP_NAMES

        attempts = sorted(
            (s for s in steps if _is_attempt(s)),
            key=lambda s: int(_step_suffix(s.get("JobID", ""))),
        )
        row["_attempts"] = attempts
        row["attempts"] = str(len(attempts)) if attempts else ""
        row["_steps"] = all_steps
        row["_provenance"] = prov
        out.append(row)

    out.sort(key=lambda r: int(r["JobID"]) if str(r["JobID"]).isdigit() else 0)
    return out


def _attempt_details_html(attempts: list[dict[str, str]], attempt_by_step: dict[str, int] | None = None) -> str:
    """`<details>` roster of a job's resume attempts, in a pre-escaped cell.

    Job grain folds N attempts into one row; without this the CANCELLED/COMPLETED breakdown
    is lost, which is the cost [Q153] exists to avoid paying. Uses the shim's pre-escaped
    cell-HTML property -- no cell formatter, no Tabulator, no new mechanism.

    Each entry carries TWO identifiers and they come from different places, which is the whole
    point. The STEP index is a fact about the artifact and is read off `JobID`. The ATTEMPT
    number is a fact about the run's history and is read only from the LEDGER
    (`attempt_by_jobstep`, resolved once during enrichment). Deriving the second from the first
    fabricates a value for every step the ledger does not record -- which aggregation newly
    makes possible, because folding N steps into one row puts an ordered list in front of the
    renderer and invites numbering it. A step the ledger does not record keeps its true step
    index and is explicitly unnumbered: it ran, and which attempt it was is not known here.
    """
    if not attempts:
        return ""
    by_step = attempt_by_step or {}
    items = "".join(
        "<li>step {step} ({n}): {state}, {el}, {rss}</li>".format(
            step=_esc(_step_suffix(a.get("JobID", ""))),
            n=_esc(
                _attempt_label(by_step[a.get("JobID", "")]) if a.get("JobID", "") in by_step else "attempt not recorded"
            ),
            state=_esc(a.get("State", "") or "state not recorded"),
            el=_esc(a.get("Elapsed", "") or "elapsed not recorded"),
            rss=_esc(a.get("MaxRSS", "") or "memory not recorded"),
        )
        for a in attempts
    )
    return (
        f"<details><summary>{len(attempts)}</summary>"
        f"<ul style='margin:4px 0 0 14px;padding:0;font-size:11px'>{items}</ul></details>"
    )


def _undisplayed_csv_columns(header: list[str]) -> list[str]:
    """CSV columns the curated table neither displays nor knowingly suppresses.

    The curated column set is what makes this table readable, and widening it to whatever
    a CSV happens to carry would make the width unpredictable. But dropping a column
    SILENTLY is the part that would be a defect: a future plugin version that adds a
    genuinely new measurement would lose it with nothing on the page to say so. So the
    drop is deliberate and the surprise is disclosed -- the same shape as the
    not-captured note for GPU utilisation.
    """
    displayed = {col.key for col in _EFF_COLUMNS}
    return sorted(c for c in header if c not in displayed and c not in _EFF_KNOWN_UNDISPLAYED)


def _undisplayed_columns_note(columns: list[str]) -> str:
    return (
        "<p class='note'><strong>Unrecognised column(s) in the efficiency report:</strong> "
        + ", ".join(_code(c) for c in columns)
        + ". These were present in the source CSV but are not part of this table's curated "
        "column set, so their values are not shown. That usually means the SLURM executor "
        "plugin started reporting something new — worth adding here if it is useful.</p>"
    )


def _build_slurm_efficiency_html(
    csv_texts: list[tuple[str, str]],
    purpose_map: dict[str, dict[str, str]],
    scenario_map: dict[tuple[str, str, str], dict[str, str]],
    gpu_hardware_for_partition,
    recovery: dict[str, dict[str, dict[str, str]]] | None = None,
) -> str:
    """Render the UNION of every efficiency report at JOB grain ([Q145]), reduced per [Q153]."""
    recovery = recovery or {}
    heading = _heading("SLURM Efficiency")
    if not csv_texts:
        return heading + "\n" + _banner("The SLURM resource-efficiency report is present but empty.")

    _header, merged = _parse_efficiency_csvs(csv_texts)
    if not merged:
        return heading + "\n" + _banner("The SLURM resource-efficiency reports contain no job rows.")

    step_rows = _enrich_efficiency_rows(merged, purpose_map, scenario_map, gpu_hardware_for_partition)
    # Carry the toolkit-joined fields onto the job they belong to before collapsing, so the
    # join survives aggregation. Keyed on MainJobID, which the join already used.
    joined: dict[str, dict[str, str]] = {}
    for s in step_rows:
        main = (s.get("MainJobID") or s.get("JobID", "").split(".", 1)[0]).strip()
        if main and main not in joined:
            joined[main] = s

    rows = _aggregate_jobs(merged, recovery)
    for row in rows:
        src = joined.get(str(row["JobID"]), {})
        row.setdefault("_scen_joined", src.get("_scen_joined", ""))
        for col in _EFF_COLUMNS:
            if col.reduction is _TOOLKIT_JOIN and col.key not in ("attempts",):
                row.setdefault(col.key, src.get(col.key, ""))
        # `_QUEUE_JOIN` is deliberately NOT `_TOOLKIT_JOIN`, so the loop above skips it and it
        # is resolved here against BOTH its sources. Toolkit map first -- it is per-job and
        # exact -- then SLURM's `Planned`. Each arm writes its own provenance, so the cell
        # tooltip names the source that actually produced the number rather than restating the
        # header's general rule ([Q153] trackability; the same shape `_reduce_cpu_sum` uses).
        _q_toolkit = (src.get("queue_seconds_this_job") or "").strip()
        _q_prov = row.setdefault("_provenance", {})
        if _q_toolkit:
            row["queue_seconds_this_job"] = _q_toolkit
            _q_prov["queue_seconds_this_job"] = "hhemt's own per-job queue record for this job id"
        elif row.get("_planned_seconds"):
            row["queue_seconds_this_job"] = row["_planned_seconds"]
            _q_prov["queue_seconds_this_job"] = (
                "SLURM `Planned` on the job accounting record — hhemt's own queue map had no "
                "entry for this job (it derives from _walltime ledgers a bundle does not carry)"
            )
        else:
            row.setdefault("queue_seconds_this_job", "")

    undisplayed = _undisplayed_csv_columns(_header)
    n_recovered = sum(1 for r in rows if r.get("Elapsed"))
    # Scenario-join rate, counted over the JOB rows the table displays rather than over the
    # step rows the enrichment walked -- `[Q145]` fixes the grain at one row per job, so a
    # step-grained denominator would disclose a number no column on the page reports.
    joinable_rows = sum(1 for r in rows if r.get("model_type"))
    scen_joined = sum(1 for r in rows if r.get("_scen_joined"))
    # The ledger's attempt resolution, read ONCE during enrichment and carried here so the
    # roster renders that resolution instead of re-deriving one from step suffixes.
    attempt_by_step: dict[str, int] = {
        s.get("JobID", ""): s["_attempt_n"] for s in step_rows if "_attempt_n" in s
    }

    def _cell(row: dict[str, Any], col: _EffColumn) -> str:
        """One cell, carrying its own provenance as a native `title` tooltip.

        The tooltip TEXT is produced by the reducer (`_provenance`), never authored here --
        so a cell cannot claim a reduction the code did not perform.
        """
        if col.key == "attempts" and row.get("_attempts"):
            return _attempt_details_html(row["_attempts"], attempt_by_step)
        value = row.get(col.key, "")
        if not value:
            return "—"
        why = (row.get("_provenance") or {}).get(col.key, "")
        return f'<span title="{_esc(why)}">{_esc(value)}</span>' if why else _esc(value)

    grid = [[_cell(row, col) for col in _EFF_COLUMNS] for row in rows]
    summary = (
        f"<p class='note'>{_esc(len(rows))} job(s) across {_esc(len(csv_texts))} efficiency "
        f"report(s), one row per JOB with its steps aggregated. {_esc(n_recovered)} job(s) "
        f"carry a recovered accounting record; a job without one shows an em-dash in the "
        f"columns read from it. {_TABLE_INTERACTION_NOTE}</p>"
    )
    # Join-rate disclosure (develop `34b2624`). Kept because it separates two states the
    # em-dash alone renders identically: a JOIN FAILURE and legitimately uncaptured data.
    # develop's own first paragraph is NOT carried -- it is step-grained, which `[Q145]`
    # retires, and it restates the sort/filter sentence that `_TABLE_INTERACTION_NOTE`
    # single-sources ([Q153]).
    summary += (
        f"<p class='note'>Scenario-record join: {_esc(scen_joined)} of {_esc(joinable_rows)} "
        f"simulation row(s) matched a <code>scenario_status.csv</code> record. Every column "
        f"from <code>Partition</code> rightward is sourced from that record, so a zero here "
        f"means the whole block is blank for a JOIN failure rather than for missing data. "
        f"<code>Partition</code> and <code>Nodes (config)</code> fall back to this analysis's "
        f"own configuration when the record predates the per-row partition column.</p>"
    )
    if joinable_rows and not scen_joined:
        summary += _banner(
            "No simulation row matched a scenario_status.csv record, so every "
            "scenario-sourced column below is blank because the join failed — not "
            "because the values were not captured. This is a key-vocabulary mismatch "
            "between the CSV and the _status/*.flag.json sidecars; the sidecar records "
            "the scenario slug as its event id."
        )
    summary += _reduction_caption()
    table = _sortable_grid_table(
        [col.header for col in _EFF_COLUMNS],
        grid,
        # DISTINCT from the section heading's own anchor. `_heading` derives the <h3> id from
        # the title via `_anchor` ("SLURM Efficiency" -> "slurm-efficiency"), so a table id of
        # the same string put the attribute on TWO elements -- invalid HTML, and the nav's
        # data-jump="slurm-efficiency" then resolved to whichever the parser reached first.
        table_id="slurm-efficiency-table",
        column_panel=True,
        header_tooltips=tuple(col.reduction.rule for col in _EFF_COLUMNS),
    )
    parts = [heading, summary, table]
    if undisplayed:
        parts.append(_undisplayed_columns_note(undisplayed))
    parts.append(_EFF_UNCAPTURED_NOTE)
    return "\n".join(parts)


def _resolve_all_efficiency_csvs(eff_dir: Path) -> list[Path]:
    """Resolve EVERY SLURM efficiency-report CSV FILE under ``eff_dir``, oldest first.

    The snakemake-executor-plugin-slurm treats ``--slurm-efficiency-report-path``
    as a DIRECTORY and writes ``efficiency_report_{run_uuid}.csv`` inside it
    (efficiency_report.py::create_efficiency_report). The toolkit driver passes a
    ``.csv``-suffixed path, so on disk the glob match
    ``slurm_efficiency_report_{ts}.csv`` is itself a DIRECTORY that CONTAINS the
    real CSV -- a bare ``read_text()`` on it raises ``IsADirectoryError``. Return
    every actual FILE, descending into any directory-shaped match; return ``[]``
    when no CSV file is present (absent-banner fallback).

    A HISTORY, not a snapshot: one report is written per Snakemake invocation, so
    a report that is present in an older file and absent from the newest one is
    RETAINED HISTORY, never a gap. Anything that later prunes these directories
    for tidiness would silently re-break that, and the rendered table would once
    again show only the most recent invocation.

    Glob / is_file / is_dir / stat only (os.scandir + os.stat) -- no file is
    opened, so this adds no file-open audit surface (Gotcha 53).
    """
    if not eff_dir.is_dir():
        return []
    candidates: list[Path] = []
    for match in sorted(eff_dir.glob(_SLURM_EFF_GLOB)):
        if match.is_file():
            # Hypothetical future flat-file layout (driver cleanup / other plugin).
            candidates.append(match)
        elif match.is_dir():
            # Current plugin layout: descend to the inner efficiency_report_*.csv.
            candidates.extend(p for p in sorted(match.glob(_SLURM_EFF_INNER_GLOB)) if p.is_file())
    # Defensive: a plugin that wrote the inner file directly under eff_dir.
    candidates.extend(p for p in sorted(eff_dir.glob(_SLURM_EFF_INNER_GLOB)) if p.is_file())
    if not candidates:
        return []
    # UNION, not latest. Each Snakemake invocation writes its own report covering only
    # that invocation's jobs (plugin: `sacct --name={run_uuid}`), so `max(mtime)` showed
    # the newest INVOCATION rather than the newest DATA -- on a re-render that is two
    # render jobs and nothing else. Returned newest-last so the caller's dict-merge
    # resolves a duplicate JobID in favour of the more recent report. Inner run-uuid
    # filenames do not sort chronologically, hence mtime rather than name.
    return sorted(set(candidates), key=lambda p: p.stat().st_mtime)


def _read_scenario_status(analysis_dir: Path) -> tuple[dict[tuple[str, str, str], dict[str, str]], Path | None]:
    """`{(sa_id, event_id, model_type): row}` from `scenario_status.csv`; and the file read.

    The file is returned so the caller can DECLARE it (ADR-6 Gate-A). It is already
    carried in every bundle by `_copy_supporting_files`, so declaring it adds a
    manifest row and no payload bytes.

    THREE event spellings are registered per row, because the CSV and the
    `_status/*.flag.json` sidecar do not share one vocabulary:

    * the integer ``event_iloc`` this CSV records;
    * ``event_index.{iloc}``, which is the scenario slug ONLY when the analysis has a
      single weather indexer literally named ``event_index`` whose value equals the
      iloc -- true of the synthetic fixture and of no real analysis;
    * the basename of ``scenario_directory``, which IS the slug the sidecar carries.
      That equality is by CONSTRUCTION, not convention: `scenario.py` assigns
      ``self.event_id = self.sim_id_str`` and ``sim_folder = .../ self.sim_id_str``
      from one variable in adjacent lines, and `analysis.py` writes that path here.

    This is a key SET, not a guarantee. A sidecar vocabulary outside the set produces
    a MISS, and the miss is disclosed as the scenario-join rate in the rendered
    efficiency table rather than left to read as absent data.
    """
    path = analysis_dir / _SCENARIO_STATUS_FILENAME
    if not path.is_file():
        return ({}, None)
    try:
        text = path.read_text()
    except OSError:
        return ({}, None)
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for record in csv.DictReader(io.StringIO(text)):
        sa_id = (record.get("sa_id") or "").strip()
        model_type = (record.get("model_type") or "").strip()
        iloc = (record.get("event_iloc") or "").strip()
        slug = os.path.basename((record.get("scenario_directory") or "").rstrip("/\\"))
        for event_key in {iloc, f"event_index.{iloc}", slug}:
            if event_key:
                out[(sa_id, event_key, model_type)] = record
    return (out, path)


#: Column spellings the partition has been written under. `hpc.partition` is the
#: canonical sensitivity-overlay column; `analysis.hpc_ensemble_partition` is the
#: accepted legacy overlay spelling; `hpc_ensemble_partition` is the resolved
#: analysis-config field name, which is what older sensitivity exports carry and
#: what `_apply_config_fallbacks` writes for a non-sensitivity analysis.
_PARTITION_COLUMN_SPELLINGS: tuple[str, ...] = (
    "hpc.partition",
    "analysis.hpc_ensemble_partition",
    "hpc_ensemble_partition",
)


def _first_nonempty(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def _apply_config_fallbacks(scenario_map, analysis) -> None:
    """Fill partition / node count from IN-MEMORY config where the CSV carries neither.

    `hpc.partition` is a sensitivity CSV overlay column, so a non-sensitivity export
    has no partition column at all and `n_nodes` is written only under
    `analysis.in_slurm`. Without this, `Partition`, `GPU hardware` and
    `Nodes (config)` stay blank even on a perfect join, which is indistinguishable
    on the page from data that was never captured.

    In-memory config only -- no file is opened, so this adds no render-time read
    surface (Gotcha 53), exactly like `_gpu_hardware_for_partition`. The CSV always
    wins where it carries a value, so a sensitivity master's per-row partitions are
    never overwritten by the master default. Each row dict is shared across the
    several keys that address it, so it is mutated at most once.
    """
    cfg = getattr(analysis, "cfg_analysis", None)
    if cfg is None:
        return
    partition = str(getattr(cfg, "hpc_ensemble_partition", "") or "")
    n_nodes = str(getattr(cfg, "n_nodes", "") or "")
    seen: set[int] = set()
    for row in scenario_map.values():
        if id(row) in seen:
            continue
        seen.add(id(row))
        if partition and not _first_nonempty(row, _PARTITION_COLUMN_SPELLINGS):
            row["hpc_ensemble_partition"] = partition
        if n_nodes and not (row.get("n_nodes") or "").strip():
            row["n_nodes"] = n_nodes


# --- page shell --------------------------------------------------------------


_VALIDATION_REPORT_FILENAME = "validation_report.json"
_DATA_AVAILABILITY_CHECK = "Data availability"


def _build_data_availability_html(report_path: Path) -> str:
    """Project the persisted `Data availability` CheckResult onto the Metadata page.

    Reads the SAME `{analysis_dir}/validation_report.json` the Errors-and-Warnings
    renderer reads -- no second read-model. ADR-14 D1 rejected emitting a parallel
    `metadata_report.json` on the grounds that a second projection of the same record is a
    FAIR drift anti-pattern, and that reasoning applies unchanged here.

    Every dynamic value is `_esc`-escaped: the detail rows carry scenario ids and
    filesystem paths, which legitimately contain characters that would otherwise close a
    tag.
    """
    if not report_path.exists():
        return _absent_banner(
            "Data Availability",
            "Data-availability record not available -- validation_report.json was not "
            "found. It is written at consolidation; re-run consolidation to populate.",
        )
    try:
        payload = json.loads(report_path.read_text())
    except (OSError, ValueError):
        return _absent_banner(
            "Data Availability",
            "Data-availability record could not be read -- validation_report.json is present but did not parse.",
        )
    check = next(
        (c for c in payload.get("checks", []) if c.get("name") == _DATA_AVAILABILITY_CHECK),
        None,
    )
    if check is None:
        return _absent_banner(
            "Data Availability",
            "This analysis was consolidated by a toolkit build that predates the "
            "post-processing reclaim, so no data-availability record exists. Every "
            "per-scenario artifact the run produced is still on disk.",
        )
    rows = [
        ("Status", _esc("reclaim recorded and consistent" if check.get("passed") else "INCONSISTENT")),
        ("Record", _esc(str(check.get("summary", "")))),
    ]
    parts = [_heading("Data Availability"), _kv_table(rows)]
    parts.append(
        _banner(
            "Reclaimed artifact classes were removed deliberately, after this toolkit "
            "verified the corresponding summary outputs were present and openable on "
            "disk. An absent timeseries or raw output is a disclosed reclaim, not a "
            "failed run -- but reprocessing those scenarios is not possible without "
            "re-running the simulations."
        )
    )
    detail_rows = [
        [_esc(str(d.get("sa_id", ""))), _esc(str(d.get("scenario", ""))), _esc(str(d.get("detail", "")))]
        for d in check.get("details", [])
    ]
    if detail_rows:
        parts.append(_grid_table(["Sub-analysis", "Scenario", "Detail"], detail_rows))
    return "\n".join(parts)


#: The page's sections, in render order. Module-level so the jump-nav and the test's
#: anchor coverage read ONE declaration -- a literal in each drifted silently when the
#: Data Availability section landed and only the nav learned about it.
_SECTION_TITLES = ("Provenance", "Data Availability", "Reproduction Guide", "SLURM Efficiency")


#: Iter-12 item {23}. The jump-nav links carry `data-jump` as well as `href`, and the
#: shim below cancels the fragment navigation and scrolls directly.
#:
#: This page renders inside an iframe (see `_wrap_html_doc`). A fragment href cannot be
#: resolved in place against that iframe's base document, so the click is handled as a
#: NAVIGATION rather than an in-page scroll -- which is what re-enters the report shell's
#: render path and, per the user's iteration-11 report, appends a heading per click with
#: increasing indentation instead of scrolling. `preventDefault()` means the fragment is
#: never resolved and the navigation never happens.
#:
#: `scrollIntoView({block: "start"})` is also the better scroll on its own terms: the
#: section headings sit above `div.table-scroll` containers with sticky `thead`, and a
#: native fragment jump does not account for a nested scroll container.
#:
#: The `href` is RETAINED so the nav degrades to today's behaviour with JavaScript
#: unavailable -- the state an archived / emailed copy of this page must survive in.
_JUMP_NAV_JS = """
(function () {
  document.querySelectorAll("nav.jump-nav a[data-jump]").forEach(function (a) {
    a.addEventListener("click", function (ev) {
      var target = document.getElementById(a.getAttribute("data-jump"));
      if (!target) { return; }
      ev.preventDefault();
      target.scrollIntoView({ block: "start" });
    });
  });
})();
"""


def _jump_nav() -> str:
    links = " &middot; ".join(
        f'<a href="#{_anchor(t)}" data-jump="{_anchor(t)}">{_esc(t)}</a>'
        for t in _SECTION_TITLES
    )
    return f'<nav class="jump-nav">{links}</nav>'


#: Moves each fragment's markup from its <template> into the mount point the guide
#: emitted. Runs BEFORE the shared filter blob and the per-table scripts, so the
#: container div each `new Tabulator("#reprex-{bucket}")` targets is already in the DOM.
_TRF_MOUNT_JS = """
(function () {
  var tpls = document.querySelectorAll("template[data-trf-mount]");
  for (var i = 0; i < tpls.length; i++) {
    var t = tpls[i];
    var host = document.getElementById(t.getAttribute("data-trf-mount") + "-mount");
    if (host) { host.appendChild(t.content.cloneNode(true)); }
  }
})();
"""


def _wrap_html_doc(
    analysis_id: str,
    inline_css: str,
    *sections: str,
    fragments: list[TableFragment] | None = None,
) -> str:
    """<!DOCTYPE> + inline <style> + <h2> title + jump-nav (one anchor per
    `_SECTION_TITLES` entry) + the section fragments.

    Each renderer .html is shown in an iframe by the Snakemake report engine, so
    inline CSS is mandatory (no shared stylesheet reaches the iframe) and the
    in-page anchors scroll the iframe rather than the parent document.
    """
    body = "\n".join(s for s in sections if s)
    frags = fragments or []
    # [Q148] document-level Tabulator wiring. The CDN assets and the ~40 KB shared
    # filter blob are emitted ONCE for all three reproduction-guide tables -- which is
    # the entire reason `build_html_document` was split into head-assets + fragment.
    # Calling it once per table would have produced three <!DOCTYPE html> roots and
    # three copies of the blob inside one page.
    #
    # Each fragment's markup is relocated into its mount point at load time rather than
    # being interpolated into `body`: the fragments are built inside
    # `_build_reprex_guide_html`, which returns markup and mount points together, and a
    # mount keeps the table's DOM adjacent to its own <script> without threading the
    # fragment list through every section builder.
    tabulator_head = tabulator_head_assets() if frags else ""
    frag_styles = "".join(f.styles for _, f in frags)
    frag_mounts = "".join(
        f'<template data-trf-mount="reprex-{bucket}">{f.markup}</template>'
        for bucket, f in frags
    )
    frag_scripts = (
        "<script>"
        + _TRF_MOUNT_JS
        + tabulator_shared_js()
        + "".join(f.script for _, f in frags)
        + "</script>"
        if frags
        else ""
    )
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"{tabulator_head}"
        f"<style>{inline_css}{_SUPPLEMENTAL_CSS}{frag_styles}</style></head><body>"
        f"<h2>Metadata — {_esc(analysis_id)}</h2>"
        f"{_jump_nav()}"
        f"{body}"
        f"{frag_mounts}"
        f"<script>{_SORT_FILTER_JS}{_JUMP_NAV_JS}</script>"
        f"{frag_scripts}"
        "</body></html>"
    )


#: Iter-12 item 17. The affordance for a tooltip-bearing cell, emitted from the
#: RESOLVER rather than from `_SUPPLEMENTAL_CSS`. That placement is the whole point:
#: `_SUPPLEMENTAL_CSS` is a module-level static built at import time, so a rule living
#: there cannot read a colour off the runtime `report_cfg` and would have to hardcode
#: one or fall back to `currentColor` -- either of which re-introduces the second colour
#: source the brand_theme stipulation exists to prevent, and the latter of which fails
#: this rule's own paired test (which asserts the colour EQUALS what the resolved style
#: yields, never a literal). `primary_color` is the theme-driven brand colour the h2/h3
#: chrome already uses; the config comment beside it names which colours are NOT
#: theme-driven and this is not among them.
#:
#: Scoped to `strong.tip-affordance`, which `_sensitivity_varied_values` emits ONLY on
#: the varied branch. The single-distinct-value branch returns an EMPTY tooltip, so it
#: deliberately carries no affordance -- styling it would advertise a hover that shows
#: nothing.
_TIP_AFFORDANCE_CSS = (
    # Attribute selection, not class selection: a class rule reaches only the one
    # site that remembers to apply it, and the requirement is report-wide. Every
    # element carrying a native `title` tooltip matches, in every renderer, with no
    # renderer change. button/a/iframe are excluded deliberately -- a control hint
    # and a frame label are not tooltip-bearing text.
    "strong.tip-affordance, td[title], span[title], abbr[title] "
    "{{ cursor: help; border-bottom: 1px dotted {affordance_color}; color: {affordance_color}; }}\n"
    # `th` takes the underline only: the header background is {primary_color} navy,
    # against which #A85000 measures 2.46:1 versus white's 13.56:1.
    "th[title] {{ cursor: help; border-bottom: 1px dotted currentColor; }}\n"
    # The hovered row drops the colour: analysis.py's D-5 overlay sets
    # row_hover_bg_color to the full-saturation accent, against which the affordance
    # colour measures 1.77:1 -- at exactly the moment the reader hovers to read the
    # tooltip. The dotted underline is the redundant channel that survives, so the
    # affordance is never lost, only de-emphasised.
    "tr:hover td[title] {{ color: inherit; border-bottom-color: currentColor; }}\n"
    # Tabulator's `tooltip` column option returns a string that Tabulator renders
    # itself; it sets no DOM `title`, so no attribute selector can reach it. The
    # cell carries `cssClass: trf-has-tip` instead (see the companion spec).
    ".tabulator-cell.trf-has-tip "
    "{{ cursor: help; border-bottom: 1px dotted {affordance_color}; color: {affordance_color}; }}\n"
)


def _resolve_inline_css(report_cfg: report_config) -> str:
    """Brand chrome from the brand_theme-driven errors_and_warnings style block.

    The metadata page reuses the sibling static renderer's resolved palette so
    the report's chrome stays consistent and no brand hex literal is introduced
    here (brand_theme stipulation). Falls back to a bare default when the caller
    supplied a report_cfg without the block.
    """
    style = getattr(report_cfg, "errors_and_warnings", None)
    if style is None:
        from hhemt.config.report import ErrorsAndWarningsConfig

        style = ErrorsAndWarningsConfig()
    return style.render_inline_css() + _TIP_AFFORDANCE_CSS.format(
        primary_color=style.primary_color,
        affordance_color=getattr(style, "affordance_color", "#A85000"),
    )


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
) -> Path:
    """Render the Metadata page (provenance + reproduction guide + SLURM efficiency)."""
    from hhemt.report_renderers._figure_emission import emit_plot_with_sources
    from hhemt.report_renderers._provenance import ProvenanceLog, ProvenanceRef

    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    analysis_id = str(getattr(analysis.cfg_analysis, "analysis_id", "") or "")
    sidecar_path = analysis_dir / _SIDECAR_FILENAME

    # ADR-6 Gate-A anchor + ADR-6 D3: declare the expected source UNCONDITIONALLY,
    # even when absent. `_validate_source_path` accepts non-existent paths, so the
    # info-icon still names the source the page would have read.
    source_paths: list[Path] = [sidecar_path]

    prov = ProvenanceLog()
    with prov.artist(
        axes_id="html_section",
        kind="table",
        note="metadata page (RO-Crate provenance sidecar + reprex taxonomy + SLURM efficiency)",
    ) as artist:
        artist.add_channel("provenance", ProvenanceRef(source_path=_SIDECAR_FILENAME))
        # The whole-experiment records the crate cannot carry: the per-rule status
        # sidecars (run timeline + the rule-name<->JobID map the SLURM table joins on)
        # and the consolidated tree's ADR-15 producing stamps. Every sidecar actually
        # OPENED is declared below, per the declared-subset-of-actual invariant;
        # `_status/` is already copytree'd into the bundle, so this adds manifest rows
        # and no payload bytes.
        status_payloads, status_files = _read_status_flag_payloads(analysis_dir)
        source_paths.extend(status_files)
        for sidecar in status_files:
            artist.add_channel("status", ProvenanceRef(source_path=str(sidecar.relative_to(analysis_dir))))
        tree_path = _resolve_consolidated_tree(analysis_dir, analysis)
        if tree_path is not None:
            source_paths.append(tree_path)
            artist.add_channel(
                "producing_stamp",
                ProvenanceRef(source_path=str(tree_path.relative_to(analysis_dir))),
            )
        # (1) Provenance -- one declared open() on the sidecar -> audit Tier-1 ap == d.
        if sidecar_path.exists():
            doc = json.loads(sidecar_path.read_text())
            provenance_html = _build_provenance_html(
                doc, analysis_dir=analysis_dir, analysis=analysis, status_payloads=status_payloads
            )
        else:
            provenance_html = _absent_banner(
                "Provenance",
                "Provenance metadata not available — the RO-Crate sidecar "
                "(ro-crate-metadata.json) was not found. It is written at "
                "consolidation; re-run consolidation to populate.",
            )

    # (2) Data availability -- reads the SAME persisted validation_report.json the
    # Errors-and-Warnings renderer reads (ADR-14 D1: no second read-model). Declared
    # unconditionally per ADR-6 D3, so the info-icon names the source even when absent.
    validation_report_path = analysis_dir / _VALIDATION_REPORT_FILENAME
    source_paths.append(validation_report_path)
    with prov.artist(
        axes_id="html_section",
        kind="table",
        note="data-availability record (post-processing reclaim disclosure)",
    ) as artist:
        artist.add_channel("availability", ProvenanceRef(source_path=_VALIDATION_REPORT_FILENAME))
        data_availability_html = _build_data_availability_html(validation_report_path)

    # (3) Reproduction guide -- config-schema introspection (no file read) for the
    # descriptions / requiredness / defaults, plus in-memory config values for the
    # HPC and EXPERIMENT buckets only. `_config_field_rows` still takes no analysis,
    # so its zero-user-info guarantee stays structural rather than disciplinary.
    reprex_guide = _build_reprex_guide_html(
        _config_field_values(analysis),
        _sensitivity_varied_values(analysis),
    )

    # (4) SLURM efficiency -- glob + descend (os.scandir/os.stat; audit-invisible)
    # to EVERY inner efficiency_report_*.csv FILE, then declare them all. The plugin
    # writes the real CSV INSIDE a `.csv`-NAMED DIRECTORY (see
    # _resolve_all_efficiency_csvs), so read_text() on the glob match itself raises
    # IsADirectoryError, and declaring the directory would raise in
    # _validate_source_path (directory-as-source rejected unless zarr).
    #
    # Declaring the whole union is also what carries it into the render bundle:
    # `_harvest_and_copy_sources` copies exactly the declared set and
    # `_copy_supporting_files` never touches logs/, so bundle carriage follows
    # declaration and needs no bundle-side change.
    eff_dir = analysis_dir.joinpath(*_SLURM_EFF_RELDIR)
    eff_csvs = _resolve_all_efficiency_csvs(eff_dir)
    # Stage-A recovery of the job and `.batch` rows the plugin's parsing drops. Declared
    # unconditionally per ADR-6 D3 so the info-icon names it even before the back-fill runs.
    _recovery_map, _recovery_path = _load_job_recovery(analysis_dir)
    scenario_map, scenario_status_path = _read_scenario_status(analysis_dir)
    _apply_config_fallbacks(scenario_map, analysis)
    if scenario_status_path is not None:
        source_paths.append(scenario_status_path)

    def _gpu_hardware_for_partition(partition: str) -> str:
        """Partition -> GPU hardware, via the toolkit's own deterministic resolver.

        In-memory config only (no file read): `resolve_gpu_target` returns
        (None, None) for a CPU partition, an undeclared partition, or a missing
        HPC-system config, so this degrades to an empty cell rather than raising.
        """
        try:
            from hhemt.config.hpc_system import resolve_gpu_target

            hardware, _backend = resolve_gpu_target(getattr(analysis, "cfg_hpc_system", None), partition)
        except Exception:  # noqa: BLE001 -- a display column must not break the render
            return ""
        return hardware or ""

    if eff_csvs:
        source_paths.extend(eff_csvs)
        # Stage-A recovery artifact, declared only in the branch that reads it: the
        # absent-SLURM path declares no CSV at all, and an unconditional append here
        # broke that contract (caught by test_absent_slurm_csv_degrades_gracefully).
        if _recovery_path is not None:
            source_paths.append(_recovery_path)
        with prov.artist(
            axes_id="html_section",
            kind="table",
            note="SLURM resource-efficiency reports (union across all workflow submissions)",
        ) as artist:
            for csv_path in eff_csvs:
                artist.add_channel(
                    "data",
                    ProvenanceRef(source_path=str(csv_path.relative_to(analysis_dir))),
                )
            if scenario_status_path is not None:
                artist.add_channel("scenario_status", ProvenanceRef(source_path=_SCENARIO_STATUS_FILENAME))
            # Tier 2 of the purpose join. Declared INSIDE this branch, matching the
            # `_recovery_path` contract directly above: the absent-SLURM path reads
            # nothing here and must declare nothing. Declared even when absent (ADR-6
            # D3) so the info-icon names the source, and because the renderer-IO audit
            # requires declared >= actual reads. Graceful-absent: an unreadable or
            # missing index yields {} and the join degrades to exactly Tier 1.
            _job_index_path = analysis_dir / "_status" / "_job_index.json"
            source_paths.append(_job_index_path)
            artist.add_channel("job_index", ProvenanceRef(source_path="_status/_job_index.json"))
            _job_index: dict[str, str] = {}
            if _job_index_path.exists():
                try:
                    _loaded = json.loads(_job_index_path.read_text())
                    if isinstance(_loaded, dict):
                        _job_index = {str(k): str(v) for k, v in _loaded.items()}
                except (OSError, ValueError):
                    _job_index = {}
            slurm_html = _build_slurm_efficiency_html(
                [(str(p.relative_to(analysis_dir)), p.read_text()) for p in eff_csvs],
                _job_purpose_map(status_payloads, _job_index),
                scenario_map,
                _gpu_hardware_for_partition,
                recovery=_recovery_map,
            )
    else:
        slurm_html = _absent_banner(
            "SLURM Efficiency",
            "No SLURM resource-efficiency data — this analysis ran in local/native "
            "mode, or the end-of-workflow efficiency report has not yet been written. "
            "It is finalized at workflow teardown, AFTER the report is rendered, so it "
            "is expected to be absent on the run that produces this page; re-render "
            "after the run completes to populate it.",
        )

    html = _wrap_html_doc(
        analysis_id,
        _resolve_inline_css(report_cfg),
        provenance_html,
        data_availability_html,
        reprex_guide.html,
        slurm_html,
        fragments=reprex_guide.fragments,
    )
    return emit_plot_with_sources(
        html,
        output_path,
        source_paths,
        analysis_dir=analysis_dir,
        output_format="html",
        manifest_data={
            "renderer": "metadata",
            "sidecar_present": sidecar_path.exists(),
            "validation_report_present": validation_report_path.exists(),
            "slurm_csv_present": bool(eff_csvs),
            "slurm_csv_count": len(eff_csvs),
            "status_flag_count": len(status_files),
            "scenario_status_present": scenario_status_path is not None,
        },
        provenance=prov,
    )
