"""Metadata report renderer (ADR-14 / C10).

Renders ONE self-contained static HTML page under the "Metadata" ReportingSet
category, with three sub-sections:

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

All-static inline-CSS HTML (data-viz research): this page is itself a
portability/provenance artifact -- it is read detached from a live network
(inside a render bundle, archived at a DOI, emailed to a reviewer) -- so a
CDN-Tabulator dependency would contradict its own thesis, and inline-Tabulator
bundling is unimplemented (Gotcha 51). Mirrors errors_and_warnings.py.

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
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    "user": "Supply — you provide these (USER)",
    "hpc": "Amend — bundled, but revise for your machine (HPC)",
    "experiment": "Keep — these define the experiment (EXPERIMENT)",
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
   `_grid_table` emits no `div.table-scroll` wrapper, so widening the table
   instead (by dropping `break-all`) is not an option here. */
table td:first-child { min-width: 18ch; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
nav.jump-nav { margin: 8px 0 16px; font-size: 13px; }
nav.jump-nav a { text-decoration: none; }
p.instruction { font-weight: 600; margin: 4px 0 8px; }
p.note { font-size: 12px; color: #555; margin: 4px 0 10px; }
span.badge { display: inline-block; padding: 1px 7px; border-radius: 8px;
             color: white; font-size: 11px; font-weight: 700; }
/* Sortable/filterable table affordances. Layout-only -- no brand hex literal
   (brand_theme stipulation); the sort indicator is a glyph, not a color. */
table.sortable th { cursor: pointer; user-select: none; white-space: nowrap; }
table.sortable th::after { content: " \\2195"; opacity: 0.35; font-size: 10px; }
table.sortable th.sorted-asc::after { content: " \\2191"; opacity: 1; }
table.sortable th.sorted-desc::after { content: " \\2193"; opacity: 1; }
input.table-filter { width: 100%; box-sizing: border-box; margin: 4px 0 8px;
                     padding: 4px 6px; font-size: 12px; }
div.table-scroll { overflow-x: auto; }
/* Typeset units and symbolic operations (Iter-11, item 15). `line-height: 0` is not
   cosmetic: without it every superscripted exponent inflates its table row's height, so
   the data-dictionary rows stop aligning with the rest of the page. No math engine --
   see `_expr_html` for why a bundled KaTeX/MathJax was rejected. */
sub, sup { line-height: 0; font-size: 0.75em; }
span.units { white-space: nowrap; }
span.expr { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
"""

# Inline vanilla-JS sort + filter shim. Deliberately NOT Tabulator: this page is read
# detached from a live network (inside a render bundle, archived at a DOI, emailed to a
# reviewer), so a CDN dependency would contradict the page's own thesis, and
# inline-Tabulator bundling is unimplemented (Gotcha 51). ~40 lines keeps the page
# self-contained and does not pre-empt the reporting-system_inline-tabulator plan.
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
  document.querySelectorAll("table.sortable").forEach(function (table) {
    var body = table.tBodies[0];
    if (!body) { return; }
    table.querySelectorAll("thead th").forEach(function (th, idx) {
      th.addEventListener("click", function () {
        var desc = th.classList.contains("sorted-asc");
        table.querySelectorAll("thead th").forEach(function (o) {
          o.classList.remove("sorted-asc", "sorted-desc");
        });
        th.classList.add(desc ? "sorted-desc" : "sorted-asc");
        var rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (r1, r2) {
          var d = cmp(cellText(r1, idx), cellText(r2, idx));
          return desc ? -d : d;
        });
        rows.forEach(function (r) { body.appendChild(r); });
      });
    });
  });
  document.querySelectorAll("input.table-filter").forEach(function (input) {
    input.addEventListener("input", function () {
      var needle = input.value.toLowerCase();
      var table = document.getElementById(input.getAttribute("data-table"));
      if (!table || !table.tBodies[0]) { return; }
      Array.prototype.slice.call(table.tBodies[0].rows).forEach(function (r) {
        r.style.display = r.innerText.toLowerCase().indexOf(needle) === -1 ? "none" : "";
      });
    });
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
    """Static 2-column Field/Value table. Values are PRE-ESCAPED HTML fragments."""
    if not rows:
        return ""
    body = "\n    ".join(f"<tr><td>{_esc(k)}</td><td>{v}</td></tr>" for k, v in rows)
    return (
        "<table>\n"
        "  <thead><tr><th>Field</th><th>Value</th></tr></thead>\n"
        "  <tbody>\n    " + body + "\n  </tbody>\n</table>"
    )


def _grid_table(headers: list[str], rows: list[list[str]]) -> str:
    """Static n-column table. Row cells are PRE-ESCAPED HTML fragments."""
    if not rows:
        return ""
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "\n    ".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>\n  <thead><tr>{head}</tr></thead>\n  <tbody>\n    " + body + "\n  </tbody>\n</table>"


def _sortable_grid_table(headers: list[str], rows: list[list[str]], *, table_id: str, filter_label: str) -> str:
    """`_grid_table` plus a click-to-sort header and a free-text filter box.

    Static HTML + the inline `_SORT_FILTER_JS` shim only -- no CDN, no Tabulator
    (see `_SORT_FILTER_JS` for why). Degrades to a plain readable table when
    JavaScript is unavailable, which is the state the page must survive in an
    archived/emailed copy. Row cells are PRE-ESCAPED HTML fragments.
    """
    if not rows:
        return ""
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "\n    ".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (
        f'<input class="table-filter" type="text" data-table="{_esc(table_id)}" '
        f'placeholder="{_esc(filter_label)}">\n'
        '<div class="table-scroll">\n'
        f'<table class="sortable" id="{_esc(table_id)}">\n'
        f"  <thead><tr>{head}</tr></thead>\n"
        "  <tbody>\n    " + body + "\n  </tbody>\n</table>\n</div>"
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


def _job_purpose_map(payloads: list[dict]) -> dict[str, dict[str, str]]:
    """`{slurm_job_id: {purpose, rule_name, sa_id, event_id, model_type, written_at}}`.

    `status_flags.write_status_flag` records `slurm_job_id` from the ``SLURM_JOB_ID``
    environment variable, which is the PARENT job id -- exactly the efficiency CSV's
    `MainJobID` column, so the join is direct. Locally-executed rules carry a null
    job id and are skipped rather than keyed on the empty string.
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

    No math engine. This page carries none, and the module already refuses a CDN
    dependency for its sort/filter shim on the grounds that the page is read detached from
    a live network. A BUNDLED engine would clear that bar on self-containment but would
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


def _provenance_outputs(graph: list[dict], root: dict) -> str:
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
            _provenance_outputs(graph, root),
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
        f"<p class='note'><strong>Run timeline.</strong> {_esc(len(rows))} completed workflow "
        "step(s), recorded one file at a time as each finished. This is the whole experiment "
        "from setup through consolidation, not the most recent run — regenerating this report "
        "cannot shorten it. Click a column heading to sort; type to filter.</p>"
    )
    table = _sortable_grid_table(
        ["Completed at", "What it did", "Rule", "Sub-analysis", "Event", "Model", "SLURM job"],
        rows,
        table_id="run-timeline",
        filter_label="Filter steps — try a purpose, sub-analysis, or model",
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
        items = "".join(f"<dt>{_code(k)}</dt><dd>{_esc(v)}</dd>" for k, v in options.items())
        parts.append(f"<dl class='options'>{items}</dl>")
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
        return f"<strong>Conditional</strong> — required when {_esc(render_clauses(clauses))}"
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
    """Render one config value for display: paths analysis-dir-relative, others repr-ish."""
    if value is None:
        return "<em>null</em>"
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


def _sensitivity_varied_values(analysis: TRITONSWMM_analysis) -> dict[str, str]:
    """`{"{config_label}.{field_name}": rendered cell}` for every SWEPT parameter.

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

    out: dict[str, str] = {}
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
        detail = ", ".join(_code(v) for v in distinct[:_VARIED_VALUE_PREVIEW])
        if len(distinct) > _VARIED_VALUE_PREVIEW:
            detail += f", … ({len(distinct)} values)"
        out[label] = "<strong>Varied by the sensitivity analysis</strong>" + (f" — {detail}" if detail else "")
    return out


def _build_reprex_guide_html(
    values_by_field: dict[str, str] | None = None,
    varied_by_field: dict[str, str] | None = None,
) -> str:
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
        "directory.</p>"
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
        headers = ["Field", "Description", "Required", "Placeholder"]
        _values = values_by_field or {}
        _varied = varied_by_field or {}
        if bucket in ("hpc", "experiment") and (_values or _varied):
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
            headers.insert(3, "Value used")
            _fallback = _code(_BUCKET_PLACEHOLDER.get(bucket, "—"))
            rows = [
                row[:3] + [_varied.get(_strip_code(row[0])) or _values.get(_strip_code(row[0]), _fallback)] + row[3:]
                for row in rows
            ]
        parts.append(_grid_table(headers, rows))

    return "\n".join(parts)


# --- (3) SLURM efficiency ----------------------------------------------------


#: Columns carried through from the plugin's CSV, in display order. The plugin's unnamed
#: index column and its intermediate unit-conversion columns are dropped -- they are
#: restatements of neighbours (MaxRSS vs MaxRSS_MB) and add width without adding meaning.
_EFF_PASSTHROUGH_COLUMNS: tuple[tuple[str, str], ...] = (
    ("JobID", "Job ID"),
    ("Elapsed", "Elapsed"),
    ("NNodes", "Nodes"),
    ("NCPUS", "CPUs"),
    ("CPU Efficiency (%)", "CPU eff (%)"),
    ("MaxRSS_MB", "Max RSS (MB)"),
    ("RequestedMem_MB", "Req mem (MB)"),
    ("Memory Usage (%)", "Mem used (%)"),
)

#: Enrichment columns joined in from the toolkit's own records. `_status/*.flag.json`
#: supplies the first four; `scenario_status.csv` (keyed on sa_id/event_id/model_type)
#: supplies the hardware and concurrency block.
_EFF_ENRICHMENT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("purpose", "What the job did"),
    ("sa_id", "Sub-analysis"),
    ("event_id", "Event"),
    ("model_type", "Model"),
    ("partition", "Partition"),
    ("gpu_hardware", "GPU hardware"),
    ("n_gpus", "GPUs"),
    ("n_mpi_procs", "MPI ranks"),
    ("n_omp_threads", "OMP threads"),
    ("n_nodes_cfg", "Nodes (config)"),
    ("run_mode", "Run mode"),
    ("backend_used", "Backend"),
    # O21. TWO queue columns, deliberately. The table has one row per SLURM job and a
    # resumed sim occupies several rows, so a lone sim-total would repeat across them and
    # read as each allocation having waited the total. The first column is that row's own
    # wait; the second is the sim's end-to-end wait and is correctly identical across the
    # sim's rows, which the label says out loud. Coverage discloses a partial sum.
    ("queue_seconds_this_job", "Queue, this job (s)"),
    ("queue_seconds_total", "Queue, sim total (s)"),
    ("queue_seconds_coverage", "Queue coverage"),
)

#: Columns the user asked for that SLURM accounting does not capture. Rendered as an
#: explicit disclosure rather than omitted: an absent measurement that is silently left
#: out reads as "not applicable", and an empty column reads as zero. Neither is true.
_EFF_UNCAPTURED_NOTE = (
    "<p class='note'><strong>Not shown, and why.</strong> <em>GPU utilisation</em> is absent "
    "from SLURM accounting entirely — <code>AllocTRES</code> records how many GPUs were "
    "allocated, never how hard they worked — so reporting it needs a sampler running "
    "alongside the simulation, and existing runs cannot be back-filled without re-running "
    "them. <em>CPU model</em> is likewise not recorded by the toolkit today; it is "
    "recoverable on the cluster from <code>sacct -o NodeList</code> plus "
    "<code>scontrol show node</code> while the accounting database still holds the job. "
    "<em>Queue time</em> is captured only for runs submitted one job per simulation "
    "(<code>multi_sim_run_method: batch_job</code>). Under "
    "<code>1_job_many_srun_tasks</code> the whole ensemble shares a single allocation, so "
    "there is no per-simulation queue to measure and the column is left blank rather than "
    "filled with the allocation's own wait — a repeated number there would be a per-sim "
    "figure that no simulation actually experienced. A blank cell here means not "
    "measured; it never means the job did not wait. Where queue time IS captured, "
    "<em>Queue coverage</em> reads <code>k/n</code> over the simulation's allocations, so "
    "a run that began before queue capture shipped shows a partial sum as partial. "
    "<em>CPU efficiency</em> is shown as not-measured wherever SLURM reported no CPU time "
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
    "table cannot currently reach them. Recovering them needs no new simulation: SLURM "
    "accounting still holds each job's partition and allocated resources, and the workflow "
    "engine's own per-job log tree names the rule behind every job id.</p>"
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
                    enriched["purpose"] += " (initial run)" if _n == 0 else f" (resume {_n})"
        partition = scen.get("hpc.partition", "")
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
#: shown as `Job ID`; `JobName` is the executable name, always `python`; `RuleName` appears
#: only on clusters that store job comments and is superseded by the toolkit's own purpose).
#: Listing them here is what lets an UNRECOGNISED column be reported instead of dropped.
_EFF_KNOWN_UNDISPLAYED: frozenset[str] = frozenset(
    {
        "",  # the plugin's unnamed DataFrame index column
        "JobName",  # always the executable (`python`); the real purpose is joined in
        "TotalCPU",  # shown as the derived CPU-efficiency percentage
        "Elapsed_sec",  # parsed restatement of Elapsed
        "TotalCPU_sec",  # parsed restatement of TotalCPU
        "MaxRSS",  # raw string form of MaxRSS_MB
        "ReqMem",  # raw string form of RequestedMem_MB
        "MainJobID",  # the join key; the step id is shown as Job ID
        "RuleName",  # present only where SLURM stores job comments; superseded
        "Comment",
        "State",
    }
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
    displayed = {key for key, _label in _EFF_PASSTHROUGH_COLUMNS}
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
) -> str:
    """Render the UNION of every efficiency report, enriched and sortable/filterable."""
    heading = _heading("SLURM Efficiency")
    if not csv_texts:
        return heading + "\n" + _banner("The SLURM resource-efficiency report is present but empty.")

    _header, merged = _parse_efficiency_csvs(csv_texts)
    if not merged:
        return heading + "\n" + _banner("The SLURM resource-efficiency reports contain no job rows.")

    rows = _enrich_efficiency_rows(merged, purpose_map, scenario_map, gpu_hardware_for_partition)
    columns = list(_EFF_PASSTHROUGH_COLUMNS) + list(_EFF_ENRICHMENT_COLUMNS)
    matched = sum(1 for r in rows if r.get("purpose"))
    undisplayed = _undisplayed_csv_columns(_header)

    # A rendered `0.0` in a PERCENTAGE column asserts a measured zero -- "this job used no
    # CPU" -- which is false about jobs that demonstrably ran for minutes. Measured on the
    # delivered generation: CPU eff read exactly 0.0 on 664/664 rows, because the executor
    # plugin inherits RequestedMem_MB from the main job row down to its steps but does NOT
    # inherit TotalCPU, then discards the main rows (efficiency_report.py: mem_map /
    # job_steps / `df = job_steps.copy()`). So the ratio is computed from a zero numerator.
    # Confirmed on-cluster: parent 18396501 reports TotalCPU=00:04.828 while step .0 reports
    # 00:00:00. The parent row never reaches the CSV, so this is NOT repairable here -- the
    # em-dash means NOT CAPTURED, and 0.0 would mean MEASURED ZERO. Blanking is deliberately
    # narrow: only the derived-from-TotalCPU column, only when the source parsed to zero.
    _zeroish = {"", "0", "0.0", "0.00", "0.0%"}

    def _cell(row: dict[str, str], key: str) -> str:
        value = row.get(key, "")
        if key == "CPU Efficiency (%)" and str(value).strip() in _zeroish:
            if str(row.get("TotalCPU_sec", "")).strip() in _zeroish:
                return "—"
        return _esc(value or "—")

    grid = [[_cell(row, key) for key, _label in columns] for row in rows]
    summary = (
        f"<p class='note'>{_esc(len(rows))} job(s) across {_esc(len(csv_texts))} efficiency "
        f"report(s) — one report is written per workflow submission, and all of them are "
        f"combined here so the table covers the whole experiment rather than the most recent "
        f"run. Rows are keyed on SLURM job ID, so re-running part of the experiment adds rows "
        f"and never rewrites the ones that did not re-run. {_esc(matched)} row(s) carry a "
        f"toolkit-recorded purpose; the rest are jobs the toolkit did not flag (SLURM reports "
        f"every job step's command name as <code>python</code>, which is why the purpose is "
        f"joined in from the toolkit's own records rather than read off the job). Click a "
        f"column heading to sort; type to filter.</p>"
    )
    table = _sortable_grid_table(
        [label for _key, label in columns],
        grid,
        table_id="slurm-efficiency",
        filter_label="Filter jobs — try a purpose, sub-analysis, partition, or job id",
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

    The key is built to match the `_status/*.flag.json` payload's own vocabulary:
    the sidecar records ``event_id`` as the scenario slug (``event_index.0``) while
    this CSV records ``event_iloc`` as the integer index, so BOTH spellings are
    registered and the join succeeds whichever the sidecar carried.
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
        for event_key in {iloc, f"event_index.{iloc}"}:
            out[(sa_id, event_key, model_type)] = record
    return (out, path)


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


def _jump_nav() -> str:
    links = " &middot; ".join(f'<a href="#{_anchor(t)}">{_esc(t)}</a>' for t in _SECTION_TITLES)
    return f'<nav class="jump-nav">{links}</nav>'


def _wrap_html_doc(analysis_id: str, inline_css: str, *sections: str) -> str:
    """<!DOCTYPE> + inline <style> + <h2> title + jump-nav (one anchor per
    `_SECTION_TITLES` entry) + the section fragments.

    Each renderer .html is shown in an iframe by the Snakemake report engine, so
    inline CSS is mandatory (no shared stylesheet reaches the iframe) and the
    in-page anchors scroll the iframe rather than the parent document.
    """
    body = "\n".join(s for s in sections if s)
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<style>{inline_css}{_SUPPLEMENTAL_CSS}</style></head><body>"
        f"<h2>Metadata — {_esc(analysis_id)}</h2>"
        f"{_jump_nav()}"
        f"{body}"
        f"<script>{_SORT_FILTER_JS}</script>"
        "</body></html>"
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
    return style.render_inline_css()


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
    reprex_html = _build_reprex_guide_html(
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
    scenario_map, scenario_status_path = _read_scenario_status(analysis_dir)
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
            slurm_html = _build_slurm_efficiency_html(
                [(str(p.relative_to(analysis_dir)), p.read_text()) for p in eff_csvs],
                _job_purpose_map(status_payloads),
                scenario_map,
                _gpu_hardware_for_partition,
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
        reprex_html,
        slurm_html,
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
