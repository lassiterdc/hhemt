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
  (3) SLURM efficiency    -- latest globbed slurm_efficiency_report_*.csv
      (Decision D2). EMPTY on the producing run (the CSV is written at
      Snakemake teardown, AFTER render_report); populates on a later
      re-render / reprocess. This is inherent, not a defect.

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
        "Revise them for your target system. Bundle.reprex(reprex_config, "
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
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
nav.jump-nav { margin: 8px 0 16px; font-size: 13px; }
nav.jump-nav a { text-decoration: none; }
p.instruction { font-weight: 600; margin: 4px 0 8px; }
p.note { font-size: 12px; color: #555; margin: 4px 0 10px; }
span.badge { display: inline-block; padding: 1px 7px; border-radius: 8px;
             color: white; font-size: 11px; font-weight: 700; }
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
        rows.append(("Toolkit", _esc(_prop(src, "name"))))
    if _prop(src, "codeRepository"):
        rows.append(("Code repository", _code(_prop(src, "codeRepository"))))
    git_sha = _prop(app, "softwareVersion") or _prop(src, "version")
    if git_sha:
        rows.append(("Git SHA (exact code that produced this)", _code(git_sha)))
    if not rows:
        return "<h4>2. Software</h4>\n" + _banner("No software provenance captured in this crate.")
    return "<h4>2. Software</h4>\n" + _kv_table(rows)


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
        rows.append(("Sub-datasets (hasPart)", " ".join(_code(p) for p in sub_parts)))
    parts.append(_kv_table(rows))

    var_rows: list[list[str]] = []
    for ref in _ref_ids(_prop(dataset, "variableMeasured")):
        pv = _by_id(graph, ref)
        if pv is None:
            continue
        var_rows.append(
            [
                _code(_prop(pv, "name")),
                _esc(_prop(pv, "description") or "—"),
                _esc(_prop(pv, "unitText") or "—"),
                _code(_prop(pv, "propertyID")) if _prop(pv, "propertyID") else "—",
                _esc(_prop(pv, "measurementTechnique") or "—"),
            ]
        )
    if var_rows:
        parts.append("<p class='note'>CF-conformant data dictionary for the consolidated variables.</p>")
        parts.append(
            _grid_table(
                ["Variable", "Long name", "Units", "CF standard_name", "cell_methods"],
                var_rows,
            )
        )
    return "\n".join(parts)


def _build_provenance_html(doc: dict) -> str:
    """Project the RO-Crate JSON-LD @graph into a static provenance recreation chain.

    Ordered disclosed -> verifiable: BLUF verifiability anchors, then
    Identity -> Software -> Environment -> Inputs -> Process -> Outputs.
    Allow-list BY CONSTRUCTION: each sub-block reaches only for the named safe
    fields it enumerates; `_prop` refuses the volatile keys as a backstop.
    """
    graph = _graph(doc)
    root = _by_id(graph, _ROOT_ID) or {}
    app = _by_id(graph, _APP_ID) or {}
    src = _by_id(graph, _TOOLKIT_SRC_ID) or {}
    sif = _find_sif(graph)
    inputs = _find_input_files(graph)
    runs = _of_type(graph, "CreateAction")

    return "\n".join(
        [
            _heading("Provenance"),
            _provenance_bluf(app, sif, inputs),
            _provenance_identity(graph, root),
            _provenance_software(app, src),
            _provenance_environment(sif),
            _provenance_inputs(inputs),
            _provenance_process(runs),
            _provenance_outputs(graph, root),
        ]
    )


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

    def _row(label: str, note: str | None, bucket: str, field_name: str) -> list[str]:
        placeholder = _BUCKET_PLACEHOLDER.get(bucket, f"{{your-{field_name}}}")
        return [_code(label), _esc(note or "—"), _code(placeholder)]

    # (b) the target user's supply set. Listed FIRST inside each bucket so the
    # Supply block opens with what the reproducer literally types.
    for field_name, field_info in reprex_config.model_fields.items():
        bucket = "hpc" if field_name in _REPREX_SELECTOR_FIELDS else "user"
        rows_by_bucket[bucket].append(_row(f"reprex_config.{field_name}", field_info.description, bucket, field_name))

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
            rows_by_bucket[bucket].append(
                _row(f"{config_label}.{field_name}", field_info.description, bucket, field_name)
            )
    return rows_by_bucket, unclassified


def _bucket_badge(bucket: str) -> str:
    color = _BUCKET_COLOR[bucket]
    return f'<span class="badge" style="background-color:{color}">{_esc(_BUCKET_VERB[bucket])}</span>'


def _build_reprex_guide_html() -> str:
    """Static grouped table: every config field -> USER=Supply / HPC=Amend / EXPERIMENT=Keep.

    Grouped (not a flat sortable grid) because the primary task a reproducer
    performs is "what do I DO with this field?" -- answered pre-attentively by
    the bucket. Each group is redundant-coded with an Okabe-Ito badge AND the
    instruction verb, so it survives grayscale and CVD.
    """
    rows_by_bucket, unclassified = _config_field_rows()

    parts: list[str] = [_heading("Reproduction Guide")]
    parts.append(
        "<p class='note'>Every configuration field below is grouped by what a reproducer must "
        "do with it. Value cells are placeholders and schema descriptions only — this page "
        "never carries the producing user's configuration values, so it is safe to ship "
        "inside a bundle.</p>"
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
        parts.append(_grid_table(["Field", "What it is", "Placeholder"], rows))

    return "\n".join(parts)


# --- (3) SLURM efficiency ----------------------------------------------------


def _build_slurm_efficiency_html(csv_text: str) -> str:
    """Parse the efficiency CSV text into a static HTML table.

    Parsed from an in-memory string (the caller already read the declared file),
    so this adds no file-open audit surface.
    """
    rows = [r for r in csv.reader(io.StringIO(csv_text)) if r]
    if not rows:
        return (
            _heading("SLURM Efficiency") + "\n" + _banner("The SLURM resource-efficiency report is present but empty.")
        )
    header, body = rows[0], rows[1:]
    if not body:
        return (
            _heading("SLURM Efficiency") + "\n" + _banner("The SLURM resource-efficiency report contains no job rows.")
        )
    grid = [[_code(cell) if "/" in cell else _esc(cell) for cell in row] for row in body]
    return _heading("SLURM Efficiency") + "\n" + _grid_table(header, grid)


# --- page shell --------------------------------------------------------------


def _jump_nav() -> str:
    links = " &middot; ".join(
        f'<a href="#{_anchor(t)}">{_esc(t)}</a>' for t in ("Provenance", "Reproduction Guide", "SLURM Efficiency")
    )
    return f'<nav class="jump-nav">{links}</nav>'


def _wrap_html_doc(analysis_id: str, inline_css: str, *sections: str) -> str:
    """<!DOCTYPE> + inline <style> + <h2> title + 3-anchor jump-nav + the section fragments.

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
        # (1) Provenance -- one declared open() on the sidecar -> audit Tier-1 ap == d.
        if sidecar_path.exists():
            doc = json.loads(sidecar_path.read_text())
            provenance_html = _build_provenance_html(doc)
        else:
            provenance_html = _absent_banner(
                "Provenance",
                "Provenance metadata not available — the RO-Crate sidecar "
                "(ro-crate-metadata.json) was not found. It is written at "
                "consolidation; re-run consolidation to populate.",
            )

    # (2) Reproduction guide -- pure config-schema introspection, no file read.
    reprex_html = _build_reprex_guide_html()

    # (3) SLURM efficiency -- glob (os.scandir; audit-invisible) then declare the
    # specific FILE. Declaring the directory would raise in _validate_source_path
    # once the dir exists (directory-as-source is rejected unless zarr).
    eff_dir = analysis_dir.joinpath(*_SLURM_EFF_RELDIR)
    csvs = sorted(eff_dir.glob(_SLURM_EFF_GLOB)) if eff_dir.is_dir() else []
    if csvs:
        latest = csvs[-1]  # timestamped filenames sort chronologically
        source_paths.append(latest)
        with prov.artist(
            axes_id="html_section",
            kind="table",
            note="SLURM resource-efficiency report",
        ) as artist:
            artist.add_channel(
                "data",
                ProvenanceRef(source_path=str(latest.relative_to(analysis_dir))),
            )
            slurm_html = _build_slurm_efficiency_html(latest.read_text())
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
            "slurm_csv_present": bool(csvs),
        },
        provenance=prov,
    )
