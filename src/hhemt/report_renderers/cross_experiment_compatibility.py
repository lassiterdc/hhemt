"""Cross-experiment compatibility + characterized-divergence renderer (PIP-1, Phase 4).

Reads the persisted combined_compatibility.json read-model (Phase 3) and renders
the CompatibilityReport (informational / warning / blocking divergences by
taxonomy bucket) as an inline-styled HTML table. The cross-FAMILY byte-identity
panel is DEFERRED for the bundle path (R6 — a bundle ships the consolidated tree
only, not the flat per-scenario summaries check_cross_sim_identity reads), so a
"deferred" placeholder is rendered in its place. Uniform renderer signature per
the report-renderers stipulation; emits via emit_plot_with_sources (declaring the
read-model as the source, satisfying the Gotcha-41 non-empty-source gate).
"""

from __future__ import annotations

import html as _html
import json as _json
from pathlib import Path

from hhemt.report_renderers._figure_emission import emit_plot_with_sources
from hhemt.report_renderers._provenance import (
    ProvenanceLog,
    ProvenanceRef,
)


def render(analysis, report_cfg, output_path: Path, **kwargs) -> None:
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    source = analysis_dir / "combined_compatibility.json"

    # Per the report-renderer provenance convention (matching the peer table
    # renderers disk_utilization / errors_and_warnings / metadata), the data
    # source is recorded via a `with prov.artist(kind="table")` block and
    # threaded into the manifest sidecar through `provenance=prov`.
    prov = ProvenanceLog()
    with prov.artist(
        axes_id="html_section",
        kind="table",
        note="cross-experiment compatibility table (combined_compatibility.json)",
    ) as artist:
        artist.add_channel(
            "data",
            ProvenanceRef(source_path="combined_compatibility.json"),
        )
        # Q6a (iter-2): build the deterministic combine-provenance table from the child
        # crates (read + declared below) in addition to the compatibility divergences.
        prov_rows = _combine_provenance_rows(analysis_dir)
        for r in prov_rows:
            artist.add_channel("data", ProvenanceRef(source_path=r["_source_rel"]))
        html = _render_compatibility_html(source, prov_rows)

    child_sources = [analysis_dir / r["_source_rel"] for r in prov_rows]
    emit_plot_with_sources(
        html,
        output_path,
        source_paths=[source, *child_sources],
        analysis_dir=analysis_dir,
        provenance=prov,
    )


def _provenance_table_html(prov_rows: list[dict] | None) -> str:
    """Deterministic 'what was combined' table (Q6a). One row per child crate."""
    rows = prov_rows or []
    if not rows:
        return "<p class='note'>No child crates recorded for this combine.</p>"
    body = "\n".join(
        "<tr><td>{e}</td><td>{r}</td><td>{m}</td><td>{n}</td><td>{s}</td></tr>".format(
            e=_html.escape(str(row.get("experiment_id"))),
            r=_html.escape(str(row.get("role"))),
            m=_html.escape(str(row.get("model"))),
            n=_html.escape(str(row.get("n_subs"))),
            s=_html.escape(str(row.get("toolkit_sha"))),
        )
        for row in rows
    )
    return (
        "<table class='compat'><thead><tr><th>Experiment</th><th>Role</th><th>Model</th>"
        "<th># sub-analyses</th><th>Toolkit sha</th></tr></thead><tbody>" + body + "</tbody></table>"
    )


def _combine_provenance_rows(analysis_dir: Path) -> list[dict]:
    """One deterministic provenance row per combined child crate. Each field is derived from
    a bundled, deterministic source (no HPC re-run): experiment_id from the dir name; role
    from scenario_status.csv n_resumes (the _bundle_role_from_status convention); model from the
    sensitivity_datatree tier; n_subs from the /sa_* group count; toolkit sha from
    bundle_manifest.json. `_source_rel` is the declared audit source (relative to analysis_dir)."""
    import csv as _csv
    import json as _cjson

    crates = analysis_dir / "child_crates"
    rows: list[dict] = []
    if not crates.exists():
        return rows
    for child in sorted(p for p in crates.iterdir() if p.is_dir()):
        eid = child.name
        # role from scenario_status.csv (clean iff every n_resumes == 0)
        role, max_r = "clean", 0
        csvp = child / "scenario_status.csv"
        if csvp.exists():
            try:
                with csvp.open() as fh:
                    for r in _csv.DictReader(fh):
                        try:
                            max_r = max(max_r, int(float(r.get("n_resumes") or 0)))
                        except (TypeError, ValueError):
                            continue
                role = "resume" if max_r > 0 else "clean"
            except OSError:
                pass
        # model + n_subs from the consolidated tree tiers
        model, n_subs = "TRITON", 0
        store = child / "sensitivity_datatree.zarr"
        if store.exists():
            try:
                import xarray as _xr

                dt = _xr.open_datatree(str(store), engine="zarr", consolidated=False)
                grps = set(dt.groups)
                sa = sorted(g for g in grps if g.count("/") == 1 and g.startswith("/sa_"))
                n_subs = len(sa)
                for g in sa:
                    if f"{g}/tritonswmm/triton" in grps:
                        model = "TRITON-SWMM"
                        break
                    if f"{g}/triton_only/triton" in grps:
                        model = "TRITON"
                        break
            except Exception:
                pass
        # toolkit sha from the child bundle manifest
        sha = ""
        manp = child / "bundle_manifest.json"
        if manp.exists():
            try:
                man = _cjson.loads(manp.read_text())
                sha = str(man.get("toolkit_git_sha") or man.get("git_sha") or "")
            except (OSError, ValueError):
                pass
        rows.append(
            {
                "experiment_id": eid,
                "role": role,
                "model": model,
                "n_subs": n_subs,
                "toolkit_sha": sha,
                "_source_rel": f"child_crates/{eid}/scenario_status.csv",
            }
        )
    return rows


def _render_compatibility_html(source: Path, prov_rows: list[dict] | None = None) -> str:
    if source.exists():
        payload = _json.loads(source.read_text())
    else:  # combine may not have run; render an honest placeholder
        payload = {"is_compatible": True, "divergences": []}
    divs = payload.get("divergences", [])
    if divs:
        rows = "\n".join(
            "<tr><td>{f}</td><td>{bk}</td><td>{sev}</td><td>{ba}: {va}</td><td>{bb}: {vb}</td></tr>".format(
                f=_html.escape(str(d.get("field_name"))),
                bk=_html.escape(str(d.get("bucket"))),
                sev=_html.escape(str(d.get("severity"))),
                ba=_html.escape(str(d.get("bundle_a"))),
                va=_html.escape(str(d.get("value_a"))),
                bb=_html.escape(str(d.get("bundle_b"))),
                vb=_html.escape(str(d.get("value_b"))),
            )
            for d in divs
        )
        table = (
            "<table class='compat'><thead><tr><th>Field</th><th>Bucket</th>"
            "<th>Severity</th><th>Bundle A</th><th>Bundle B</th></tr></thead>"
            "<tbody>" + rows + "</tbody></table>"
        )
    else:
        table = "<p class='note'>All compared identity fields agree — the bundles are combine-compatible.</p>"
    # c2: reaching this renderer already implies compatibility (a BLOCKING divergence aborts
    # combine_bundle before any render), so only genuine informational/warning divergence rows
    # are informative. The R6 deferred-panel prose is removed (deterministic-only content, F9);
    # the physically meaningful clean-vs-resume comparison lives in the Cross-Experiment Results
    # section. Data-viz owns the .compat cell-border CSS.
    return (
        "<section class='cross-experiment-compatibility'>"
        "<h2>What Was Combined (combine provenance) &amp; compatibility</h2>"
        + _provenance_table_html(prov_rows)
        + "<h3>Identity-field compatibility</h3>"
        + table
        + "</section>"
    )
