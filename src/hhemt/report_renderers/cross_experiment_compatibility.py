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


_EXPECTED_TOGGLE_FIELDS = frozenset(
    {"toggle_triton_model", "toggle_tritonswmm_model", "toggle_swmm_model"}
)


def _divergence_is_expected(d: dict) -> bool:
    """True when this divergence is the combine's intended structure, not a finding.

    ``check_bundle_compatibility`` compares bundles PAIRWISE, so a four-bundle combine
    (two model arms x clean/resume) yields six pairs and the model-toggle fields differ in
    the four cross-arm ones. ``_downgrade_paired_model_arms`` already admits those rows --
    it detects the collapse to fewer base experiments and downgrades them BLOCKING ->
    WARNING -- but that admission is invisible in the rendered table, so a reader sees
    warnings with no way to learn the toolkit deliberately allowed them.
    """
    field = str(d.get("field_name") or "")
    bucket = str(d.get("bucket") or "")
    return field in _EXPECTED_TOGGLE_FIELDS or bucket == "hpc"


def _divergence_message(d: dict) -> str:
    """Deterministic plain-language reason for one divergence row.

    A pure function of ``(field_name, bucket, severity)`` -- every branch reads fields that
    are already in ``combined_compatibility.json``, so this adds no read and needs no
    combine re-run. Branch precedence is load-bearing: an unknown field name in the ``hpc``
    bucket must render the hpc message, not the fallback.
    """
    field = str(d.get("field_name") or "")
    bucket = str(d.get("bucket") or "")
    severity = str(d.get("severity") or "")
    if field in _EXPECTED_TOGGLE_FIELDS:
        return (
            "Expected: these bundles are the two model arms of one experiment; the toggle "
            "divergence is what makes them arms. Admitted by the paired-model-arm rule."
        )
    if bucket == "hpc":
        return (
            "Expected: the bundles ran on different HPC systems or partitions. Compute "
            "environment is not part of experiment identity."
        )
    if field == "schemaVersion":
        return (
            "Layout-version skew between bundles; figures render, but cross-bundle field "
            "semantics may differ."
        )
    if bucket == "experiment" and severity == "warning":
        return (
            "Sensitivity-axis divergence: the two bundles sweep different rows or columns "
            "of the compute matrix."
        )
    return f"Divergence in `{field}` ({bucket} bucket, {severity})."


def _render_compatibility_html(source: Path, prov_rows: list[dict] | None = None) -> str:
    if source.exists():
        payload = _json.loads(source.read_text())
    else:  # combine may not have run; render an honest placeholder
        payload = {"is_compatible": True, "divergences": []}
    divs = payload.get("divergences", [])
    verdict = (
        "<p class='note'>{n} identity-field divergence(s) across the combined bundles; "
        "{e} are expected under a both-arms / cross-system combine.</p>".format(
            n=len(divs),
            e=sum(1 for d in divs if _divergence_is_expected(d)),
        )
        if divs
        else ""
    )
    if divs:
        rows = "\n".join(
            "<tr><td>{f}</td><td>{status}</td><td>{ba}: {va}</td><td>{bb}: {vb}</td></tr>".format(
                f=_html.escape(str(d.get("field_name"))),
                status="Expected" if _divergence_is_expected(d) else "Review",
                ba=_html.escape(str(d.get("bundle_a"))),
                va=_html.escape(str(d.get("value_a"))),
                bb=_html.escape(str(d.get("bundle_b"))),
                vb=_html.escape(str(d.get("value_b"))),
            )
            for d in divs
        )
        # Iteration 4 (FQ8): `Bucket` is dropped -- it is a three-valued taxonomy whose only
        # consumer is _classify, and once the message states WHY a row is informational or
        # warning the bucket is fully carried by the message. Keeping both invites the
        # reader to reconcile two codings of one fact.
        #
        # Iteration 5: the message column is dropped at the user's request, and the raw
        # severity string goes with it. Severity is single-valued on this artifact -- all
        # twelve divergences of the reviewed combine are `warning` -- so the column
        # discriminated nothing while reading as an alarm. `_divergence_is_expected` is the
        # distinction that does carry information: it splits the same rows 8/4, and the
        # verdict line above the table already reports that split in aggregate. Per-row
        # Expected/Review makes the aggregate locatable, which is what the dropped message
        # column was doing.
        table = (
            "<table class='compat'><thead><tr><th>Field</th>"
            "<th>Status</th><th>Bundle A</th><th>Bundle B</th></tr></thead>"
            "<tbody>" + rows + "</tbody></table>"
        )
    else:
        table = "<p class='note'>All compared identity fields agree — the bundles are combine-compatible.</p>"
    # c2: reaching this renderer already implies compatibility (a BLOCKING divergence aborts
    # combine_bundle before any render), so only genuine informational/warning divergence rows
    # are informative. The R6 deferred-panel prose is removed (deterministic-only content, F9);
    # the physically meaningful clean-vs-resume comparison lives in the Cross-Experiment Results
    # section.
    #
    # The .compat cell-border CSS ships INSIDE this fragment. report.css.j2 carries a
    # complete table.compat rule set, but emit_plot_with_sources writes this fragment
    # verbatim with no <head>, and Snakemake embeds figure HTML in an iframe -- a separate
    # document the parent stylesheet does not cascade into. The shell rule is therefore
    # unreachable from here, which is why the rendered table has no cell boundaries even
    # though a correct rule for it exists. A fragment that must survive iframe embedding
    # carries its own declarations.
    return (
        "<section class='cross-experiment-compatibility'>"
        "<style>"
        "table.compat{border-collapse:collapse;margin:1rem 0;font-size:0.95rem;"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}"
        "table.compat th,table.compat td{border:1px solid #DADADA;padding:6px 12px;"
        "text-align:left;vertical-align:top}"
        "table.compat th{background-color:#232D4B;color:white;font-weight:600}"
        "table.compat tbody tr:nth-child(even){background-color:#F1F1EF}"
        "</style>"
        "<h2>What Was Combined (combine provenance) &amp; compatibility</h2>"
        + _provenance_table_html(prov_rows)
        + "<h3>Identity-field compatibility</h3>"
        + verdict
        + table
        + "</section>"
    )
