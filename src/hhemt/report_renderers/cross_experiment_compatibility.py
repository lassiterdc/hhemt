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
    # CP-5 (user ruling 2026-08-11): abbreviate the solver sha and MARK the rows whose
    # solver differs from the others, so a split pin is visible at a glance instead of
    # requiring a reader to diff four 40-char shas across rows. The marker is keyed on
    # the row set actually present, never on a hardcoded pin, so it stays correct when
    # every arm shares one solver (no marker, no caption) and when they do not.
    _solvers = {str(row.get("solver_sha")) for row in rows if row.get("solver_sha")}
    _split = len(_solvers) > 1
    # Tie-break on the SHA, not on set iteration order: an even split (two clean arms
    # and two resume arms is the ordinary shape) has no true minority, and `min` over a
    # set would let Python's hash ordering decide which side gets marked -- a
    # nondeterministic report. Marking is stable either way, because with an even split
    # "differs from the others" is true of whichever side carries the marker.
    _minority = min(
        _solvers,
        key=lambda s: (sum(1 for row in rows if str(row.get("solver_sha")) == s), s),
    ) if _split else None

    def _sv(row) -> str:
        raw = row.get("solver_sha")
        if not raw:
            return "n/a"
        short = _html.escape(str(raw)[:8])
        return f"{short} *" if _split and str(raw) == _minority else short

    # CP-5 amendment: mark the hhemt-build axis with the SAME present-set-keyed logic
    # the solver column uses, so the reader can see what did NOT vary as well as what
    # did. On the delivered generation the solver splits (two clean arms at one build,
    # two resume arms at another) while hhemt is constant -- and that combination is
    # exactly the claim the clean-vs-resume comparison rests on: one variable moved.
    # Marking only the axis that varied leaves the constancy of the other to be
    # inferred from silence, and silence is not evidence.
    _builds = {str(row.get("toolkit_version")) for row in rows if row.get("toolkit_version")}
    _build_split = len(_builds) > 1
    # Same sha-keyed tie-break as _sv, for the same reason: an even split has no true
    # minority and `min` over a set would let hash ordering pick a side, making the
    # rendered bytes nondeterministic across re-renders -- which would also make every
    # re-bundle look like a content change to _render_sha.
    _build_minority = min(
        _builds,
        key=lambda b: (sum(1 for row in rows if str(row.get("toolkit_version")) == b), b),
    ) if _build_split else None

    def _bv(row) -> str:
        raw = row.get("toolkit_version")
        if not raw:
            return "n/a"
        shown = _html.escape(str(raw))
        return f"{shown} *" if _build_split and str(raw) == _build_minority else shown

    body = "\n".join(
        "<tr><td>{e}</td><td>{r}</td><td>{m}</td><td>{n}</td>"
        "<td>{s}</td><td>{tv}</td><td>{sv}</td></tr>".format(
            e=_html.escape(str(row.get("experiment_id"))),
            r=_html.escape(str(row.get("role"))),
            m=_html.escape(str(row.get("model"))),
            n=_html.escape(str(row.get("n_subs"))),
            s=_html.escape(str(row.get("toolkit_sha"))),
            tv=_bv(row),
            sv=_sv(row),
        )
        for row in rows
    )
    # Two independent axes, so two independent clauses -- and the "identical across
    # every row" sentence is emitted for the axis that did NOT split, because a reader
    # verifying that this was a controlled comparison needs the constancy stated, not
    # left to be inferred from an unmarked column.
    _notes = []
    if _split:
        _notes.append(
            "* on the Solver sha column marks an experiment run on a different solver "
            "build than the others. Full shas are recorded on each child crate; the "
            "abbreviation here is for reading, not for identification."
        )
    elif rows:
        _notes.append("Solver sha is identical across every row.")
    if _build_split:
        _notes.append(
            "* on the hhemt build column marks an experiment produced by a different "
            "hhemt build than the others."
        )
    elif rows:
        _notes.append(
            "hhemt build is identical across every row, so any difference between these "
            "experiments is not attributable to the toolkit code."
        )
    _caption = ("<p class='note'>" + " ".join(_notes) + "</p>") if _notes else ""
    # CP-5: ONE solver column, not a TRITON column plus a TRITON-SWMM column. The
    # measured pin is IDENTICAL across the coupled and pure-TRITON arms because
    # TRITON-SWMM is the coupled build and a pure-TRITON run is that same binary
    # with SWMM off -- two columns would print n/a opposite a sha and imply the
    # arms ran different codebases. The Model column already states which arm the
    # row is. `n/a` renders when a child ships no tree or a pre-provenance one.
    return (
        "<table class='compat'><thead><tr><th>Experiment</th><th>Role</th><th>Model</th>"
        # "Toolkit" named the product nowhere -- the user measured `hhemt` zero times
        # on this figure. "build" rather than "version" because the value is a
        # git-describe-derived PEP-440 local version, and calling it a "version"
        # invites reading it as an install target, which the ToolkitPin stipulation
        # reserves for a resolvable published artifact.
        # These two columns are DIFFERENT provenance axes and were reading as one
        # build's sha plus its version. `hhemt sha` is `toolkit_sha`, from the child
        # bundle_manifest -- the build that EMITTED the bundle. `hhemt build` derives
        # from `hhemt_producing_sha` on the consolidated tree -- the build that
        # PRODUCED the data. On the delivered generation they were 17 commits apart
        # (ad70cd3b416f vs 01655abb60c2) and a reader had no way to see that.
        "<th># sub-analyses</th><th>hhemt bundle sha</th><th>hhemt build (data-producing)</th>"
        "<th>Solver sha</th></tr></thead><tbody>" + body + "</tbody></table>" + _caption
    )


def _derive_version_from_sha(sha: str) -> str | None:
    """PEP-440 local version for an ARCHIVED sha, or None when underivable.

    This is a DERIVATION, not a splice. It reads nothing but the sha the artifact
    already carries and the local git object DB, and it is recomputable by anyone with
    the repo -- which is what lets a reader check it rather than trust it. Returns None
    (never a guess) when the sha is not in the object DB, when git is unavailable, or
    when the derived sha does not prefix-match the input; the caller then keeps the raw
    stamped value. The prefix-match is the falsifiability check: a mis-derivation fails
    it, where an unchecked lookup would not.

    Deliberately does NOT back-fill any stage that was never captured. A never-captured
    value cannot be recovered, only guessed, and a guessed provenance value is worse
    than an absent one -- it launders uncertainty into apparent certainty, which is the
    EW-3 failure shape (a field stamped from a value that had no meaning at run time,
    reading authoritative and wrong).
    """
    import subprocess as _sp

    from hhemt.bundle._emit import _toolkit_source_dir

    try:
        out = _sp.run(
            ["git", "describe", "--tags", "--long", "--abbrev=12", sha],
            # Anchor on the toolkit source, never the process CWD. Under the render
            # rule the CWD is the analysis/bundle directory, so an unanchored
            # `git describe` either fails outright (non-repo -> check=True raises ->
            # None) or resolves a FOREIGN repo that does not contain this sha (-> non-
            # zero -> None). Both kept the raw stamped "0.1.0", which is what the
            # delivered generation rendered on all four rows. `_emit._toolkit_source_dir`
            # exists for exactly this: measured 2026-07-15, an unanchored git query from
            # a foreign repo's CWD returned the foreign SHA while hhemt was imported
            # from a different checkout. The `None` contract is unchanged -- with the
            # anchor, git failure still means genuinely-absent git, not wrong-directory.
            cwd=_toolkit_source_dir(),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        tag, n, gsha = out.rsplit("-", 2)
    except Exception:
        return None
    if not gsha.startswith("g") or not sha.startswith(gsha[1:]):
        return None  # derived sha does not match the input -- refuse rather than guess
    return f"{tag.lstrip('v')}+{n}.g{sha}"
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
        solver_sha, toolkit_version = "", ""
        store = child / "sensitivity_datatree.zarr"
        if store.exists():
            try:
                import xarray as _xr

                dt = _xr.open_datatree(str(store), engine="zarr", consolidated=False)
                # CP-5: the solver pin and toolkit version are ROOT ATTRS on this
                # same tree, stamped at consolidation (processing_analysis stamps
                # triton_producing_sha; analysis_validation already reads it back).
                # No new read: `dt` is already open for the model/n_subs derivation
                # below, so this costs nothing and cannot introduce a new failure.
                solver_sha = str(dt.attrs.get("triton_producing_sha") or "")
                hhemt_sha = str(dt.attrs.get("hhemt_producing_sha") or "")
                # The DELIVERED generation stamped the static pyproject pin here, so
                # this reads "0.1.0" on every pre-fix tree -- true, and useless. Rather
                # than splice a corrected value into an archived DOI-candidate artifact,
                # derive it from the sha the artifact ALREADY carries: the mapping is a
                # pure function of the commit and is recomputable at any time
                # (`git describe --tags --long 01655abb60c2` -> `v0.1.0-241-g01655ab`).
                # A tree stamped by the fixed minter already carries the derived form
                # and is passed through untouched -- detected by the "+" that a
                # PEP-440 local version has and a bare pin does not, never by a
                # hardcoded "0.1.0" comparison, which would silently stop firing at
                # the next release.
                toolkit_version = str(dt.attrs.get("hhemt_producing_version") or "")
                if toolkit_version and "+" not in toolkit_version and hhemt_sha:
                    derived = _derive_version_from_sha(hhemt_sha)
                    if derived:
                        toolkit_version = derived
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
                "toolkit_version": toolkit_version,
                "solver_sha": solver_sha,
                "_source_rel": f"child_crates/{eid}/scenario_status.csv",
            }
        )
    return rows


#: The model-toggle fields ALONE. Kept separate from the expected-set below because
#: `_divergence_message`'s first branch is keyed on it and renders "these bundles are
#: the two model arms of one experiment" -- a sentence that is TRUE only for a toggle.
_MODEL_TOGGLE_FIELDS = frozenset(
    {"toggle_triton_model", "toggle_tritonswmm_model", "toggle_swmm_model"}
)

#: Identity fields whose divergence IS the combine's intended structure. The model
#: toggles differ because the bundles are the two model arms of one experiment.
#: `sensitivity_analysis` differs because a cross-experiment combine PRESUPPOSES
#: differently-scoped experiments -- two bundles that swept identical matrices would
#: be a degenerate combine with nothing to compare. Unconditional rather than gated
#: on clean-vs-resume roles: the divergence dict carries no role, only bundle NAMES
#: and file paths, and keying correctness on a filename is a proxy instrument.
#:
#: This is a SUPERSET of `_MODEL_TOGGLE_FIELDS`, not a replacement. Every measured
#: divergence carries `bucket=experiment, severity=warning` -- the toggles included --
#: so `_divergence_message` cannot be fixed by reordering its branches: that would
#: route the toggle rows into the sensitivity-axis sentence. The two sets are what
#: keep "is this expected" and "what do I say about it" independently correct.
#: `TRITONSWMM_branch_key` is admitted UNCONDITIONALLY, and that is sound rather than
#: permissive: the field is in `_EXPERIMENT_IDENTITY_FIELDS` and classifies BLOCKING by
#: default (`bundle/_compatibility.py:137`), only `declare_solver_split=True` downgrades it
#: to WARNING (`:356-370`), and a BLOCKING divergence aborts `combine_bundle` before any
#: render (`bundle/_combine.py:131`). An UNdeclared split therefore never reaches a
#: renderer at all -- so the presence of this field here IS the declaration, and the
#: renderer needs no access to the flag.
_EXPECTED_IDENTITY_FIELDS = _MODEL_TOGGLE_FIELDS | frozenset(
    {"sensitivity_analysis", "TRITONSWMM_branch_key"}
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
    return field in _EXPECTED_IDENTITY_FIELDS or bucket == "hpc"


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
    # Keyed on the TOGGLE set, never on `_EXPECTED_IDENTITY_FIELDS`: the sentence below
    # is true only of a model toggle, and every divergence -- toggles included -- carries
    # bucket=experiment/severity=warning, so branch order cannot separate them.
    if field in _MODEL_TOGGLE_FIELDS:
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
    # Must precede the `bucket == "experiment" and severity == "warning"` catch-all: a
    # solver-sha divergence carries exactly that bucket/severity pair, so without its own
    # branch it printed "the two bundles sweep different rows or columns of the compute
    # matrix" -- a sentence that is factually wrong about a pinned solver sha. Measured on
    # the delivered combined bundle: 4 TRITONSWMM_branch_key rows, 4 sensitivity-axis
    # sentences.
    if field == "TRITONSWMM_branch_key":
        return (
            "Expected: the bundles run different pinned TRITON-SWMM solvers, declared at "
            "combine time. Running one arm at a fix and the other at its ancestor is what "
            "makes a bit-identical result evidence for both."
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
    # CP-4: the identity-field section is removed at the user's request and the heading
    # below is reframed as data provenance. What it reported was not actionable: a
    # BLOCKING divergence aborts combine_bundle before any render, so only non-blocking
    # rows reached this table, and on the measured generation all twelve were the
    # combine's intended structure. The one class that WOULD have mattered --
    # schemaVersion layout skew between bundles -- now routes to
    # cross_experiment_errors_and_warnings, the sanctioned warnings surface; without
    # that hop this deletion would have retired that warning silently.
    #
    # The payload read and the declared `source` are KEPT deliberately even though this
    # function no longer consumes `divergences`: render() declares
    # combined_compatibility.json as its provenance source, and a declared source that
    # is never opened makes that declaration false (the ADR-6 non-empty-source concern).
    # `_divergence_is_expected` / `_divergence_message` are likewise kept -- they are
    # the exact logic the errors-and-warnings hop imports.
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
        "<h2>What Was Combined (data provenance)</h2>"
        + _provenance_table_html(prov_rows)
        + "</section>"
    )
