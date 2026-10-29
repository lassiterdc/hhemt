"""Record which build of hhemt produced each artifact, and refuse to mix artifacts from different builds.

An analysis's figures, consolidated stores and partial processing outputs are written by
separate stages, usually in separate processes, so nothing guarantees that the code which
wrote an artifact is the code now reading it. This module mints the identity stamp a stage
writes (`producing_stamp`), keeps each stage's append-only history of the builds that
produced it, and supplies the gates that refuse -- or, when the operator declares it,
print and proceed -- when a figure set, a consolidated store or a chapter set was written
by a different build from the one running.

It also renders the RO-Crate provenance graph written at consolidation (`emit_provenance`):
one PROV `CreateAction` per (event, model) run and a `wasGeneratedBy` edge per output,
returned twice -- a deterministic core for the consolidated tree's attrs and the full graph
for the `ro-crate-metadata.json` sidecar. The graph is read from the analysis and per-model
JSON logs, and nothing in this module sets a log field: the processing runner's
`_already_written` check decides whether to skip an output by reading those same records,
so a write from here would make it skip an output that was never written, or rewrite one
that was.
"""

from __future__ import annotations

import importlib.metadata
import socket

# Module scope, and needed at run time as well as for typing: several functions below call
# `Path(...)` on their arguments, and ruff resolves the `Path` annotations against module
# scope even under `from __future__ import annotations` -- an unresolved one is F821, inside
# CI's gating `--select=E9,F63,F7,F82` set. Some function bodies still bind a local `_Path`
# alias; it is the same class.
from pathlib import Path
from types import SimpleNamespace

from rocrate.model.contextentity import ContextEntity

from hhemt.constants import LAYOUT_VERSION
from hhemt.exceptions import StalePlotsError
from hhemt.metadata import (
    build_analysis_crate,
    canonical_jsonld,
    canonical_jsonld_from_doc,
    partition_core_vs_sidecar,
    resolve_dataset_license,
)


def _default_code_repository() -> str:
    """Canonical public repo URL for the RO-Crate codeRepository, sourced from the
    INSTALLED package metadata (pyproject.toml [project.urls].homepage = "https://github.com/lassiterdc/hhemt")
    — single source of truth, no duplicated literal. RAISES (never silently falls back to
    a stale/guessed literal) when the metadata homepage is absent: a missing homepage is a
    genuinely broken install that MUST surface loudly, not be papered into durable archival
    provenance. CI pins the expected value (test_provenance.py::test_default_code_repository_pins_homepage)
    so a pyproject regression is caught at CI, before deployment, not at hour-3 of an HPC consolidation."""
    from hhemt.exceptions import ProcessingError  # lazy: matches this module's import style; avoids any cycle

    for entry in importlib.metadata.metadata("hhemt").get_all("Project-URL") or []:
        label, _sep, url = entry.partition(",")
        if label.strip().lower() == "homepage":
            return url.strip()
    raise ProcessingError(
        operation="provenance_code_repository",
        filepath=None,
        reason="hhemt package metadata exposes no 'homepage' Project-URL; cannot resolve the RO-Crate "
        "codeRepository. Fix pyproject.toml [project.urls].homepage. (No silent fallback by design — "
        "a guessed/stale URL must never be baked into durable provenance.)",
    )


def _toolkit_git_sha() -> str:
    from hhemt.bundle._emit import _get_toolkit_git_sha

    return _get_toolkit_git_sha(strict=False)


def _describe_version() -> str:
    """PEP-440 local version derived from `git describe`, not from the static pin.

    `pyproject.toml` carries a STATIC `version = "0.1.0"` that commits do not update,
    so `importlib.metadata.version("hhemt")` is a constant that cannot tell two builds
    apart. This derives `{tag}+{N}.g{sha}` from `git describe --tags --long --abbrev=12`,
    which DOES distinguish builds and is deterministically recomputable from an archived
    sha alone: `git describe --tags --long --abbrev=12 01655abb60c2` prints
    `v0.1.0-241-g01655abb60c2`, which this returns as `0.1.0+241.g01655abb60c2`.

    When `git describe` fails (no git, as in a container image, or no reachable tag),
    this composes `{pin}+g{sha[:12]}` from `running_identity().sha` when that resolves,
    returns the bare pin when it does not (a wheel install has no identity), and
    returns `"0+unknown"` when the installed metadata version itself cannot be read.
    NEVER raises: a provenance minter that can abort a 3-hour consolidation is worse
    than one that degrades.
    """
    import subprocess

    from hhemt.bundle._emit import _toolkit_source_dir

    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--long", "--abbrev=12"],
            cwd=_toolkit_source_dir(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        tag, n, gsha = out.rsplit("-", 2)
        return f"{tag.lstrip('v')}+{n}.{gsha}"
    except Exception:
        # git is unavailable -- the IMAGE shape, where the recipe removed it. The bare pin
        # is not identifying (it has not moved in hundreds of commits), so compose one from
        # the sha the identity authority already holds. This is a FORMAT composition over a
        # value already resolved, not a git lookup: `cross_experiment_compatibility.
        # _derive_version_from_sha` looks like the right helper and is not, because it
        # shells git against the local object DB and returns None in exactly this shape.
        try:
            from hhemt.validation import running_identity

            _sha = running_identity().sha
        except Exception:
            _sha = ""
        try:
            _pin = importlib.metadata.version("hhemt")
        except Exception:
            return "0+unknown"
        return f"{_pin}+g{_sha[:12]}" if _sha else _pin


def _identity_dirty() -> str:
    """The dirty flag, resolved through the ONE identity source rather than by shelling git.

    `_is_dirty` below returns False on ANY exception, so a tree where git cannot run reads
    CLEAN -- an affirmative claim of cleanliness produced by a command that did not run.
    In the `image` and `archive` shapes that False is accidentally correct, because those
    shapes are exact commits; in the `refuse` shape (a wheel, a bare copy, a checkout whose
    git fails for a reason other than absence) it is a fabrication.

    `running_identity()` either ANSWERS or REFUSES. Its `dirty` is measured for a checkout
    and False by construction for an image or an archive. A refusal returns the literal
    "unknown" -- the same sentinel `_from_tree_attrs` already emits for this state, pinned
    here rather than left free because `check_provenance_completeness` buckets any
    non-"true"/"false" string as undetermined, so the value chosen decides the rendered cell.
    """
    from hhemt.exceptions import ConfigurationError
    from hhemt.validation import running_identity

    try:
        return "true" if running_identity().dirty else "false"
    except ConfigurationError:
        return "unknown"


def _is_dirty() -> bool:
    """True when the toolkit checkout has uncommitted changes to TRACKED files.

    Nothing calls this today: `producing_stamp` takes its dirty flag from
    `_identity_dirty` above, which reads it from `running_identity()` and reports a
    toolkit it cannot identify as "unknown" rather than as clean.

    False both when there are none and when the question cannot be answered: the
    `except` arm below returns the same False a verified-clean tree returns, so a
    non-git install reads clean. The two-valued result has no value for
    "undeterminable", which is the property `_identity_dirty` does not share.

    Scoped to tracked content deliberately. `git status --porcelain` also reports
    untracked (`??`) paths, and an untracked path is not a change to the checkout:
    it cannot alter what the toolkit computes unless it is importable package
    source, which a path outside the tracked tree is not. Counting untracked paths
    made this fire on any tree where a harness, an editor or an agent had written a
    stray file, and the stamp then recorded a divergence from HEAD that
    `git diff HEAD` did not show.

    Why a dirty flag has to travel beside the sha at all: `git rev-parse HEAD` succeeds
    on a dirty tree and returns the COMMITTED sha, so a run with uncommitted edits
    resolves the same sha as the clean commit and, read alone, looks authoritative.
    `process_simulation._resolve_producing_stamp`'s docstring says the sha is
    `"unknown"` when the working tree is dirty; it is not -- `running_identity()`
    returns the committed sha for a dirty checkout and reports the edits only as
    `dirty`, which `producing_stamp` stamps as `hhemt_dirty`. False on any error: an
    undeterminable tree must not be asserted dirty.
    """
    import subprocess

    from hhemt.bundle._emit import _toolkit_source_dir

    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=_toolkit_source_dir(),
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return False


def producing_stamp() -> dict[str, str]:
    """THE single minter for every stage's hhemt version-provenance stamp.

    A stage that stamps an artifact calls THIS, at write time, from the running
    process, so every carrier holds the same three fields in the same shape. The
    minting rule this enforces: a stamp records the code that ACTUALLY ran the stage,
    so it may never be resolved from a module-level constant, a config field, or a value
    carried over from an earlier stage. The failure that rule rules out is a field
    stamped from a constant that had no value yet at run time: it read authoritative and
    was wrong.

    Corollary the CALL SITES must honor and this function cannot: if the stage did not
    execute, do not call this. An unconditional stamp cannot distinguish
    "re-ran at this sha" from "skipped, carrying an older artifact".
    """
    return {
        "hhemt_sha": _toolkit_git_sha(),
        "hhemt_version": _describe_version(),
        "hhemt_dirty": _identity_dirty(),
    }


#: The append-only per-stage provenance history, relative to analysis_dir (or, for the
#: chapter-set stage, to the chapter set's own directory). Every code path reads the name
#: from here, but two user-facing message strings also spell it out -- the chapter-set
#: guard's mismatch message below and the `allow_mixed_version_chapters` field description
#: in `config/analysis.py` -- so a rename must update those two strings as well. Deliberately
#: NOT one of the consolidated-tree names: this artifact is orthogonal to the zarr layout and
#: must not acquire a dependency on a filename that is a live rename candidate.
_HISTORY_FILENAME = "provenance_history.json"


def stamp_report_stage(analysis_dir: Path, report_path: Path) -> None:
    """Capture the REPORT stage's producing stamp. Called by BOTH render_report twins.

    Called AFTER the render has demonstrably succeeded and BEFORE the format-gated early
    return, so the capture fires on every requested format. The ordering property the
    previous end-of-method placement protected is PRESERVED and is about the report FILE
    existing, not about its format: a stamp written at method entry would survive a failed
    render and claim a report that was never produced, and both twins hold the rendered
    file in hand before the format gate.

    `report_path` is the FORMAT-AGNOSTIC output, never an html-only binding -- recording a
    path that does not exist for a zip is the defect one rung down from the one this fixes.

    Best-effort and never raises: a provenance write must not fail a completed render.
    """
    import json as _json

    try:
        (Path(analysis_dir) / "report_manifest.json").write_text(
            _json.dumps({"report_path": str(report_path), **producing_stamp()}, indent=2),
            encoding="utf-8",
        )
        # Appends, never overwrites: the manifest above holds only the LATEST build, and
        # this is what keeps an earlier one. De-duplicated, so a re-render at an unchanged
        # build appends nothing and rewrites nothing.
        append_stage_provenance(Path(analysis_dir), "report")
    except Exception as _e:  # never fail a completed render on a provenance write
        print(f"[render_report] report_manifest.json stamp failed (non-fatal): {_e}", flush=True)


def _write_consolidate_manifest(analysis_dir: Path, stamp: dict) -> None:
    """Write the CONSOLIDATE stage's own full-triple carrier. Called from BOTH seams.

    Lives here rather than in either consolidation module because it is called from two
    of them -- `processing_analysis` at the member tier and `sensitivity_analysis` at the
    master tier -- and a helper called from two modules belongs in a third. Both callers
    already import from this module at the call site.

    `stamp` is passed in rather than minted here: the caller mints ONCE and writes twice,
    so the gate field and this carrier can never disagree. Minting a second time here
    would be two sites a later edit can separate, which is the failure this chain is about.

    Compare-and-write, so an idempotent re-consolidation preserves mtime and does not
    perturb the analysis-scope DU own-files walk. Best-effort and never raises.
    """
    import json as _json

    try:
        target = Path(analysis_dir) / "consolidate_manifest.json"
        payload = _json.dumps({k: stamp[k] for k in sorted(stamp)}, indent=2) + "\n"
        if target.exists() and target.read_text(encoding="utf-8") == payload:
            return  # byte-identical: preserve mtime
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    except Exception as _e:  # never fail a consolidation that succeeded
        print(f"[consolidate] consolidate_manifest.json stamp failed (non-fatal): {_e}", flush=True)


def append_stage_provenance(analysis_dir: Path, stage: str, *, stamp: dict | None = None) -> bool:
    """Append this stage's producing stamp to the analysis's append-only history.

    Appends, never overwrites. Returns True if an entry was appended, False if the
    stage's latest entry already names this exact build.

    WHY THIS IS NOT IN THE RO-CRATE SIDECAR, stated here because the sidecar is the
    intuitive home and is the wrong one. Both `emit_provenance` call sites and both
    `write_rocrate_sidecar` call sites are inside CONSOLIDATION. The two stages that write
    this history into `analysis_dir` -- plots and report -- run strictly AFTER
    consolidation, so a history projected into the sidecar could not contain the render
    that has not happened yet: on a single pass it would record zero render entries, and on
    the two-renders case this exists for, neither. The third caller, the chapter-set stage,
    records during processing, BEFORE consolidation, and into the chapter set's own
    directory, so the sidecar is not its home either. A standalone read-model written by
    each stage at its own capture site is the same persist-then-read shape
    `validation_report.json` already uses, and for the same ordering reason.

    The sidecar's own idempotence -- a re-consolidation with unchanged content leaves
    `ro-crate-metadata.json` byte- and mtime-identical -- is preserved BY CONSTRUCTION rather
    than by keying: this function never touches that file, so the sidecar's compare-and-write,
    `_EMBEDDED_PROV_KEYS`, and the byte-identity goldens are all untouched. The de-duplication
    below is what keeps THIS file idempotent -- an unchanged build appends nothing and the
    file's bytes and mtime are preserved, so the analysis-scope disk-usage total, which counts
    the analysis directory's own top-level files, is unperturbed too.

    The key is `(hhemt_sha, hhemt_dirty)`, not the sha alone: two builds at one commit with
    different working-tree states are different producers, which `producing_stamp` already
    models. NO timestamp is recorded -- an emit-time clock read would make every invocation
    append and would defeat the idempotence this contract depends on. Ordering IS the
    history; wall-clock is not needed to read it.

    Best-effort and never raises: a provenance write must not fail a stage that succeeded.

    `stamp` is the identity to record, and defaults to `producing_stamp()` called here.
    The plots and report stages pass nothing and record exactly that. The chapter-set
    stage (`record_chapter_build`) passes the identity its guard compares against, because
    the guard names the running build by `processing_build_key()`, which falls back to the
    container image label when `producing_stamp()` is "unknown"; recording the default
    there would write "unknown" for a build the guard's message calls by its label, and
    the guard drops "unknown" entries, so a later run could not compare against it.
    """
    import json as _json
    from pathlib import Path as _Path

    try:
        target = _Path(analysis_dir) / _HISTORY_FILENAME
        entries: list[dict] = []
        if target.exists():
            loaded = _json.loads(target.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                entries = loaded
        stamp = producing_stamp() if stamp is None else stamp
        key = (stamp.get("hhemt_sha"), stamp.get("hhemt_dirty"))
        for prior in reversed(entries):
            if prior.get("stage") == stage:
                if (prior.get("hhemt_sha"), prior.get("hhemt_dirty")) == key:
                    return False  # unchanged build for this stage -- no write, mtime preserved
                break  # a DIFFERENT build superseded it; fall through and append
        entries.append({"stage": stage, **stamp})
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_json.dumps(entries, indent=2) + "\n", encoding="utf-8")
        return True
    except Exception:
        return False


def read_stage_provenance_history(analysis_dir: Path) -> dict[str, list[dict]]:
    """Per-stage revision lists, oldest first. Graceful-absent: {} when the file is absent.

    Absence is the honest reading for every analysis produced before this capture existed,
    and is NOT distinguishable from "one revision" by design -- a stage with a single
    recorded build has one entry, and a stage with none has no key.
    """
    import json as _json
    from collections import defaultdict
    from pathlib import Path as _Path

    out: defaultdict[str, list[dict]] = defaultdict(list)
    try:
        target = _Path(analysis_dir) / _HISTORY_FILENAME
        if not target.exists():
            return {}
        loaded = _json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            return {}
        for entry in loaded:
            if isinstance(entry, dict) and entry.get("stage"):
                out[str(entry["stage"])].append(entry)
    except Exception:
        return {}
    return dict(out)


def collect_plot_stamps(
    analysis_dir: Path, *, report_rule_stems: frozenset[str] = frozenset()
) -> tuple[set[tuple[str, str]], int]:
    """Distinct (sha, dirty) keys across every figure sidecar, and the sidecar count.

    Reads ALL sidecars rather than one, because uniform-within-a-render is a measured
    property and not a guaranteed one: a targeted re-render refreshes some sidecars and
    leaves others, which yields two distinct keys and is its own staleness class
    (figures inconsistent with EACH OTHER, remedied by a full stage:render force rather
    than by re-rendering the report). An unreadable or unstamped sidecar contributes no
    key and is counted, so `keys == set()` with `n > 0` means "figures present, none
    stamped" -- absent, which the caller treats as never-equal.

    A sidecar under `constants.UNREGENERABLE_ANALYSIS_SUBTREES` (which includes
    `plots/eda/`) is skipped and not counted UNLESS its figure is one the analysis's
    active reporting set writes through a plot rule, as named by `report_rule_stems`
    (from `utils.report_rule_figure_stems`). The forced figure re-render spares exactly
    the same figures (`utils.select_regenerable_figures` takes the same argument), so a
    stale figure the check counts is one that re-render deletes and a report rule
    rebuilds, and a skipped one, such as a figure only `analysis.eda()` writes, is one
    the re-render could never clear, where counting it would refuse every later render.
    Empty `report_rule_stems` (the default, which the bundle-side checks pass) skips
    every sidecar under those subtrees.
    """
    import json
    from pathlib import Path as _Path

    from hhemt.constants import UNREGENERABLE_ANALYSIS_SUBTREES
    from hhemt.utils import figure_exists_for, is_report_rule_figure  # lazy: matches this module's import style

    keys: set[tuple[str, str]] = set()
    n = 0
    root = _Path(analysis_dir)
    plots = root / "plots"
    if not plots.exists():
        return keys, 0
    for sidecar in sorted(plots.rglob("*.manifest.json")):
        # Matched as `rel == p or rel.startswith(p + "/")`, the expression
        # select_regenerable_figures uses, so `plots/eda_summary/` is NOT skipped.
        rel = sidecar.relative_to(root).as_posix()
        if any(rel == p or rel.startswith(p + "/") for p in UNREGENERABLE_ANALYSIS_SUBTREES):
            if not is_report_rule_figure(rel.removesuffix(".manifest.json"), report_rule_stems):
                continue
        # An ORPHAN sidecar -- one whose figure is gone -- is not evidence about the
        # figures on disk, and this function's whole subject is the figures on disk.
        # Skipped ABOVE the counter deliberately: counted-but-excluded would make the
        # caller's "no figure carries a build stamp (N sidecar(s) found)" report a
        # nonzero N for zero live figures. Nothing is quieted by this -- an all-orphan
        # tree yields an empty key set, which `assert_plots_match_running_build` RAISES
        # on rather than passing.
        if not figure_exists_for(sidecar):
            continue
        n += 1
        try:
            payload = json.loads(sidecar.read_text())
        except Exception:
            continue
        sha = str(payload.get("hhemt_sha") or "").strip()
        if sha:
            keys.add((sha, str(payload.get("hhemt_dirty") or "unknown")))
    return keys, n


def assert_plots_match_running_build(
    analysis_dir: Path, *, declare_stale_plots: bool = False, report_rule_stems: frozenset[str] = frozenset()
):
    """Refuse a render whose figures were not produced by the build now running.

    The report-side operand is `producing_stamp()` computed IN-PROCESS, never a
    persisted report stamp: reading `report_manifest.json` would compare a render
    against itself (the writer runs at the tail of the same call), would refuse every
    first render for want of a prior stamp, and would depend on a writer that fails
    non-fatally. Equality on the (sha, dirty) tuple, never an ordering.
    """
    keys, n = collect_plot_stamps(analysis_dir, report_rule_stems=report_rule_stems)
    running = producing_stamp()
    mine = (str(running.get("hhemt_sha") or "").strip(), str(running.get("hhemt_dirty") or "unknown"))
    if len(keys) > 1:
        why = (
            f"the {n} figure sidecar(s) carry {len(keys)} DIFFERENT build stamps "
            f"({sorted(f'{s[:12]}(dirty={d})' for s, d in keys)}) -- the figures are inconsistent with each "
            "other, which a partial re-render produces. Force a full re-render with "
            'force_rerun {"subject": "all", "stage": "render"}'
        )
    elif not keys:
        why = (
            f"no figure carries a build stamp ({n} sidecar(s) found) -- the figures predate "
            "the provenance capture site and cannot be shown to match this build"
        )
    elif not mine[0]:
        why = "the running build could not be identified (no toolkit sha resolvable)"
    elif keys == {mine}:
        return None
    else:
        (psha, pdirty) = next(iter(keys))
        why = f"figures built at {psha[:12]} (dirty={pdirty}), report rendering at {mine[0][:12]} (dirty={mine[1]})"
    msg = (
        f"Report/figure build mismatch: {why}. The figures may not reflect the renderer "
        "code now producing this report. Re-render with force_rerun "
        '{"subject": "all", "stage": "render"}. To proceed anyway, call render_report() on '
        "the analysis, or regenerate_report() on the bundle, from Python with "
        "declare_stale_plots=True: the mismatch is then printed to standard output, and "
        "hhemt does not store it. There is no command-line option for this."
    )
    if declare_stale_plots:
        print(f"[render_report] DECLARED STALE PLOTS: {msg}", flush=True)
        return msg
    raise StalePlotsError(msg)


#: Break-glass for the consolidation build gate below. An ENV VAR rather than a
#: keyword argument because consolidation runs in a SUBPROCESS
#: (`python -m hhemt.consolidate_workflow`), so a caller-passed kwarg cannot reach
#: the gate without a new CLI arg threaded through argparse and the Snakefile shell.
#: Precedent for the env-var shape: HHEMT_ENABLE_PROVENANCE_AUDIT, HHEMT_ALLOW_INSTALLED.
_DECLARE_STALE_BUILD_ENV = "HHEMT_DECLARE_STALE_BUILD"

#: Values a sha resolver returns when it resolved NOTHING. `_toolkit_git_sha` delegates
#: to `bundle/_emit._get_toolkit_git_sha(strict=False)`, which returns the literal
#: "unknown" on CalledProcessError/FileNotFoundError or an empty rev-parse -- the
#: wheel-install case its own strict-mode message names. "unknown" is TRUTHY, so a
#: falsy-check misses it and two different builds both resolving "unknown" compare
#: EQUAL. These are absences wearing a value, and absent is never equal.
_SENTINEL_SHAS = frozenset({"", "unknown", "0+unknown"})
#: CONSUMERS: `store_build_mismatch` below, and `analysis_validation.check_provenance_
#: completeness`, which imports this set across the module boundary rather than restating
#: it. The rule travels with the set; a second literal list would drift from this one.


def store_build_mismatch(store: Path, stored: str | None) -> str | None:
    """Return a reason string when `store` was NOT produced by the running build, else None.

    The consolidation-tier sibling of `assert_plots_match_running_build` above. Each
    property below is deliberate, and softening any one of them reopens the failure it
    closes:

    ABSENT IS NEVER EQUAL, including absent on both sides -- and ABSENCE INCLUDES A
    SENTINEL. A store whose supplied consolidation build stamp is absent is a MISMATCH,
    and so is one whose stamp is `"unknown"`, because that is what `_toolkit_git_sha`
    returns when it resolved nothing. A falsy-check does not catch it: `"unknown"` is
    truthy, so two different wheel installs both stamping `"unknown"` would compare equal
    and reuse each other's store with no output at all. Reading absence as "no objection"
    would silently reuse every tree consolidated before consolidation began recording its
    build stamp, which is precisely the population most likely to be stale. This is the
    one property that distinguishes this function from `check_provenance_completeness`,
    whose graceful-absent posture is correct for a REPORT (a provenance check that can
    abort validate_analysis takes the whole Errors-and-Warnings sidebar down with it) and
    wrong for a GATE.

    A DIRTY RUNNING CHECKOUT IS A MISMATCH WHEN THE SHAS OTHERWISE AGREE, and this arm
    is deliberate rather than incidental. `git rev-parse` succeeds on a dirty tree and
    returns the COMMITTED sha (`_is_dirty`'s own docstring says so), so a developer
    editing a consolidation module and re-running gets an unchanged sha -- which is
    exactly the workflow this gate exists to serve: the consolidation code changed and the
    sha did not, so a sha comparison alone would reuse a store the edited code would not
    have written. When the shas DIFFER the store already rebuilds and dirty adds nothing;
    when they AGREE the sha check says "reuse" and dirty is precisely the condition under
    which that verdict cannot be trusted. So the arm fires only in the second case.

    THE COST, AND THE PROPERTY THAT COST BUYS: while the checkout is dirty this gate
    DOES NOT CONVERGE -- a rebuild re-stamps the same committed sha against the same
    dirty tree, so the next consolidation rebuilds again. That is intended. What it
    costs is the CONSOLIDATION tier only: `process_*` sits upstream of this gate and is
    not re-triggered by it (measured on the synth sensitivity tree: 28.9 s to rebuild
    every consolidate rule, against 274 s of process rules the gate never touches).

    AND THAT COST SCALES LINEARLY IN MEMBER COUNT, which is the part a reader at
    ensemble scale will otherwise be surprised by. One `consolidate_member_*` rule runs
    per member, so the 28.9 s above is a per-tree figure for four synth members, about
    7 s per member. Projected linearly from that one measurement, and not itself
    measured, a fifty-member production ensemble would spend about six minutes. Seconds
    at development scale and minutes at
    ensemble scale is the same non-convergence priced very differently, and an operator
    who knows their edits cannot affect consolidation arms
    HHEMT_DECLARE_STALE_BUILD=1, which prints and reuses.

    THE RUNNING VALUE IS MINTED, NEVER READ BACK. `producing_stamp()` is called inline
    here, per its own documented rule that a stamp "may never be resolved from a
    module-level constant, a config field, or a value carried over from an earlier stage."
    The persisted `validation_report.json` is exactly such a carried-over value: it is
    written after consolidation, by `export_scenario_status` once it has written
    `scenario_status.csv`, and again by `analysis.eda` after the EDA calc, so whichever
    of the two ran last wrote it, and at the moment this gate runs it is a receipt of
    indeterminate authorship about a PRIOR run.

    THE ESCAPE PRINTS, AND THE PRINT IS ITS ONLY TRACE. With `HHEMT_DECLARE_STALE_BUILD=1`
    this returns None, so the caller reuses the store exactly as it would on a match, but
    only after printing the reason to stdout. Neither caller stores the declaration: it
    survives only in the step's log file, which each workflow consolidate rule writes by
    teeing the step's output, and a consolidation run outside the workflow keeps it
    nowhere. A silent bypass would reintroduce the failure with a flag on it.

    THE STORED OPERAND IS SUPPLIED, NOT READ OFF THE STORE, because the value on the
    store cannot converge. The tree root's `hhemt_producing_sha` is derived by
    `cf_conventions.apply_producing_stamp` from the per-`event_iloc` coordinates that
    `process_simulation._write_output` freezes at PROCESSING time -- deliberately, so a
    later reprocess reads the OLD stamp. Comparing that against a CONSOLIDATION-stage
    running build can never converge: the rebuild this gate triggers re-derives the same
    processing-stage value and the next call mismatches identically. Measured: a warm
    tree whose members' summaries were written at 3fe13978bcf5 re-stamped 3fe13978bcf5
    on every rebuild while the running build was e69c4fae7fcb. The caller now passes the
    CONSOLIDATION build stamp its own log recorded at the previous write, which is the
    convergent form already used by `consolidation_inputs_fingerprint` in the same
    caller -- writer and reader agree because they are the same expression.

    `store` is still taken, and only for the existence branch below.
    """
    import os

    if not store.exists():
        # Inert by construction: every branch that consumes this value also requires
        # `fname_out.exists()`, so a mismatch reported here can never trigger a rebuild
        # of a store that is not there. Reported as a mismatch anyway, because
        # absent-is-never-equal is the rule and a silent None here would be the one
        # place the rule is quietly suspended.
        why = "no consolidated store present at the reuse gate"
    else:
        stored = str(stored or "").strip()
        _stamp = producing_stamp()
        running = str(_stamp.get("hhemt_sha") or "").strip()
        running_dirty = str(_stamp.get("hhemt_dirty") or "").strip().lower() == "true"
        # Order matters: the two sentinel arms come FIRST so a sentinel can never reach
        # the equality test, where two builds that both resolved "unknown" compare equal.
        if stored in _SENTINEL_SHAS:
            why = (
                f"the consolidation build stamp supplied for this store is the sentinel {stored!r} "
                "-- it names no commit, so the store cannot be shown to match this build"
            )
        elif running in _SENTINEL_SHAS:
            why = f"the running build resolved to the sentinel sha {running!r} (no git checkout?)"
        elif stored != running:
            why = f"store built at {stored[:12]}, consolidation running at {running[:12]}"
        elif running_dirty:
            why = (
                f"the shas agree at {running[:12]} but the running checkout is DIRTY, so the "
                "sha names a commit whose content is not what is running"
            )
        else:
            return None

    msg = (
        f"Consolidated-store build mismatch at {store}: {why}. The store may not reflect "
        "the consolidation code now reading it. Rebuild it with force_rerun "
        '{"subject": "all", "stage": "consolidate"}, or set '
        f"{_DECLARE_STALE_BUILD_ENV}=1 to reuse it anyway: the mismatch is then printed to "
        "standard output and, when the workflow runs this step, kept in that step's log file."
    )
    if os.environ.get(_DECLARE_STALE_BUILD_ENV) == "1":
        print(f"[consolidate] DECLARED STALE BUILD: {msg}", flush=True)
        return None
    return msg


#: Stage key for the per-chapter-set producing-build history. `record_chapter_build` (the
#: writer) and `collect_chapter_build_keys` (the reader) both take it from here, so the two
#: cannot disagree on the key a chapter set's history is filed under.
_CHAPTER_STAGE = "chapters"


def processing_build_key() -> tuple[str, str]:
    """The (sha, dirty) identity of the build running the PROCESSING stage.

    `producing_stamp()` alone is not sufficient inside a container whose source has been
    replaced. In an unmodified image it is: `running_identity()` reads the image's staged
    `HHEMT_SHA`. But it refuses, and `_toolkit_git_sha()` then returns "unknown", when a
    `.git` sits under the toolkit root inside a container (a development checkout mounted
    over the image's source) or the staged `HHEMT_SHA` disagrees with the image label.
    When the sha is unresolvable this falls back to the image's own `org.hhemt.hhemt_sha`
    label, read from `/.singularity.d/labels.json`, which the SIF recipe writes from the
    staged `HHEMT_SHA` at build time. The label names the image, not a checkout mounted
    over it, so edits in such a checkout are invisible to this key. Graceful-absent on
    BOTH paths: an unresolvable identity returns "unknown", which the chapter-set guard
    treats as NOT-COMPARED rather than as unequal.
    """
    stamp = producing_stamp()
    sha = str(stamp.get("hhemt_sha") or "").strip()
    dirty = str(stamp.get("hhemt_dirty") or "unknown")
    if sha and sha != "unknown":
        return sha, dirty
    try:
        from hhemt.container_labels import looks_like_sha, read_container_labels

        label = read_container_labels("/").hhemt_sha
    except Exception:
        label = None
    if looks_like_sha(label):
        # An image label identifies a BUILT artifact, which has no working tree, so
        # "false" is the honest dirty value rather than a carried-over one.
        return str(label).strip(), "false"
    return "unknown", dirty


def record_chapter_build(chapters_dir) -> bool:
    """Record this process's build in the chapter set's own provenance history.

    The recorded identity is `processing_build_key()` -- the same (sha, dirty) the
    chapter-set guard compares against and names as "this process" in its message --
    with the remaining fields of `producing_stamp()`. The two differ only when
    `producing_stamp()` cannot resolve the running build inside a container (for
    example a development checkout mounted over the image's source): the guard then
    names the build by the image's `org.hhemt.hhemt_sha` label, and this records that
    label rather than "unknown", so the history lists the build the message names and a
    later run can compare against it. Where neither resolves, both are "unknown" and the
    guard leaves the entry uncompared, as before.

    Delegates to `append_stage_provenance`, which already accepts an arbitrary
    directory, de-duplicates on (sha, dirty), preserves mtime on an unchanged
    build, and never raises.
    """
    sha, dirty = processing_build_key()
    stamp = {**producing_stamp(), "hhemt_sha": sha, "hhemt_dirty": dirty}
    return append_stage_provenance(Path(chapters_dir), _CHAPTER_STAGE, stamp=stamp)


def collect_chapter_build_keys(chapters_dir) -> list[tuple[str, str]]:
    """The (sha, dirty) keys recorded against this chapter set, oldest first."""
    entries = read_stage_provenance_history(Path(chapters_dir)).get(_CHAPTER_STAGE, [])
    return [
        (str(e.get("hhemt_sha") or "").strip(), str(e.get("hhemt_dirty") or "unknown"))
        for e in entries
        if str(e.get("hhemt_sha") or "").strip()
    ]


def assert_chapters_match_running_build(chapters_dir, *, declare_mixed_version_chapters: bool = False):
    """Refuse to extend a chapter set that a DIFFERENT processing build wrote.

    The invariant: every flagged chapter merged into one unified store was written
    by one build. Enforced at loop ENTRY, which is the only point at which a mixed
    merge is still preventable. By induction over invocations this holds for every
    chapter whose producing build was RECORDED -- it does NOT extend to a legacy set
    that predates the capture site, which is NOT-COMPARED by design (below). No
    second gate at merge time is needed for the recorded population.

    ABSENCE IS NOT-COMPARED, which INVERTS `StalePlotsError`'s absent-is-never-equal
    rule and does so deliberately. Figures are cheap to re-render, so failing loud on
    a missing stamp is right there. A chapter set is a partially-completed run whose
    loss is the failure this whole mechanism exists to prevent, and every set already
    on disk predates this capture site — so an absent-is-unequal reader would refuse
    every resume of every existing analysis on the day it lands.

    Comparison is by `shas_match` prefix in both directions, so an abbreviated sha
    validates against a full one rather than reading as a different commit.

    `prior` is the PROVENANCE of the flagged chapters, not a log of arrivals, because
    the build is recorded in `utils.verify_and_flag_chapter` -- immediately after the
    flag write and only when a chapter was actually flagged. Recording at this
    function's call site instead would enter builds that wrote zero chapters, and the
    `all(...)` below would then range over builds that contributed nothing.
    """
    from hhemt.container_labels import shas_match
    from hhemt.exceptions import ProcessingError
    from hhemt.utils import completed_chapters

    chapters_dir = Path(chapters_dir)
    n_flagged = len(completed_chapters(chapters_dir))
    if not n_flagged:
        return None
    prior = [k for k in collect_chapter_build_keys(chapters_dir) if k[0] != "unknown"]
    if not prior:
        return None
    mine = processing_build_key()
    if mine[0] == "unknown":
        return None
    if all(shas_match(k[0], mine[0]) and k[1] == mine[1] for k in prior):
        return None
    msg = (
        f"Chapter-set build mismatch under {chapters_dir}: {n_flagged} flagged chapter(s) "
        f"were written by {sorted({f'{s[:12]}(dirty={d})' for s, d in prior})} but this "
        f"process is {mine[0][:12]}(dirty={mine[1]}). Merging them would publish one store "
        "built by two processing builds. Either discard the partial chapter set with "
        'force_rerun {"subject": "all", "stage": "process"}, or -- if nothing material '
        "changed -- set allow_mixed_version_chapters: true in the analysis config to "
        "proceed anyway: the mismatch is then printed to standard output and, when the "
        "workflow runs this step, kept in that step's log file. Once this run flags a "
        "chapter, provenance_history.json in that directory lists the earlier build(s) "
        "named above and this one."
    )
    if declare_mixed_version_chapters:
        print(f"[Chunked Processing] DECLARED MIXED-VERSION CHAPTERS: {msg}", flush=True)
        return msg
    raise ProcessingError(
        operation="assert_chapters_match_running_build",
        filepath=str(chapters_dir),
        reason=msg,
    )


def _resolve_case_manifest(analysis):
    """CaseManifest for input File parts, or a minimal stand-in (empty manifest).

    `TRITONSWMM_analysis` sets `_case_manifest` from its `case_manifest_yaml` argument
    when that file is given and loads, and to None otherwise (no argument, a missing
    file, or a load or validation failure). On None this degrades to a minimal
    descriptor with zero input parts, so the crate stays valid.
    """
    cm = getattr(analysis, "_case_manifest", None)
    if cm is not None:
        return cm
    return SimpleNamespace(
        case_name=str(analysis.cfg_analysis.analysis_id),
        description="",
        manifest={},
    )


def _input_parts_from_case(cfg_case) -> list[dict]:
    return [
        {"@id": fname, "sha256": hexsha, "contentSize": None, "encodingFormat": None}
        for fname, hexsha in getattr(cfg_case, "manifest", {}).items()
    ]


def _iter_run_units(analysis):
    """Yield (member_id, event_id, model_type) per real invocation unit.

    Regular analysis: member_id is "" ; one unit per (event_iloc, enabled model_type).
    """
    # A member analysis is identified by its OWN config, never by an `sa_id` attribute:
    # `grep -rnE "\.(sa_id|member_id)\s*=\s*[^=]"` over src/ returns nothing, so the prior
    # `getattr(analysis, "sa_id", "")` ALWAYS took its "" default. That is correct for a regular
    # analysis (documented above) but silently collapsed every member to the same empty segment,
    # so `#run-{member_id}-{event}-{model}` collided across members inside one crate.
    # `analysis_id` is used WHOLE rather than stripped of `member_prefix`: that prefix is an
    # instance attribute set in `TRITONSWMM_sensitivity_analysis`'s constructor, not a module
    # constant, and duplicating the literal here would be a second copy that can drift. Nothing
    # parses these ids -- the only other `#run-` occurrence in the tree is a hand-built test
    # fixture -- so uniqueness is the whole requirement.
    member_id = (
        str(analysis.cfg_analysis.analysis_id) if getattr(analysis.cfg_analysis, "is_experiment_member", False) else ""
    )
    enabled = analysis._get_enabled_model_types()  # reads the system config's toggle_*_model fields
    for event_iloc in analysis.df_sims.index:
        for model_type in enabled:
            yield (member_id, str(event_iloc), model_type)


def _output_ids(analysis, member_id, event_id, model_type) -> list[str]:
    """Per-output @ids for one run unit, derived from the per-model processing_log."""
    from hhemt.scenario import TRITONSWMM_scenario

    scen = TRITONSWMM_scenario(int(event_id), analysis)
    mlog = scen.get_log(model_type)
    outs = getattr(mlog.processing_log, "outputs", {}) or {}
    # NOTE: the canonical event-id slug lives on the scenario itself (`scen.event_id`,
    # which `TRITONSWMM_scenario`'s constructor sets from `sim_id_str`), NOT on
    # `scen.scen_paths` (`ScenarioPaths` carries no event_id field).
    return [f"sims/{scen.event_id}/processed/{name}" for name in sorted(outs)]


def _sif_spec_from_system_log(analysis) -> dict | None:
    """Build the crate's by-reference SIF entity from the digest captured at SETUP.

    THE GAP THIS CLOSES. `metadata.build_analysis_crate` has carried a complete SIF
    entity — `{@id, softwareVersion, sha256, downloadUrl}` — and `sif_spec` has been a
    wired-through parameter, but NO production caller ever passed a non-None value: both
    `processing_analysis` and `sensitivity_analysis` omit it. So the whole downstream
    chain was dormant. `_reprex._verify_sif` is genuinely FAIL-CLOSED (it raises on a
    digest mismatch) and never executed, because `_find_sif_entity` selects on an entity
    nobody emitted. A fail-closed check that never runs reads as protective and is not;
    this is the one wire that makes it run.

    Mirrors `processing_analysis._stamp_triton_provenance`: read the system log, refresh
    it first because setup and consolidation are different processes on HPC, and be
    graceful-absent throughout. A native run, a sandbox container, a pre-fix toolkit, or a
    failed digest read all yield None — and None means NO SIF ENTITY, which every consumer
    already handles.

    NO `downloadUrl`. There is no deposit target, so there is no URL to record, and
    inventing one would be worse than omitting it. The digest still does real work without
    it: a reproducer who obtains the SIF by ANY route — a manual copy between systems, a
    colleague's copy, a future deposit — can verify it is the right image. Without the
    entity there is nothing to verify it against.
    """
    _sys_log = getattr(getattr(analysis, "_system", None), "log", None)
    if _sys_log is None:
        return None
    try:
        _sys_log.refresh()  # pick up the setup-process write in the cross-process case
    except Exception:
        pass
    try:
        digest = _sys_log.sif_sha256.get()
    except Exception:
        return None
    if not digest:
        return None
    return {"@id": f"#sif-{str(digest)[:12]}", "sha256": str(digest)}


def _agent_id(node: str | None) -> str:
    return f"#agent-{node or socket.gethostname()}"


def emit_provenance(
    analysis,
    *,
    sif_spec=None,
    code_repository: str | None = None,
    consolidated_zarr_relpath: str = "analysis_datatree.zarr",
    sub_dataset_relpaths=None,
    with_run_units: bool = True,
    emitted_vars: set[str] | None = None,
    # THE INTERMEDIATE HOP. The thread is caller -> emit_provenance -> build_analysis_crate, so
    # store-sourcing the crate's descriptors changes TWO signatures, not one; this function
    # carries the mapping through and reads none of it. A precomputed {name: attrs} mapping is
    # the smaller of the two possible contract changes and keeps `metadata.py` free of xarray.
    emitted_attrs: dict[str, dict[str, str | None]] | None = None,
) -> tuple[str, str]:
    """Build the analysis crate + render the per-run CreateAction graph from log.py.

    Returns (embedded_core_jsonld, sidecar_jsonld). The caller writes the first to
    tree.attrs["ro_crate_metadata"] (via cf_conventions.apply_provenance_core) and the
    second to analysis_dir/ro-crate-metadata.json (via metadata.write_rocrate_sidecar).
    log.py is the SOURCE-OF-RECORD; this reads it and NEVER writes it.
    """
    code_repository = (
        code_repository or _default_code_repository()
    )  # resolve at call time (throw-on-absence; no import-time read)
    cfg_case = _resolve_case_manifest(analysis)
    input_parts = _input_parts_from_case(cfg_case)
    if sif_spec is None:
        sif_spec = _sif_spec_from_system_log(analysis)
    alog = analysis.log  # TRITONSWMM_analysis_log (read-only)
    crate = build_analysis_crate(
        analysis_id=str(analysis.cfg_analysis.analysis_id),
        system_id=None,
        layout_version=LAYOUT_VERSION,
        toolkit_git_sha=_toolkit_git_sha(),
        code_repository=code_repository,
        cfg_case=cfg_case,
        # Resolved from BOTH license sources, never through str(): an unlicensed analysis
        # carries None, and str(None) is the text "None", which the license table does not
        # contain (KeyError at consolidation). An SPDX id resolves to itself, so a table
        # member's crate is byte-identical to before the details-file form existed; a
        # details file resolves to a record the crate writes as its own CreativeWork entity.
        # getattr rather than a bare read because this is handed config-like objects that
        # may predate the details-file field (test doubles built from a SimpleNamespace).
        dataset_license=resolve_dataset_license(
            dataset_license=analysis.cfg_analysis.dataset_license,
            dataset_license_file=getattr(analysis.cfg_analysis, "dataset_license_file", None),
        ),
        sif_spec=sif_spec,
        consolidated_zarr_relpath=consolidated_zarr_relpath,
        input_parts=input_parts,
        sub_dataset_relpaths=sub_dataset_relpaths,
        emitted_vars=emitted_vars,
        emitted_attrs=emitted_attrs,
    )

    for member_id, event_id, model_type in _iter_run_units(analysis) if with_run_units else ():
        out_ids = _output_ids(analysis, member_id, event_id, model_type)
        action = crate.add(
            ContextEntity(
                crate,
                f"#run-{member_id}-{event_id}-{model_type}",
                properties={
                    "@type": "CreateAction",
                    "name": f"TRITON-SWMM run {event_id} ({model_type})",
                    "instrument": [{"@id": "#hhemt-app"}] + ([{"@id": sif_spec["@id"]}] if sif_spec else []),
                    "object": [{"@id": p["@id"]} for p in input_parts],
                    "result": [{"@id": oid} for oid in out_ids],
                    # VOLATILE — present only in the sidecar full graph; stripped from the core by partition:
                    "startTime": alog.workflow_submission_time.get(),
                    "agent": {"@id": _agent_id(alog.workflow_submission_node.get())},
                },
            )
        )
        for oid in out_ids:
            node = crate.dereference(oid)  # ro-crate-py returns None (NOT KeyError) for an absent @id
            if node is not None:
                node["wasGeneratedBy"] = {"@id": action.id}
            # else: output not yet a graph node (no summary) — skip the inverse edge

    full_doc = crate.metadata.generate()
    sidecar = canonical_jsonld(crate)  # full graph (incl. volatile)
    core = canonical_jsonld_from_doc(partition_core_vs_sidecar(full_doc))  # deterministic subset
    return core, sidecar
