"""Dual-write provenance emitter (C2) for the reproducibility-system (ADR-7).

Reads log.py READ-ONLY at consolidation and renders a per-run PROV CreateAction +
per-output wasGeneratedBy graph. NEVER mutates log.py (the _already_written
completion-gate is load-bearing across the DAG — Gotchas 28/34/40).
"""

from __future__ import annotations

import importlib.metadata
import socket
from pathlib import Path  # module-scope: the `"Path"` parameter annotations below are

# type-only (this module sets `from __future__ import annotations`), but ruff resolves
# annotation names against module scope and reported F821 -- which is inside CI's gating
# `--select=E9,F63,F7,F82` set. Function bodies keep their local `_Path` alias unchanged.
from types import SimpleNamespace

from rocrate.model.contextentity import ContextEntity

from hhemt.constants import LAYOUT_VERSION
from hhemt.exceptions import StalePlotsError
from hhemt.metadata import (
    build_analysis_crate,
    canonical_jsonld,
    canonical_jsonld_from_doc,
    partition_core_vs_sidecar,
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

    `pyproject.toml` carries a STATIC `version = "0.1.0"` that has not moved in 241
    commits, so `importlib.metadata.version("hhemt")` is a constant that distinguishes
    nothing — it was identical on all four arms of the delivered generation. This
    derives `{tag}+{N}.g{sha}` from `git describe --tags --long`, which DOES
    distinguish generations and is deterministically recomputable from an archived
    sha alone (verified: `git describe --tags --long 01655abb60c2` -> `v0.1.0-241-g01655ab`).

    Falls back to the installed metadata version when git is unavailable — a wheel
    install is the intended fallback case, not a failure. NEVER raises: a provenance
    minter that can abort a 3-hour consolidation is worse than one that degrades.
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
        try:
            return importlib.metadata.version("hhemt")
        except Exception:
            return "0+unknown"


def _is_dirty() -> bool:
    """True when the toolkit checkout has uncommitted changes at mint time.

    This closes a hole the codebase currently states but does not enforce:
    `process_simulation._resolve_producing_stamp`'s docstring claims the sha is
    `"unknown"` when the working tree is dirty, but `_get_toolkit_git_sha` only
    degrades on CalledProcessError/FileNotFoundError — `git rev-parse HEAD` succeeds
    on a dirty tree, so a run with uncommitted edits stamps a CLEAN sha and reads
    authoritative. The only dirty check in the toolkit today is `_carry_source_tree`'s
    emit-time warning, which fires long after capture. False on any error: an
    undeterminable tree must not be asserted dirty.
    """
    import subprocess

    from hhemt.bundle._emit import _toolkit_source_dir

    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=_toolkit_source_dir(),
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return False


def producing_stamp() -> dict[str, str]:
    """THE single minter for every stage's hhemt version-provenance stamp (ADR-15 widening).

    Every stage that writes an artifact calls THIS, at write time, from the running
    process. Three fields, one shape, six carriers. The minting rule this enforces:
    a stamp records the code that ACTUALLY ran the stage, so it may never be resolved
    from a module-level constant, a config field, or a value carried over from an
    earlier stage. EW-3 is the counter-example — a field stamped from a constant that
    had no value yet at run time, which read authoritative and was wrong.

    Corollary the CALL SITES must honor and this function cannot: if the stage did not
    execute, do not call this. An unconditional stamp cannot distinguish
    "re-ran at this sha" from "skipped, carrying an older artifact".
    """
    return {
        "hhemt_sha": _toolkit_git_sha(),
        "hhemt_version": _describe_version(),
        "hhemt_dirty": "true" if _is_dirty() else "false",
    }


#: The append-only per-stage provenance history, relative to analysis_dir. Declared ONCE
#: here and referenced by no other literal, so a rename is a single edit. Deliberately NOT
#: one of the consolidated-tree names: this artifact is orthogonal to the zarr layout and
#: must not acquire a dependency on a filename that is a live rename candidate.
_HISTORY_FILENAME = "provenance_history.json"


def append_stage_provenance(analysis_dir: Path, stage: str) -> bool:
    """Append this stage's producing stamp to the analysis's append-only history.

    Contract property 3 ("appends, never overwrites"). Returns True if an entry was
    appended, False if the stage's latest entry already names this exact build.

    WHY THIS IS NOT IN THE RO-CRATE SIDECAR, stated here because the sidecar is the
    intuitive home and is the wrong one. Both `emit_provenance` call sites and both
    `write_rocrate_sidecar` call sites are inside CONSOLIDATION. Every stage this history
    is about -- plots, report, bundle, combine -- runs strictly AFTER consolidation, so a
    history projected into the sidecar could not contain the render that has not happened
    yet: on a single pass it would record zero render entries, and on the two-renders case
    this exists for, neither. A standalone read-model written by each stage at its own
    capture site is the same persist-then-read shape `validation_report.json` already uses,
    and for the same ordering reason.

    R4 (Gotcha 59) is preserved BY CONSTRUCTION rather than by keying: this function never
    touches `ro-crate-metadata.json`, so the sidecar's compare-and-write, `_EMBEDDED_PROV_KEYS`,
    and the byte-identity goldens are all untouched. The de-duplication below is what keeps
    THIS file idempotent -- an unchanged build appends nothing and the file's bytes and mtime
    are preserved, so the Gotcha-38 analysis-scope DU own-files walk is unperturbed too.

    The key is `(hhemt_sha, hhemt_dirty)`, not the sha alone: two builds at one commit with
    different working-tree states are different producers, which `producing_stamp` already
    models. NO timestamp is recorded -- an emit-time clock read would make every invocation
    append and would defeat the idempotence this contract depends on. Ordering IS the
    history; wall-clock is not needed to read it.

    Best-effort and never raises: a provenance write must not fail a stage that succeeded.

    Annotations are UNQUOTED deliberately, unlike the two older `analysis_dir: "Path"`
    signatures below. This module sets `from __future__ import annotations` (line 8) and
    imports `Path` at module scope (line 12), so the quotes buy nothing and ruff flags them
    UP037 -- the older pair are pre-existing findings, and matching their style would have
    propagated the debt into new code rather than merely inheriting it.
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
        stamp = producing_stamp()
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


def collect_plot_stamps(analysis_dir: Path) -> tuple[set[tuple[str, str]], int]:
    """Distinct (sha, dirty) keys across every figure sidecar, and the sidecar count.

    Reads ALL sidecars rather than one, because uniform-within-a-render is a measured
    property and not a guaranteed one: a targeted re-render refreshes some sidecars and
    leaves others, which yields two distinct keys and is its own staleness class
    (figures inconsistent with EACH OTHER, remedied by a full stage:render force rather
    than by re-rendering the report). An unreadable or unstamped sidecar contributes no
    key and is counted, so `keys == set()` with `n > 0` means "figures present, none
    stamped" -- absent, which the caller treats as never-equal.
    """
    import json
    from pathlib import Path as _Path

    keys: set[tuple[str, str]] = set()
    n = 0
    plots = _Path(analysis_dir) / "plots"
    if not plots.exists():
        return keys, 0
    for sidecar in sorted(plots.rglob("*.manifest.json")):
        n += 1
        try:
            payload = json.loads(sidecar.read_text())
        except Exception:
            continue
        sha = str(payload.get("hhemt_sha") or "").strip()
        if sha:
            keys.add((sha, str(payload.get("hhemt_dirty") or "unknown")))
    return keys, n


def assert_plots_match_running_build(analysis_dir: Path, *, declare_stale_plots: bool = False):
    """Refuse a render whose figures were not produced by the build now running.

    The report-side operand is `producing_stamp()` computed IN-PROCESS, never a
    persisted report stamp: reading `report_manifest.json` would compare a render
    against itself (the writer runs at the tail of the same call), would refuse every
    first render for want of a prior stamp, and would depend on a writer that fails
    non-fatally. Equality on the (sha, dirty) tuple, never an ordering.
    """
    keys, n = collect_plot_stamps(analysis_dir)
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
        '{"subject": "all", "stage": "render"}, or pass --declare-stale-plots to proceed '
        "and record the mismatch."
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


def store_build_mismatch(store: Path) -> str | None:
    """Return a reason string when `store` was NOT produced by the running build, else None.

    The consolidation-tier sibling of `assert_plots_match_running_build` above, and the
    fifth instance of that pattern. Three properties are inherited deliberately and a
    reviewer must not soften any of them:

    ABSENT IS NEVER EQUAL, including absent on both sides -- and ABSENCE INCLUDES A
    SENTINEL. A store with no `hhemt_producing_sha` is a MISMATCH, and so is one whose
    sha is `"unknown"`, because that is what `_toolkit_git_sha` returns when it resolved
    nothing. A falsy-check does not catch it: `"unknown"` is truthy, so two different
    wheel installs both stamping `"unknown"` would compare equal and reuse each other's
    store with no output at all. Reading absence as "no objection" would silently reuse
    every tree built before the ADR-15 capture site existed, which is precisely the
    population most likely to be stale. This is the one property that distinguishes this
    function from `check_provenance_completeness`, whose graceful-absent posture is
    correct for a REPORT (a provenance check that can abort validate_analysis takes the
    whole Errors-and-Warnings sidebar down with it) and wrong for a GATE.

    A DIRTY RUNNING CHECKOUT IS A MISMATCH WHEN THE SHAS OTHERWISE AGREE, and this arm
    is deliberate rather than incidental. `git rev-parse` succeeds on a dirty tree and
    returns the COMMITTED sha (`_is_dirty`'s own docstring says so), so a developer
    editing a consolidation module and re-running gets an unchanged sha -- which is
    exactly the workflow this gate exists to serve, and exactly the change class that
    caused the incident it was written for. When the shas DIFFER the store already
    rebuilds and dirty adds nothing; when they AGREE the sha check says "reuse" and
    dirty is precisely the condition under which that verdict cannot be trusted. So the
    arm fires only in the second case.

    THE COST, AND THE PROPERTY THAT COST BUYS: while the checkout is dirty this gate
    DOES NOT CONVERGE -- a rebuild re-stamps the same committed sha against the same
    dirty tree, so the next consolidation rebuilds again. That is intended. What it
    costs is the CONSOLIDATION tier only: `process_*` sits upstream of this gate and is
    not re-triggered by it (measured on the synth sensitivity tree: 28.9 s to rebuild
    every consolidate rule, against 274 s of process rules the gate never touches).

    AND THAT COST SCALES LINEARLY IN MEMBER COUNT, which is the part a reader at
    ensemble scale will otherwise be surprised by. One `consolidate_member_*` rule runs
    per member, so the measured figures are per-tree and not per-analysis: 28.9 s at four
    synth members, ~60 s at Norfolk case-study scale, and near EIGHT MINUTES at a
    fifty-member production ensemble. Seconds at development scale and minutes at
    ensemble scale is the same non-convergence priced very differently, and an operator
    who knows their edits cannot affect consolidation arms
    HHEMT_DECLARE_STALE_BUILD=1, which prints and reuses.

    THE RUNNING VALUE IS MINTED, NEVER READ BACK. `producing_stamp()` is called inline
    here, per its own documented rule that a stamp "may never be resolved from a
    module-level constant, a config field, or a value carried over from an earlier stage."
    The persisted `validation_report.json` is exactly such a carried-over value: it is
    written AFTER consolidation completes (consolidate_workflow.py, immediately before
    `_emit_runner_flag`) and by whichever of three call sites ran last
    (consolidate_workflow, export_scenario_status, analysis.eda), so at the moment this
    gate runs it is a receipt of indeterminate authorship about a PRIOR run.

    THE ESCAPE PRINTS AND RECORDS. `HHEMT_DECLARE_STALE_BUILD=1` returns None so the
    caller reuses the store, but only after emitting the reason to stdout. A silent
    bypass reintroduces the failure with a flag on it.

    Reads the store's root attributes directly rather than through `xr.open_datatree`:
    the gate runs on every consolidation and a JSON read of one file is the cheap form.
    Both zarr layouts are handled because a pre-V0021 store may be v2.
    """
    import json
    import os

    def _root_attrs() -> dict:
        try:
            meta = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
            return meta.get("attributes") or {}
        except Exception:
            pass
        try:
            return json.loads((store / ".zattrs").read_text(encoding="utf-8"))
        except Exception:
            return {}

    if not store.exists():
        # Inert by construction: every branch that consumes this value also requires
        # `fname_out.exists()`, so a mismatch reported here can never trigger a rebuild
        # of a store that is not there. Reported as a mismatch anyway, because
        # absent-is-never-equal is the rule and a silent None here would be the one
        # place the rule is quietly suspended.
        why = "no consolidated store present at the reuse gate"
    else:
        stored = str(_root_attrs().get("hhemt_producing_sha") or "").strip()
        _stamp = producing_stamp()
        running = str(_stamp.get("hhemt_sha") or "").strip()
        running_dirty = str(_stamp.get("hhemt_dirty") or "").strip().lower() == "true"
        # Order matters: the two sentinel arms come FIRST so a sentinel can never reach
        # the equality test, which is the whole of the AB-1 fold.
        if stored in _SENTINEL_SHAS:
            why = (
                f"the store's hhemt_producing_sha is the sentinel {stored!r} -- it names no "
                "commit, so the store cannot be shown to match this build"
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
        f"{_DECLARE_STALE_BUILD_ENV}=1 to reuse it and record the mismatch."
    )
    if os.environ.get(_DECLARE_STALE_BUILD_ENV) == "1":
        print(f"[consolidate] DECLARED STALE BUILD: {msg}", flush=True)
        return None
    return msg


#: Stage key for the per-chapter-set producing-build history. Declared ONCE here and
#: referenced by no other literal, exactly as _HISTORY_FILENAME above is.
_CHAPTER_STAGE = "chapters"


def processing_build_key() -> tuple[str, str]:
    """The (sha, dirty) identity of the build running the PROCESSING stage.

    `producing_stamp()` alone is not sufficient in container mode: the SIF build
    deletes the toolkit's `.git` (containers/uva-cpu.def), so an in-image
    `_toolkit_git_sha()` returns "unknown" for every SIF and two different images
    stamp identically. When the git sha is unresolvable this falls back to the
    image's own `org.hhemt.hhemt_sha` label, which build_sifs_uva.sh stamps
    host-side. Graceful-absent on BOTH paths: an unresolvable identity returns
    "unknown", which every caller treats as NOT-COMPARED rather than as unequal.
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

    Delegates to `append_stage_provenance`, which already accepts an arbitrary
    directory, de-duplicates on (sha, dirty), preserves mtime on an unchanged
    build, and never raises.
    """
    return append_stage_provenance(Path(chapters_dir), _CHAPTER_STAGE)


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
        "proceed and record the mixed provenance."
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

    The analysis object does not currently carry a CaseManifest; a thin accessor
    (`analysis._case_manifest`) is the narrow wiring point. Until it lands, degrade
    to a minimal descriptor with zero input parts — the crate stays valid.
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
    # instance attribute on the sensitivity object (`sensitivity_analysis.py:280`), not a module
    # constant, and duplicating the literal here would be a second copy that can drift. Nothing
    # parses these ids -- the only other `#run-` occurrence in the tree is a hand-built test
    # fixture -- so uniqueness is the whole requirement.
    member_id = (
        str(analysis.cfg_analysis.analysis_id) if getattr(analysis.cfg_analysis, "is_experiment_member", False) else ""
    )
    enabled = analysis._get_enabled_model_types()  # encapsulates self._system.cfg_system.toggle_* (analysis.py:1431)
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
    # scenario.py:60 — `self.event_id = self.sim_id_str`), NOT on `scen.scen_paths`
    # (ScenarioPaths, paths.py:83, carries no event_id). Verified at Phase-2 preflight.
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
    it: a reproducer who obtains the SIF by ANY route — the documented ADR-2 manual
    transfer, a colleague's copy, a future deposit — can now verify it is the right image.
    Today they cannot, by any means.
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
        dataset_license=str(analysis.cfg_analysis.dataset_license),
        sif_spec=sif_spec,
        consolidated_zarr_relpath=consolidated_zarr_relpath,
        input_parts=input_parts,
        sub_dataset_relpaths=sub_dataset_relpaths,
        emitted_vars=emitted_vars,
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
