"""The experiment-root store field is a WRITE path; readers use the resolver.

`analysis.py` binds `AnalysisPaths.sensitivity_datatree_zarr` to the literal unified
filename, and `sensitivity_analysis.consolidate_sensitivity_datatree` writes THROUGH
that field. A reader that takes the same field is pinned to the unified name and fails
on a tree that legitimately still carries the retired one -- which is the defect this
guard exists to stop recurring.

The asymmetry is deliberate and it is NOT safe to remove by making the field itself
tolerant: reader and writer share the field, so a tolerant field would make the writer
re-create the retired name and the rename would never finish.

This guard is static and reads no tree. It is not a substitute for the resolver's own
behaviour coverage, which lives in `test_experiment_tree_resolution.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

FIELD = "sensitivity_datatree_zarr"

#: KEYED ON THE RELATIVE PATH, NEVER THE BASENAME. ``src/hhemt/config/analysis.py``
#: exists alongside ``src/hhemt/analysis.py``, so a basename key would hand the config
#: module this exemption for free. Nothing names the field there today, which makes the
#: basename form UNFALSIFIABLE at that path rather than wrong -- the worse of the two,
#: because it reports green from a check that cannot fire.
#:
#: THE SET IS PARTITIONED because a single ``{path: reason}`` dict cannot distinguish a
#: PERMANENT exemption from a TEMPORARY one. This repo has already paid for that: the
#: ``non_breaking_allowlist`` in ``_layout_relevant_files.yaml`` pins each entry to a
#: ``layout_signature`` content hash precisely so an exemption is CHANGE-SCOPED rather
#: than path-permanent. The partition below is that discipline in the form this subject
#: allows -- a pinned count and a dated reason instead of a hash.

#: Permanent by design: the write path and the two sites that define it.
_STRUCTURAL: dict[str, str] = {
    "paths.py": "declares the dataclass field",
    "analysis.py": "the single binding site",
    "sensitivity_analysis.py": "the WRITE path and its configured-ness guard",
}

#: TEMPORARY: clause-1 readers not yet converted. Each reason MUST be dated so a later
#: reader can age it out. Remove an entry when its file stops naming the field.
_RESIDUAL: dict[str, str] = {
    "recompute.py": "2026-09-08: clause-1 reader; converts with the follow-up",
    "analysis_validation.py": "2026-09-08: clause-1 reader; converts with the follow-up",
    "report_renderers/metadata.py": "2026-09-08: clause-1 reader; converts with the follow-up",
}

#: EXACT, never a ceiling. A ``<=`` bound would permit silent growth up to the bound,
#: which is the very shape this partition exists to stop; ``==`` makes admitting a new
#: reader FAIL and forces a deliberate act.
_RESIDUAL_COUNT = 3

_ALLOWED: dict[str, str] = {**_STRUCTURAL, **_RESIDUAL}

_DATED_REASON = re.compile(r"^\d{4}-\d{2}-\d{2}: ")


def _src_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "hhemt"


def test_field_is_named_only_where_the_write_path_needs_it():
    """Any other src/ file naming the field is a reader that should use the resolver."""
    offenders: list[str] = []
    for path in sorted(_src_root().rglob("*.py")):
        if FIELD not in path.read_text(encoding="utf-8"):
            continue
        rel = str(path.relative_to(_src_root()))
        if rel not in _ALLOWED:
            offenders.append(rel)
    assert not offenders, (
        f"{len(offenders)} src/hhemt file(s) name {FIELD!r} outside the write path: "
        f"{offenders}. A reader must resolve the store with "
        "hhemt.utils.resolve_experiment_tree(analysis_dir), which accepts a tree that "
        "still carries the retired name. Only the write path may name the field, "
        "because a tolerant write would re-create the retired name."
    )


def test_the_residual_exemptions_are_change_scoped():
    """The residual set is pinned, dated, and disjoint from the structural set.

    Without these three the partition is decoration: an unpinned count readmits silent
    growth, an undated reason cannot be aged out by anyone but its author, and a path in
    both sets is laundered from temporary to permanent by duplication.
    """
    findings: list[str] = []
    if len(_RESIDUAL) != _RESIDUAL_COUNT:
        findings.append(
            f"residual count {len(_RESIDUAL)} != pinned {_RESIDUAL_COUNT}; converting a "
            "reader means REMOVING its entry and lowering the pin, and admitting one "
            "means saying so here"
        )
    undated = sorted(rel for rel, why in _RESIDUAL.items() if not _DATED_REASON.match(why))
    if undated:
        findings.append(f"residual reason(s) missing a leading 'YYYY-MM-DD: ': {undated}")
    both = sorted(set(_STRUCTURAL) & set(_RESIDUAL))
    if both:
        findings.append(f"path(s) in BOTH _STRUCTURAL and _RESIDUAL: {both}")
    assert not findings, findings


def test_the_allowlist_is_not_stale():
    """Every allowlisted path still NAMES the field, so the set cannot rot upward.

    Existence is the wrong predicate and was the first form of this test: a file that
    still exists but no longer names the field leaves a permanent exemption for a site
    that no longer needs one, and an existence check reports it clean. The predicate
    must be the one the exemption is FOR.
    """
    stale: list[str] = []
    for rel in sorted(_ALLOWED):
        candidate = _src_root() / rel
        if not candidate.is_file() or FIELD not in candidate.read_text(encoding="utf-8"):
            stale.append(rel)
    assert not stale, (
        f"allowlisted path(s) no longer name {FIELD!r}: {stale}. Remove the entry -- an "
        "exemption for a site that no longer needs one is an exemption nothing polices."
    )
