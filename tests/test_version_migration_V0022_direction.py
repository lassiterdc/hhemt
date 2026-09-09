"""V0022's direction decision: promote by producer EVIDENCE, fall back to position.

A separate module from `test_version_migration_V0022.py` on purpose. That module's
`third_state` fixture constructs its retired-name store as a copytree of the unified one
and labels the halves with a synthetic `provenance_probe`, so it ENCODES the premise that
whatever sits at the retired path is newer. It is left untouched as a characterization of
the NO-EVIDENCE case. This module supplies the case it cannot express: two stores whose
own `hhemt_producing_version` attributes order them, in both directions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt.version_migration.exceptions import MigrationBlockedError
from hhemt.version_migration.versions.V0022__promote_producer_written_experiment_tree import (
    _producer_generation,
    _vocabulary_inverted,
    upgrade,
)

RETIRED = "sensitivity_datatree.zarr"
UNIFIED = "experiment_datatree.zarr"
SUPERSEDED = "experiment_datatree.zarr.superseded-v0022"


class _Ctx:
    """Minimal MigrationContext stand-in: records planned ops, executes none.

    `plan` emptiness on the refusing arm is an implementation detail, NOT a safety property:
    `MigrationContext` only appends, and `runner.py` skips both the plan print and
    `ctx.execute()` when `upgrade()` raises, so nothing reaches disk from any position.
    """

    def __init__(self, target_dir: Path) -> None:
        self.target_dir = target_dir
        self.dry_run = True
        self.plan: list[str] = []

    def zarr_set_attrs(self, *a, **k) -> None:
        self.plan.append("zarr_set_attrs")

    def rewrite_text_preserving_mtime(self, *a, **k) -> None:
        """`_plan_sidecar` reaches this whenever a sidecar exists that `_repoint` changes.
        Omitting it made the refusing arm pass for a reason unrelated to what it tests --
        the fixture simply wrote no sidecar. Demonstrated as an AttributeError."""
        self.plan.append("rewrite_text_preserving_mtime")

    def move_dir(self, *a, **k) -> None:
        self.plan.append("move_dir")

    def record_applied(self, *a, **k) -> None:
        self.plan.append("record_applied")


def _store(root: Path, name: str, describe: str | None, group: str) -> Path:
    """A skeleton store: root attrs plus one named group. No chunks -- every predicate on
    this surface reads names, is_dir(), and zarr.json, so a skeleton exercises all of it."""
    d = root / name
    (d / group).mkdir(parents=True)
    attrs: dict = {"analysis_id": "synth"}
    if describe is not None:
        attrs["hhemt_producing_version"] = describe
    (d / "zarr.json").write_text(json.dumps({"attributes": attrs}), encoding="utf-8")
    return d


def _sidecar(root: Path) -> None:
    """A repointable ro-crate sidecar, so `_plan_sidecar` actually reaches
    `rewrite_text_preserving_mtime` and the stand-in's fidelity is exercised rather than
    insulated by an absent file."""
    doc = {"@graph": [{"@id": "sensitivity_datatree.zarr/", "@type": "Dataset"}]}
    (root / "ro-crate-metadata.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


def test_vocabulary_inversion_is_refused_not_promoted(tmp_path: Path) -> None:
    """THE LOAD-BEARING ARM, and the shape a measured on-disk tree actually had: the
    retired-name store carries `sa_*` groups and the unified-name store carries `member_*`.
    Promoting the retired store regresses the V0019 vocabulary. This is the DECIDABLE case."""
    work = tmp_path / "inverted"
    work.mkdir()
    _store(work, RETIRED, "0.1.0+472.gaaaaaaaaaaaa", "sa_0")
    _store(work, UNIFIED, "0.1.0+584.gbbbbbbbbbbbb", "member_0")
    _sidecar(work)
    ctx = _Ctx(work)

    with pytest.raises(MigrationBlockedError) as exc:
        upgrade(ctx)

    assert "REGRESS the member vocabulary" in str(exc.value)
    assert "by hand" not in str(exc.value), "a refusal must never instruct a manual deletion"
    assert ctx.plan == [], "no op is planned on the refusing arm (detail, not the safety property)"
    assert not (work / SUPERSEDED).exists()
    assert (work / UNIFIED / "member_0").is_dir(), "the member-keyed store stays where it was"


def test_matching_vocabulary_still_promotes(tmp_path: Path) -> None:
    """The gate must not block the legitimate third state it was written to allow, and must
    not block it merely because the counts happen to invert."""
    work = tmp_path / "legitimate"
    work.mkdir()
    _store(work, RETIRED, "0.1.0+90.gaaaaaaaaaaaa", "member_0")
    _store(work, UNIFIED, "0.1.0+101.gbbbbbbbbbbbb", "member_0")
    ctx = _Ctx(work)

    upgrade(ctx)

    assert "move_dir" in ctx.plan, "an inverted COUNT must not refuse; only vocabulary refuses"
    assert "record_applied" in ctx.plan


def test_sidecar_repoint_reaches_the_context_method(tmp_path: Path) -> None:
    """Regression pin on the stand-in's fidelity: `_plan_sidecar` calls
    `rewrite_text_preserving_mtime`, and a `_Ctx` lacking it fails with AttributeError rather
    than passing for the wrong reason."""
    work = tmp_path / "sidecar"
    work.mkdir()
    _store(work, RETIRED, None, "member_0")
    _store(work, UNIFIED, None, "member_0")
    _sidecar(work)
    ctx = _Ctx(work)

    upgrade(ctx)

    assert "rewrite_text_preserving_mtime" in ctx.plan


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0.1.0+595.ge5d73d1012dd", ("0.1.0", 595)),
        ("pre-public-cut+92.gaaaaaaaaaaaa", ("pre-public-cut", 92)),
        ("0.1.0", None),
        ("0+unknown", None),
        ("", None),
    ],
    ids=["describe", "other-tag-base", "bare-version", "unknown-fallback", "empty"],
)
def test_generation_captures_the_tag_because_the_count_alone_is_not_an_ordering(tmp_path: Path, raw, expected) -> None:
    """The tag is captured because `_describe_version` resolves against the NEAREST REACHABLE
    tag and this repo carries three, so a count is a distance from a moving origin. Measured:
    two mainline commits 34 minutes apart carry `pre-public-cut+92` then `0.1.0+89`."""
    d = tmp_path / "s.zarr"
    d.mkdir()
    (d / "zarr.json").write_text(json.dumps({"attributes": {"hhemt_producing_version": raw}}))
    assert _producer_generation(d) == expected


def test_zarr_v2_zattrs_layout_is_read(tmp_path: Path) -> None:
    """A store predating the unification may be zarr v2."""
    d = tmp_path / "v2.zarr"
    d.mkdir()
    (d / ".zattrs").write_text(json.dumps({"hhemt_producing_version": "0.1.0+100.gaaaaaaaaaaaa"}))
    assert _producer_generation(d) == ("0.1.0", 100)


@pytest.mark.parametrize(
    "retired_groups, unified_groups, inverted",
    [
        (["sa_0"], ["member_0"], True),
        (["member_0"], ["member_0"], False),
        (["sa_0"], ["sa_0"], False),
        (["member_0"], ["sa_0"], False),
        (["parameters"], ["parameters"], False),
    ],
    ids=["sa-over-member", "both-current", "both-retired", "reverse", "neither"],
)
def test_vocabulary_predicate_fires_only_on_a_real_regression(
    tmp_path: Path, retired_groups, unified_groups, inverted
) -> None:
    work = tmp_path / "vocab"
    work.mkdir()
    r = _store(work, RETIRED, None, retired_groups[0])
    u = _store(work, UNIFIED, None, unified_groups[0])
    assert _vocabulary_inverted(r, u) is inverted
