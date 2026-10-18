"""Tests for the HPC example-profile currency gate (scripts/check_hpc_profile_currency.py).

The gate exists so that the copyable profile on docs/how-to/hpc-profile-setup.md cannot
fall behind the ``hpc_system_config`` model unnoticed. These tests pin both directions:
a profile that has drifted from the model fails, and a profile that matches the model
passes however its keys and partitions are arranged.

Every case that needs a correct profile derives it from the live page rather than from a
hand-written fixture, so a correct future change to the model and the page together does
not red a test here. Nothing in this module builds an analysis, compiles, or simulates.
"""

from __future__ import annotations

import copy
import textwrap

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, create_model

import scripts.check_hpc_profile_currency as gate  # repo root is on sys.path under pytest
from hhemt.config.hpc_system import ContainerSpec, PartitionSpec, hpc_system_config


def _live_profile() -> dict:
    text = (gate.REPO_ROOT / gate.PAGE_REL).read_text(encoding="utf-8")
    profile, findings = gate.extract_profile(text)
    assert profile is not None, findings
    return profile


def _codes(findings: list[gate.Finding]) -> set[tuple[str, str]]:
    return {(f.location, f.code) for f in findings}


def _page(*yaml_bodies: str) -> str:
    """A markdown page carrying the given yaml bodies as column-0 fenced blocks."""
    parts = ["# A page\n"]
    for body in yaml_bodies:
        parts.append("Some prose.\n\n```yaml\n" + body + "```\n")
    return "\n".join(parts)


def test_live_page_is_current(capsys) -> None:
    """Property: the page as committed shows every model field and loads through the model.

    Class: the live page itself, run through ``main`` exactly as CI or a maintainer would.
    Why required: this is the page readers copy; a missing key is a setting they never
    learn exists, and an unknown key is a load failure on their cluster.
    What kills it: deleting any field's line from the page's yaml block, or adding a field
    to ``hpc_system_config``, ``PartitionSpec`` or ``ContainerSpec`` without adding it to
    the page.
    Second correct implementation: a gate that compares the page against
    ``model_json_schema()`` property names instead of walking ``model_fields`` passes this
    the same way, because the two enumerate the same fields.
    """
    assert gate.main([]) == 0
    out = capsys.readouterr().out
    assert "0 findings" in out


@pytest.mark.parametrize(
    "drop",
    [
        ("top", "max_concurrent_jobs"),
        ("partition", "max_gpu"),
        ("container", "binds"),
    ],
    ids=["top-level", "partition-field", "container-field"],
)
def test_absent_model_field_fails(drop: tuple[str, str]) -> None:
    """Property: a model field the profile does not show is a finding naming its dotted path.

    Class: one field removed, at each of the three nesting positions the model has.
    Why required: an omitted field is the drift a model change produces silently.
    What kills it: a walk that stops at the top level (the partition and container cases
    then pass), or treating a partition field as covered by the model rather than by an entry.
    Second correct implementation: checking ``set(model_fields) - set(keys)`` per nesting
    level with explicit recursion instead of a shared walker passes this unchanged.
    """
    profile = _live_profile()
    where, name = drop
    if where == "top":
        del profile[name]
        expected = (name, "field-absent")
    elif where == "partition":
        for entry in profile["partitions"].values():
            entry.pop(name, None)
        expected = (f"partitions.*.{name}", "field-absent")
    else:
        del profile["container"][name]
        expected = (f"container.{name}", "field-absent")
    findings, _ = gate.check_profile(profile)
    assert expected in _codes(findings)


def test_misnamed_key_fails() -> None:
    """Property: a key the model does not have is a finding even when the real field is shown elsewhere.

    Class: a rename inside ONE partition, where the other partitions still carry the real name.
    Why required: the union rule (a partition field shown on any entry counts as shown)
    must not also hide a wrong key on another entry; that wrong key is a load failure.
    What kills it: reporting only absent fields, or relying on the absence half alone.
    Second correct implementation: running ``model_validate`` and reporting its
    ``extra_forbidden`` errors in place of the walk's own unknown-key pass passes this.
    """
    profile = _live_profile()
    first = next(iter(profile["partitions"]))
    entry = profile["partitions"][first]
    entry["max_runtime_minutes"] = entry.pop("max_runtime")
    findings, _ = gate.check_profile(profile)
    codes = {code for _, code in _codes(findings)}
    assert "key-not-in-model" in codes
    # Reported exactly once: the walk and validation both see the key, and printing every
    # rename twice is noise a reader learns to skim past.
    naming = [f for f in findings if "max_runtime_minutes" in f.location]
    assert len(naming) == 1, [f.render() for f in naming]


def test_unknown_key_where_the_walk_does_not_descend_is_reported() -> None:
    """Property: an unknown key is a finding even inside a field the coverage walk cannot enter.

    Class: a field typed as a union of two different specs, which has no single nested
    model for the walk to follow, carrying a key neither spec has.
    Why required: the key is a load failure on the reader's cluster. Validation sees it;
    if the gate discards validation's unknown-key errors on the assumption that the walk
    reported them, nothing reports it and the gate exits 0.
    What kills it: skipping every ``extra_forbidden`` validation error rather than only
    those whose location the walk already reported.
    Second correct implementation: reporting every ``extra_forbidden`` error and dropping
    the walk's own unknown-key pass instead -- each key is then reported once, by validation.
    """

    class _A(BaseModel):
        model_config = ConfigDict(extra="forbid")
        a: int | None = None

    class _B(BaseModel):
        model_config = ConfigDict(extra="forbid")
        b: int | None = None

    model = create_model("GrownProfile", __base__=hpc_system_config, either=(_A | _B | None, None))
    model.model_rebuild(_types_namespace={"ContainerSpec": ContainerSpec})
    profile = _live_profile()
    profile["either"] = {"a": 1, "zz": 3}
    findings, _ = gate.check_profile(profile, model=model)
    assert any(f.code == "key-not-in-model" and "zz" in f.location for f in findings), [f.render() for f in findings]


def test_wrong_value_type_fails() -> None:
    """Property: a profile whose keys are all right but whose value does not load is a finding.

    Class: a numeric cap given as prose.
    Why required: the page promises a profile that loads; a key check alone would pass this.
    What kills it: dropping the ``model_validate`` step.
    Second correct implementation: validating with ``hpc_system_config(**profile)`` instead
    of ``model_validate`` passes this.
    """
    profile = _live_profile()
    first = next(iter(profile["partitions"]))
    profile["partitions"][first]["max_runtime"] = "four days"
    findings, _ = gate.check_profile(profile)
    assert "profile-invalid" in {code for _, code in _codes(findings)}


def test_null_container_block_shows_no_container_fields() -> None:
    """Property: a nested block set to null shows none of the nested model's fields.

    Class: ``container: null``, which the model itself accepts.
    Why required: the key being present is not the same as its fields being shown; a
    reader copying ``container: null`` learns nothing about container settings.
    What kills it: counting a present nested key as covering its whole sub-model.
    Second correct implementation: substituting ``{}`` for a null nested block before
    walking it passes this.
    """
    profile = _live_profile()
    profile["container"] = None
    findings, _ = gate.check_profile(profile)
    absent = {loc for loc, code in _codes(findings) if code == "field-absent"}
    assert {f"container.{name}" for name in ContainerSpec.model_fields} <= absent


@pytest.mark.parametrize("nested", [False, True], ids=["top-level-field", "container-field"])
def test_field_added_to_model_fails_unchanged_page(nested: bool) -> None:
    """Property: the field set comes from the model, so a field added to it fails an unchanged page.

    Class: a model grown by one optional field, at the top level and inside ``container``,
    checked against the live page's profile.
    Why required: this is the staleness the gate exists for; the page is untouched when
    the model grows.
    What kills it: a gate that checks a written-down list of expected fields -- it passes an
    unchanged page here, which is exactly the case it is supposed to catch.
    Second correct implementation: deriving the field set from ``model_json_schema()``
    (with ``$defs`` followed for nested models) passes this.
    """
    if nested:
        grown_container = create_model("GrownContainer", __base__=ContainerSpec, new_flag=(bool, False))
        model = create_model("GrownProfile", __base__=hpc_system_config, container=(grown_container | None, None))
        expected = ("container.new_flag", "field-absent")
    else:
        model = create_model("GrownProfile", __base__=hpc_system_config, new_cap=(int | None, None))
        expected = ("new_cap", "field-absent")
    findings, _ = gate.check_profile(_live_profile(), model=model)
    assert expected in _codes(findings)


def test_differently_positioned_correct_profile_passes(tmp_path) -> None:
    """Property: a profile that shows every field passes regardless of key order and partition layout.

    Class: the live profile with its top-level keys reversed, its partitions renamed and
    rebuilt so each partition field appears on exactly ONE of two partitions, and a second
    unrelated yaml block placed before it on the page.
    Why required: the coverage rule is a property of the field set, not of where the live
    page happens to put each field; a gate tied to the live layout would red every
    legitimate rewrite of the page.
    What kills it: requiring every partition field on every partition, or taking the first
    yaml block on the page as the profile.
    Second correct implementation: selecting the profile block as the one that loads
    through the model, instead of by its two marker keys, passes this.
    """
    base = _live_profile()
    reordered = {key: copy.deepcopy(base[key]) for key in reversed(list(base))}
    union: dict[str, object] = {}
    for entry in base["partitions"].values():
        for key, value in entry.items():
            union.setdefault(key, value)
    left: dict[str, object] = {}
    right: dict[str, object] = {}
    for i, (key, value) in enumerate(sorted(union.items())):
        (left if i % 2 == 0 else right)[key] = value
    reordered["partitions"] = {"zeta": left, "alpha": right}
    assert set(left) & set(right) == set()
    assert set(left) | set(right) == set(PartitionSpec.model_fields)

    other = textwrap.dedent(
        """\
        hpc_ensemble_partition: alpha
        """
    )
    page = tmp_path / "page.md"
    page.write_text(_page(other, yaml.safe_dump(reordered, sort_keys=False)), encoding="utf-8")
    assert gate.main(["--page", str(page)]) == 0


@pytest.mark.parametrize("n_profiles", [0, 2], ids=["no-profile-block", "two-profile-blocks"])
def test_page_must_carry_exactly_one_profile_block(tmp_path, n_profiles: int, capsys) -> None:
    """Property: the gate refuses to pass a page on which it cannot identify one profile.

    Class: a page with no profile block, and a page with two.
    Why required: a page with no profile would otherwise check nothing and print a pass;
    a page with two leaves it undecided which one readers copy.
    What kills it: returning success when no candidate block is found, or checking the
    first of several.
    Second correct implementation: counting candidate blocks with a markdown parser
    instead of a fence regex passes this.
    """
    body = yaml.safe_dump(_live_profile(), sort_keys=False)
    page = tmp_path / "page.md"
    page.write_text(_page(*([body] * n_profiles)) if n_profiles else _page("key: value\n"), encoding="utf-8")
    assert gate.main(["--page", str(page)]) == 1
    err = capsys.readouterr().err
    expected = "no-profile-block" if n_profiles == 0 else "ambiguous-profile-block"
    assert expected in err


def test_unknown_key_two_containers_deep_is_reported_exactly_once() -> None:
    """Property: an unknown key under a mapping of lists of specs is reported once, at the walk's location.

    Class: a field typed ``dict[str, list[Spec]]`` whose only entry carries every real
    sub-field plus an unknown key ``zz``.
    Why required: the walk and validation both see the key. Validation names the mapping key
    and the list index, the walk names each as ``*``, and only when the translation consumes
    one location segment per container level do the two agree; otherwise the key is printed
    twice, once per path, which is noise a reader learns to skim past.
    What kills it: translating a validation location with one ``*`` per FIELD rather than one
    per container level, or deciding depth from the value's shape.
    Second correct implementation: reporting unknown keys from validation only and dropping
    the walk's own unknown-key pass -- the key is then reported once, by validation.
    """

    class _Spec(BaseModel):
        model_config = ConfigDict(extra="forbid")
        a: int | None = None

    model = create_model("GrownProfile", __base__=hpc_system_config, extra_specs=(dict[str, list[_Spec]], None))
    model.model_rebuild(_types_namespace={"ContainerSpec": ContainerSpec})
    profile = _live_profile()
    profile["extra_specs"] = {"x": [{"a": 1, "zz": 3}]}
    findings, _ = gate.check_profile(profile, model=model)
    naming = [f for f in findings if "zz" in f.location]
    assert [(f.location, f.code) for f in naming] == [("extra_specs.*.*.zz", "key-not-in-model")], [
        f.render() for f in findings
    ]
