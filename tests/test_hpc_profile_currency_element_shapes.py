"""Reviewer tests, round 2: a spec nested inside a container's ELEMENT type must be walked too.

The gate's classifier (scripts/check_hpc_profile_currency.py, ``_nested_model``) states in its
docstring that it returns None only "for anything that holds no model", and that an optional
mapping or list of specs is walked exactly like a required one. It unwraps ``X | None`` at
the FIELD level, but it classifies a container by testing whether the element type is itself
a model class. So when the ELEMENT is wrapped -- ``dict[str, Sub | None]``, ``list[Sub | None]``,
or a mapping of lists, ``dict[str, list[Sub]]`` -- the element is not a model class, the
classifier returns None, and the walk never descends. Each of these annotations holds a model.

Measured against the gate in round 2: in all three shapes an entry that omits sub-field ``b``
produces NO finding, so a page omitting a nested field passes with exit 0. (The unknown-key
direction IS now caught, by the validation-error path, so it is not re-tested here.)

For each shape the gate must report a sub-field no entry shows, and must not report a sub-field
that some entry shows. Coverage follows the gate's own rule for partitions: a sub-field shown
on ANY entry counts as shown.

Nothing here builds an analysis, compiles, or simulates.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, create_model

import scripts.check_hpc_profile_currency as gate  # repo root is on sys.path under pytest
from hhemt.config.hpc_system import ContainerSpec, hpc_system_config


class _Sub(BaseModel):
    """A two-field nested spec; forbids extra keys, as every live nested spec does."""

    model_config = ConfigDict(extra="forbid")

    a: int | None = None
    b: int | None = None


_SHAPES = {
    "mapping-of-optional": (dict[str, _Sub | None], None, lambda item: {"x": item}),
    "list-of-optional": (list[_Sub | None], [], lambda item: [item]),
    "mapping-of-lists": (dict[str, list[_Sub]], None, lambda item: {"x": [item]}),
}


def _live_profile() -> dict:
    text = (gate.REPO_ROOT / gate.PAGE_REL).read_text(encoding="utf-8")
    profile, findings = gate.extract_profile(text)
    assert profile is not None, findings
    return profile


def _check(shape: str, item: dict) -> list[gate.Finding]:
    annotation, default, wrap = _SHAPES[shape]
    model = create_model("GrownProfile", __base__=hpc_system_config, extra_specs=(annotation, default))
    # Resolve the base's postponed `container` forward reference so the harness, not the
    # gate, is never what reds.
    model.model_rebuild(_types_namespace={"ContainerSpec": ContainerSpec})
    profile = _live_profile()
    profile["extra_specs"] = wrap(item)
    findings, _ = gate.check_profile(profile, model=model)
    return [f for f in findings if "extra_specs" in f.location]


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_sub_field_no_entry_shows_fails(shape: str) -> None:
    """Property: a sub-field of a spec held in a container's wrapped element type is checked.

    Class: one entry carrying ``a`` and omitting ``b``, in each of the three element shapes.
    Why required: the omission direction is the staleness the gate exists to detect, and in
    these shapes it is currently silent (exit 0 on an incomplete page).
    What kills it: classifying a container by whether its element type IS a model class,
    rather than by whether it HOLDS one, so ``Sub | None`` and ``list[Sub]`` as elements stop
    the descent.
    Second correct implementation: classify the element type by recursing through the
    classifier itself (unwrap, then classify), or derive the nested field set from
    ``model_json_schema()`` following ``$defs``.
    """
    findings = _check(shape, {"a": 1})
    absent = [f for f in findings if f.code == "field-absent" and f.location.endswith(".b")]
    assert absent, "no field-absent finding for extra_specs' sub-field b; findings:\n" + "\n".join(
        f.render() for f in findings
    )


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_entry_showing_every_sub_field_passes(shape: str) -> None:
    """Property: an entry showing every sub-field yields no finding about the field.

    Class: one entry carrying both ``a`` and ``b``, in each of the three element shapes.
    Why required: a fix that descends but misreads the nesting depth (walking the list as if it
    were the spec, or the mapping's values as if they were entries) reports a complete page
    stale, and a gate that reds correct pages is routed around.
    What kills it: descending one level too few or too many for the mapping-of-lists shape.
    Second correct implementation: any of the two named in the test above.
    """
    findings = _check(shape, {"a": 1, "b": 2})
    assert findings == [], "\n".join(f.render() for f in findings)
