"""Reviewer tests for the HPC profile currency gate: nested-model shapes the walk must follow.

The gate's contract (scripts/check_hpc_profile_currency.py, module docstring) is that the
field list is read from the model at run time, walking nested models, so that a field
added to the model fails an unchanged page. The live model nests models in exactly two
shapes today -- ``ContainerSpec | None`` and ``dict[str, PartitionSpec]`` -- and the gate's
annotation classifier handles exactly those. These tests grow the model, through the
gate's own ``model=`` parameter, by a field nested in each of the three neighbouring
shapes a config model commonly takes next:

- ``dict[str, Sub] | None`` -- an OPTIONAL mapping of specs, the shape ``partitions``
  takes the day it becomes optional;
- ``list[Sub]`` -- a list of specs;
- ``list[Sub] | None`` -- an optional list of specs.

For each shape the gate must report a sub-field the profile does not show, must report a
key the sub-model does not have, and must pass a profile that shows every sub-field. The
coverage rule is the gate's own rule for partitions: a sub-field shown on ANY entry of
the mapping or list counts as shown.

Why this matters even though no live field has these shapes: the gate silently skips the
nested fields of an optional mapping or list (no finding, exit 0), so the field-added-to-
the-model guarantee its docstring states holds only for the shapes the model happens to
use today. And the unknown-key case is a second, compounding gap: ``check_profile``
drops ``extra_forbidden`` validation errors on the premise that the walk already reported
every unknown key, which is true only at the levels the walk actually descends into.

Nothing here builds an analysis, compiles, or simulates; every input is an in-memory
mapping derived from the live page's profile.
"""

from __future__ import annotations

from collections.abc import Callable

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
    "optional-mapping": (dict[str, _Sub] | None, None, lambda item: {"x": item}),
    "list": (list[_Sub], [], lambda item: [item]),
    "optional-list": (list[_Sub] | None, None, lambda item: [item]),
}


def _live_profile() -> dict:
    text = (gate.REPO_ROOT / gate.PAGE_REL).read_text(encoding="utf-8")
    profile, findings = gate.extract_profile(text)
    assert profile is not None, findings
    return profile


def _grown(shape: str) -> tuple[type[BaseModel], Callable[[dict], object]]:
    annotation, default, wrap = _SHAPES[shape]
    model = create_model("GrownProfile", __base__=hpc_system_config, extra_specs=(annotation, default))
    # The base class's `container` annotation is a postponed ForwardRef; resolve it so the
    # grown model validates. Without this every case would red on a PydanticUserError
    # raised by the harness rather than on the gate's behaviour.
    model.model_rebuild(_types_namespace={"ContainerSpec": ContainerSpec})
    return model, wrap


def _check(shape: str, item: dict) -> list[gate.Finding]:
    model, wrap = _grown(shape)
    profile = _live_profile()
    profile["extra_specs"] = wrap(item)
    findings, _ = gate.check_profile(profile, model=model)
    return findings


def _about_extra_specs(findings: list[gate.Finding]) -> list[gate.Finding]:
    return [f for f in findings if "extra_specs" in f.location]


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_profile_showing_every_nested_field_passes(shape: str) -> None:
    """Property: a profile that shows every sub-field of a nested spec yields no finding about it.

    Class: one entry carrying both sub-fields, in each of the three shapes.
    Why required: a gate that reports a correct profile as stale is routed around.
    What kills it: classifying ``list[Sub]`` as a single nested model, so the list value is
    discarded and every sub-field reads absent although the page shows it.
    Second correct implementation: treating a list of specs exactly as the mapping of specs
    is treated (walk the union of its elements), or skipping the list's elements entirely
    -- both pass THIS test; the next two tests separate them.
    """
    findings = _about_extra_specs(_check(shape, {"a": 1, "b": 2}))
    assert findings == [], "\n".join(f.render() for f in findings)


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_nested_field_the_profile_omits_fails(shape: str) -> None:
    """Property: a sub-field no entry shows is a field-absent finding naming that sub-field.

    Class: the entry carries ``a`` and omits ``b``, in each of the three shapes.
    Why required: this is the field-added-to-the-model case one level down; it is the
    staleness the gate exists to detect.
    What kills it: an annotation classifier that returns no nested model for an OPTIONAL
    mapping or list (the union branch only recognises a bare model among the union's
    members), so the walk never descends and the gate passes an incomplete page.
    Second correct implementation: unwrapping ``X | None`` to ``X`` before classifying, or
    deriving the nested field set from ``model_json_schema()`` with ``$defs`` followed.
    """
    findings = _about_extra_specs(_check(shape, {"a": 1}))
    absent = [f for f in findings if f.code == "field-absent" and f.location.endswith(".b")]
    assert absent, "no field-absent finding for extra_specs' sub-field b; findings:\n" + "\n".join(
        f.render() for f in findings
    )


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_nested_key_the_model_lacks_fails(shape: str) -> None:
    """Property: a key a nested spec does not have is a finding, whatever its nesting shape.

    Class: the entry carries both real sub-fields plus an unknown key ``zz``.
    Why required: the page promises a profile that loads; ``zz`` is a load failure on the
    reader's cluster. ``check_profile`` drops pydantic's ``extra_forbidden`` errors on the
    premise that the walk reports every unknown key, so where the walk does not descend
    the key is reported by neither path.
    What kills it: the non-descending classifier above, or the ``extra_forbidden`` filter
    applied at levels the walk did not visit.
    Second correct implementation: reporting ``extra_forbidden`` errors from validation
    instead of from the walk, at every level.
    """
    findings = _about_extra_specs(_check(shape, {"a": 1, "b": 2, "zz": 3}))
    named = [f for f in findings if "zz" in f.location or "zz" in f.detail]
    assert named, "no finding names the unknown nested key 'zz'; findings:\n" + "\n".join(f.render() for f in findings)
