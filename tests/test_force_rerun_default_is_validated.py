"""An omitted ``force_rerun`` holds a ``ForceRerunSpec``, so dumping the config is silent.

``analysis_config.force_rerun`` is typed as a ``ForceRerunSpec`` and its default is the
bare string ``"none"``. Pydantic does not validate a default, so a config that omits the
field used to carry the raw string in a field typed as a model, and every ``model_dump``
of it emitted a serializer warning. ``validate_default=True`` on that one field coerces
the default through the spec's ``mode="before"`` validator, which maps ``"none"`` to
``ForceRerunSpec(subject="none", stage="simulate", models=None)``.

No solver, compile or simulation-fixture request is used: the tests build an
``analysis_config`` from a dict of ``tmp_path`` files.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from hhemt.config.analysis import ForceRerunSpec, analysis_config
from tests.test_config_validation import _minimal_analysis_config_dict


def _config_without_force_rerun(tmp_path: Path) -> analysis_config:
    """Build the minimal config helper's dict with ``force_rerun`` left out."""
    cfg = _minimal_analysis_config_dict(tmp_path)
    del cfg["force_rerun"]
    return analysis_config.model_validate(cfg)


def test_omitted_force_rerun_is_a_spec_and_dumps_silently(tmp_path: Path):
    """Property: with ``force_rerun`` omitted, the field holds a ``ForceRerunSpec`` and
    ``model_dump`` raises no warning.

    Class: every ``analysis_config`` that omits ``force_rerun``, dumped in python mode and
    in json mode. Why required: the field is typed as a spec, so a raw string in it makes
    the serializer emit a ``PydanticSerializationUnexpectedValue`` warning on every dump of
    the config (the bundle's config pair, the persisted ``cfg_analysis.yaml``), and the
    validator that reads the field must then handle both shapes. Killed by: removing
    ``validate_default=True`` from the field (the default stays the string ``"none"``).
    Second correct implementation under which this still passes: a ``default_factory``
    returning ``ForceRerunSpec(subject="none")`` in place of the literal default.
    """
    cfg = _config_without_force_rerun(tmp_path)

    assert type(cfg.force_rerun) is ForceRerunSpec
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        python_dump = cfg.model_dump()
        json_dump = cfg.model_dump(mode="json")

    assert json_dump["force_rerun"] == {"subject": "none", "stage": "simulate", "models": None}
    assert python_dump["force_rerun"] == json_dump["force_rerun"]


def test_omitted_and_explicit_none_are_the_same_value(tmp_path: Path):
    """Property: omitting ``force_rerun`` and writing ``force_rerun: none`` give equal
    specs and equal dumps.

    Class: the two spellings of "force nothing", the second being a differently-positioned
    input that already held a spec before the default was validated. Why required: it is
    the statement that the validated default changes no behaviour, only the representation
    of an omitted field, so a config that names the sentinel and one that omits it cannot
    diverge. Killed by: a default that validates to any other spec (a different ``stage``
    or ``subject``). Second correct implementation under which this still passes: the
    default declared as ``ForceRerunSpec(subject="none")`` directly.
    """
    omitted = _config_without_force_rerun(tmp_path)
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["force_rerun"] = "none"
    explicit = analysis_config.model_validate(cfg)

    assert type(explicit.force_rerun) is ForceRerunSpec
    assert omitted.force_rerun == explicit.force_rerun
    assert omitted.model_dump(mode="json")["force_rerun"] == explicit.model_dump(mode="json")["force_rerun"]


@pytest.mark.parametrize("dump_mode", ["python", "json"])
def test_omitted_force_rerun_round_trips_to_an_equal_spec(tmp_path: Path, dump_mode: str):
    """Property: an omitted-key config survives ``model_dump`` then ``model_validate`` with
    an equal ``force_rerun`` spec.

    Class: the omitted-key config dumped in each mode and reloaded through the model, which
    is the path the render bundle's ``cfg_analysis.yaml`` takes. Why required: the dumped
    shape of an omitted field changed from the string ``"none"`` to the spec's dict, so the
    dict must reload to the same value or a bundle would read a different policy than the
    one it was emitted with. Killed by: a spec whose dumped dict does not validate back to
    itself (for example a serializer that drops ``stage``). Second correct implementation
    under which this still passes: the dump carrying the spec as the bare sentinel string
    for ``subject="none"`` and the default stage, provided the string reloads to the same
    spec.
    """
    cfg = _config_without_force_rerun(tmp_path)

    dumped = cfg.model_dump(mode=dump_mode)
    reloaded = analysis_config.model_validate(dumped)

    assert reloaded.force_rerun == cfg.force_rerun
