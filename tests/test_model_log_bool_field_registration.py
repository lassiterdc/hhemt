"""Every boolean field of the per-model log is registered with its validator and serializer.

``TRITONSWMM_model_log`` declares its ``LogField[bool]`` fields one by one, and pydantic
only knows how to coerce and write a ``LogField`` through the field validator and field
serializer that NAME it. A bool field left out of those two name lists is a declared field
that can be neither loaded as ``true`` (validation fails) nor written once set (every
write of the log raises ``PydanticSerializationError``). Nothing at declaration time
catches the omission, because the default ``None`` serializes either way.

These tests read the two name lists from pydantic's own decorator registry, not from the
source text, so a later unregistered bool field reds here as well as the three that were
missing when this file was written. Pure Python: no analysis, no compile, no solver.
"""

import typing

import pytest

from hhemt.log import LogField, TRITONSWMM_model_log


def _bool_fields() -> list[str]:
    """Names of the model-log fields annotated ``LogField[bool]`` or ``LogField[bool] | None``."""
    names = []
    for name, info in TRITONSWMM_model_log.model_fields.items():
        annotation = info.annotation
        # The annotation itself for a required field; its union arms for ``LogField[bool] | None``.
        # ``get_args`` alone is not enough: on ``LogField[bool]`` it returns ``(bool,)``.
        candidates = (annotation, *typing.get_args(annotation))
        if any(c == LogField[bool] for c in candidates):
            names.append(name)
    return names


def _registered(kind: str, decorator: str) -> set[str]:
    """Field names one of the class's named field validators or serializers applies to."""
    registry = getattr(TRITONSWMM_model_log.__pydantic_decorators__, kind)
    return set(registry[decorator].info.fields)


def test_every_bool_field_is_named_by_the_bool_validator_and_serializer():
    """Property: each ``LogField[bool]`` field is in BOTH the bool validator and the bool serializer.

    Class: every bool field the class declares, optional or not -- the population is read
    from ``model_fields``, so a field added later joins it without editing this test.

    Why required: an unregistered bool field cannot be written once set and cannot be
    loaded when stored as true, and nothing else in the suite fails on that state.

    What kills it: removing any bool field name from either list, or adding a new
    ``LogField[bool]`` field without registering it.

    A second correct implementation under which it still passes: one that builds both
    decorators from a single module-level tuple of bool field names, in any order; the test
    reads only which names each decorator covers. Not covered: an implementation that
    registers bools under a decorator with a different name, which this test would red.
    """
    fields = _bool_fields()
    # Both required bool fields must be in the population, or the scan is reading only unions.
    assert {"simulation_completed", "force_rerun_pending"} <= set(fields)
    validated = _registered("field_validators", "_validate_bool_fields")
    serialized = _registered("field_serializers", "_serialize_bool_fields")
    missing = sorted(f for f in fields if f not in validated or f not in serialized)
    assert missing == [], f"LogField[bool] fields absent from the bool validator or serializer: {missing}"


@pytest.mark.parametrize("field", _bool_fields())
def test_each_bool_field_round_trips_true_through_json(field, tmp_path):
    """Property: a model log with one bool field set True writes to JSON and reads back True.

    Class: each ``LogField[bool]`` field, one parametrized case per field.

    Why required: this is the observable the registration exists for -- the write must
    not raise and the stored ``true`` must load.

    What kills it: an unregistered field, which raises ``PydanticSerializationError`` on
    ``model_dump_json`` or fails validation on the read-back.

    A second correct implementation under which it still passes: a serializer that writes
    the bool as a JSON object wrapping the value, with a validator that unwraps it; the
    test asserts only that the value read back is True.
    """
    log = TRITONSWMM_model_log.model_validate({"logfile": tmp_path / "log_tritonswmm.json"})
    setattr(log, field, LogField(True))
    payload = log.model_dump_json()
    reloaded = TRITONSWMM_model_log.model_validate_json(payload)
    assert getattr(reloaded, field).get() is True
