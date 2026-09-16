"""The document-level LOAD MODE: three intents carried in pydantic's validation context.

Test 1 is the FENCE-DEPENDENCY GUARD. The mode is a no-op unless `_check_paths_exist`
runs in `mode="after"` -- under `mode="before"` a Path-annotated field loaded from YAML
arrives as `str`, the `isinstance(v, Path)` selector declines, and the existence branch
the mode gates never fires. If someone reverts that arming, this goes red and names why,
rather than the mode silently becoming dead code.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from hhemt.config.base import cfgBaseModel
from hhemt.config.loaders import EXISTENCE_CONTEXT_KEY
from hhemt.config.system import system_config


def test_check_paths_exist_is_registered_as_an_after_mode_wildcard_validator():
    """FENCE-DEPENDENCY GUARD -- see the module docstring.

    Probe the pydantic REGISTRY, not the class namespace. ModelMetaclass harvests the
    PydanticDescriptorProxy into __pydantic_decorators__ at class construction and leaves
    a plain `classmethod` behind, so `cls.__dict__[name]` carries no decorator info on any
    v2 model -- a probe there is red on a tree that satisfies its own precondition, and
    says the opposite of the truth.
    """
    registry = cfgBaseModel.__pydantic_decorators__.field_validators
    assert "_check_paths_exist" in registry, (
        "_check_paths_exist is not in __pydantic_decorators__.field_validators, so it is "
        "not a registered field validator at all and the load mode gates nothing."
    )
    info = registry["_check_paths_exist"].info
    assert info.mode == "after", (
        "_check_paths_exist must run in mode='after'. In mode='before' a YAML-loaded "
        "Path field arrives as str, the isinstance(v, Path) selector declines, and the "
        "existence branch the load mode gates never fires -- the mode becomes dead code."
    )
    assert info.fields == ("*",), (
        "_check_paths_exist must stay a WILDCARD validator. Narrowing its field set "
        "disarms the check on the unlisted fields just as thoroughly as a mode flip, "
        "and is the failure a mode-only probe cannot see."
    )


def _absent_path_dict(tmp_path: Path) -> dict:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_config_validation import _minimal_system_config_dict

    d = _minimal_system_config_dict(tmp_path)
    d["landuse_raster"] = str(tmp_path / "inputs" / "ABSENT_landuse.tif")
    return d


def test_absent_context_still_raises(tmp_path):
    """The DEFAULT is strict. No existing call site loosens by omission."""
    with pytest.raises(ValidationError, match="landuse_raster"):
        system_config.model_validate(_absent_path_dict(tmp_path))


@pytest.mark.parametrize("intent", ["template", "metadata"])
def test_non_runnable_intents_skip_the_existence_check(tmp_path, intent):
    cfg = system_config.model_validate(_absent_path_dict(tmp_path), context={EXISTENCE_CONTEXT_KEY: intent})
    assert not Path(cfg.landuse_raster).exists()


def test_explicit_runnable_intent_raises(tmp_path):
    """`runnable` passed explicitly is the same as omitting it -- no third behaviour."""
    with pytest.raises(ValidationError, match="landuse_raster"):
        system_config.model_validate(_absent_path_dict(tmp_path), context={EXISTENCE_CONTEXT_KEY: "runnable"})


def test_direct_constructor_cannot_opt_out(tmp_path):
    """A bare `system_config(...)` carries no context and therefore cannot loosen.
    Pinned because it is the one entry shape with no caller-declared intent."""
    d = _absent_path_dict(tmp_path)
    with pytest.raises(ValidationError, match="landuse_raster"):
        system_config(**d)
