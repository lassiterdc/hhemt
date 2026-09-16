"""A SET-but-absent optional system input path is rejected at config load.

Two arms, and the second is what stops the check from over-firing: these five fields are
`Path | None` with `required=False`, so `None` means "not requested" and must stay
accepted. Requiredness is the toggle validators' job, not this one's.

Asserted at LOAD rather than at preflight because `cfgBaseModel._check_paths_exist` is
where the check lives once it runs in `mode="after"`; a preflight-level assertion for the
same property would sit behind a model that already refused the input.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent))
from test_config_validation import _minimal_system_config_dict  # noqa: E402

from hhemt.config.system import system_config  # noqa: E402


def test_set_but_absent_optional_path_is_rejected_at_load(tmp_path):
    cfg_dict = _minimal_system_config_dict(tmp_path)
    cfg_dict["landuse_raster"] = str(tmp_path / "inputs" / "ABSENT_landuse.tif")
    with pytest.raises(ValidationError, match="landuse_raster"):
        system_config.model_validate(cfg_dict)


def test_none_optional_path_is_accepted_at_load(tmp_path):
    cfg_dict = _minimal_system_config_dict(tmp_path)
    cfg_dict["landuse_raster"] = None
    assert system_config.model_validate(cfg_dict).landuse_raster is None
