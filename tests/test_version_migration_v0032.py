"""V0032 (layout 31 to 32): the SWMM and TRITON cfg templates are removed.

The migration changes nothing on disk, so a layout-31 tree is stamped 32 with no other change. A stored config that
still carries a template key is refused at load with the teaching message. No solver, no Snakemake.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from hhemt.config.system import system_config
from hhemt.version_migration import registry, runner, state
from hhemt.version_migration.constants import LAYOUT_VERSION

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts"


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and p.name != "_version.json" and not p.name.endswith(".lock")
    }


def test_v0032_is_registered_and_takes_layout_31_to_32():
    (module,) = [m for m in registry.discover_migrations() if m.version_to == 32]
    assert (module.version_from, module.version_to) == (31, 32)
    assert LAYOUT_VERSION == max(m.version_to for m in registry.discover_migrations())


def test_a_layout_31_tree_is_stamped_32_with_no_other_change(tmp_path):
    work = tmp_path / "v31"
    shutil.copytree(FIXTURE_ROOT / "v31", work)
    runner.baseline(work, 31)
    before = _files(work)
    assert before, "control: the fixture holds files"
    result = runner.run_migration(work, target=32, apply=True)
    assert result.migrations_applied == ["V0032__retire_swmm_and_triton_templates"]
    assert state.read_version_file(work).layout_version == 32
    assert _files(work) == before


def test_the_v31_and_v32_fixtures_hold_the_same_files():
    assert _files(FIXTURE_ROOT / "v31") == _files(FIXTURE_ROOT / "v32")


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("SWMM_hydraulics", "SWMM_model"),
        ("SWMM_hydrology", "SWMM_model"),
        ("SWMM_full", "SWMM_model"),
        ("triton_swmm_configuration_template", "renders the TRITON cfg itself"),
    ],
)
def test_a_stored_config_with_a_template_key_is_refused_naming_the_replacement(key, replacement):
    with pytest.raises(ValidationError) as excinfo:
        system_config.model_validate({key: "x.inp"})
    message = str(excinfo.value)
    assert key in message and "not a system config field" in message and replacement in message
