"""Reviewer checks for change 1122-0004.

``toggle_consolidate_timeseries`` and ``timeseries_keep_triton_gridded`` lose their defaults.

The builder's own file (``tests/test_timeseries_keys_have_no_default.py``) enumerates tracked YAML files and tracked
dict literals. This file closes the populations that file does not reach and the assertions it does not scan.

- A tracked zip archive holds analysis config YAML (the golden archives), and no YAML scan reads inside one.
- A test that still asserts the OLD default of either field reads ``PydanticUndefined`` now and is red.
- A config a run writes (``model_dump`` then YAML) states both keys, including the ``None`` one, so a member or a
  reloaded copy loads; a stored config lacking a key is refused, which pins the behaviour for older copies.

No solver, no Snakemake and no synthetic analysis is built, so every test is laptop-admissible.
"""

from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hhemt.config.analysis import analysis_config
from tests.test_config_validation import _minimal_analysis_config_dict

_REPO = Path(__file__).resolve().parent.parent
_KEYS = ("toggle_consolidate_timeseries", "timeseries_keep_triton_gridded")
_MARKERS = ("weather_event_indices", "weather_timeseries")


def _tracked(*patterns: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", *patterns], cwd=_REPO, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return [_REPO / rel for rel in out]


def _zip_analysis_members() -> list[tuple[Path, str]]:
    found = []
    for archive in _tracked("*.zip"):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not name.endswith((".yaml", ".yml")):
                    continue
                try:
                    doc = yaml.safe_load(z.read(name))
                except yaml.YAMLError:
                    continue
                if isinstance(doc, dict) and all(m in doc for m in _MARKERS):
                    found.append((archive, name))
    return found


def test_the_zip_enumeration_finds_the_golden_analysis_configs():
    """Property: the zip population is not empty, so the next test cannot pass over nothing.

    Must-fire control: both golden archives carry an ``analysis_config.yaml`` and the sensitivity one carries members.
    """
    members = _zip_analysis_members()
    assert {(a.name, n) for a, n in members} >= {
        ("multisim.zip", "analysis_config.yaml"),
        ("sensitivity.zip", "analysis_config.yaml"),
    }
    assert any("/members/member_" in n for _, n in members)


@pytest.mark.parametrize("archive,member", _zip_analysis_members(), ids=lambda v: v.name if isinstance(v, Path) else v)
def test_every_analysis_yaml_inside_a_tracked_zip_states_both_keys(archive, member):
    """Property: no analysis config held inside a tracked archive leaves either key to a default.

    Killed by: regenerating a golden from a generator that omits either key.
    """
    with zipfile.ZipFile(archive) as z:
        doc = yaml.safe_load(z.read(member))
    missing = [k for k in _KEYS if k not in doc]
    assert not missing, f"{archive.name}:{member} does not state {missing}"


def test_no_tracked_test_still_asserts_a_default_for_either_field():
    """Property: nothing under ``tests/`` reads ``.default`` of either field, because neither has one now.

    A field with no default reports ``PydanticUndefined`` for ``.default``, so an assertion that it is ``True`` or
    ``None`` is red. Must-fire control: the pattern matches a synthetic line of the old shape.
    Plausible wrong implementation: updating the dict builders and leaving an assertion on the old default.
    """
    pattern = re.compile(
        r"model_fields\[\s*['\"](?:toggle_consolidate_timeseries|timeseries_keep_triton_gridded)['\"]\s*\]\.default"
    )
    assert pattern.search('assert analysis_config.model_fields["toggle_consolidate_timeseries"].default is True')
    offenders = []
    for path in _tracked("tests/*.py", "tests/**/*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(_REPO)}:{lineno}: {line.strip()}")
    assert not offenders, "a test still asserts an old default:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "consolidate,keep", [(True, None), (False, None), (True, "default"), (True, ["max_flood_depth"])]
)
def test_a_written_config_reloads_with_both_keys_stated(tmp_path, consolidate, keep):
    """Property: ``model_dump`` then YAML then load keeps both keys, the ``None`` one included, so a member overlay
    (``{**master.model_dump(), **overlay}``) and a reloaded copy never drop a key.

    Must-fire control: the dumped YAML text names both keys; dropping the ``None`` one reds the reload.
    """
    cfg = {
        **_minimal_analysis_config_dict(tmp_path),
        "toggle_consolidate_timeseries": consolidate,
        "timeseries_keep_triton_gridded": keep,
    }
    if not consolidate:
        cfg["toggle_clear_scenario_timeseries_after_consolidation"] = False
    master = analysis_config.model_validate(cfg)
    dumped = yaml.safe_dump(master.model_dump(mode="json"), sort_keys=False)
    on_disk = yaml.safe_load(dumped)
    assert all(k in on_disk for k in _KEYS)
    reloaded = analysis_config.model_validate(on_disk)
    assert reloaded.toggle_consolidate_timeseries is consolidate
    assert reloaded.timeseries_keep_triton_gridded == keep
    member = analysis_config.model_validate({**master.model_dump(), "analysis_id": "member_0"})
    assert member.timeseries_keep_triton_gridded == keep
    # Control: a stored copy that lost the ``None``-valued key (an old copy, or a dump that excludes None) is refused.
    lost = {k: v for k, v in on_disk.items() if k != "timeseries_keep_triton_gridded"}
    with pytest.raises(ValidationError, match="timeseries_keep_triton_gridded"):
        analysis_config.model_validate(lost)


def test_the_clearing_toggle_still_resolves_from_the_stated_consolidation_toggle(tmp_path):
    """Property: with the clearing key unset, its value follows the stated consolidation toggle in both states.

    Must-fire control: the two arms differ, so a constant fails one of them.
    """
    base = _minimal_analysis_config_dict(tmp_path)
    on = analysis_config.model_validate({**base, "toggle_consolidate_timeseries": True})
    off = analysis_config.model_validate({**base, "toggle_consolidate_timeseries": False})
    assert on.toggle_clear_scenario_timeseries_after_consolidation is True
    assert off.toggle_clear_scenario_timeseries_after_consolidation is False


def test_the_sensitivity_member_overlay_helper_never_carries_either_key_away():
    """Property: ``_create_members`` builds a member from the master dump plus overlay cells, so both keys travel.

    Source-level pin: the merge is ``{**base_cfg, **overlay_cells}`` over ``model_dump()``, and ``model_dump`` is called
    without an ``exclude_none`` argument at both merge sites that would drop the ``None``-valued key.
    """
    text = (_REPO / "src" / "hhemt" / "sensitivity_analysis.py").read_text()
    assert "self.experiment.cfg_analysis.model_dump()" in text
    assert not re.search(r"cfg_analysis\.model_dump\([^)]*exclude_(none|unset|defaults)\s*=\s*True", text)
