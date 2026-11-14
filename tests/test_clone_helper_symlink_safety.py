"""The per-test clone helper must copy scenario symlinks AS LINKS.

Each cached scenario dir carries ``build -> {_software}/triton/build_tritonswmm_cpu``,
a link into the SHARED compile tier. Another chunk rebuilding that tier
``rm -rf``s the build dir's contents, and a follow-symlinks ``copytree`` landing
inside that window raises ``shutil.Error ... [Errno 2]`` on the SOURCE.

Measured on Rivanna run ``20260824T021951Z``: ``build_tritonswmm_cpu`` was rebuilt
at 23:41 mid-run (666 files under it, 4037 under ``_software``), and the 3
``test_synth_static_plots`` setup errors were exactly that read. The target existed
again by the time the tree was inspected, which is why the failure reads as
mysterious after the fact -- the absence is transient by construction.

A DANGLING link reproduces the same read deterministically and needs no race, so
this test is a real guard rather than a re-description of the incident. It is
deliberately fixture-free: it stubs the two attributes ``prepare_clone_dir`` reads,
so it costs no compile tier and runs in milliseconds.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from tests._failing_fixture_helpers import prepare_clone_dir


def _stub_analysis(system_dir: Path) -> SimpleNamespace:
    """The minimal shape ``prepare_clone_dir`` reads: ``._system.cfg_system.system_directory``."""
    return SimpleNamespace(_system=SimpleNamespace(cfg_system=SimpleNamespace(system_directory=str(system_dir))))


def _seed_system_dir(root: Path) -> Path:
    src = root / "src_system"
    (src / "sims" / "event_index.0").mkdir(parents=True)
    (src / "system_config.yaml").write_text(yaml.safe_dump({"system_directory": str(src)}))
    (src / "analysis_config.yaml").write_text(yaml.safe_dump({"analysis_id": "synth_multi_sim"}))
    return src


def test_clone_survives_a_scenario_build_link_whose_target_is_absent(tmp_path: Path) -> None:
    """A scenario ``build`` link into an absent shared tier must not fail the clone."""
    src = _seed_system_dir(tmp_path)
    absent_target = tmp_path / "_software" / "triton" / "build_tritonswmm_cpu"
    link = src / "sims" / "event_index.0" / "build"
    link.symlink_to(absent_target)  # dangling BY CONSTRUCTION -- the mid-rebuild window

    assert link.is_symlink() and not link.exists(), (
        "precondition: the seeded link must be dangling, or this test cannot "
        "distinguish a working fix from a target that happens to exist"
    )

    paths = prepare_clone_dir(_stub_analysis(src), tmp_path / "dst")

    copied = paths["system_dir"] / "sims" / "event_index.0" / "build"
    assert copied.is_symlink(), (
        "the clone dereferenced a link into the shared compile tier instead of copying "
        "it as a link; that tier is not the clone's to own, and following it makes the "
        "clone fail whenever another chunk is mid-rebuild"
    )


def test_clone_preserves_a_live_build_link_as_a_link(tmp_path: Path) -> None:
    """The link is copied as a link even when its target DOES exist.

    Second arm on purpose: arm one alone would pass under a hypothetical fix that
    special-cased broken links, which would still dereference (and so deep-copy the
    whole compile tier for) every healthy one.
    """
    src = _seed_system_dir(tmp_path)
    live_target = tmp_path / "_software" / "triton" / "build_tritonswmm_cpu"
    live_target.mkdir(parents=True)
    (live_target / "triton.exe").write_text("binary")
    link = src / "sims" / "event_index.0" / "build"
    link.symlink_to(live_target)

    assert link.exists(), "precondition: this arm's target must exist"

    paths = prepare_clone_dir(_stub_analysis(src), tmp_path / "dst")

    copied = paths["system_dir"] / "sims" / "event_index.0" / "build"
    assert copied.is_symlink(), (
        "a live shared-tier link was dereferenced and deep-copied into the clone; the "
        "clone would then carry a private copy of the compile tier per test"
    )
