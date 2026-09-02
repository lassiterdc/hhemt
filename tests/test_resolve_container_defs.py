"""`resolve_container_defs` — the descriptor-vs-flag decision, extracted from the CLI.

The decision needs no CLI-only context, so it lives beside `resolve_def_recipe` as a
free function and is unit-tested without a `CliRunner`, a `TRITONSWMM_system`, or a
`TRITONSWMM_analysis`. `bundle_command` reduces to a one-line call.

Mirrors the module's own precedent: `resolve_overrides` / `_confirm_override_gate`
already reconcile a CLI argument against the descriptor here, refuse rather than
silently prefer either source, and raise `ConfigurationError`.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from hhemt.exceptions import ConfigurationError


def _bundle_dir(tmp_path: Path, def_recipe: str = "containers/x.def") -> Path:
    d = tmp_path / "expt"
    (d / "containers").mkdir(parents=True)
    (d / "containers" / "x.def").write_text("Bootstrap: docker\n")
    (d / "experiment.yaml").write_text(
        textwrap.dedent(f"""\
        description: fixture
        system_config: configs/s.yaml
        analysis_config: configs/a.yaml
        toolkit_pin:
          version: "0.1.0"
        container:
          def_recipe: {def_recipe}
          sha256_source: ro-crate
        """)
    )
    return d


# --- descriptor ABSENT: the existing behaviour must be byte-preserved ---
@pytest.mark.parametrize("explicit", [None, [], [Path("/tmp/a.def")]])
def test_no_descriptor_passes_explicit_defs_through_unchanged(explicit) -> None:
    """The default MUST NOT fire without a descriptor.

    _emit.py's hard ConfigurationError on a container-mode analysis with no
    container_defs is a real safety property; this function must not defeat it.
    """
    from hhemt.experiment_bundle import resolve_container_defs

    assert resolve_container_defs(None, explicit) == list(explicit or [])


# --- descriptor PRESENT, flag ABSENT: the descriptor supplies ---
def test_descriptor_supplies_when_flag_absent(tmp_path: Path) -> None:
    from hhemt.experiment_bundle import resolve_container_defs

    d = _bundle_dir(tmp_path)
    assert resolve_container_defs(d, None) == [d / "containers" / "x.def"]


# --- descriptor PRESENT, flag AGREES: accepted ---
def test_agreeing_flag_is_accepted(tmp_path: Path) -> None:
    from hhemt.experiment_bundle import resolve_container_defs

    d = _bundle_dir(tmp_path)
    same = d / "containers" / "x.def"
    assert resolve_container_defs(d, [same]) == [same]


# --- descriptor PRESENT, flag DISAGREES: REFUSE, naming both ---
def test_disagreeing_flag_is_refused_naming_both_values(tmp_path: Path) -> None:
    from hhemt.experiment_bundle import resolve_container_defs

    d = _bundle_dir(tmp_path)
    other = tmp_path / "other.def"
    other.write_text("Bootstrap: docker\n")
    with pytest.raises(ConfigurationError) as exc:
        resolve_container_defs(d, [other])
    msg = str(exc.value)
    assert "other.def" in msg and "x.def" in msg, f"both values must be named; got {msg}"


def test_multiple_explicit_defs_disagree_with_a_single_descriptor_recipe(tmp_path: Path) -> None:
    """A repeated --container-defs cannot equal one descriptor value, so it must refuse."""
    from hhemt.experiment_bundle import resolve_container_defs

    d = _bundle_dir(tmp_path)
    same = d / "containers" / "x.def"
    other = tmp_path / "other.def"
    other.write_text("Bootstrap: docker\n")
    with pytest.raises(ConfigurationError) as exc:
        resolve_container_defs(d, [same, other])
    msg = str(exc.value)
    # The remedy must be REACHABLE. "correct the descriptor" is impossible for a
    # multi-arch operator: ContainerRef.def_recipe is scalar, so the descriptor
    # cannot express two arches. The multi-def branch must name the flag path.
    assert "--experiment-config" in msg, f"multi-def remedy must name the escape hatch; got {msg}"
    assert "correct the descriptor" not in msg, f"impossible remedy offered; got {msg}"


def test_single_disagreeing_def_offers_the_descriptor_fix(tmp_path: Path) -> None:
    """With ONE flag value, correcting the descriptor IS a reachable remedy."""
    from hhemt.experiment_bundle import resolve_container_defs

    d = _bundle_dir(tmp_path)
    other = tmp_path / "other.def"
    other.write_text("Bootstrap: docker\n")
    with pytest.raises(ConfigurationError) as exc:
        resolve_container_defs(d, [other])
    assert "correct the descriptor" in str(exc.value)
