"""`container.def_recipe` self-rooting contract ([Q161]).

One field, two shapes, discriminated by the value itself: a `${VAR}` prefix means
explicitly-rooted (the shared-container arm), anything else means bundle-relative
(the ruled default). A literal absolute path declares neither and is rejected.

The POSITIVE arms fail against pre-fix source -- the validator does not exist, so
`ContainerRef` accepts an absolute path, and `resolve_def_recipe` is unimportable.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from hhemt.config.experiment_bundle import ContainerRef


def _bundle_dir(tmp_path: Path, def_recipe: str) -> Path:
    d = tmp_path / "expt"
    (d / "containers").mkdir(parents=True)
    (d / "containers" / "x.def").write_text("Bootstrap: docker\n")
    (d / "experiment.yaml").write_text(
        textwrap.dedent(f"""\
        experiment_id: expt
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


# --- File 1: the shape validator. Both arms discriminate pre-fix. ---
@pytest.mark.parametrize(
    "value",
    [
        "/scratch/dcl3nd/containers/uva-cuda.def",  # literal absolute
        "/home/dcl3nd/containers/uva-cuda.def",  # literal home-absolute
        "~/containers/uva-cuda.def",  # tilde-rooted: operator-specific, same as absolute
        "$HHEMT_CONTAINERS/uva-cuda.def",  # UNBRACED $VAR: the resolver only expands ${...}
    ],
)
def test_operator_rooted_def_recipe_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="declare its own root"):
        ContainerRef(def_recipe=value)


@pytest.mark.parametrize(
    "value",
    ["containers/uva-cuda.def", "${HHEMT_CONTAINERS}/uva-cuda.def", "recipes/sub/x.def"],
)
def test_self_rooting_shapes_are_accepted(value: str) -> None:
    assert ContainerRef(def_recipe=value).def_recipe == value


# --- File 2: the resolver. Each arm reaches a distinct branch. ---
def test_bundle_relative_is_the_default_root(tmp_path: Path) -> None:
    from hhemt.experiment_bundle import load_bundle, resolve_def_recipe

    d = _bundle_dir(tmp_path, "containers/x.def")
    assert resolve_def_recipe(load_bundle(d), d) == d / "containers" / "x.def"


def test_var_rooted_resolves_outside_the_bundle(tmp_path: Path, monkeypatch) -> None:
    from hhemt.experiment_bundle import load_bundle, resolve_def_recipe

    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "x.def").write_text("Bootstrap: docker\n")
    monkeypatch.setenv("HHEMT_TEST_CONTAINERS", str(shared))
    d = _bundle_dir(tmp_path, "${HHEMT_TEST_CONTAINERS}/x.def")
    assert resolve_def_recipe(load_bundle(d), d) == shared / "x.def"


def test_unset_var_raises_naming_the_placeholder(tmp_path: Path, monkeypatch) -> None:
    from hhemt.exceptions import ConfigurationError
    from hhemt.experiment_bundle import load_bundle, resolve_def_recipe

    monkeypatch.delenv("HHEMT_TEST_CONTAINERS", raising=False)
    d = _bundle_dir(tmp_path, "${HHEMT_TEST_CONTAINERS}/x.def")
    with pytest.raises(ConfigurationError) as exc:
        resolve_def_recipe(load_bundle(d), d)
    # Anchored on the placeholder NAME, which both the corrected and the legacy
    # token rendering contain -- so this asserts behaviour, not message cosmetics.
    assert "HHEMT_TEST_CONTAINERS" in str(exc.value)


def test_missing_recipe_raises_naming_the_resolved_path(tmp_path: Path) -> None:
    from hhemt.exceptions import ConfigurationError
    from hhemt.experiment_bundle import load_bundle, resolve_def_recipe

    d = _bundle_dir(tmp_path, "containers/absent.def")
    with pytest.raises(ConfigurationError) as exc:
        resolve_def_recipe(load_bundle(d), d)
    assert "absent.def" in str(exc.value)
