"""The in-image fallback executable names agree with what the shipped image recipes install.

When a container profile leaves ``exe_in_sif`` empty for a model type, the simulation
runs ``/opt/hhemt/bin/{name}`` with ``name`` taken from ``run_simulation._DEFAULT_EXE_NAME``.
That fallback is only correct if every image recipe the toolkit ships installs a file of
that name, and if the TRITON-only model's name is the build compiled without SWMM. The
TRITON-only model cannot run the coupled build: the coupled build reads a SWMM input
file at start-up, and a TRITON-only configuration names none, so the run dies at start.

These checks read the recipe text and the module's dictionary. They compile nothing and
run no solver.
"""

import importlib.resources
import re

from hhemt.run_simulation import _DEFAULT_EXE_NAME

#: Where the recipes install the executables, and the build directory each one is copied
#: from. ``build`` is the coupled TRITON-SWMM build; ``build_triton_only`` is TRITON
#: compiled with SWMM disabled.
_BIN = "/opt/hhemt/bin"
_COUPLED_BUILD = "/opt/triton/build/"
_TRITON_ONLY_BUILD = "/opt/triton/build_triton_only/"


def _recipe_texts() -> dict[str, str]:
    """Every shipped ``.def`` recipe, keyed by file name, read from the installed package."""
    recipes = importlib.resources.files("hhemt") / "sif" / "recipes"
    return {r.name: r.read_text() for r in recipes.iterdir() if r.name.endswith(".def")}


def _installs(text: str) -> dict[str, str]:
    """Map each file a recipe copies into ``/opt/hhemt/bin`` to the source it copies.

    A ``cp`` split across lines with a trailing backslash is joined first, because the
    SWMM install names its destination on the line after its source.
    """
    joined = re.sub(r"\\\n\s*", " ", text)
    found = {}
    for source, name in re.findall(rf"^\s*cp\s+(.+?)\s+{re.escape(_BIN)}/(\S+)\s*$", joined, re.MULTILINE):
        found[name] = source
    return found


def test_default_exe_names_are_installed_and_triton_maps_to_the_swmm_disabled_build():
    """Every built-in executable name is a file each shipped recipe installs, from the right build.

    Property: for every shipped recipe, each value of ``_DEFAULT_EXE_NAME`` is a file the
    recipe copies into ``/opt/hhemt/bin``; the name used for the ``triton`` model is copied
    from the SWMM-disabled build directory; and the name used for ``tritonswmm`` is copied
    from the coupled build directory.

    Class: the three model types crossed with the four shipped recipes. A recipe added
    later is read by the same loop.

    Why it is required: a profile that leaves ``exe_in_sif`` empty runs exactly these
    names. A name no recipe installs fails at launch with a missing file. A ``triton``
    name pointing at the coupled build starts the coupled solver on a configuration with
    no SWMM input file, and the run dies at start.

    What kills it: setting ``_DEFAULT_EXE_NAME["triton"]`` back to ``"triton.exe"``,
    which every recipe copies from the coupled build, or naming any file a recipe does
    not install.

    A second correct implementation under which it still passes: one where the recipes
    install the SWMM-disabled build under a different file name and the dictionary names
    that file. The test keys on what each recipe copies and from where, never on the
    literal ``triton_only.exe``.
    """
    assert set(_DEFAULT_EXE_NAME) == {"triton", "tritonswmm", "swmm"}
    recipes = _recipe_texts()
    # Must-be-nonzero control on the parser: a matcher that finds nothing would make
    # every membership check below fail for the wrong reason, or pass vacuously.
    assert len(recipes) >= 4, sorted(recipes)
    for recipe, text in sorted(recipes.items()):
        installs = _installs(text)
        assert len(installs) >= 3, (recipe, installs)
        for model_type, name in _DEFAULT_EXE_NAME.items():
            assert name in installs, f"{recipe} installs no {_BIN}/{name} for model type {model_type!r}"
        triton_source = installs[_DEFAULT_EXE_NAME["triton"]]
        assert triton_source.startswith(_TRITON_ONLY_BUILD), (
            f"{recipe}: model type 'triton' runs {_DEFAULT_EXE_NAME['triton']}, copied from "
            f"{triton_source}, which is not the SWMM-disabled build"
        )
        assert installs[_DEFAULT_EXE_NAME["tritonswmm"]].startswith(_COUPLED_BUILD), recipe
