"""The coupled-SWMM version label must be measured from the `project(...)` block.

WHICH INSTRUMENT THIS IS, stated before anything else because the three candidates are not
interchangeable. These tests **extract the extraction command from each recipe at test time and
execute it** against a fixture `CMakeLists.txt`. That pins BEHAVIOUR (the command's sed grammar and
its pipeline order, not its spelling) and it pins the CLASS of sites together, because every recipe
under `src/hhemt/sif/recipes/` is discovered rather than listed. A text-only assertion would pin
spelling and would silently un-pin on the next reformat; these do not.

THE BOUND, stated so the test is not read as stronger than it is. The recipe reads the IN-IMAGE path
`/opt/triton/external/swmm/CMakeLists.txt`, which does not exist on a host. The test therefore
rebases that one literal onto the fixture path before executing — and asserts the literal was
PRESENT first, so a renamed in-image path fails loudly instead of making the rebase a no-op. What
this exercises is the shipped command's grammar and ordering on a controlled file. What it cannot
exercise is an image build, and it does not claim to.

WHY THIS TEST IS THE ONLY DURABLE GUARD. `org.hhemt.swmm_coupled_version` is written by `%post` at
build time and read by nothing: `SWMM_COUPLED_VERSION_LABEL` is declared in `container_labels.py`
and consumed nowhere, and `sif/preflight.py` compares four other labels. So no runtime gate and no
preflight can catch this value regressing — only a test can. That is the whole reason the package
spends a chunk on it.

THE DEFECT BEING PINNED. `sed -n 's/.*VERSION \\([0-9][0-9.]*\\).*/\\1/p' <file> | head -1` returns the
file's FIRST version-shaped token, which in the vendored solver's `CMakeLists.txt` is
`cmake_minimum_required (VERSION 3.13)` — a CMake minimum, not a SWMM version. The label therefore
recorded 3.13 instead of the project's 5.2.3.

THE `HHEMT_SIF_RECIPES_DIR` SEAM, and why a shipped test carries one. The pre-fix failure of these
same tests must be demonstrable, and the only faithful way to demonstrate it is to run THIS MODULE
against pre-fix recipe text. The environment variable points the module at a directory of recipes;
it defaults to this repository's own. It is null-safe for the ordinary run and it is not a
behavioural switch — every assertion is the same either way.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

#: The recipe directory under test: this repository's, unless a pre-fix arm points elsewhere.
RECIPE_DIR = Path(
    os.environ.get(
        "HHEMT_SIF_RECIPES_DIR",
        str(Path(__file__).resolve().parents[1] / "src" / "hhemt" / "sif" / "recipes"),
    )
)

#: The in-image path every recipe reads. Rebasing THIS literal is the only edit the test makes to
#: the shipped command, and `_run` refuses to run unless it is present.
IN_IMAGE_CMAKE = "/opt/triton/external/swmm/CMakeLists.txt"

#: The PRE-FIX command, kept verbatim as the FIXTURE'S CONTROL. It is not what the recipes carry any
#: more; it is here so that the fixture can be shown to reproduce the trap. Without this, a fixture
#: whose first version-shaped token happened to be the project's would let both forms pass and the
#: real test below would be measuring nothing.
PREFIX_COMMAND = "sed -n 's/.*VERSION \\([0-9][0-9.]*\\).*/\\1/p' {path} | head -1"

#: Captures the shell command inside the recipe's `_swmm_coupled="$( ... )"` assignment. Anchored on
#: the assignment so the later `echo "org.hhemt.swmm_coupled_version $_swmm_coupled"` cannot match.
_ASSIGNMENT = re.compile(r'^[ \t]*_swmm_coupled="\$\((.*)\)"[ \t]*$', re.M)

#: Every recipe shipped in the directory under test, discovered rather than listed so a FIFTH
#: recipe is covered the day it lands instead of the day someone remembers to add it here.
RECIPES = sorted(RECIPE_DIR.glob("*.def"))


def _fixture_cmakelists(tmp_path: Path) -> Path:
    """A `CMakeLists.txt` that reproduces the trap.

    Its FIRST version-shaped token is the `cmake_minimum_required (VERSION 3.13)` line, and the
    project block declares `VERSION 5.2.3` below it — the vendored solver's own shape, reduced to
    the two lines the extraction can see.
    """
    path = tmp_path / "CMakeLists.txt"
    path.write_text(
        "#\n"
        "# CMakeLists.txt - CMake configuration file for swmm-solver\n"
        "#\n"
        "\n"
        "cmake_minimum_required (VERSION 3.13)\n"
        "\n"
        "project(swmm-solver\n"
        "    VERSION 5.2.3\n"
        "    LANGUAGES C CXX\n"
        ")\n"
    )
    return path


def _extracted_command(recipe: Path) -> str:
    """The shell command inside one recipe's `_swmm_coupled=` assignment."""
    match = _ASSIGNMENT.search(recipe.read_text())
    assert match is not None, (
        f'{recipe.name} carries no `_swmm_coupled="$( ... )"` assignment; either the extraction '
        "was renamed or this test has drifted from the file it guards"
    )
    return match.group(1).strip()


def _shell(command: str) -> str:
    """Run one shell command and return its stripped stdout, raising on a non-zero exit."""
    completed = subprocess.run(  # noqa: S603 - repo content by construction; see the module docstring
        ["bash", "-c", command], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _run(command: str, cmakelists: Path) -> str:
    """Execute a RECIPE command with the in-image path rebased onto the fixture."""
    assert IN_IMAGE_CMAKE in command, (
        f"this command does not read {IN_IMAGE_CMAKE}, so rebasing it would be a silent no-op and "
        f"the assertion below would measure nothing: {command!r}"
    )
    return _shell(command.replace(IN_IMAGE_CMAKE, str(cmakelists)))


def test_the_fixture_reproduces_the_trap(tmp_path):
    """CONTROL — the fixture really does defeat the pre-fix command.

    Property: on the fixture, the PRE-FIX command returns `3.13` — the CMake minimum — and NOT the
    project's `5.2.3`.

    Class: CONTROL, not a finding, and it is labelled that way rather than dressed as a defect
    test. It exists to make the test below falsifiable: if the fixture's first version-shaped token
    were the project's, the pre-fix command would return `5.2.3` too and
    ``test_every_recipe_measures_the_project_version`` would pass against a broken recipe.

    Why it is required: the whole defect is an ORDERING accident in the file, so a fixture without
    the ordering cannot distinguish the two commands.

    What kills it: editing the fixture so `project(...)` precedes `cmake_minimum_required`.

    A second correct formulation that still passes: asserting the fixture's first
    `VERSION`-matching line directly, rather than by running the pre-fix sed.
    """
    fixture = _fixture_cmakelists(tmp_path)
    assert _shell(PREFIX_COMMAND.format(path=fixture)) == "3.13", (
        "the fixture must put the CMake minimum first, or the real test below cannot tell the two commands apart"
    )


@pytest.mark.parametrize("recipe_name", [p.name for p in RECIPES])
def test_every_recipe_measures_the_project_version(recipe_name, tmp_path):
    """The shipped extraction returns the project's VERSION, not the file's first one.

    Property: for EVERY recipe in the directory under test, the extraction command that recipe
    actually carries, executed against a fixture reproducing the trap, returns `5.2.3`.

    Class: REGRESSION — red on the tree before package `1c`, where every recipe carried the
    pre-fix command and returned `3.13`. The failing output is pasted in the round file against
    recipes from `22daab73^`.

    Why it is required, as a consequence a reader can check against the code: `%post` emits this
    value into `org.hhemt.swmm_coupled_version` at build time and nothing reads it back, so a
    regression here changes what every newly built image CLAIMS about its solver and is caught by
    no gate, no preflight and no run. The label is the only record of which SWMM the coupling
    actually used.

    What kills it: reverting the recipe to the pre-fix command, or any edit whose sed loses the
    `project(` address range.

    A second correct implementation that still passes: any command that binds the number to the
    project declaration — asking CMake itself for `PROJECT_VERSION`, or anchoring on the VERSION
    line's own indentation — returns the same string here, so this test does not over-constrain the
    mechanism the way a text assertion would.
    """
    recipe = RECIPE_DIR / recipe_name
    value = _run(_extracted_command(recipe), _fixture_cmakelists(tmp_path))
    assert value == "5.2.3", (
        f"{recipe_name}'s extraction returned {value!r} on a fixture whose project version is "
        "5.2.3; a `3.13` here means the command is reading the CMake minimum again, and a `9.9` "
        "would mean it is reading a stray VERSION line"
    )


def test_the_recipes_carry_one_identical_extraction():
    """The class is ONE defect replicated, so every site must carry the same command.

    Property: the set of extraction commands across every recipe in the directory under test has
    exactly one member, and there is more than one recipe, so the property is not vacuous.

    Class: REGRESSION — this is the assertion that catches a chunk landing 1 of 4 and reporting
    success. It is red on a tree where SOME sites are fixed and others are not, which neither the
    behavioural test above (it is parametrized per site and reports per site) nor any single-site
    test can express.

    Why it is required: the defect is a duplicated command, and duplication is what makes partial
    repair possible. A per-site test cannot say "these four are one class".

    What kills it: fixing fewer than all sites — demonstrated in the round file by a directory of
    three fixed recipes and one pre-fix recipe, where this assertion alone reds.

    A second correct implementation that still passes: extracting the command by a different
    mechanism (a shell parse, a line-prefix match) — the assertion is over EQUALITY ACROSS SITES,
    never over the text any one site carries. **Its bound, stated plainly:** this is the one
    assertion here that touches spelling, and only as SAMENESS, so a uniform reformat across all
    four leaves it green while the behavioural test still guards the grammar.

    NOT pinned: the NUMBER of recipes. Discovered by glob, so a fifth recipe added later inherits
    the obligation automatically and this test does not have to be edited to cover it.
    """
    assert len(RECIPES) > 1, (
        f"only {len(RECIPES)} recipe(s) found under {RECIPE_DIR}; the equality below would be "
        "vacuous, which usually means the pre-fix seam points at the wrong directory"
    )
    commands = {_extracted_command(recipe) for recipe in RECIPES}
    assert len(commands) == 1, (
        "the recipes no longer carry one identical extraction — the class is a duplicated command, "
        f"so a partial repair shows up here. Distinct forms ({len(commands)}):\n" + "\n\n".join(sorted(commands))
    )
