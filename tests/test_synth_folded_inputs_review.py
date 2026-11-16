"""Reviewer tests for the TRITON inputs folded into the single-sim Rivanna tier.

The ``test_synth_`` name sends this file to the Rivanna selector. It is not run on the laptop: the last test drives the
single-sim pipeline, and the static tests below share the venue with it so the file is one unit.

What the five tests guard, in the words of the design of the simulation-test folds: the overlay is applied
in ``synth_all_models_analysis_cached`` alone, the shared case builder and the sibling fixture are untouched, no outfall
is listed, the points and the zero raster are valid for the synthetic DEM, the docs page claims nothing the merged code
does not name, and the run-stage assertion holds when the file is entered a second time on the cached tree.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import tests.conftest as suite_conftest
import tests.test_synth_01_singlesim as single

pytestmark = [pytest.mark.usefixtures("tritonswmm_cpu_compiled")]

_REPO = Path(__file__).resolve().parents[1]
_TESTS = _REPO / "tests"
_CONFTEST = _TESTS / "conftest.py"
_OVERLAY = "_overlay_folded_inputs"
_CACHED_FIXTURE = "synth_all_models_analysis_cached"

# The keys the overlay may write: the TRITON input fields the shape adds and nothing else. A SWMM,
# system-geometry or outfall key is outside it because the shape rules the outfall arm out of this fold (R14-1) and
# keeps the synthetic cache key and every synthetic golden unchanged (DR-15).
_SHAPE_ALLOWED_OVERLAY_KEYS = frozenset(
    {
        "observation_points",
        "TRITON_observation_interval_s",
        "TRITON_courant_number",
        "TRITON_kokkos_print_configuration",
        "TRITON_initial_depth_raster",
        "TRITON_initial_depth_raster_kind",
        "TRITON_initial_qx_raster",
        "TRITON_initial_qy_raster",
    }
)

# The files that name the overlaid fixture: its definition and the two consumers that census lists.
_SHAPE_CONSUMER_FILES = frozenset(
    {"tests/conftest.py", "tests/test_synth_01_singlesim.py", "tests/test_version_migration_state.py"}
)


def _overlay_function() -> ast.FunctionDef:
    tree = ast.parse(_CONFTEST.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == _OVERLAY)


def _written_keys(function: ast.FunctionDef) -> set[str]:
    """The string keys a function stores by subscript or by a ``.update({...})`` literal."""
    keys: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) and isinstance(node.slice, ast.Constant):
            keys.add(str(node.slice.value))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update":
            for argument in node.args:
                if isinstance(argument, ast.Dict):
                    keys |= {str(k.value) for k in argument.keys if isinstance(k, ast.Constant)}
    return keys


def test_the_overlay_writes_only_triton_input_fields():
    """Every key the overlay stores is a TRITON input field of the shape; no outfall, SWMM or geometry key.

    Class: NEW CAPABILITY, plausible wrong implementation: the overlay also writes ``swmm_outfalls_to_couple``, which
    the shape rules out of this fold. Reachability: the overlay writes both YAML files, so any key it stores reaches
    the cached tree; artifact: ``tests/conftest.py`` ``_overlay_folded_inputs``. Both-states anchor: the assertion
    turns on the set of stored keys, present in every version of the overlay.

    Input class: every key stored by the overlay function. Why required: a SWMM or system-geometry key reaches the
    two consumers (``test_synth_01_singlesim.py`` and ``test_version_migration_state.py``) and, for an outfall, adds
    the conversion arm that ``tests/test_swmm_runnable_derive.py`` and ``tests/test_runnable_norfolk_preflight.py``
    already pin on the laptop, against a synthetic model whose only outfall has no link. Killed by: adding
    ``swmm_outfalls_to_couple`` to the ``system_cfg`` store. A second correct implementation also sets
    ``TRITON_initial_qx_raster``, which the allowlist admits.
    Invocation: not isolatable from this tree (Rivanna-only file) - discrimination shown instead: adding the key
    to the overlay reds this test.
    """
    keys = _written_keys(_overlay_function())
    assert {"observation_points", "TRITON_courant_number"} <= keys, f"control: the matcher reads the overlay: {keys}"
    stray = sorted(keys - _SHAPE_ALLOWED_OVERLAY_KEYS)
    assert not stray, f"the overlay writes keys outside the TRITON input fields: {stray}"


def test_the_overlay_reaches_one_fixture_and_two_consumer_files():
    """The overlay is applied by the cached fixture alone, and only the two census files name that fixture.

    Class: NEW CAPABILITY, plausible wrong implementation: the overlay moved into the shared case builder or applied
    in the sibling fixture ``synth_all_models_analysis`` (six consumer files). Reachability: a call of the overlay in
    another fixture, or a third file requesting the cached fixture, receives courant 0.45, the Kokkos print and the
    observation inputs; artifacts: ``tests/conftest.py`` and ``tests/fixtures/``. Both-states anchor: the assertion
    turns on call sites and file names, present before and after the fold.

    Input class: every function of ``tests/conftest.py``, every file under ``tests`` and every file of
    ``tests/fixtures``. Why required: the numerics change for two consumers only, and a
    third consumer silently receiving the overlay would change a test whose author never saw it. Killed by: calling
    the overlay from ``synth_all_models_analysis``, or a new test file requesting the cached fixture. A second correct
    implementation applies the overlay through a fixture of another name and moves the consumer files with it.
    Invocation: not isolatable from this tree - discrimination shown instead: the call in the sibling fixture reds
    the first assertion.
    """
    tree = ast.parse(_CONFTEST.read_text())
    appliers = sorted(
        f.name
        for f in ast.walk(tree)
        if isinstance(f, ast.FunctionDef)
        and f.name != _OVERLAY
        and any(isinstance(c, ast.Call) and getattr(c.func, "id", None) == _OVERLAY for c in ast.walk(f))
    )
    assert appliers == [_CACHED_FIXTURE], f"the overlay is applied by {appliers}"

    pattern = re.compile(rf"\b{_CACHED_FIXTURE}\b")
    naming = sorted(
        str(path.relative_to(_REPO))
        for path in _TESTS.rglob("*.py")
        if path != Path(__file__).resolve() and pattern.search(path.read_text(errors="replace"))
    )
    assert naming == sorted(_SHAPE_CONSUMER_FILES), f"files naming the overlaid fixture: {naming}"

    leaked = sorted(
        str(path.relative_to(_REPO))
        for path in (_TESTS / "fixtures").rglob("*.py")
        if _OVERLAY in path.read_text(errors="replace") or "FOLDED_" in path.read_text(errors="replace")
    )
    assert not leaked, f"the shared case builder side names the overlay: {leaked}"


def test_the_folded_points_sit_on_interior_floodplain_cells_of_the_synthetic_dem():
    """Both observation points fall on the cell the constants claim, inside the watershed and off every wall.

    Class: NEW CAPABILITY, plausible wrong implementation: a point on a wall ring or the tidal shelf, which TRITON
    computes no water on and the preflight refuses (a ``ConfigurationError`` at scenario preparation, red in
    ``test_prepare_all_scenarios`` and in ``test_hydrology_variant_local_runoff_invariant``). Reachability: the
    synthetic grid is 16 columns by 30 rows of 10 m cells at the origin, and each point maps to one cell by TRITON's
    own formula; artifacts: ``src/hhemt/synthetic_model/geometry.py`` ``dem_elev_at`` and
    ``src/hhemt/synthetic_model/vectors.py`` ``build_watershed``. Both-states anchor: the assertion turns on the DEM
    geometry, not on wording the fold introduces.

    Input class: every folded point. Why required: the constants ``FOLDED_OBSERVATION_CELLS`` are asserted against the
    processed store after the run, and the points are checked against the DEM at preparation, so a point off the
    interior floodplain reds the pipeline at its second stage with a message about the DEM and not about this fold.
    Killed by: moving ``river_centerline`` to y=45, the shelf. A second correct implementation picks two other
    interior cells and updates the constants together.
    Invocation: not isolatable from this tree - discrimination shown instead: a point at y=45 gives cell row 25, the
    sea wall, elevation 50, and reds the elevation assertion.
    """
    from hhemt.synthetic_model._build import SyntheticModelParams
    from hhemt.synthetic_model.geometry import _WALL_ELEV, dem_elev_at
    from hhemt.triton_cfg import ObservationPoint, observation_cell

    params = SyntheticModelParams()
    header = {
        "ncols": params.n_cols,
        "nrows": params.n_rows,
        "xllcorner": params.xllcorner,
        "yllcorner": params.yllcorner,
        "cellsize": params.cell_size_m,
    }
    cs = params.cell_size_m
    # The watershed polygon: inset one cell on three sides, its south edge above the shelf and the sea-wall row.
    west, east = params.xllcorner + cs, params.xllcorner + params.n_cols * cs - cs
    south, north = params.yllcorner + 5 * cs, params.yllcorner + params.n_rows * cs - cs
    cells = []
    for point_id, x, y in suite_conftest.FOLDED_OBSERVATION_POINTS:
        row, col = observation_cell(ObservationPoint(point_id, x, y), header)
        cells.append((row, col))
        assert west <= x <= east and south <= y <= north, f"{point_id} lies outside the watershed polygon"
        elevation = dem_elev_at(params, col, params.n_rows - 1 - row)
        assert elevation < _WALL_ELEV, f"{point_id} falls on a wall cell ({row}, {col})"
        assert 0.0 < elevation < 2.0, f"{point_id} elevation {elevation} is not floodplain terrain"
    assert tuple(cells) == suite_conftest.FOLDED_OBSERVATION_CELLS


def test_the_docs_page_names_only_identifiers_the_merged_code_defines():
    """Every snake_case identifier in backticks on the how-to page occurs in ``src/hhemt``.

    Class: NEW CAPABILITY, plausible wrong implementation: a page that names a config field, variable, coordinate or
    cfg key the merged code does not define (a misspelt or newly named field). Reachability: the page names the fields
    in backticks and the code defines each in the config models, the cfg renderer or the processing step; artifact:
    ``docs/how-to/triton-inputs.md``. Both-states anchor: the assertion turns on the identifier text, present in the
    page and in the code before and after any edit.

    Input class: every backticked token of the page that holds an underscore and starts with a letter. Why required:
    the brief's rule is that the docs make no claim the merged code does not make, and an identifier the reader
    types into a config is the claim a misspelling breaks. Killed by: renaming `TRITON_observation_interval_s` to
    `TRITON_observation_interval` on the page. A second correct implementation lists the tokens of the page in a
    table and the test reads the table; the token class is the same.
    Invocation: not isolatable from this tree - discrimination shown instead: the newly named field is absent from every
    source file, so it is the one reported.
    """
    page = (_REPO / "docs" / "how-to" / "triton-inputs.md").read_text()
    tokens = sorted(set(re.findall(r"`([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*)`", page)))
    assert len(tokens) >= 20, f"control: the matcher reads the page's identifiers: {tokens}"
    source = "\n".join(path.read_text(errors="replace") for path in (_REPO / "src" / "hhemt").rglob("*.py"))
    # Whole-word: a field name that is a prefix of a real one (``TRITON_observation_interval``) must not match it.
    missing = [t for t in tokens if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])", source)]
    assert not missing, f"the page names identifiers that no source file defines: {missing}"


def test_the_run_stage_assertion_holds_when_the_file_is_entered_again_on_the_cached_tree(
    synth_all_models_analysis_cached,
):
    """The run stage passes on a second entry, after the processing stage cleared the raw outputs.

    Class: REGRESSION on a warm tree. Pre-fix run: not available on a worktree (Rivanna-only); the reproduction is the
    state after ``test_process_sim``. Reachability: ``test_process_sim`` clears the raw outputs with
    ``override_clear_raw="all"``, which deletes every ``_CLEAR_RAW_DELETE_SUBDIRS`` member, ``cfg`` among them
    (``src/hhemt/process_simulation.py`` ``_clear_raw_outputs``), and the second entry of ``test_run_sim`` launches no
    solver because ``model_run_completed`` is true (``src/hhemt/run_simulation.py`` ``prepare_simulation_command``
    returns ``None``, and ``run_simulation_runner.py`` logs "simulation already completed, skipping execution"), so
    the assertion reads a ``cfg/`` folder that no run of this entry wrote. Both-states anchor: the assertion turns on
    the re-entry state of the cached tree, not on any wording of the folded inputs.

    Input class: every re-entry of ``test_run_sim`` on a tree whose sims are complete and whose raw outputs are
    cleared, which is every second run of the single-sim file on one cached tree. Why required: the file is run
    again on the same cached tree by every developer rerun, and a stage that is red on the second run reports a
    failure of the solver where there is none. Killed by: a run-stage assertion that needs the raw ``cfg`` dumps and
    does not make the solver run. A second correct implementation forces the run stage to execute the solver every
    time (the toolkit's force marker), or keeps the solver-read evidence in a place the clearing does not reach.
    Invocation: not isolatable from this tree - discrimination shown instead: a run stage that forces the solver to
    execute, or one that reads persisted evidence, greens this test.
    """
    analysis = synth_all_models_analysis_cached
    single.test_prepare_all_scenarios(analysis)
    single.test_run_sim(analysis)
    single.test_process_sim(analysis)
    single.test_run_sim(analysis)
