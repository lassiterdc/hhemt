"""REVIEWING-SEAT suite for chunk ``2a.1`` / item ``1b.3`` (shape § 7.3).

This file is the reviewing seat's independent re-derivation of the § 7.3 done-when
items. It is NOT a copy of the coder's ``tests/test_swmm_marker_producer_check.py``:
every fixture, fake and instrument here is authored from the criterion, so a shared
mistake in the coder's file cannot make both suites green.

VENUE: laptop-safe. Every fixture is a text file. ``subprocess.run`` is stubbed for
the script-generation arms and, for the arms that EXECUTE the emitted fragment,
only ``bash`` runs -- no solver is compiled or executed, no network, no ``uv sync``.

WHY THE SUITE IS SHAPED THIS WAY. § 7.3 names two arms as the ones that catch a
plausible wrong implementation:

- item 2 -- generate the script for a ``build_dir`` that already holds a
  ``CMakeLists.txt`` so the clone branch does NOT fire, and assert the checks are
  still emitted. An implementation that emits them inside the clone ``if`` passes
  every other assertion and fails that one.
- item 4 -- the helper is memoised, so a per-log-read check costs one ``stat`` after
  the first rather than a file read on every completion evaluation.

A third property the criterion states in prose and no shape assertion can reach is
that the container arm is a WARNING. § 7.3 says the check "deliberately does NOT
change ``success``", and Decision 3's priority (4) is "a warning rather than a
refusal". A diagnostic layered on a per-log-read status predicate must therefore be
incapable of aborting that predicate. ``test_review_container_arm_is_failure_isolated_*``
is the arm that reads that sentence as a behavioural requirement rather than as a
comment on the ``success`` assignment, and it is the one FAILING arm in this suite.

``MAIN_C`` fixture bodies are transcribed from USEPA v5.2.4 ``src/run/main.c`` as
recorded at ``tests/test_resume_event_ledger.py:_SWMM_DEFAULT_MEASUREMENT``'s
neighbouring comment (lines 91-92 at that tag).
"""

from __future__ import annotations

import ast
import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt import run_simulation as rs
from hhemt import swmm_markers as sm
from hhemt.system import TRITONSWMM_system

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src" / "hhemt"
_RECIPE_DIR = _SRC_ROOT / "sif" / "recipes"

# --- synthetic main.c bodies (USEPA v5.2.4 src/run/main.c, lines 91-92) -------

MAIN_C_BOTH = (
    '        printf("\\n\\n... EPA SWMM completed in %.2f seconds.", runTime);\n'
    '        if      ( swmm_getError(errMsg, msgLen) > 0 ) printf(" There are errors.\\n");\n'
)
MAIN_C_NO_COMPLETION = (
    '        printf("\\n\\n Simulation finished.", runTime);\n'
    '        if      ( swmm_getError(errMsg, msgLen) > 0 ) printf(" There are errors.\\n");\n'
)
MAIN_C_NO_ERRORS = (
    '        printf("\\n\\n... EPA SWMM completed in %.2f seconds.", runTime);\n'
    '        if      ( swmm_getError(errMsg, msgLen) > 0 ) printf(" had problems.\\n");\n'
)
#: `src/solver/text.h` carries `" There are errors."` as `#define FMT03`. A matcher
#: that searched the whole source tree would pass even when `main.c` stopped
#: printing the clause -- which is why the helper must be scoped to `main.c`.
TEXT_H_WITH_ERRORS = '#define FMT03 " There are errors."\n'


def _write_main_c(root: Path, text: str) -> Path:
    main_c = root / "src" / "run" / "main.c"
    main_c.parent.mkdir(parents=True, exist_ok=True)
    main_c.write_text(text)
    return main_c


# ---------------------------------------------------------------------------
# One home for the two literals
# ---------------------------------------------------------------------------


def test_review_literals_are_assigned_in_exactly_one_python_file_repo_wide() -> None:
    """Property: the two log literals have exactly ONE declaration in the repository.

    MATCHER: ``{" = "}{literal}{'"'}`` is not used; the instrument is the source line
    ``SWMM_<NAME> = "<literal>"`` parsed by ``ast``, so a prose or C-string mention
    cannot satisfy it. UNIT: declared ASSIGNMENTS, over every ``*.py`` under
    ``src/``, ``tests/`` and ``scripts/``.

    MUST-FIRE CONTROL: the same instrument must find BOTH assignments in
    ``swmm_markers.py``. A matcher that had silently broken (an ``ast`` API change, a
    wrong root) would report the same empty list it reports for a real duplicate-free
    tree, so the control is what separates the two.
    """
    found: list[tuple[str, str]] = []
    scanned = 0
    for root in (_REPO_ROOT / "src", _REPO_ROOT / "tests", _REPO_ROOT / "scripts"):
        for path in sorted(root.rglob("*.py")):
            if ".venv" in path.parts or "__pycache__" in path.parts:
                continue
            scanned += 1
            try:
                tree = ast.parse(path.read_text(errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                    continue
                if node.value.value not in sm.SWMM_MARKERS:
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        found.append((path.relative_to(_REPO_ROOT).as_posix(), node.value.value))

    # must-fire control: the instrument sees the canonical declarations
    assert scanned > 100, f"instrument scanned only {scanned} files; root resolution is wrong"
    canonical = [f for f in found if f[0].endswith("swmm_markers.py")]
    assert sorted(v for _, v in canonical) == sorted(sm.SWMM_MARKERS), (
        "MUST-FIRE CONTROL FAILED: the instrument did not see the canonical assignments, "
        f"so its answer about duplicates means nothing. found={found}"
    )
    assert [f for f in found if not f[0].endswith("swmm_markers.py")] == [], (
        f"a second declaration of a SWMM log literal exists: {found}"
    )


def test_review_run_simulation_reexports_the_literals_without_redeclaring() -> None:
    """``run_simulation.SWMM_COMPLETION_MARKER`` must resolve, and must be the SAME object."""
    assert rs.SWMM_COMPLETION_MARKER is sm.SWMM_COMPLETION_MARKER
    assert rs.SWMM_ERROR_MARKER is sm.SWMM_ERROR_MARKER

    text = (_SRC_ROOT / "run_simulation.py").read_text()
    for name in ("SWMM_COMPLETION_MARKER", "SWMM_ERROR_MARKER"):
        assert f'{name} = "' not in text, (
            f"{name} is re-declared in run_simulation.py; the import is supposed to be the "
            "only binding there, and a second copy is the drift this consolidation removed."
        )


# ---------------------------------------------------------------------------
# Item 1 -- the helper against three synthetic main.c fixtures
# ---------------------------------------------------------------------------


def test_review_item1_helper_verdicts_for_the_three_fixtures(tmp_path: Path) -> None:
    """One fixture per arm; ``()`` means healthy, and each missing literal is named."""
    assert sm.missing_markers(_write_main_c(tmp_path / "both", MAIN_C_BOTH)) == ()

    missing_completion = sm.missing_markers(_write_main_c(tmp_path / "no_completion", MAIN_C_NO_COMPLETION))
    assert missing_completion == (sm.SWMM_COMPLETION_MARKER,)

    missing_errors = sm.missing_markers(_write_main_c(tmp_path / "no_errors", MAIN_C_NO_ERRORS))
    assert missing_errors == (sm.SWMM_ERROR_MARKER,), (
        "the errors literal is a NEGATIVE conjunct of the verdict: its absence makes a FAILED "
        "run read COMPLETE. This arm must name it on its own."
    )


def test_review_item1_errors_arm_does_not_also_report_completion(tmp_path: Path) -> None:
    """The two arms are distinguishable, so a single 'something is missing' cannot pass both."""
    missing = sm.missing_markers(_write_main_c(tmp_path, MAIN_C_NO_ERRORS))
    assert sm.SWMM_COMPLETION_MARKER not in missing


def test_review_item1_scoped_to_main_c_so_text_h_cannot_mask_a_missing_literal(tmp_path: Path) -> None:
    """``src/solver/text.h`` carries the errors clause too; a tree-wide search is vacuous."""
    _write_main_c(tmp_path, " EPA SWMM completed in 1.0 seconds.\n")
    text_h = tmp_path / "src" / "solver" / "text.h"
    text_h.parent.mkdir(parents=True, exist_ok=True)
    text_h.write_text(TEXT_H_WITH_ERRORS)

    sm.clear_producer_check_cache()
    assert sm.producer_check(tmp_path) == (sm.SWMM_ERROR_MARKER,), (
        "text.h's copy of the errors clause masked main.c's absence: the helper is searching "
        "too broadly and the producer check would be vacuous."
    )


def test_review_absent_tree_is_none_not_both_missing(tmp_path: Path) -> None:
    """``None`` (nothing to check) must not collapse into 'both missing'."""
    sm.clear_producer_check_cache()
    assert sm.producer_check(tmp_path / "absent") is None
    assert sm.missing_markers_in_source_tree(tmp_path / "absent") is None


def test_review_swmm_markers_order_is_completion_then_errors() -> None:
    """The tuple is in verdict order, and it is what the bash emitter iterates."""
    assert sm.SWMM_MARKERS == (sm.SWMM_COMPLETION_MARKER, sm.SWMM_ERROR_MARKER)


# ---------------------------------------------------------------------------
# Script generation harness (item 2 / interpolation / non-failing)
# ---------------------------------------------------------------------------


def _fake_self(tmp_path: Path, *, tag: str = "v5.2.4") -> SimpleNamespace:
    """Duck-typed ``self`` sufficient to drive ``_compile_SWMM_locked`` to its script write.

    ``swmm_source/CMakeLists.txt`` is present, which IS the § 7.3 item-2 fixture
    condition: the clone branch does not fire, so anything the generated script still
    carries proves it was emitted OUTSIDE that branch.
    """
    src = tmp_path / "swmm_source"
    src.mkdir(exist_ok=True)
    (src / "CMakeLists.txt").write_text("")
    fake = SimpleNamespace(
        additional_modules=None,
        compilation_swmm_successful=True,
        swmm_executable=tmp_path / "runswmm",
        cfg_system=SimpleNamespace(
            SWMM_tag_key=tag,
            SWMM_git_URL="https://example.invalid/swmm.git",
        ),
        log=SimpleNamespace(
            compilation_swmm_successful=SimpleNamespace(set=lambda _v: None),
            write=lambda: None,
        ),
    )
    # Stubbed rather than bound to the real emitters: this suite asserts only the
    # producer-check block, and a stub keeps the fixture independent of environment.
    fake._emit_libstdcpp_ld_preamble_lines = lambda: []
    fake._emit_module_load_lines = lambda modules: []
    fake._capture_swmm_provenance = lambda *_a, **_k: None
    return fake


def _generate_script(tmp_path: Path, monkeypatch, *, redownload: bool = False) -> str:
    import hhemt.system as system_mod

    def _fake_run(cmd, stdout=None, stderr=None, **kwargs):
        # The emitter polls its logfile for this marker before returning.
        if stdout is not None:
            stdout.write("script finished\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(system_mod.subprocess, "run", _fake_run)
    TRITONSWMM_system._compile_SWMM_locked(
        _fake_self(tmp_path),
        recompile_if_already_done_successfully=True,
        redownload_swmm_if_exists=redownload,
        verbose=False,
        build_dir=tmp_path,
    )
    return (tmp_path / "compile_swmm.sh").read_text()


def _grep_needle(marker: str) -> str:
    return f"grep -qF -- '{marker}'"


# ---------------------------------------------------------------------------
# Item 2 -- emitted OUTSIDE the clone branch
# ---------------------------------------------------------------------------


def test_review_item2_checks_emitted_when_the_clone_branch_does_not_fire(tmp_path, monkeypatch) -> None:
    """THE arm. A pre-existing clone at another tag never re-runs the clone branch.

    Two-arm differential: the VIOLATING state is 'checks only emitted where the clone
    lives', which this arm detects because the clone never runs and the checks must
    still be present. The satisfying state -- clone fires, checks after it -- is the
    paired arm below, so this assertion is not merely strict, it is positioned.
    """
    text = _generate_script(tmp_path, monkeypatch, redownload=False)

    # fixture precondition, asserted rather than assumed
    assert "git clone" not in text, "fixture precondition: a present CMakeLists.txt must skip the clone"
    assert not (tmp_path / "swmm_source" / "CMakeLists.txt").read_text(), "fixture sanity"

    for marker in sm.SWMM_MARKERS:
        assert _grep_needle(marker) in text, (
            f"the generated script omits the producer check for {marker!r} when the clone "
            "branch does not fire -- the check is inside the clone `if` and a pre-existing "
            "clone at a different tag is never inspected."
        )
        named = [ln for ln in text.splitlines() if "WARNING:" in ln and marker in ln]
        assert named, f"no WARNING line names {marker!r}"


def test_review_item2_checks_follow_the_clone_when_it_fires(tmp_path, monkeypatch) -> None:
    """The paired arm: when the clone DOES fire, the checks come after it."""
    text = _generate_script(tmp_path, monkeypatch, redownload=True)
    clone_at = text.index("git clone")
    for marker in sm.SWMM_MARKERS:
        assert text.index(_grep_needle(marker)) > clone_at


def test_review_emitted_check_reads_the_variable_the_clone_wrote_to(tmp_path, monkeypatch) -> None:
    """``SWMM_SOURCE_DIR`` must be bound BEFORE the check uses it.

    The wrong-but-plausible implementation writes the check against a shell variable
    that is only assigned inside the clone branch; on the pre-existing-clone path that
    variable is empty, `[ -f "" ]` is false, and the check silently never runs -- a
    green-looking script whose item-2 arm is vacuous at execution time.
    """
    text = _generate_script(tmp_path, monkeypatch, redownload=False)
    assignment = text.index("SWMM_SOURCE_DIR=")
    first_use = text.index("_swmm_main_c=")
    assert assignment < first_use, (
        "the producer check references SWMM_SOURCE_DIR before it is assigned; on the "
        "pre-existing-clone path the variable would be empty and the check would never run."
    )
    # and the check must use that variable, not a hard-coded path
    assert "${SWMM_SOURCE_DIR}/src/run/main.c" in text


def test_review_emitted_check_targets_main_c_only(tmp_path, monkeypatch) -> None:
    """Scoped to ``main.c``: a tree-wide grep would match ``src/solver/text.h``."""
    text = _generate_script(tmp_path, monkeypatch, redownload=False)
    grep_lines = [ln for ln in text.splitlines() if "grep -qF" in ln]
    assert grep_lines, "no grep lines emitted"
    for line in grep_lines:
        assert '"$_swmm_main_c"' in line, f"grep is not scoped to main.c: {line!r}"
    assert "src/run/main.c" in text


# ---------------------------------------------------------------------------
# Non-failing under `set -e` -- executed, not merely shape-asserted
# ---------------------------------------------------------------------------


def _run_emitted_fragment(tmp_path: Path, main_c_text: str | None) -> subprocess.CompletedProcess:
    """Execute the emitted check block under ``set -e`` against a synthetic tree."""
    import hhemt.system as system_mod

    tree = tmp_path / "swmm_source"
    tree.mkdir(exist_ok=True)
    if main_c_text is not None:
        _write_main_c(tree, main_c_text)
    fragment = "\n".join(["set -e", f'SWMM_SOURCE_DIR="{tree}"', *system_mod._emit_swmm_marker_check_lines()])
    return subprocess.run(["bash", "-c", fragment], capture_output=True, text=True)


def test_review_emitted_check_non_failing_and_names_the_missing_literal(tmp_path: Path) -> None:
    """``set -e`` is in force, so a bare ``grep`` would exit 1 here; it must not."""
    completed = _run_emitted_fragment(tmp_path, MAIN_C_NO_ERRORS)
    assert completed.returncode == 0, f"the check aborted a `set -e` script: {completed.stderr}"
    assert "WARNING:" in completed.stderr
    assert sm.SWMM_ERROR_MARKER in completed.stderr
    assert sm.SWMM_COMPLETION_MARKER not in completed.stderr.split("WARNING:", 1)[-1]


def test_review_emitted_check_silent_negative_control(tmp_path: Path) -> None:
    """Negative control: a ``grep`` that never matched would satisfy the arm above."""
    completed = _run_emitted_fragment(tmp_path, MAIN_C_BOTH)
    assert completed.returncode == 0
    assert "WARNING:" not in completed.stderr


def test_review_emitted_check_silent_when_main_c_absent(tmp_path: Path) -> None:
    """An absent ``main.c`` is not a finding: no producer to be wrong about."""
    completed = _run_emitted_fragment(tmp_path, None)
    assert completed.returncode == 0, completed.stderr
    assert "WARNING:" not in completed.stderr


def test_review_emitted_check_ignores_text_h_when_executed(tmp_path: Path) -> None:
    """Executed counterpart of the helper-level scoping arm."""
    import hhemt.system as system_mod

    tree = tmp_path / "swmm_source"
    _write_main_c(tree, " EPA SWMM completed in 1.0 seconds.\n")
    text_h = tree / "src" / "solver" / "text.h"
    text_h.parent.mkdir(parents=True, exist_ok=True)
    text_h.write_text(TEXT_H_WITH_ERRORS)
    fragment = "\n".join(["set -e", f'SWMM_SOURCE_DIR="{tree}"', *system_mod._emit_swmm_marker_check_lines()])
    completed = subprocess.run(["bash", "-c", fragment], capture_output=True, text=True)
    assert completed.returncode == 0
    assert sm.SWMM_ERROR_MARKER in completed.stderr, (
        "text.h's copy of the errors clause masked main.c's absence in the EXECUTED check"
    )


# ---------------------------------------------------------------------------
# Literals interpolated from the module, never retyped into bash
# ---------------------------------------------------------------------------


def test_review_literals_are_interpolated_from_the_module(tmp_path, monkeypatch) -> None:
    """One source: changing the module tuple must reach the generated bash.

    This patches ``system.py``'s module-level binding (the seam the emitter actually
    reads at call time) and asserts the REAL literals do not appear, so an emitter
    with a hard-coded copy fails even though it would match the unpatched tuple.
    """
    import hhemt.system as system_mod

    monkeypatch.setattr(system_mod, "SWMM_MARKERS", ("REVIEW-SENTINEL-A", "REVIEW-SENTINEL-B"))
    text = _generate_script(tmp_path, monkeypatch, redownload=False)
    assert "REVIEW-SENTINEL-A" in text
    assert "REVIEW-SENTINEL-B" in text
    assert sm.SWMM_COMPLETION_MARKER not in text, (
        "the generated script still carries the real completion literal after the module "
        "tuple was changed: the literals are retyped into bash rather than interpolated."
    )
    assert sm.SWMM_ERROR_MARKER not in text


# ---------------------------------------------------------------------------
# Item 3 -- the pinning defaults are unchanged (read from the code, not the doc)
# ---------------------------------------------------------------------------


def _recorded_swmm_measurement() -> tuple[str, str, str]:
    """Extract ``_SWMM_DEFAULT_MEASUREMENT`` from the existing test WITHOUT executing it."""
    source = (_REPO_ROOT / "tests" / "test_resume_event_ledger.py").read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_SWMM_DEFAULT_MEASUREMENT" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, tuple) and len(value) == 3
            return value  # type: ignore[return-value]
    raise AssertionError("_SWMM_DEFAULT_MEASUREMENT not found: item 3's pin has moved or been deleted")


def test_review_item3_defaults_still_match_the_recorded_measurement() -> None:
    """§ 7.3 item 3: the two ``system_config`` defaults are unchanged."""
    from hhemt.config.system import system_config

    url, _commit, tag = _recorded_swmm_measurement()
    assert system_config.model_fields["SWMM_git_URL"].default == url
    assert system_config.model_fields["SWMM_tag_key"].default == tag


def test_review_item3_the_pinning_test_is_still_present() -> None:
    """The chunk must ASSERT this, not edit it: the test function must still exist."""
    source = (_REPO_ROOT / "tests" / "test_resume_event_ledger.py").read_text()
    assert "def test_the_swmm_defaults_are_the_ones_the_recorded_measurement_covers(" in source


# ---------------------------------------------------------------------------
# Item 4 -- memoised: one file read, one stat per call
# ---------------------------------------------------------------------------


def test_review_item4_one_file_read_total_and_one_stat_per_call(tmp_path, monkeypatch) -> None:
    """MATCHER: wraps ``Path.read_text``/``Path.stat`` and counts calls whose receiver
    is THIS ``main.c``. UNIT: method CALLS (not lines, not matches).

    Two ``producer_check`` calls must give exactly ONE read and exactly TWO stats.
    """
    sm.clear_producer_check_cache()
    main_c = _write_main_c(tmp_path, MAIN_C_BOTH)

    reads: list[Path] = []
    real_read_text = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        if self == main_c:
            reads.append(self)
        return real_read_text(self, *args, **kwargs)

    stats: list[Path] = []
    real_stat = Path.stat

    def counting_stat(self, *args, **kwargs):
        if self == main_c:
            stats.append(self)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    monkeypatch.setattr(Path, "stat", counting_stat)

    assert sm.producer_check(tmp_path) == ()
    assert sm.producer_check(tmp_path) == ()

    assert len(reads) == 1, f"the file was read {len(reads)} times; the memo is not holding"
    assert len(stats) == 2, f"expected exactly one stat per call, got {len(stats)}"


def test_review_item4_memo_is_keyed_on_a_normalised_path(tmp_path) -> None:
    """Equivalent spellings of one root must not each pay a first-read cost."""
    sm.clear_producer_check_cache()
    main_c = _write_main_c(tmp_path, MAIN_C_BOTH)

    reads = [0]
    real_read_text = Path.read_text

    def counting(self, *args, **kwargs):
        if self == main_c:
            reads[0] += 1
        return real_read_text(self, *args, **kwargs)

    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr(Path, "read_text", counting)
        assert sm.producer_check(tmp_path) == ()
        assert sm.producer_check(str(tmp_path)) == ()
        assert sm.producer_check(tmp_path / ".") == ()
    finally:
        mp.undo()
    assert reads[0] == 1, (
        f"three equivalent spellings of one root caused {reads[0]} reads; the memo key is not normalised."
    )


def test_review_item4_the_chokepoint_is_a_single_function() -> None:
    """The per-log-read call site must go through ONE helper that calls ``producer_check``."""
    import inspect

    wrapper_src = inspect.getsource(rs._warn_if_swmm_producer_drifted)
    assert "producer_check(" in wrapper_src

    verdict_src = inspect.getsource(rs.TRITONSWMM_run.model_run_completed)
    # exactly one producer-check call site in the verdict function
    assert verdict_src.count("_warn_if_swmm_producer_drifted(") == 1


# ---------------------------------------------------------------------------
# Container arm -- warns, names the literal, once per root, does not change `success`
# ---------------------------------------------------------------------------


def test_review_container_arm_warns_and_names_the_missing_literal(tmp_path, caplog) -> None:
    tree = tmp_path / "swmm-src"
    _write_main_c(tree, MAIN_C_NO_ERRORS)
    sm.clear_producer_check_cache()
    rs._SWMM_PRODUCER_WARNED.clear()

    with caplog.at_level(logging.WARNING, logger="hhemt.run_simulation"):
        missing = rs._warn_if_swmm_producer_drifted(tree)

    assert missing == (sm.SWMM_ERROR_MARKER,)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == "hhemt.run_simulation"]
    assert len(warnings) == 1
    assert sm.SWMM_ERROR_MARKER in warnings[0].getMessage()


def test_review_container_arm_warns_once_per_root_but_not_across_roots(tmp_path, caplog) -> None:
    """Dedup is per root: a second drifted root must still be reported."""
    first = tmp_path / "root-a"
    second = tmp_path / "root-b"
    _write_main_c(first, MAIN_C_NO_COMPLETION)
    _write_main_c(second, MAIN_C_NO_COMPLETION)
    sm.clear_producer_check_cache()
    rs._SWMM_PRODUCER_WARNED.clear()

    with caplog.at_level(logging.WARNING, logger="hhemt.run_simulation"):
        rs._warn_if_swmm_producer_drifted(first)
        rs._warn_if_swmm_producer_drifted(first)  # duplicate: suppressed
        rs._warn_if_swmm_producer_drifted(second)  # different root: reported

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == "hhemt.run_simulation"]
    assert len(warnings) == 2, f"expected one warning per drifted root; got {len(warnings)}"


def test_review_container_arm_does_not_change_the_verdict() -> None:
    """§ 7.3 + Decision 3: the check is a WARNING, never a refusal.

    Asserted structurally because the call's return value being DISCARDED is the
    property that makes it incapable of altering ``success``: the call must be a bare
    expression statement, and the verdict line must be the unchanged conjunction.
    """
    import inspect
    import re

    src = inspect.getsource(rs.TRITONSWMM_run.model_run_completed)
    assert re.search(r"^\s*_warn_if_swmm_producer_drifted\(\)\s*$", src, re.MULTILINE), (
        "the producer check's result is bound somewhere; it must be a bare call so it cannot influence `success`."
    )
    assert "success = SWMM_COMPLETION_MARKER in log_content and SWMM_ERROR_MARKER not in log_content" in src


def test_review_container_source_root_matches_every_recipe_clone_target() -> None:
    """Every recipe must clone the standalone source to the helper's read root.

    If one recipe cloned elsewhere, the container arm would return ``None`` for that
    image forever and the coverage Decision 3 claims would be silently partial.
    """
    recipes = sorted(_RECIPE_DIR.glob("*.def"))
    assert len(recipes) == 4, f"expected four container recipes, found {[p.name for p in recipes]}"

    url = "https://github.com/USEPA/Stormwater-Management-Model.git"
    targets: dict[str, str] = {}
    for recipe in recipes:
        for line in recipe.read_text().splitlines():
            if url in line:
                tokens = line.split()
                idx = next(i for i, tok in enumerate(tokens) if tok == url)
                assert idx + 1 < len(tokens), f"{recipe.name}: clone URL carries no destination"
                targets[recipe.name] = tokens[idx + 1]
                break
        assert recipe.name in targets, f"{recipe.name}: no standalone-SWMM clone found"

    expected = str(sm.CONTAINER_SWMM_SOURCE_ROOT)
    assert set(targets.values()) == {expected}, (
        f"recipe clone targets {targets} disagree with CONTAINER_SWMM_SOURCE_ROOT={expected!r}"
    )


# ---------------------------------------------------------------------------
# The refusal: no recipe and no sif/args.py was edited
# ---------------------------------------------------------------------------


def test_review_no_marker_check_was_injected_into_any_recipe_or_args_module() -> None:
    """Decision 3 / § 7.3: editing a recipe re-keys every built image.

    The check is a bash ``grep`` over the literal, so a recipe that carried the check
    would carry the literal too. Content-based rather than git-diff-based, so it holds
    whether or not the chunk is committed.
    """
    for recipe in sorted(_RECIPE_DIR.glob("*.def")):
        text = recipe.read_text()
        for marker in sm.SWMM_MARKERS:
            assert marker not in text, (
                f"{recipe.name} carries {marker!r}: a recipe edit re-keys every built image "
                "through sif/identity.py's recipe_sha256, which this chunk must not do."
            )

    args_text = (_SRC_ROOT / "sif" / "args.py").read_text()
    for marker in sm.SWMM_MARKERS:
        assert marker not in args_text, f"src/hhemt/sif/args.py carries {marker!r}"


def test_review_sif_identity_still_keys_on_the_recipe_hash() -> None:
    """The ground for the refusal: a recipe edit is not image-identity-neutral."""
    from hhemt.sif.identity import SifIdentity

    assert "recipe_sha256" in SifIdentity.model_fields
    import inspect

    key_src = inspect.getsource(SifIdentity.key.fget)  # type: ignore[attr-defined]
    assert "model_dump" in key_src, (
        "SifIdentity.key no longer hashes model_dump(); the recipe-hash re-key mechanism this "
        "chunk's refusal rests on would have changed."
    )


# ---------------------------------------------------------------------------
# FINDING -- the container arm is not failure-isolated on the read path
# ---------------------------------------------------------------------------


def test_review_container_arm_is_failure_isolated_when_main_c_is_a_directory(tmp_path) -> None:
    """FINDING. A retained tree whose ``main.c`` path is not a readable regular file
    must degrade exactly like the absent-tree case, not abort the verdict.

    ``producer_check`` catches ``OSError`` from ``stat`` and returns ``None`` ("no source
    tree here"), but ``missing_markers``'s ``read_text`` is unprotected, so
    ``IsADirectoryError`` -- an ``OSError`` subclass, i.e. the same class the stat branch
    already tolerates -- escapes. ``_warn_if_swmm_producer_drifted`` is called from
    ``TRITONSWMM_run.model_run_completed``, a status predicate reached from report
    renderers and ``analysis.py``'s status sweep, so the exception turns a diagnostic
    WARNING into a REFUSAL that prevents the completion verdict from being computed at
    all. That contradicts § 7.3's "deliberately does NOT change ``success``" and
    Decision 3's priority (4), "a warning rather than a refusal".

    Root-independent: creating a directory at ``main.c`` needs no chmod.
    """
    tree = tmp_path / "swmm-src"
    main_c = tree / "src" / "run" / "main.c"
    main_c.mkdir(parents=True)  # a directory where the source file is expected
    sm.clear_producer_check_cache()
    rs._SWMM_PRODUCER_WARNED.clear()

    try:
        verdict = sm.producer_check(tree)
    except OSError as exc:  # noqa: BLE001 - the finding IS that this escapes
        pytest.fail(
            "producer_check raised "
            f"{type(exc).__name__} from the read path. The stat branch deliberately tolerates "
            "OSError (returns None); the read branch does not, so a non-regular `main.c` aborts "
            "model_run_completed instead of producing a warning. Failure-isolate the read the "
            "same way the stat is isolated."
        )
    assert verdict is None, f"expected the absent-tree degradation (None); got {verdict!r}"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a mode-000 file, so the arm is vacuous")
def test_review_container_arm_is_failure_isolated_when_main_c_is_unreadable(tmp_path: Path) -> None:
    """FINDING (second reachable arm). An EACCES on a retained ``main.c`` must not abort.

    Reachable when the retained source is present but not world-readable -- e.g. an
    image whose source tree was copied with restrictive umask, or a FUSE/NFS-mounted
    ``/opt/swmm-src``. Consequence is identical to the directory arm: a status predicate
    raising instead of warning.
    """
    tree = tmp_path / "swmm-src"
    main_c = _write_main_c(tree, MAIN_C_BOTH)
    os.chmod(main_c, 0o000)
    sm.clear_producer_check_cache()
    rs._SWMM_PRODUCER_WARNED.clear()
    try:
        try:
            verdict = rs._warn_if_swmm_producer_drifted(tree)
        except OSError as exc:  # noqa: BLE001
            pytest.fail(
                "_warn_if_swmm_producer_drifted raised "
                f"{type(exc).__name__} on an unreadable retained main.c. A per-log-read "
                "diagnostic reached from model_run_completed must be incapable of aborting the "
                "verdict; catch OSError around the read as the stat branch already does."
            )
        assert verdict in (None, ())
    finally:
        os.chmod(main_c, 0o600)


# ---------------------------------------------------------------------------
# ROUND 2 -- F1 fix verification. The guard must isolate the failure WITHOUT
# becoming a hole: equivalence with the absent-tree case, and the warning path
# for a source that IS readable and DOES lack a marker must survive.
# ---------------------------------------------------------------------------


def _warning_count(caplog) -> int:
    return len([r for r in caplog.records if r.levelno == logging.WARNING and r.name == "hhemt.run_simulation"])


def test_review_fix_unavailable_producers_are_indistinguishable_to_the_caller(tmp_path, caplog) -> None:
    """Absent, directory and unreadable must all give the SAME answer and NO warning.

    "A warning rather than a refusal" requires the caller be unable to distinguish
    "there is no producer" from "the producer could not be read": both are 'nothing to
    check', neither is a verdict, and neither may raise. If the fix made unreadable
    degrade to some THIRD state -- or emitted a warning for it -- the caller would
    behave differently for the two, which is the hole this arm exists to catch.
    """
    if os.geteuid() == 0:
        pytest.skip("root can read a mode-000 file, so the unreadable arm is vacuous")

    absent = tmp_path / "absent"
    directory = tmp_path / "directory"
    (directory / "src" / "run" / "main.c").mkdir(parents=True)
    unreadable = tmp_path / "unreadable"
    unreadable_main_c = _write_main_c(unreadable, MAIN_C_BOTH)
    os.chmod(unreadable_main_c, 0o000)

    sm.clear_producer_check_cache()
    rs._SWMM_PRODUCER_WARNED.clear()
    try:
        with caplog.at_level(logging.WARNING, logger="hhemt.run_simulation"):
            results = [
                sm.producer_check(absent),
                sm.producer_check(directory),
                sm.producer_check(unreadable),
            ]
            wrapped = [
                rs._warn_if_swmm_producer_drifted(absent),
                rs._warn_if_swmm_producer_drifted(directory),
                rs._warn_if_swmm_producer_drifted(unreadable),
            ]
    finally:
        os.chmod(unreadable_main_c, 0o600)

    assert results == [None, None, None], f"unavailable producers disagreed: {results}"
    assert wrapped == [None, None, None], f"the wrapper disagreed: {wrapped}"
    assert _warning_count(caplog) == 0, (
        "an unavailable producer emitted a warning; 'nothing to check' must be silent, or the "
        "caller can distinguish an absent tree from an unreadable one."
    )


def test_review_fix_readable_source_missing_a_marker_still_warns(tmp_path, caplog) -> None:
    """THE other half of the guard: the fix must not swallow a GENUINE drift finding.

    A readable `main.c` that lacks a marker is a real verdict, not an unavailable
    producer. If the guard were widened until it swallowed this too, the check would be
    silent in exactly the case it exists to report. Both literals get an arm.
    """
    for missing_marker, body in (
        (sm.SWMM_COMPLETION_MARKER, MAIN_C_NO_COMPLETION),
        (sm.SWMM_ERROR_MARKER, MAIN_C_NO_ERRORS),
    ):
        root = tmp_path / missing_marker.replace(" ", "_").replace(".", "")
        _write_main_c(root, body)
        sm.clear_producer_check_cache()
        rs._SWMM_PRODUCER_WARNED.clear()
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger="hhemt.run_simulation"):
            verdict = rs._warn_if_swmm_producer_drifted(root)

        assert verdict == (missing_marker,), (
            f"a readable source missing {missing_marker!r} returned {verdict!r}; the guard has "
            "swallowed a genuine drift finding."
        )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == "hhemt.run_simulation"]
        assert len(warnings) == 1, f"expected one warning naming {missing_marker!r}; got {len(warnings)}"
        assert missing_marker in warnings[0].getMessage()


def test_review_fix_read_failure_is_not_memoised(tmp_path: Path) -> None:
    """A transient read failure must not become a standing verdict for the process.

    The docstring claims a read FAILURE is not cached. Asserted behaviourally: an
    unreadable tree answers None, and once readable the SAME root must produce the real
    verdict with `stat` unchanged (mtime and size untouched by chmod), which is only
    possible if the failure was never memoised.
    """
    if os.geteuid() == 0:
        pytest.skip("root can read a mode-000 file, so the failure cannot be induced")

    tree = tmp_path / "swmm-src"
    main_c = _write_main_c(tree, MAIN_C_NO_ERRORS)
    os.chmod(main_c, 0o000)
    sm.clear_producer_check_cache()
    try:
        assert sm.producer_check(tree) is None
    finally:
        os.chmod(main_c, 0o600)

    assert sm.producer_check(tree) == (sm.SWMM_ERROR_MARKER,), (
        "the unreadable read was memoised: after the file became readable the same root still "
        "answered None, so a transient permission problem silenced the check for the process."
    )


def test_review_fix_call_site_in_model_run_completed_is_unchanged() -> None:
    """The fix must live in the helper, not move or reshape the call site."""
    import inspect
    import re

    src = inspect.getsource(rs.TRITONSWMM_run.model_run_completed)
    assert src.count("_warn_if_swmm_producer_drifted(") == 1
    assert re.search(r"^\s*_warn_if_swmm_producer_drifted\(\)\s*$", src, re.MULTILINE), (
        "the call is no longer a bare expression statement"
    )
    assert "success = SWMM_COMPLETION_MARKER in log_content and SWMM_ERROR_MARKER not in log_content" in src, (
        "the verdict line changed alongside the fix"
    )


def test_review_fix_guard_is_scoped_to_oserror_not_a_bare_except() -> None:
    """Lexical control: the guard must name OSError, and `missing_markers` must stay strict.

    A bare `except:` or `except Exception:` would swallow programming errors too, and
    moving the guard INTO `missing_markers` would lose the None verdict the caller needs.
    """
    import inspect

    tree_src = inspect.getsource(sm.missing_markers_in_source_tree)
    assert "except OSError" in tree_src
    assert "except Exception" not in tree_src
    assert "except:" not in tree_src

    check_src = inspect.getsource(sm.producer_check)
    assert "except OSError" in check_src
    assert "except Exception" not in check_src

    # `missing_markers` propagates on purpose; its signature has no None verdict.
    assert "OSError" in inspect.getsource(sm.missing_markers)
