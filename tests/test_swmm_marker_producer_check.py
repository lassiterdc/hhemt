"""The standalone-SWMM producer check: one home for the two log literals, and the check that
the third-party program which prints them still does.

Item ``1b.3`` of the 2026-10-29 small-correctness fixes. Two producers, one helper:
``TRITONSWMM_system._compile_SWMM_locked`` emits the check into the generated
``compile_swmm.sh`` for the native clone, and ``TRITONSWMM_run.model_run_completed`` calls it
against the source tree a container image retains at ``/opt/swmm-src``.

SCOPE, stated because it is a residual rather than an absence: nothing here asserts that the
REAL clone at ``SWMM_tag_key`` still prints either literal. That needs the network and a clone,
so it is done-when item 6 -- a human runs it on Rivanna scratch. These tests cover the
toolkit's side of the contract: one home for the literals, the check emitted where a
pre-existing clone still meets it, non-failing shape under ``set -e``, and a memoised cost.

Item 3 of the criterion needs no test here by design: ``system_config``'s ``SWMM_tag_key`` and
``SWMM_git_URL`` defaults are pinned by
``tests/test_resume_event_ledger.py::test_the_swmm_defaults_are_the_ones_the_recorded_measurement_covers``,
which this chunk leaves untouched.

VENUE: laptop-safe. Every fixture is a text file, no solver is compiled or run, and
``subprocess.run`` is stubbed for the script-generation arms -- the idiom
``tests/test_setup_target_resources.py`` established for this same generator.
"""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt import run_simulation as rs
from hhemt import swmm_markers as sm
from hhemt.system import TRITONSWMM_system

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "hhemt"

# The two print sites, transcribed from USEPA v5.2.4 ``src/run/main.c`` (the measurement
# recorded at ``tests/test_resume_event_ledger.py:_SWMM_DEFAULT_MEASUREMENT``).
_BOTH_LITERALS = (
    '        printf("\\n\\n... EPA SWMM completed in %.2f seconds.", runTime);\n'
    '        if      ( swmm_getError(errMsg, msgLen) > 0 ) printf(" There are errors.\\n");\n'
)
_NO_COMPLETION = (
    '        printf("\\n\\n Simulation finished.", runTime);\n'
    '        if      ( swmm_getError(errMsg, msgLen) > 0 ) printf(" There are errors.\\n");\n'
)
_NO_ERRORS = (
    '        printf("\\n\\n... EPA SWMM completed in %.2f seconds.", runTime);\n'
    '        if      ( swmm_getError(errMsg, msgLen) > 0 ) printf(" had problems.\\n");\n'
)


def _write_main_c(root: Path, text: str) -> Path:
    main_c = root / "src" / "run" / "main.c"
    main_c.parent.mkdir(parents=True, exist_ok=True)
    main_c.write_text(text)
    return main_c


# ---------------------------------------------------------------------------
# One home for the two literals
# ---------------------------------------------------------------------------


def test_the_two_literals_are_declared_in_exactly_one_module():
    """A second copy is the drift class this consolidation exists to remove.

    MATCHER: ``f'= "{literal}"'`` (an assignment to the literal) over every ``*.py`` under
    ``src/hhemt``. UNIT: matching FILES, counted per literal occurrence in that form.
    A broken matcher returns 0 and fails this assertion, so the arm is self-controlling.
    """
    literals = (sm.SWMM_COMPLETION_MARKER, sm.SWMM_ERROR_MARKER)
    declared = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        text = path.read_text()
        for literal in literals:
            if f'= "{literal}"' in text:
                declared.append((path.relative_to(_SRC_ROOT).as_posix(), literal))
    assert declared == [("swmm_markers.py", literals[0]), ("swmm_markers.py", literals[1])]


def test_run_simulation_still_exposes_both_names():
    """``run_simulation.SWMM_COMPLETION_MARKER`` must keep resolving for existing callers."""
    assert rs.SWMM_COMPLETION_MARKER == sm.SWMM_COMPLETION_MARKER
    assert rs.SWMM_ERROR_MARKER == sm.SWMM_ERROR_MARKER


# ---------------------------------------------------------------------------
# Done-when 1: the helper against three synthetic main.c fixtures
# ---------------------------------------------------------------------------


def test_both_literals_present_reports_no_warning(tmp_path):
    assert sm.missing_markers(_write_main_c(tmp_path, _BOTH_LITERALS)) == ()


def test_missing_completion_literal_is_named(tmp_path):
    assert sm.missing_markers(_write_main_c(tmp_path, _NO_COMPLETION)) == (sm.SWMM_COMPLETION_MARKER,)


def test_missing_errors_literal_is_named_and_has_its_own_arm(tmp_path):
    """The errors literal is a NEGATIVE conjunct of the verdict, so its absence is the
    dangerous one: a failed run would read COMPLETE. It gets its own fixture rather than
    sharing the completion arm, because a single "something is missing" assertion cannot
    distinguish which literal moved.
    """
    assert sm.missing_markers(_write_main_c(tmp_path, _NO_ERRORS)) == (sm.SWMM_ERROR_MARKER,)


def test_no_source_tree_reports_none_not_both_missing(tmp_path):
    """``None`` (nothing to check) must not collapse into "both missing", or every native run
    would warn about a container tree that was never built."""
    assert sm.missing_markers_in_source_tree(tmp_path / "absent") is None
    assert sm.producer_check(tmp_path / "absent") is None


# ---------------------------------------------------------------------------
# The two ways a RETAINED tree can be present-but-uncheckable. Both must degrade to the
# absent answer rather than raise: this runs inside a completion verdict, and a warning
# that can raise aborts the verdict it was meant to annotate (reviewer finding F1).
# ---------------------------------------------------------------------------


def test_a_main_c_that_is_a_directory_degrades_to_none(tmp_path, caplog):
    """``IsADirectoryError`` from the read must not escape to the caller.

    A directory at the ``main.c`` path withholds a verdict; it does not carry one. Reported as
    "both missing" it would be a false alarm; raised, it would abort the completion verdict.
    """
    (tmp_path / "src" / "run" / "main.c").mkdir(parents=True)
    sm.clear_producer_check_cache()
    rs._SWMM_PRODUCER_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="hhemt.run_simulation"):
        assert sm.missing_markers_in_source_tree(tmp_path) is None
        assert sm.producer_check(tmp_path) is None
        assert rs._warn_if_swmm_producer_drifted(tmp_path) is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == "hhemt.run_simulation"]


def test_an_unreadable_main_c_degrades_to_none(tmp_path, caplog):
    """``PermissionError`` from the read must not escape to the caller."""
    main_c = _write_main_c(tmp_path, _BOTH_LITERALS)
    main_c.chmod(0o000)
    try:
        if os.access(main_c, os.R_OK):
            pytest.skip("a privileged session reads a 0o000 file, so the PermissionError arm cannot be exercised here")
        sm.clear_producer_check_cache()
        rs._SWMM_PRODUCER_WARNED.clear()
        with caplog.at_level(logging.WARNING, logger="hhemt.run_simulation"):
            assert sm.missing_markers_in_source_tree(tmp_path) is None
            assert sm.producer_check(tmp_path) is None
            assert rs._warn_if_swmm_producer_drifted(tmp_path) is None
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == "hhemt.run_simulation"]
    finally:
        main_c.chmod(0o644)


# ---------------------------------------------------------------------------
# The container-path chokepoint: warns, names the literal, once per process
# ---------------------------------------------------------------------------


def test_container_arm_warns_and_names_the_missing_literal(tmp_path, caplog):
    tree = tmp_path / "swmm-src"
    _write_main_c(tree, _NO_ERRORS)
    sm.clear_producer_check_cache()
    rs._SWMM_PRODUCER_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="hhemt.run_simulation"):
        missing = rs._warn_if_swmm_producer_drifted(tree)
    assert missing == (sm.SWMM_ERROR_MARKER,)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == "hhemt.run_simulation"]
    assert len(warnings) == 1
    assert sm.SWMM_ERROR_MARKER in warnings[0].getMessage()


def test_the_container_warning_fires_once_per_root(tmp_path, caplog):
    """A resumed campaign evaluates one verdict per event; the standing fact is reported once."""
    tree = tmp_path / "swmm-src"
    _write_main_c(tree, _NO_COMPLETION)
    sm.clear_producer_check_cache()
    rs._SWMM_PRODUCER_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="hhemt.run_simulation"):
        first = rs._warn_if_swmm_producer_drifted(tree)
        second = rs._warn_if_swmm_producer_drifted(tree)
    assert first == second == (sm.SWMM_COMPLETION_MARKER,)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == "hhemt.run_simulation"]
    assert len(warnings) == 1


def test_the_container_arm_is_wired_into_the_standalone_swmm_verdict():
    """The check must sit in the ``swmm`` arm of the ONE verdict function, not elsewhere."""
    import inspect

    src = inspect.getsource(rs.TRITONSWMM_run.model_run_completed)
    call_at = src.index("_warn_if_swmm_producer_drifted()")
    # After the TRITON/TRITON-SWMM arm, i.e. inside the `else:  # swmm` branch.
    assert call_at > src.index('success = "Simulation ends" in log_content')
    assert call_at < src.index("SWMM_COMPLETION_MARKER in log_content")


# ---------------------------------------------------------------------------
# Done-when 4: memoised, so a per-log-read check costs one stat after the first
# ---------------------------------------------------------------------------


def test_producer_check_reads_the_file_once_and_stats_once_per_call(tmp_path, monkeypatch):
    """MATCHER: wraps ``Path.read_text`` and ``Path.stat`` and records the receives.

    UNIT: calls reaching each method with THIS ``main.c`` as the receiver. Two calls to
    ``producer_check`` must give exactly one file READ and exactly one STAT per call, which
    is the done-when's "one stat after the first" stated as a count rather than a claim.
    """
    sm.clear_producer_check_cache()
    main_c = _write_main_c(tmp_path, _BOTH_LITERALS)

    reads = []
    real_read_text = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        reads.append(self)
        return real_read_text(self, *args, **kwargs)

    stats = []
    real_stat = Path.stat

    def counting_stat(self, *args, **kwargs):
        stats.append(self)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    monkeypatch.setattr(Path, "stat", counting_stat)

    assert sm.producer_check(tmp_path) == ()
    assert sm.producer_check(tmp_path) == ()
    assert reads == [main_c]
    assert stats == [main_c, main_c]


def test_a_rewritten_source_is_re_read_rather_than_answered_from_cache(tmp_path):
    """The memo is invalidated on ``(st_mtime_ns, st_size)``, so a swapped tree is not trusted."""
    sm.clear_producer_check_cache()
    _write_main_c(tmp_path, _BOTH_LITERALS)
    assert sm.producer_check(tmp_path) == ()
    _write_main_c(tmp_path, _NO_COMPLETION)
    assert sm.producer_check(tmp_path) == (sm.SWMM_COMPLETION_MARKER,)


# ---------------------------------------------------------------------------
# Done-when 2: the generated compile script carries the checks OUTSIDE the clone branch
# ---------------------------------------------------------------------------


def _fake_swmm_self(tmp_path: Path, *, tag: str | None) -> SimpleNamespace:
    """Duck-typed ``self`` sufficient to drive ``_compile_SWMM_locked`` to its script write.

    ``swmm_source/CMakeLists.txt`` is present, which is the done-when's fixture condition:
    the clone branch does not fire, so anything the script still carries proves it was
    emitted OUTSIDE that branch.
    """
    src = tmp_path / "swmm_source"
    src.mkdir()
    (src / "CMakeLists.txt").write_text("")
    fake = SimpleNamespace(
        _execution_container_mode=False,
        additional_modules=None,
        compilation_swmm_successful=True,
        swmm_executable=tmp_path / "runswmm",
        cfg_system=SimpleNamespace(SWMM_tag_key=tag, SWMM_git_URL="https://example.invalid/swmm.git"),
        log=SimpleNamespace(
            compilation_swmm_successful=SimpleNamespace(set=lambda _v: None),
            write=lambda: None,
        ),
    )
    fake._emit_libstdcpp_ld_preamble_lines = lambda: TRITONSWMM_system._emit_libstdcpp_ld_preamble_lines(fake)
    fake._emit_module_load_lines = lambda modules: TRITONSWMM_system._emit_module_load_lines(fake, modules)
    # The script TEXT is written before the subprocess runs, so the provenance capture in the
    # success branch cannot influence anything asserted here.
    fake._capture_swmm_provenance = lambda *_a, **_k: None
    return fake


def _generate_swmm_script(tmp_path: Path, monkeypatch, *, redownload: bool = False, tag: str | None = "v5.2.4"):
    import hhemt.system as system_mod

    def _fake_run(cmd, stdout=None, stderr=None, **kwargs):
        # The emitter polls its logfile for this marker before returning.
        if stdout is not None:
            stdout.write("script finished\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(system_mod.subprocess, "run", _fake_run)
    TRITONSWMM_system._compile_SWMM_locked(
        _fake_swmm_self(tmp_path, tag=tag),
        recompile_if_already_done_successfully=True,
        redownload_swmm_if_exists=redownload,
        verbose=False,
        build_dir=tmp_path,
    )
    return (tmp_path / "compile_swmm.sh").read_text()


def _grep_line(marker: str) -> str:
    return f"if ! grep -qF -- '{marker}' \"$_swmm_main_c\"; then"


def test_checks_are_emitted_when_the_clone_branch_does_not_fire(tmp_path, monkeypatch):
    """THE arm that catches the obvious wrong implementation.

    A pre-existing clone at a different tag never re-runs the clone branch, and it is exactly
    that clone whose source decides the verdict. An implementation that emits the checks
    inside the ``if`` passes every other assertion in this file and fails this one.
    """
    text = _generate_swmm_script(tmp_path, monkeypatch)
    assert "git clone" not in text, "fixture precondition: a present CMakeLists.txt must skip the clone"
    for marker in sm.SWMM_MARKERS:
        assert _grep_line(marker) in text


def test_checks_follow_the_clone_block_when_it_does_fire(tmp_path, monkeypatch):
    text = _generate_swmm_script(tmp_path, monkeypatch, redownload=True)
    clone_at = text.index("git clone")
    for marker in sm.SWMM_MARKERS:
        assert text.index(f"grep -qF -- '{marker}'") > clone_at


def test_the_check_cannot_abort_a_set_e_script(tmp_path, monkeypatch):
    """Non-failing is a shape assertion, not an intention: a bare ``grep`` aborts under ``set -e``."""
    text = _generate_swmm_script(tmp_path, monkeypatch)
    assert "set -e  # Exit on error" in text
    grep_lines = [line for line in text.splitlines() if "grep -qF" in line]
    assert len(grep_lines) == len(sm.SWMM_MARKERS)
    for line in grep_lines:
        assert line.lstrip().startswith("if ! grep -qF"), line
    for marker in sm.SWMM_MARKERS:
        named = [line for line in text.splitlines() if "WARNING:" in line and marker in line]
        assert named, f"no WARNING line names {marker!r}"


def test_the_script_interpolates_the_literals_rather_than_retyping_them(tmp_path, monkeypatch):
    """One source: changing the module's tuple must reach the generated bash."""
    import hhemt.system as system_mod

    monkeypatch.setattr(system_mod, "SWMM_MARKERS", ("SENTINEL-MARKER-A", "SENTINEL-MARKER-B"))
    text = _generate_swmm_script(tmp_path, monkeypatch)
    assert "SENTINEL-MARKER-A" in text
    assert "SENTINEL-MARKER-B" in text
    assert rs.SWMM_COMPLETION_MARKER not in text


# ---------------------------------------------------------------------------
# The emitted fragment, EXECUTED. A shape assertion cannot tell "non-failing" from
# "never fires"; these two arms are each other's control.
# ---------------------------------------------------------------------------


def _run_emitted_check(tmp_path: Path, main_c_text: str):
    """Run the emitted check block under ``set -e`` against a synthetic source tree."""
    import subprocess

    import hhemt.system as system_mod

    tree = tmp_path / "swmm_source"
    _write_main_c(tree, main_c_text)
    fragment = "\n".join(["set -e", f'SWMM_SOURCE_DIR="{tree}"', *system_mod._emit_swmm_marker_check_lines()])
    return subprocess.run(["bash", "-c", fragment], capture_output=True, text=True)


def test_executed_check_warns_naming_the_missing_literal_and_exits_zero(tmp_path):
    """``set -e`` is in force for the whole fragment, so a bare ``grep`` would exit 1 here."""
    completed = _run_emitted_check(tmp_path, _NO_ERRORS)
    assert completed.returncode == 0, completed.stderr
    assert "WARNING:" in completed.stderr
    assert sm.SWMM_ERROR_MARKER in completed.stderr
    assert sm.SWMM_COMPLETION_MARKER not in completed.stderr.split("WARNING:", 1)[-1]


def test_executed_check_is_silent_when_both_literals_are_present(tmp_path):
    """The negative control for the arm above: a ``grep -qF`` that never matches would pass it."""
    completed = _run_emitted_check(tmp_path, _BOTH_LITERALS)
    assert completed.returncode == 0
    assert "WARNING:" not in completed.stderr
