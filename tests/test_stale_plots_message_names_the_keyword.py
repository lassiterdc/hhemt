"""The stale-plots refusal must tell the user an escape that exists.

`provenance.assert_plots_match_running_build` refuses a report render whose figures
were made by a different toolkit build, raising `StalePlotsError`. That message is the
only guidance a user gets at the moment the render stops, so it has to name an escape
the code actually has. The escape is the Python keyword `declare_stale_plots=True` on
`TRITONSWMM_analysis.render_report`, `Bundle.regenerate_report` and
`CombinedBundle.regenerate_report`; with it the guard prints the message and returns
it, and every caller discards that return, so nothing is stored. No command-line flag
exists. A message naming a `--declare-stale-plots` flag, or promising that the
mismatch is "recorded", sends the user to something that is not there.

Every test reaches the message through `assert_plots_match_running_build` itself, never
through a copy of the string, and none constructs a `TRITONSWMM_analysis`: the figures
are real sidecar/figure pairs in a temporary directory, and only the running build's
stamp is fixed, so the result does not depend on the git state of the checkout running
the tests.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

import hhemt.provenance as provenance
from hhemt.exceptions import StalePlotsError

_RUNNING_SHA = "0" * 40
_OTHER_SHA = "f" * 40
_SECOND_OTHER_SHA = "e" * 40


def _write_figure(analysis_dir: Path, stem: str, stamp: dict | None) -> None:
    """Write one figure and its manifest sidecar under `analysis_dir/plots/`.

    `stamp=None` writes a sidecar with no build stamp, which the guard reads as a
    figure that cannot be shown to match any build.
    """
    plots = analysis_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / f"{stem}.html").write_text("<div>figure</div>", encoding="utf-8")
    (plots / f"{stem}.manifest.json").write_text(json.dumps(stamp or {}), encoding="utf-8")


def _fix_running_build(monkeypatch: pytest.MonkeyPatch, sha: str) -> None:
    """Pin the running build's stamp, which the guard otherwise reads from git."""
    monkeypatch.setattr(
        provenance,
        "producing_stamp",
        lambda: {"hhemt_sha": sha, "hhemt_version": "test", "hhemt_dirty": "false"},
    )


def _stamp(sha: str) -> dict:
    return {"hhemt_sha": sha, "hhemt_dirty": "false"}


# One case per mismatch branch of the guard. Each builds a different `why`, and the
# escape sentence is appended to all of them, so the class is "every message the guard
# raises" rather than one branch's message.
_MISMATCH_CASES = {
    "figures_from_another_build": ([("a", _stamp(_OTHER_SHA))], _RUNNING_SHA),
    "figures_inconsistent_with_each_other": (
        [("a", _stamp(_OTHER_SHA)), ("b", _stamp(_SECOND_OTHER_SHA))],
        _RUNNING_SHA,
    ),
    "figures_carry_no_stamp": ([("a", None)], _RUNNING_SHA),
    "running_build_unidentified": ([("a", _stamp(_OTHER_SHA))], ""),
}


#: A long command-line option: two hyphens immediately followed by a letter. The guard's
#: own `why` text uses " -- " as a dash, always followed by a space, so it never matches.
_LONG_OPTION = re.compile(r"--[A-Za-z]")


def _build_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> Path:
    figures, running_sha = _MISMATCH_CASES[case]
    for stem, stamp in figures:
        _write_figure(tmp_path, stem, stamp)
    _fix_running_build(monkeypatch, running_sha)
    return tmp_path


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("case", sorted(_MISMATCH_CASES))
def test_refusal_names_the_keyword_and_no_flag(tmp_path, monkeypatch, case) -> None:
    """The raised message names the real escape and no flag that does not exist.

    Property: the `StalePlotsError` message names `declare_stale_plots=True`, names both
    methods that accept it, names no command-line long option at all (the retired
    `--declare-stale-plots` spelling or any other), and does not say the mismatch is
    recorded.
    Class: every message the guard raises, one parametrized case per mismatch branch
    (figures from another build, figures inconsistent with each other, figures with no
    stamp, running build unidentified).
    Why: the message is the user's only instruction when the render stops; a flag no
    command accepts, or a promise of a record nobody writes, leaves the user with no
    working way forward.
    What kills it: restoring "pass --declare-stale-plots to proceed and record the
    mismatch", naming any other long option (for example `--declare_stale_plots=True`
    or `--allow-stale-plots`), or dropping the keyword or either method name.
    Second correct implementation: a message that puts the escape first and the
    force_rerun remedy second, or that phrases the escape as "pass declare_stale_plots=True
    to render_report() or regenerate_report()", still passes.
    """
    analysis_dir = _build_case(tmp_path, monkeypatch, case)

    with pytest.raises(StalePlotsError) as excinfo:
        provenance.assert_plots_match_running_build(analysis_dir)
    message = str(excinfo.value)

    assert "declare_stale_plots=True" in message
    assert "render_report()" in message
    assert "regenerate_report()" in message
    assert "--declare-stale-plots" not in message
    assert not _LONG_OPTION.search(message), _LONG_OPTION.search(message)
    assert "record the mismatch" not in message


def test_declared_escape_prints_the_same_message_and_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    """With the keyword, the guard prints and returns the message and stores nothing.

    Property: `declare_stale_plots=True` returns the same corrected message instead of
    raising, prints it to standard output, and leaves the analysis directory byte-for-byte
    unchanged.
    Class: the declared arm of the guard, on a single-build mismatch.
    Why: the message tells the user the mismatch is printed and not stored; this is the
    half of that claim the guard itself controls.
    What kills it: the guard writing the message to a file under `analysis_dir`, or the
    declared arm returning a message that still names the flag.
    Second correct implementation: printing with a different prefix, or to standard
    output through `logging` with a stream handler, still passes as long as the text
    reaches standard output and nothing is written under `analysis_dir`.
    Not established: that every CALLER discards the returned message. The three callers
    were read to discard it at the time of writing; this test does not read them.
    """
    analysis_dir = _build_case(tmp_path, monkeypatch, "figures_from_another_build")
    before = _tree_snapshot(analysis_dir)

    returned = provenance.assert_plots_match_running_build(analysis_dir, declare_stale_plots=True)

    assert isinstance(returned, str)
    assert "declare_stale_plots=True" in returned
    assert "--declare-stale-plots" not in returned
    assert not _LONG_OPTION.search(returned), _LONG_OPTION.search(returned)
    assert returned in capsys.readouterr().out
    assert _tree_snapshot(analysis_dir) == before


def test_matching_build_passes(tmp_path, monkeypatch) -> None:
    """Control: figures from the running build reach the comparison and pass.

    Property: when every figure carries the running build's stamp the guard returns
    None and raises nothing.
    Class: the satisfying input for the parametrized refusal above, built by the same
    helpers, so a refusal there is caused by the stamp and not by the fixture.
    Why: without it, a fixture the guard could never accept would make the refusal
    tests pass for the wrong reason.
    What kills it: the guard refusing a matching build, or the helpers writing a
    sidecar the guard cannot read.
    Second correct implementation: a guard that also checks figure timestamps still
    passes, since the figures here are freshly written.
    """
    _write_figure(tmp_path, "a", _stamp(_RUNNING_SHA))
    _fix_running_build(monkeypatch, _RUNNING_SHA)

    assert provenance.assert_plots_match_running_build(tmp_path) is None


def test_methods_the_message_names_accept_the_keyword() -> None:
    """Every method the message tells the user to call accepts `declare_stale_plots`.

    Property: `TRITONSWMM_analysis.render_report`, `Bundle.regenerate_report` and
    `CombinedBundle.regenerate_report` each take a `declare_stale_plots` parameter.
    Class: the three public entry points that call the guard with the keyword.
    Why: the message is only true while these signatures carry the keyword; a rename of
    the parameter on any of them would make the message wrong again with nothing else
    failing.
    What kills it: renaming or removing the parameter on any of the three methods.
    Second correct implementation: the parameter becoming keyword-only, or gaining a
    different default, still passes.
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.bundle import Bundle
    from hhemt.bundle._combine import CombinedBundle

    for method in (
        TRITONSWMM_analysis.render_report,
        Bundle.regenerate_report,
        CombinedBundle.regenerate_report,
    ):
        assert "declare_stale_plots" in inspect.signature(method).parameters, method.__qualname__
