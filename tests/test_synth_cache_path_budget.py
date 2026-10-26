"""The synthetic cache keeps every SWMM file path within the solver's limits.

SWMM keeps a file name only up to MAXFNAME = 259 characters. The TRITON fork refuses an
input, report or binary-output path longer than that before `swmm_open`, and it writes
the flooding-debug report only when that path is at most 258 characters. These tests
compute the paths a synthetic simulation uses, from the same cache root and tier
address the fixtures use, and check them against those limits. None of them compiles
or runs a solver.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from hhemt.scenario import compute_event_id_slug
from tests.fixtures import worktree_slug
from tests.fixtures._run_dir_reaper import collect_live_signals
from tests.fixtures._triton_source_cache import slug_runs_root

#: The SWMM files of one coupled simulation, relative to its simulation folder, with
#: the longest path the solver accepts for each. `scenario.py` places the input at
#: `swmm/hydraulics.inp` and the report at `out_tritonswmm/swmm/hydraulics.rpt`; the
#: solver names the binary output after the input and names the debug report itself.
_SWMM_FILES = (
    ("input", "swmm/hydraulics.inp", 259),
    ("report", "out_tritonswmm/swmm/hydraulics.rpt", 259),
    ("binary output", "out_tritonswmm/swmm/hydraulics.out", 259),
    ("flooding debug report", "out_tritonswmm/swmm/swmm_flooding_debug.txt", 258),
)

#: The checkout and XDG cache root of the Rivanna run whose sensitivity debug path
#: measured 272 characters. The user name is replaced by one of the same length.
_MEASURED_CWD = "/sfs/weka/scratch/user01/build_repin_c11/merged_fd1d_rep2/tree"
_MEASURED_XDG = "/scratch/user01/build_repin_c11/merged_fd1d_rep2/xdg"


def _sim_folder(runs_root: Path, analysis_name: str, member: str | None) -> Path:
    """A synthetic simulation folder.

    `test_case_builder.py` sets `system_directory = runs_root / analysis_name`, and
    `analysis.py` places the analysis at `system_directory / analysis_id`, where the
    builder sets `analysis_id` to the same name. A sensitivity member adds
    `members/member_{id}`. Simulations sit under `sims/{event_id}`.
    """
    analysis_dir = runs_root / analysis_name / analysis_name
    if member is not None:
        analysis_dir = analysis_dir / "members" / f"member_{member}"
    return analysis_dir / "sims" / compute_event_id_slug({"event_index": 9})


def _over_limit(sim_folder: Path) -> list[str]:
    over = []
    for role, rel, limit in _SWMM_FILES:
        path = str(sim_folder / rel)
        if len(path) > limit:
            over.append(f"{role} path is {len(path)} characters (limit {limit}): {path}")
    return over


@pytest.mark.skipif(sys.platform != "linux", reason="platformdirs reads XDG_CACHE_HOME on Linux only")
def test_the_measured_rivanna_geometry_now_fits(monkeypatch):
    """The geometry that produced a 272-character debug path at fork `bd18215`."""
    monkeypatch.setenv("XDG_CACHE_HOME", _MEASURED_XDG)
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: cls(_MEASURED_CWD)))
    sim_folder = _sim_folder(slug_runs_root(worktree_slug()), "synth_sensitivity", "2")
    assert _over_limit(sim_folder) == []


def _literal_analysis_names() -> set[str]:
    names: set[str] = set()
    for f in Path(__file__).resolve().parent.rglob("*.py"):
        names.update(re.findall(r'analysis_name="([A-Za-z0-9_]+)"', f.read_text(encoding="utf-8")))
    return names


def test_every_literal_synthetic_case_fits_under_this_sessions_tier():
    """Every literal case name, as a sensitivity member, under this session's own tier.

    A case that is not a sensitivity analysis has a shorter path, so treating every
    case as one bounds them all.
    """
    names = _literal_analysis_names()
    assert len(names) >= 20, f"case-name census found only {sorted(names)}; the matcher has stopped matching"
    runs_root = slug_runs_root(worktree_slug())
    over = {name: _over_limit(_sim_folder(runs_root, name, "9")) for name in sorted(names)}
    over = {name: lines for name, lines in over.items() if lines}
    assert not over, (
        f"synthetic SWMM paths under {runs_root} exceed the solver's limits; the cache root "
        f"(XDG_CACHE_HOME or HOME) or the worktree name is too long: {over}"
    )


def test_reaper_path_arm_drops_a_superseded_tier(tmp_path, monkeypatch):
    """A tier whose name is not what its live checkout derives today is not kept live."""
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    checkout = tmp_path / "checkout"
    (checkout / "tests" / "fixtures").mkdir(parents=True)
    # The checkout runs the digest derivation, so its escaped tier is superseded BY IT.
    (checkout / "tests" / "fixtures" / "_triton_source_cache.py").write_text("def checkout_slug(cwd):\n    pass\n")
    checkout = checkout.resolve()
    monkeypatch.chdir(checkout)
    current = worktree_slug()
    superseded = "path" + "".join("_s" if ch == "/" else "_u" if ch == "_" else ch for ch in str(checkout))
    assert current != superseded, "the tier address is still the escaped checkout path"
    runs_root = tmp_path / "runs"
    for slug in (current, superseded):
        (runs_root / slug).mkdir(parents=True)
        (runs_root / slug / ".origin").write_text(f"{checkout}\n")
    signals = collect_live_signals(repo, runs_root=runs_root, candidate_slugs=frozenset({current, superseded}))
    assert signals.live_path_slugs == frozenset({current})


def test_reaper_path_arm_keeps_a_tier_its_checkout_still_mints(tmp_path):
    """A checkout still at a commit from before the digest address keeps its escaped tier live.

    Checkouts at different commits share one cache root, and the reaper runs in every
    session of every one of them. A checkout whose own code still mints the escaped
    address is still using that tier, so a reaper from a newer commit must not treat it
    as superseded.
    """
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    checkout = tmp_path / "old_checkout"
    (checkout / "tests" / "fixtures").mkdir(parents=True)
    (checkout / "tests" / "fixtures" / "_triton_source_cache.py").write_text("def slug_runs_root(slug):\n    pass\n")
    checkout = checkout.resolve()
    escaped = "path" + "".join("_s" if ch == "/" else "_u" if ch == "_" else ch for ch in str(checkout))
    runs_root = tmp_path / "runs"
    (runs_root / escaped).mkdir(parents=True)
    (runs_root / escaped / ".origin").write_text(f"{checkout}\n")
    signals = collect_live_signals(repo, runs_root=runs_root, candidate_slugs=frozenset({escaped}))
    assert signals.live_path_slugs == frozenset({escaped})


def test_reaper_path_arm_keeps_a_tier_whose_origin_holds_no_toolkit(tmp_path):
    """A recorded cwd that is not itself a toolkit checkout never releases its tier.

    The predicate reads the toolkit source AT the recorded cwd only. A non-toolkit cwd
    nested inside a checkout that runs the digest code must not borrow that checkout's
    answer: the code that minted the escaped tier ran from somewhere else, and may still
    be running.
    """
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    newer = tmp_path / "newer_checkout"
    (newer / "tests" / "fixtures").mkdir(parents=True)
    (newer / "tests" / "fixtures" / "_triton_source_cache.py").write_text("def checkout_slug(cwd):\n    pass\n")
    origin = (newer / "scratch_runs").resolve()
    origin.mkdir()
    escaped = "path" + "".join("_s" if ch == "/" else "_u" if ch == "_" else ch for ch in str(origin))
    runs_root = tmp_path / "runs"
    (runs_root / escaped).mkdir(parents=True)
    (runs_root / escaped / ".origin").write_text(f"{origin}\n")
    signals = collect_live_signals(repo, runs_root=runs_root, candidate_slugs=frozenset({escaped}))
    assert signals.live_path_slugs == frozenset({escaped})
