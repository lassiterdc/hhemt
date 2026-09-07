"""The child-report re-render must actually WRITE, not be swallowed.

`_combine.py`'s per-(child, renderer) `except Exception: continue` -- and the earlier
`except` around the config load itself -- leave the STALE HARVESTED figure in place, so a
broken reader ships pre-refresh content with the suite green. This asserts the re-render
HAPPENED by planting a sentinel no renderer can emit and requiring it to be gone
afterwards. An exit-code check cannot distinguish those two worlds; the sentinel can,
because only a write erases it.
"""

from pathlib import Path

import pytest

from hhemt.bundle import _combine, combine_bundle

pytestmark = pytest.mark.requires_snakemake_subprocess

SENTINEL = "STALE-HARVEST-SENTINEL-DO-NOT-EMIT"


def _plant(child: Path) -> Path:
    target = child / "plots" / "errors_and_warnings" / "validation_report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SENTINEL)
    return target


def test_child_rerender_erases_a_planted_sentinel(synthetic_two_bundle_fixture):
    a, b = synthetic_two_bundle_fixture
    cb = combine_bundle([a, b])
    crates = cb.root / _combine._CHILD_CRATES_SUBDIR
    planted = [_plant(c) for c in sorted(p for p in crates.iterdir() if p.is_dir())]
    assert planted, "fixture produced no child crates; the assertion would be vacuous"
    _combine._rerender_child_report_figures(cb.root)
    survivors = [p for p in planted if SENTINEL in p.read_text()]
    assert not survivors, (
        f"{len(survivors)} child figure(s) still carry the sentinel: the re-render was "
        f"SWALLOWED by _combine.py's except-continue and the stale harvest shipped."
    )
