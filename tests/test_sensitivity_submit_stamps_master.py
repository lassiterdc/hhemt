"""The sensitivity facade stamps the layout version on every entry that can materialize a master.

`TRITONSWMM_sensitivity_analysis` exposes more than one execution entry, and every one of them
except the direct `submit_workflow` call stamped `_version.json` through `stamp_new_target`.
A master built by the entry that did not stamp carries outputs and no layout version, so a
later `run()` on that tree raises `LayoutVersionError` — the failure this module pins.

The stamp is lazy by design (written by the first run or submit, never at construction), and
idempotent, so a second submit on the same master is a no-op rather than a rewrite.
"""

from __future__ import annotations

import json
from pathlib import Path

from hhemt.version_migration import LAYOUT_VERSION

VERSION_FILE = "_version.json"


def _stamped_version(analysis_dir: Path) -> int | None:
    """The layout version the tree records, or None when the file is absent."""
    path = analysis_dir / VERSION_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("layout_version")


def test_direct_sensitivity_submit_stamps_the_master(synth_sensitivity_analysis):
    """Property: a direct submit on the sensitivity facade leaves `_version.json` at the current
    layout version in the master's analysis directory.

    Class: REGRESSION -- red before the stamp was added, where the directory carried outputs and
    no `_version.json` at all (the run below returns the version as None on that tree).

    Why it is required: every other execution entry stamps, and a master this entry builds is
    then unreadable by `run()` -- a LayoutVersionError on the very tree this call produced.

    What kills it: deleting the stamp, or moving it below a step that returns early (the
    cross-node preflight refuses before the stamp, which is intended: a refused submit must not
    touch disk).

    A second correct implementation that still passes: any lazy stamp that writes the current
    layout version to the master's analysis directory on this path.
    """
    analysis = synth_sensitivity_analysis
    facade = analysis.sensitivity
    analysis_dir = Path(facade.experiment.analysis_paths.analysis_dir)

    facade.submit_workflow(mode="local", dry_run=True, verbose=False)

    assert _stamped_version(analysis_dir) == LAYOUT_VERSION
