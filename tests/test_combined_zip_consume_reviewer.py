"""Reviewer tests for chunk 1a.4: consuming a combined zip through ``CombinedBundle.from_zip``.

Small zips under ``tmp_path`` with the standard library; no solver runs. Run as
``PYTHONPATH=${PWD}/src pytest tests/test_combined_zip_consume_reviewer.py``.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from hhemt.bundle._combine import CombinedBundle
from hhemt.version_migration.constants import BUNDLE_SCHEMA_VERSION


def _member_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("bundle_manifest.json", "{}")
        zf.writestr("plots/fig.txt", "member")
        zf.writestr("sims/0/out_triton/bin/H.bin", "raw")
    return buf.getvalue()


def _combined_zip(path: Path, extra: dict[str, bytes] | None = None) -> Path:
    manifest = {"bundle_schema_version": BUNDLE_SCHEMA_VERSION, "child_crates": ["child_crates/e1.zip"]}
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("bundle_manifest.json", json.dumps(manifest))
        zf.writestr("child_crates/e1.zip", _member_zip_bytes())
        zf.writestr("child_crates/e1.refreshed/plots/fig.txt", "refreshed")
        for name, payload in (extra or {}).items():
            zf.writestr(name, payload)
    return path


def test_from_zip_extracts_render_set_and_lays_refreshed_figures_over_it(tmp_path: Path) -> None:
    """Property: the unpacked child holds the member's render set, no raw output, and the refreshed bytes.

    Class: NEW CAPABILITY, held green. Wrong implementation: extracting the raw ``out_*/bin`` file, or
    laying the member's own figure over the refreshed one; red on either assertion. Reachability: the
    layout 1a.3 writes. Both-states anchor: literals. Second correct implementation: extracting the
    whole member then deleting raw files and copying refreshed ones. Invocation not isolatable from
    this tree: discrimination is swapping the overlay order.
    """
    z = _combined_zip(tmp_path / "c.zip")
    cb = CombinedBundle.from_zip(z)
    child = cb.root / "child_crates" / "e1"
    assert (child / "plots" / "fig.txt").read_text() == "refreshed"
    assert not (child / "sims" / "0" / "out_triton" / "bin" / "H.bin").exists()


def test_a_refreshed_entry_that_climbs_out_of_the_child_is_confined(tmp_path: Path) -> None:
    """Property: a ``.refreshed/`` entry whose name contains ``..`` writes nowhere outside the unpack folder.

    Input class: combined zips with a climbing name under ``child_crates/{eid}.refreshed/``.
    Approved-shape row: 5.1 row 1a.2/1a.1 ("path-confined as `extractall` is"), the confinement
    the shape requires of every extraction in the combine path; 1a.4's overlay is the one write
    in ``from_zip`` that does not go through ``extract_zip``.

    Why required: ``target = unpack / child_crates / eid / name[len(prefix):]`` followed by
    ``write_bytes`` does not sanitise ``..``, so a hostile or damaged zip overwrites a file outside
    the folder (measured below: the escape file appears beside the unpack folder).

    Class: NEW CAPABILITY. Wrong implementation: the shipped overlay; red under it. Reachability: any
    zip reaches ``report-from-bundle``. Both-states anchor: the filesystem is listed. Invocation not
    isolatable from this tree: discrimination is routing the overlay through ``extract_zip``. Second
    correct implementation: ``Path.resolve`` plus a containment check that skips the entry.
    """
    z = _combined_zip(tmp_path / "c.zip", {"child_crates/e1.refreshed/../../../escape.txt": b"x"})
    CombinedBundle.from_zip(z)
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "c" / "escape.txt").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c", "c.zip"]
