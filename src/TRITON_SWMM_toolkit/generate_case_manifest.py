"""Operator helper to compute a case.yaml sha256 manifest (ADR-12).

Run once per hosted resource version to pin the expected file contents into
``test_data/{case_name}/case.yaml``. The toolkit verifies the manifest at
download time (examples.py::_download_data_from_hydroshare).

Usage:
    python -m TRITON_SWMM_toolkit.generate_case_manifest \
        --bag-dir /path/to/extracted/bag \
        --case-yaml test_data/norfolk_coastal_flooding/case.yaml
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from TRITON_SWMM_toolkit.config.case_manifest import CaseManifest
from TRITON_SWMM_toolkit.utils import read_yaml, write_yaml


def compute_manifest(bag_dir: Path) -> dict[str, str]:
    """Return {bag-relative-posix-path: hex sha256} for every file under bag_dir."""
    bag_dir = Path(bag_dir)
    manifest: dict[str, str] = {}
    for fpath in sorted(p for p in bag_dir.rglob("*") if p.is_file()):
        rel = fpath.relative_to(bag_dir).as_posix()
        manifest[rel] = hashlib.sha256(fpath.read_bytes()).hexdigest()
    return manifest


def populate_case_yaml(bag_dir: Path, case_yaml: Path) -> CaseManifest:
    """Compute the manifest from bag_dir and write it into case_yaml (schema-checked)."""
    case_yaml = Path(case_yaml)
    existing = CaseManifest.model_validate(read_yaml(case_yaml))
    updated = existing.model_copy(update={"manifest": compute_manifest(bag_dir)})
    write_yaml(updated.model_dump(mode="json"), case_yaml)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and write the sha256 manifest block of a case.yaml (ADR-12).")
    parser.add_argument("--bag-dir", type=Path, required=True, help="Extracted resource bag dir.")
    parser.add_argument("--case-yaml", type=Path, required=True, help="Path to case.yaml to update.")
    args = parser.parse_args()
    updated = populate_case_yaml(args.bag_dir, args.case_yaml)
    print(f"Wrote {len(updated.manifest)} sha256 entries to {args.case_yaml}", flush=True)


if __name__ == "__main__":
    main()
