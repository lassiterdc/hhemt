"""python -m hhemt.sif.build_runner --spec {sif_root}/_build/{key}/spec.json"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from hhemt.sif.args import expected_measured_labels, render_build_args
from hhemt.sif.identity import SifIdentity
from hhemt.sif.recipes import family_recipe
from hhemt.sif.transaction import build_transaction


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text())
    ident = SifIdentity.model_validate(spec["identity"])
    root = Path(spec["sif_root"])
    status = root / "_status"
    sentinel = status / "_submitted" / f"{ident.key}.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({"key": ident.key, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}))
    # EXEMPT-DU: outside-analysis-tree
    (status / "_queued" / f"{ident.key}.json").unlink(missing_ok=True)  # queued -> submitted handoff
    try:
        recipes_dir = spec["sif_build"].get("recipes_dir")
        recipe, text, _ = family_recipe(ident.family, Path(recipes_dir) if recipes_dir else None)
        final = build_transaction(
            ident,
            sif_root=root,
            recipe=recipe,
            build_args=render_build_args(ident, text),
            hhemt_repo=Path(spec["sif_build"]["toolkit_root"]),
            apptainer_module=spec["build_host_container"].get("apptainer_module"),
            expected_measured=expected_measured_labels(ident),
        )
        print(f"BUILD OK  {final}", flush=True)
        return 0
    finally:
        # EXEMPT-DU: outside-analysis-tree
        sentinel.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
