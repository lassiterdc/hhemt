"""write_sif_snakefile: one wildcard-free rule per identity; manifest-only output; reconciliation first."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from hhemt.sif.identity import manifest_path, resolve_sif
from hhemt.sif.plan import SifBuildPlan, live_sentinels


def reconcile_sif_root(
    sif_root: Path, *, walltime_min: int, force_keys: set[str] = frozenset(), planned_keys: set[str] = frozenset()
) -> list[str]:
    """Unlink every manifest whose image is absent/truncated (or forced); sweep aged orphans.
    A `_build/{key}` dir is NEVER swept while its key is in the current plan or has a live
    (queued/submitted) sentinel — refute row 4: a build PENDING longer than the cap must keep its
    spec.json. Returns the actions taken."""
    acts: list[str] = []
    root = Path(sif_root)
    for man in root.glob("*/*.manifest.json"):
        sif = man.with_name(man.name[: -len(".manifest.json")] + ".sif")
        m = json.loads(man.read_text())
        if m["key"] in force_keys or not sif.is_file() or sif.stat().st_size != m["size_bytes"]:
            # EXEMPT-DU: outside-analysis-tree
            man.unlink()
            acts.append(f"unlinked stale manifest {man} (sif present={sif.is_file()})")
    cutoff = time.time() - (walltime_min + 60) * 60
    for orphan in root.glob("*/*.sif.building.*"):
        if orphan.stat().st_mtime < cutoff:
            # EXEMPT-DU: outside-analysis-tree
            orphan.unlink()
            acts.append(f"swept {orphan}")
    for work in (root / "_build").glob("*"):
        if not work.is_dir():
            continue
        key = work.name
        if key in planned_keys or live_sentinels(root, key):
            continue
        if work.stat().st_mtime < cutoff:
            # EXEMPT-DU: outside-analysis-tree
            shutil.rmtree(work, ignore_errors=True)
            acts.append(f"swept {work}")
    return acts


def write_sif_snakefile(plan: SifBuildPlan, *, sif_root: Path, build_host, sif_build_cfg) -> Path:
    root = Path(sif_root)
    (root / "_build").mkdir(parents=True, exist_ok=True)
    rules, targets = [], []
    for key, ident in plan.entries.items():
        if key in plan.already_built:
            continue
        spec = root / "_build" / key / "spec.json"
        spec.parent.mkdir(parents=True, exist_ok=True)
        spec_tmp = spec.with_suffix(".json.tmp")
        spec_tmp.write_text(
            json.dumps(
                {
                    "identity": ident.model_dump(),
                    "build_host_container": build_host.container.model_dump(),
                    "sif_build": sif_build_cfg.model_dump(mode="json"),
                    "sif_root": str(root),
                },
                indent=2,
            )
        )
        spec_tmp.replace(spec)  # atomic: a PENDING job reading spec.json never sees a partial write
        man = manifest_path(resolve_sif(root, ident))
        targets.append(str(man))
        rules.append(
            f'''
rule build_sif_{key}:
    output: "{man}"
    params: spec="{spec}"
    resources:
        slurm_partition="{sif_build_cfg.partition}", slurm_account="{build_host.default_account}",
        cpus_per_task={sif_build_cfg.cpus}, mem_mb={sif_build_cfg.mem_mb}, runtime={sif_build_cfg.walltime_min}
    retries: 1
    shell: "python -m hhemt.sif.build_runner --spec {{params.spec}}"
'''
        )
    text = "rule all:\n    input: " + json.dumps(targets) + "\n" + "".join(rules)
    (root / "Snakefile.sif").write_text(text)
    (root / "profile").mkdir(exist_ok=True)
    (root / "profile" / "config.yaml").write_text(
        f"executor: slurm\njobs: {max(1, len(targets))}\nkeep-going: true\n"
        f"default-resources:\n  slurm_account: {build_host.default_account}\n"
    )
    return root / "Snakefile.sif"
