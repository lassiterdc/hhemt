"""build_transaction: temp -> build -> read-back gate -> digest -> manifest -> rename(2), manifest last."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from hhemt.exceptions import ProcessingError
from hhemt.sif.identity import SifIdentity, manifest_path, resolve_sif


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _module_prefix(apptainer_module: str | None) -> str:
    return f"module load {apptainer_module} && " if apptainer_module else ""


def _inspect_labels(sif: Path, apptainer_module: str | None) -> dict[str, str]:
    """`apptainer inspect --json` through a login shell so a module-only apptainer resolves
    (refute row 3: a shell string as argv[0] is `FileNotFoundError`)."""
    out = subprocess.run(
        ["bash", "-lc", f"{_module_prefix(apptainer_module)}apptainer inspect --json {sif}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)["data"]["attributes"]["labels"]  # identical shape at 1.4.5 and 1.5.0 (measured)


def build_transaction(
    identity: SifIdentity,
    *,
    sif_root: Path,
    recipe: Path,
    build_args: str,
    hhemt_repo: Path,
    apptainer_module: str | None,
    expected_measured: dict[str, str],
) -> Path:
    if not os.environ.get("SLURMD_NODENAME"):
        raise ProcessingError(
            operation="build_transaction",
            filepath=sif_root,
            reason=(
                "refusing to build outside a SLURM job step ($SLURMD_NODENAME unset; an salloc "
                "shell on a login node does not set it — deliberate)"
            ),
        )
    key = identity.key
    work = Path(sif_root) / "_build" / key
    for d in ("cache", "tmp", "stage"):
        (work / d).mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "APPTAINER_CACHEDIR": str(work / "cache"),
        "APPTAINER_TMPDIR": str(work / "tmp"),
        "APPTAINER_IGNORE_PROOT": "1",
    }
    env.pop("PROOT_NO_SECCOMP", None)  # inert at 1.4.5, ineffective at 1.5.0 — never carried
    # stage: git archive of the EXACT commit; HHEMT_SHA is written by git via export-subst (Spec 43)
    tar = subprocess.run(
        ["git", "-C", str(hhemt_repo), "archive", "--format=tar", identity.hhemt_sha],
        capture_output=True,
        check=True,
    ).stdout
    stage_src = work / "stage" / "hhemt_src"
    stage_src.mkdir(exist_ok=True)  # recipe's `%files hhemt_src /opt/hhemt-src`
    subprocess.run(["tar", "-x", "-C", str(stage_src)], input=tar, check=True)
    args_file = work / "build.args"
    args_file.write_text(build_args)
    final = resolve_sif(sif_root, identity)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f"{final.name}.building.{os.environ.get('SLURM_JOB_ID', 'nojob')}.{os.getpid()}")
    # EXEMPT-DU: outside-analysis-tree
    tmp.unlink(missing_ok=True)
    t0 = time.time()
    rc = subprocess.run(
        [
            "bash",
            "-lc",
            f"{_module_prefix(apptainer_module)}cd {work / 'stage'} && apptainer build --fakeroot "
            f"--build-arg-file {args_file} {tmp} {recipe}",
        ],
        env=env,
    ).returncode
    if rc != 0:
        # EXEMPT-DU: outside-analysis-tree
        tmp.unlink(missing_ok=True)
        raise ProcessingError(
            operation="apptainer build", filepath=tmp, reason=f"rc={rc}; previous image (if any) at {final} left intact"
        )
    # READ-BACK GATE: what LANDED, not what was intended
    got = _inspect_labels(tmp, apptainer_module)
    want = {**identity.labels(), **expected_measured}  # measured half: triton_sha, hhemt_sha
    bad = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
    if bad:
        rejected = tmp.with_name(tmp.name.replace(".building.", ".rejected."))
        tmp.rename(rejected)
        raise ProcessingError(
            operation="read-back gate",
            filepath=rejected,
            reason=(
                "labels read back from the built image differ from the requested identity: "
                + json.dumps(bad)
                + " — image KEPT under .rejected. for forensics"
            ),
        )
    digest = _sha256(tmp)
    manifest = {
        "identity": identity.model_dump(),
        "key": key,
        "labels": got,
        "sha256": digest,
        "size_bytes": tmp.stat().st_size,
        "built_by_hhemt_sha": identity.hhemt_sha,
        "build_host": {
            "node": socket.gethostname(),
            "slurm_cluster": os.environ.get("SLURM_CLUSTER_NAME"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "apptainer_version": got.get("org.label-schema.usage.singularity.version"),
            "apptainer_module": apptainer_module,
        },
        "recipe_sha256": identity.recipe_sha256,
        "build_args_sha256": hashlib.sha256(build_args.encode()).hexdigest(),
        "duration_s": round(time.time() - t0),
    }  # ONE digest, unsigned, from build to deposit (item 7: (a))
    sidecar_tmp = final.with_name(final.name + ".sha256.tmp")
    sidecar_tmp.write_text(f"{digest}  {final}\n")
    man_tmp = manifest_path(final).with_suffix(".json.tmp")
    man_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    try:
        # EXEMPT-DU: outside-analysis-tree
        os.replace(tmp, final)  # rename(2), same directory, atomic
    except OSError as exc:
        raise ProcessingError(
            operation="rename",
            filepath=tmp,
            reason=(
                f"built {tmp} but could not rename over {final}: {exc}. The BUILT image is KEPT; "
                "rename it by hand once the cause is fixed, or the sweep reclaims it after the walltime cap"
            ),
        ) from exc
    # EXEMPT-DU: outside-analysis-tree
    os.replace(sidecar_tmp, final.with_name(final.name + ".sha256"))
    # EXEMPT-DU: outside-analysis-tree
    os.replace(man_tmp, manifest_path(final))  # LAST: its presence proves the whole transaction
    # EXEMPT-DU: outside-analysis-tree
    shutil.rmtree(work, ignore_errors=True)  # cache + tmp (the build-temp lives here) + stage
    return final
