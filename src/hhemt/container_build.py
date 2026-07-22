"""On-ingest Apptainer SIF builder (ADR-19).

The reprex bundle carries a digest-pinned ``.def`` build recipe + build context, NOT
the 3-8 GB built SIF (ADR-9/ADR-2). This module turns that recipe into a SIF on the
reproducer's cluster, gated on a target-side rootless-fakeroot capability preflight,
so ``from_doi`` can repoint ``container.sif_path`` at the result.

Design constraints this module realizes (all ratified in ADR-19):

- **Standalone CLI step, NEVER a Snakemake rule.** ``apptainer build`` is not
  byte-reproducible, so a SIF wired into the DAG as a rule ``output:`` would acquire
  mtime rerun-triggers and cascade a full-ensemble re-run on any rebuild. ``workflow.py``
  is untouched and generated Snakefiles stay byte-identical.
- **CPU-batch build, not a login-node build.** The build is I/O-bound and needs no GPU
  (``nvcc`` cross-compiles for the Kokkos-named arch with no device present). A login-node
  build of a compiling ``.def`` is an AUP violation: ``make -j"$(nproc)"`` sees no cgroup
  cap there and forks 40-way on a shared frontend.
- **Content-addressed cache OUTSIDE ``bundle_root``.** ``extract_reprex_bundle`` rmtree's
  ``bundle_root`` on every ingest (``bundle/_reprex.py:156-159``), so a cache under it could
  never hit -- which would make the [Q8] runbook's Leg 2 submit a second 1.6 h build from
  inside the GPU allocation. The cache root is ``$HHEMT_SIF_CACHE_DIR``, else
  ``platformdirs.user_cache_dir("hhemt")/sif_cache/`` -- mirroring the toolkit's established
  cache idiom (``<user_cache_dir>/hhemt/<family>/`` + a ``filelock`` + a completion sentinel).
- **Verification pins build INPUTS, never the output blob.** The base-image sha256 pin and
  the ``uv.lock`` hash gate BEFORE the build (ADR-4 SquashFS non-hermeticity); the built
  SIF's own sha256 is recorded as an advisory label and gates nothing.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from hhemt._filelock_compat import resolve_filelock
from hhemt.exceptions import ConfigurationError, ProcessingError

#: Cache-family subdir under the cache root (mirrors ``synthetic_test_models`` in the
#: test-tier cache; see ``tests/fixtures/synthetic_model/cache.py``).
_CACHE_FAMILY = "sif_cache"

#: Resource ask for the build job. Every value traces to the estate's own prior build of
#: this exact ``.def`` -- Rivanna job 16687623 (``standard-rivanna``, cpu=16, mem=96G,
#: Elapsed 01:37:17, TotalCPU 57:02, MaxRSS 34.6 GB, MaxDiskWrite 30.3 GB, COMPLETED).
#: This is a trim of a known-good ask, not an estimate.
_BUILD_CPUS = 16  # TotalCPU 57:02 inside Elapsed 01:37 => ~3.7% util: I/O-bound, not
#   CPU-bound. `-c 96` starts ~16 h later (routes to the scarcer afton pool) and buys
#   no speedup (sbatch --test-only probe, 2026-07-13).
_BUILD_MEM_GB = 64  # observed MaxRSS 34.6 GB => 1.85x headroom. COUPLED TO _BUILD_CPUS:
#   the .def runs `make -j"$(nproc)"` and nproc returns the CGROUP count, so peak RSS
#   scales at ~2.2 GB/core. If _BUILD_CPUS rises, raise this proportionally.
_BUILD_WALLTIME = "08:00:00"  # DEFAULT, sized from the OBSERVED SPREAD, not a single draw:
#   job 16687623 Elapsed 01:37:17 (COMPLETED); a100 job 17130540 Elapsed 02:02:56 (COMPLETED);
#   a6000 job 17140966 Elapsed 04:00:30 -- TIMEOUT at `Creating SIF file...` with every compile
#   already finished (defect-9, 2026-07-20). The build is I/O-bound (see _BUILD_CPUS), so Elapsed
#   tracks Weka/node variance across the heterogeneous `standard` pool and a ~2x draw between
#   nodes is EXPECTED, not anomalous -- size for the tail, not the median. Rivanna bills
#   ELAPSED, not the limit, so a generous limit is free insurance against pull/Weka variance.
#   Override per-cluster with $HHEMT_SIF_BUILD_WALLTIME (read at call time in build_sbatch_argv).
_BUILD_PARTITION = "standard"  # submit to the VISIBLE partition; Rivanna routes to
#   standard-rivanna / standard-afton by CPU ask. A `gpu` ask would pay ~5-10x the billing
#   weight for work that needs no device.

#: NO `--tmp` is ever emitted. Every Rivanna node reports TmpDisk=0, so ANY `--tmp=N`
#: fails submission validation outright (`--tmp=40G` => "Requested node configuration is
#: not available", probed 2026-07-13). The ~30 GB of build writes are served from
#: /scratch via APPTAINER_TMPDIR.


@dataclass(frozen=True)
class SifBuildUnavailable(Exception):
    """Raised when the target cannot build a SIF rootlessly (ADR-19 preflight FAIL).

    This is a STRUCTURED SIGNAL, not a failure: ``from_doi`` catches it and routes to the
    ADR-2 transfer fallback (fetch the off-site-built signed SIF). It carries the exact
    remediation so a caller with no fallback can act on it rather than guess.
    """

    reason: str
    remediation: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.reason}\n\nRemediation: {self.remediation}"


def sif_cache_root() -> Path:
    """Resolve the content-addressed SIF cache root.

    Precedence: ``$HHEMT_SIF_CACHE_DIR`` > ``platformdirs.user_cache_dir("hhemt")/sif_cache``.

    MUST NOT live under ``bundle_root``: ``extract_reprex_bundle`` rmtree's that tree on
    every ingest (``bundle/_reprex.py:156-159``), so an under-bundle cache can never hit --
    and the [Q8] Leg-2 runbook depends on the hit to avoid a second build inside the GPU
    allocation. Ingest-independent by construction.
    """
    override = os.environ.get("HHEMT_SIF_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    import platformdirs

    return Path(platformdirs.user_cache_dir("hhemt")) / _CACHE_FAMILY


def compute_cache_key(
    def_bytes: bytes, base_image_digest: str, lock_bytes: bytes, target_arch: str
) -> str:
    """Content-address the build: ``sha256(def || base_image_digest || uv.lock || arch)[:16]``.

    Keyed on the build INPUTS, never the output -- ``apptainer build`` is not
    byte-reproducible (ADR-4), so the output blob cannot be a cache key. Two bundles pinning
    the same recipe therefore share one SIF, which the per-analysis Snakemake state could
    never express (one more reason this is not a rule).
    """
    h = hashlib.sha256()
    for part in (def_bytes, base_image_digest.encode(), lock_bytes, target_arch.encode()):
        h.update(hashlib.sha256(part).digest())  # length-prefix-free domain separation
    return h.hexdigest()[:16]


def _slurm_available() -> bool:
    """True when a SLURM submission host is present (``sbatch`` on PATH)."""
    return shutil.which("sbatch") is not None


def render_build_script(
    *, def_path: Path, sif_out: Path, apptainer_module: str | None, tmpdir: str, cachedir: str
) -> str:
    """Render the build script body (ADR-19 Build-Step Specification).

    The ``cd "$(dirname DEF_PATH)"`` is load-bearing, not hygiene: the ``.def``'s ``%files``
    resolves relative to the BUILD CWD, so the build must run from the ``.def``'s own
    directory or every ``%files`` asset silently fails to stage.

    ``APPTAINER_IGNORE_PROOT=1`` forces mksquashfs down the version-agnostic ``-all-root``
    branch instead of the bundled proot wrapper. proot landed in Apptainer 1.5.0 and dies at
    ``ptrace(TRACEME)`` under ``kernel.yama.ptrace_scope=3`` (Rivanna); the flag is a harmless
    no-op on 1.4.5 (no proot binary) and also dodges 1.5.0's unfixed SIF-corruption bug, and
    loses nothing on an xattr-less scratch filesystem. See
    knowledge/container/apptainer_rivanna_rootless_build_blocked_by_missing_subuid.md.
    """
    lines = ["#!/bin/bash", "set -euo pipefail"]
    if apptainer_module:
        # Emit a module load only where apptainer is module-provided (UVA Rivanna). On a
        # cluster where apptainer is on the default PATH (Frontier) apptainer_module is None
        # and no load is emitted -- mirrors ContainerSpec.apptainer_module semantics.
        lines.append(f'module load "{apptainer_module}"')
    lines += [
        # Fail loud-and-early on the BUILD host if the module did not put apptainer on PATH,
        # instead of a cryptic execve failure minutes into the job. Cheap, dispositive, and on
        # the RIGHT host (unlike a submission-host capability preflight).
        'command -v apptainer >/dev/null || { echo "FATAL: apptainer not on PATH after '
        'module load (check container.apptainer_module / HHEMT_APPTAINER_MODULE)" >&2; exit 127; }',
        "export APPTAINER_IGNORE_PROOT=1",
        f'export APPTAINER_TMPDIR="{tmpdir}"',
        f'export APPTAINER_CACHEDIR="{cachedir}"',
        f'mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "{sif_out.parent}"',
        "# The .def's %files resolves relative to the build CWD -- build from the",
        "# .def's own directory or every staged asset silently goes missing.",
        f'cd "{def_path.parent}"',
        f'apptainer build --fakeroot "{sif_out}" "{def_path}"',
        "",
    ]
    return "\n".join(lines)


def build_sbatch_argv(
    *, account: str, sif_out: Path, build_script: Path, log_dir: Path
) -> list[str]:
    """Assemble the ``sbatch --wait`` argv for the build job.

    ``--wait`` blocks until the job terminates, which is what makes an ingest-time build
    synchronous with ``from_doi`` (the caller needs the SIF path before constructing the
    analysis).

    ``account`` is REQUIRED and has no default. The ratified Build-Step Specification wrote
    ``-A "${HHEMT_SLURM_ACCOUNT:-quinnlab}"``, but ``quinnlab`` is the PRODUCER's UVA
    allocation and appears in ``scripts/reprex_blocklist.txt``; defaulting public library
    code to it would make a third-party reproducer submit against an account they do not
    belong to. The account is sourced from the reproducer's own
    ``hpc_system_config.default_account`` instead -- the same source the proven-green
    ``scripts/experiments/container_validation.py:145`` uses.
    """
    return [
        "sbatch",
        "--wait",
        "-A",
        account,
        "-p",
        _BUILD_PARTITION,
        "-N",
        "1",
        "-n",
        "1",
        "-c",
        str(_BUILD_CPUS),
        f"--mem={_BUILD_MEM_GB}G",
        "-t",
        # defect-9: read at CALL time (not import time) so an operator or test can tune the
        # limit per-cluster without a source edit -- parity with the other env knobs
        # (HHEMT_SIF_CACHE_DIR, HHEMT_APPTAINER_MODULE, HHEMT_SIF_BUILD_TMPDIR,
        # HHEMT_SIF_CACHEDIR). SLURM validates `-t` at submission, so a malformed override
        # fails loudly at sbatch rather than silently.
        os.environ.get("HHEMT_SIF_BUILD_WALLTIME", _BUILD_WALLTIME),
        "-J",
        "hhemt_sif_build",
        "-o",
        str(log_dir / "sif_build_%j.log"),
        str(build_script),
    ]


def _def_declares_compiling_post(def_path: Path) -> bool:
    """True when the ``.def``'s ``%post`` compiles (make/cmake/nvcc/hipcc present).

    Gates the login-node refusal: a pull-only ``.def`` is cheap and safe to build locally;
    a compiling one forks ``make -j$(nproc)`` (40-way on an uncapped frontend).
    """
    try:
        text = def_path.read_text()
    except OSError:
        return True  # unreadable => assume the expensive case and refuse; fail safe
    in_post = False
    for line in text.splitlines():
        if line.startswith("%"):
            in_post = line.strip().startswith("%post")
            continue
        if in_post and any(
            tok in line for tok in ("make ", "make\t", "cmake", "nvcc", "hipcc", "make -j")
        ):
            return True
    return False


def build_sif(
    *,
    def_path: Path,
    sif_out: Path,
    account: str | None = None,
    apptainer_module: str | None = None,
    mode: str = "auto",
    force_rebuild: bool = False,
    log_dir: Path | None = None,
) -> Path:
    """Build ``def_path`` into ``sif_out`` and return the SIF path.

    ``mode``: ``auto`` (batch when SLURM is present, else refuse a compiling .def),
    ``batch`` (force sbatch), ``local`` (force an in-process build -- the sanctioned escape
    hatch for a pull-only, non-compiling .def).

    Raises ``SifBuildUnavailable`` on preflight FAIL (caller routes to the ADR-2 transfer
    fallback). Raises ``ConfigurationError`` on a mis-specified request and ``ProcessingError``
    when the build itself fails.
    """
    def_path = Path(def_path).resolve()
    sif_out = Path(sif_out)
    if not def_path.is_file():
        raise ConfigurationError(
            field="def_path",
            message=f"Apptainer definition file not found: {def_path}",
            config_path=None,
        )
    if sif_out.is_file() and not force_rebuild:
        print(f"[build-sif] cache HIT (existing SIF): {sif_out}", flush=True)
        return sif_out

    if mode not in ("auto", "batch", "local"):
        raise ConfigurationError(
            field="sif_build_mode",
            message=f"sif_build_mode must be one of auto|batch|local; got {mode!r}.",
            config_path=None,
        )

    use_batch = mode == "batch" or (mode == "auto" and _slurm_available())
    if not use_batch and _def_declares_compiling_post(def_path) and mode != "local":
        raise ConfigurationError(
            field="sif_build_mode",
            message=(
                f"{def_path.name} compiles in %post and no SLURM submission host was found. "
                "Building it here would fork make -j$(nproc) on an uncapped shared host."
            ),
            config_path=None,
        )

    # Zero-user-info: NO site literal in src/. Precedence: explicit arg (caller's
    # ContainerSpec.apptainer_module) > HHEMT_APPTAINER_MODULE env > None. None => no
    # `module load` is emitted (apptainer assumed on PATH); the in-script `command -v
    # apptainer` guard fails loud on a module-only host whose caller supplied nothing.
    apptainer_module = apptainer_module or os.environ.get("HHEMT_APPTAINER_MODULE")
    user = os.environ.get("USER", "user")
    tmpdir = os.environ.get("HHEMT_SIF_BUILD_TMPDIR", f"/scratch/{user}/apptainer_tmp")
    cachedir = os.environ.get("HHEMT_SIF_CACHEDIR", f"/scratch/{user}/apptainer_cache")
    sif_out.parent.mkdir(parents=True, exist_ok=True)

    if not use_batch:
        print(f"[build-sif] local build (mode={mode}): {def_path} -> {sif_out}", flush=True)
        rc = subprocess.run(
            ["apptainer", "build", "--fakeroot", str(sif_out), str(def_path)],
            cwd=def_path.parent,  # %files resolves relative to the build CWD
            env={**os.environ, "APPTAINER_IGNORE_PROOT": "1"},  # version-agnostic mksquashfs
            check=False,
        )
        if rc.returncode != 0:
            raise ProcessingError(
                operation="sif_build_local",
                filepath=str(def_path),
                reason=f"apptainer build exited {rc.returncode}.",
            )
        return sif_out

    if not account:
        raise ConfigurationError(
            field="container_build.account",
            message=(
                "A SLURM account is required to submit the SIF build and none was supplied. "
                "It is read from your hpc_system_config's `default_account`; set that (or pass "
                "--account). It is deliberately NOT defaulted: defaulting would submit against "
                "the producer's allocation."
            ),
            config_path=None,
        )

    log_dir = Path(log_dir) if log_dir is not None else sif_out.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    script = sif_out.parent / "sif_build.sh"
    script.write_text(
        render_build_script(
            def_path=def_path,
            sif_out=sif_out,
            apptainer_module=apptainer_module,
            tmpdir=tmpdir,
            cachedir=cachedir,
        )
    )
    script.chmod(0o755)

    argv = build_sbatch_argv(
        account=account, sif_out=sif_out, build_script=script, log_dir=log_dir
    )
    print(f"[build-sif] submitting (blocking ~1.6 h + queue): {' '.join(argv)}", flush=True)
    rc = subprocess.run(argv, check=False)
    if rc.returncode != 0:
        raise ProcessingError(
            operation="sif_build_sbatch",
            filepath=str(def_path),
            reason=(
                f"sbatch --wait exited {rc.returncode}. Inspect {log_dir}/sif_build_*.log "
                "and `sacct` for the job's State/ExitCode."
            ),
        )
    if not sif_out.is_file():
        raise ProcessingError(
            operation="sif_build_sbatch",
            filepath=str(sif_out),
            reason=(
                "the build job returned 0 but produced no SIF at the expected path. "
                f"Inspect {log_dir}/sif_build_*.log."
            ),
        )
    return sif_out


def get_or_build_sif(
    *,
    def_path: Path,
    base_image_digest: str,
    lock_path: Path,
    target_arch: str,
    account: str | None = None,
    apptainer_module: str | None = None,
    sif_out: Path | None = None,
    mode: str = "auto",
    force_rebuild: bool = False,
) -> Path:
    """Content-addressed entry point: return a cached SIF or build one.

    The cache home is ingest-independent (``sif_cache_root()``), so repeated ``from_doi``
    calls against the same DOI reuse one SIF -- which is what keeps the [Q8] Leg-2 GPU
    allocation from paying for a second build.
    """
    def_bytes = Path(def_path).read_bytes()
    lock_bytes = Path(lock_path).read_bytes() if Path(lock_path).is_file() else b""
    key = compute_cache_key(def_bytes, base_image_digest, lock_bytes, target_arch)
    target = (
        Path(sif_out)
        if sif_out is not None
        else sif_cache_root() / f"hhemt-{target_arch}-{key}.sif"
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    # Mirror the toolkit's established cache idiom: filelock + completion sentinel, so two
    # concurrent ingests of the same recipe do not race one SIF path.
    lock = resolve_filelock(str(target.parent / f"{target.name}.lock"))
    with lock:
        if target.is_file() and not force_rebuild:
            print(f"[build-sif] cache HIT: {target}", flush=True)
            return target
        print(f"[build-sif] cache MISS (key={key}) -> building {target}", flush=True)
        return build_sif(
            def_path=Path(def_path),
            sif_out=target,
            account=account,
            apptainer_module=apptainer_module,
            mode=mode,
            force_rebuild=force_rebuild,
        )
