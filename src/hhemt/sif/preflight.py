"""validate_container_images: the image checks container-mode preflight runs (ONE per used partition),
lifted out of validation._validate_container_config. Also `python -m hhemt.sif.preflight ...` — the
STRICT re-check the --build-sifs chain runs after the build DAG, before the experiment submits."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_container_images(cfg_analysis, cfg_hpc_system, cfg_system, result, *, build_sifs: bool = False) -> None:
    """Accumulates into `result` (the preflight ValidationResult). For every partition the matrix
    requires: recompute the identity, resolve the path, and check image + manifest + digest +
    identity label + the measured labels against the pins. `build_sifs=True` downgrades ONLY the
    absent-image finding to a warning (the pre-step will build it); everything else stays an error.
    A wheel driver (no 40-hex running sha) is REFUSED here — the item-11 statement in the VMS."""
    from hhemt.bundle._emit import _matrix_required_partitions
    from hhemt.container_labels import HHEMT_SHA_LABEL, IDENTITY_LABEL, SWMM_VERSION_LABEL, TRITON_SHA_LABEL
    from hhemt.exceptions import ConfigurationError
    from hhemt.sif.identity import derive_identity, manifest_path, resolve_sif
    from hhemt.validation import running_identity

    cspec = getattr(cfg_hpc_system, "container", None)
    root = Path(cspec.sif_root)
    running = running_identity().sha
    for part in sorted(_matrix_required_partitions(cfg_analysis, cfg_hpc_system)):
        try:
            ident = derive_identity(cfg_hpc_system, cfg_system, part, running)
        except ConfigurationError as exc:
            result.add_error(
                field="container",
                message=str(exc),
                fix_hint="see the message; container mode requires a git-checkout driver at the image's commit",
            )
            return
        p = resolve_sif(root, ident)
        if not p.is_file():
            present = sorted(str(x.relative_to(root)) for x in root.glob("*/*.sif")) if root.is_dir() else []
            msg = (
                f"no image for identity {ident.key} ({ident.family}, {ident.gpu_hardware or 'cpu'}, "
                f"TRITON {ident.triton_sha[:8]}, hhemt {ident.hhemt_sha[:8]}) at {p}. "
                f"Under {root} there ARE: {present or '(nothing)'}."
            )
            hint = (
                "hhemt build-sifs --dry-run --build-hpc-config <build host> --experiment <this experiment>  "
                "# then without --dry-run; or `hhemt run --build-sifs` to build first and submit."
            )
            if build_sifs:
                result.add_warning(
                    field="container.sif_root", message=msg + " (--build-sifs: the pre-step builds it)", fix_hint=hint
                )
            else:
                result.add_error(field="container.sif_root", message=msg, fix_hint=hint)
            continue
        man = manifest_path(p)
        if not man.is_file():
            result.add_error(
                field="container.sif_root",
                message=(
                    f"{p} has no manifest beside it — not produced by hhemt build-sifs (or its transaction did not "
                    "complete); refusing an unrecorded image."
                ),
                fix_hint="hhemt build-sifs --force --only " + ident.key,
            )
            continue
        m = json.loads(man.read_text())
        if m.get("identity") != ident.model_dump():
            result.add_error(
                field="container.sif_root",
                message=(
                    f"{man}: manifest identity != the identity recomputed from config ({ident.key}); "
                    "the file at the identity path was not built for this config."
                ),
            )
            continue
        got = _sha256_file(p)
        if got != m.get("sha256"):
            result.add_error(
                field="container.sif_root",
                message=(
                    f"{p}: sha256 {got[:16]} != manifest {str(m.get('sha256'))[:16]} — "
                    "the bytes at the identity path are not the transaction's bytes."
                ),
            )
            continue
        labels = m.get("labels") or {}
        for lbl, want in (
            (IDENTITY_LABEL, ident.key),
            (TRITON_SHA_LABEL, ident.triton_sha),
            (HHEMT_SHA_LABEL, ident.hhemt_sha),
            (SWMM_VERSION_LABEL, ident.swmm_tag),
        ):
            if labels.get(lbl) != want:
                result.add_error(
                    field="container.sif_root",
                    message=(
                        f"{p}: label {lbl}={labels.get(lbl)!r} != expected {want!r} "
                        "(the transaction's read-back gate should have rejected this image)."
                    ),
                )


def main(argv=None) -> int:
    """STRICT re-check: exit 0 iff every required image resolves clean. Used by the --build-sifs chain."""
    import argparse

    from hhemt.config.analysis import analysis_config
    from hhemt.config.hpc_system import hpc_system_config
    from hhemt.config.loaders import yaml_to_model
    from hhemt.config.system import system_config
    from hhemt.validation import ValidationResult

    ap = argparse.ArgumentParser()
    ap.add_argument("--system-config", required=True)
    ap.add_argument("--analysis-config", required=True)
    ap.add_argument("--hpc-system-config", required=True)
    a = ap.parse_args(argv)
    result = ValidationResult()
    validate_container_images(
        yaml_to_model(Path(a.analysis_config), analysis_config),
        yaml_to_model(Path(a.hpc_system_config), hpc_system_config),
        yaml_to_model(Path(a.system_config), system_config),
        result,
    )
    for w in result.warnings:
        print(f"WARNING: {w.message}", file=sys.stderr)
    for e in result.errors:
        print(f"ERROR: {e.message}\n  fix: {e.fix_hint}", file=sys.stderr)
    return 0 if result.is_valid else 2


if __name__ == "__main__":
    sys.exit(main())
