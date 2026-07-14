"""analysis.reprex — bundle round-trip runnability validation + amendment emission.

``Bundle.reprex(reprex_config, target_hpc_profile) -> ReprexResult``:

1. Resolve + VERIFY the SIF **when the crate references one** (a container-run
   bundle): mandatory sha256 digest match (fail-closed — a mismatch raises), plus a
   best-effort ``apptainer verify`` PGP check (``sif_signature_ok`` is ``None`` when
   the ``apptainer`` binary / producer key is unavailable). A NATIVE-run bundle
   records no SIF entity in its crate; reprex then reports
   ``sif_reference_present=False`` and skips verification (a vacuous pass) — native
   runs are first-class (the primary sensitivity fixture is native).
2. Re-aim ``validation.preflight_validate(cfg_hpc_system=target)`` at the target HPC
   profile, overlaying the target partition selectors from ``reprex_config``.
3. Emit per-``(sa_id, resource-column)`` problem pairs (``ValidationIssue`` shape) for
   sensitivity rows whose requested resources exceed the target partition caps.
4. Emit per-field graduated experiment amendments: **validated** where a deterministic
   ``PartitionSpec`` lookup pins the target value (the partition selectors resolve to a
   declared target partition); **advisory-with-named-reason** otherwise.
5. Run a consume-side **informational** zero-user-info scan. The HARD emit-time gate is
   deferred to the emit-hardening follow-up (ADR-9): the emit tree still carries the
   producer's absolute paths through ``bundle_manifest.json`` / harvested ``.inp`` /
   ``validation_report.json`` — surfaces the config-field scrub does not cover — so a
   hard full-tree gate over an emitted bundle would fail on a real leak the plan
   deferred fixing. Here the scan is reported (``zero_user_info_leaks``), never fatal.

Config->bundle acyclicity: any emit-time field/column bucketing uses a FUNCTION-LOCAL
import of ``hhemt.config.reprex_taxonomy`` (never module-top — see reprex_taxonomy's
package-scope import invariant).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from hhemt.exceptions import ProcessingError
from hhemt.validation import ValidationIssue

#: ``ValidationIssue.field`` prefix stamped by ``validation._validate_per_sa_row_caps``.
_ROW_ISSUE_PREFIX = "sensitivity_analysis.row["


@dataclass
class Amendment:
    """A single per-field experiment amendment for closest-possible reproduction.

    ``status`` is ``"validated"`` when a deterministic ``PartitionSpec`` lookup pins
    ``to_value`` (the target user can trust it), or ``"advisory"`` when no deterministic
    target mapping exists (``reason`` names why the reproducer must decide).
    """

    field_name: str
    from_value: object
    to_value: object
    status: Literal["validated", "advisory"]
    reason: str


@dataclass
class ReprexResult:
    """The outcome of a bundle round-trip runnability check against a target profile."""

    sif_reference_present: bool  # False => native-run bundle (no SIF in the crate)
    sif_verified: bool  # digest match, or vacuously True for a native bundle
    sif_signature_ok: bool | None  # None => apptainer/key unavailable, or native
    runnable: bool  # True => no sensitivity row exceeds a target partition cap
    problem_pairs: list[ValidationIssue] = field(default_factory=list)  # (sa_id, column)
    amendments: list[Amendment] = field(default_factory=list)
    zero_user_info_leaks: list[str] = field(default_factory=list)  # informational (ADR-9)


def _find_sif_entity(bundle_root: Path) -> dict | None:
    """Return the crate's SIF ``SoftwareApplication`` entity (the one carrying a
    ``sha256``), or ``None`` for a native bundle. The toolkit ``#hhemt-app``
    ``SoftwareApplication`` is excluded — it carries ``softwareVersion`` but no
    ``sha256`` (see ``metadata.build_analysis_crate``)."""
    crate_path = bundle_root / "ro-crate-metadata.json"
    if not crate_path.is_file():
        return None
    try:
        doc = json.loads(crate_path.read_text())
    except (ValueError, OSError):
        return None
    for entity in doc.get("@graph", []):
        etype = entity.get("@type")
        types = etype if isinstance(etype, list) else [etype]
        if "SoftwareApplication" in types and entity.get("sha256"):
            return entity
    return None


def _verify_sif(sif_path: Path, expected_sha256: str) -> tuple[bool, bool | None]:
    """Return ``(digest_ok, signature_ok | None)``. A missing SIF or a digest mismatch
    raises ``ProcessingError`` (fail-closed). ``signature_ok`` is ``None`` when the
    ``apptainer`` binary is unavailable (best-effort PGP)."""
    if not sif_path.is_file():
        raise ProcessingError(
            operation="reprex SIF verify",
            filepath=sif_path,
            reason=(
                f"reprex_config.sif_path does not exist: {sif_path}. Fetch the reference "
                f"SIF (the crate's by-reference SoftwareApplication) to this path first."
            ),
        )
    h = hashlib.sha256()
    with sif_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != expected_sha256:
        raise ProcessingError(
            operation="reprex SIF verify",
            filepath=sif_path,
            reason=(
                f"sha256 mismatch: this is NOT the reference SIF "
                f"(expected {expected_sha256}, got {digest})."
            ),
        )
    # Best-effort PGP: warn (return None) when the apptainer binary is unavailable.
    try:
        rc = subprocess.run(["apptainer", "verify", str(sif_path)], capture_output=True)
        return True, rc.returncode == 0
    except (FileNotFoundError, OSError):
        return True, None


def _scan_zero_user_info(bundle_root: Path) -> list[str]:
    """Consume-side INFORMATIONAL scan for producer user-info leaks (ADR-9). Non-fatal:
    returns the leak list rather than raising. Hard emit-time enforcement is deferred to
    the emit-hardening follow-up (the emit tree still leaks producer absolute paths
    through bundle_manifest.json / harvested .inp / validation_report.json). The scan
    matches against the LOCAL blocklist, so it is meaningful in a same-machine reprex
    (the producer's tokens are present) and best-effort across machines."""
    from hhemt.bundle._reprex_gate import assert_bundle_zero_user_info

    try:
        assert_bundle_zero_user_info(bundle_root)
        return []
    except ProcessingError as exc:
        return [exc.reason]


def extract_reprex_bundle(zip_path: Path) -> Path:
    """Extract an emitted bundle ``.zip`` to a sibling directory (same stem) and return
    it. The reprex round-trip consumes a directory root (``Bundle.from_directory``),
    while ``emit_bundle`` yields a zip; the emit-side facades call this so a reprex
    bundle is directly consumable."""
    import shutil
    import zipfile

    zip_path = Path(zip_path)
    dest = zip_path.with_suffix("")  # strip the trailing ".zip"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


def reprex(bundle_root: Path, reprex_cfg, target_hpc_profile) -> ReprexResult:
    """Verify + validate runnability + emit amendments for a reprex bundle."""
    from hhemt.bundle._emit import reconstitute_runnable_config
    from hhemt.config.analysis import analysis_config
    from hhemt.config.loaders import yaml_to_model
    from hhemt.config.system import system_config
    from hhemt.validation import ValidationResult, preflight_validate

    bundle_root = Path(bundle_root).resolve()

    # 1. SIF resolve + verify (container bundle) or native no-SIF path.
    sif_entity = _find_sif_entity(bundle_root)
    if sif_entity is not None:
        sif_reference_present = True
        sif_verified, sif_signature_ok = _verify_sif(
            Path(reprex_cfg.sif_path), sif_entity["sha256"]
        )
    else:
        sif_reference_present = False
        sif_verified = True  # native bundle: nothing to verify (vacuous pass)
        sif_signature_ok = None

    # 2. Load the bundle's configs and re-aim them at the target HPC profile.
    cfg_system = yaml_to_model(reconstitute_runnable_config(bundle_root), system_config)
    cfg_analysis = yaml_to_model(bundle_root / "cfg_analysis.yaml", analysis_config)

    # Overlay the target partition selectors (the HPC-revisable axis) + rebase the
    # sensitivity CSV onto the bundle so the per-row cap scan can read it.
    overlay: dict[str, Any] = {
        "hpc_ensemble_partition": reprex_cfg.target_ensemble_partition,
        "hpc_setup_and_analysis_processing_partition": (
            reprex_cfg.target_setup_and_analysis_processing_partition
            or reprex_cfg.target_ensemble_partition
        ),
    }
    if cfg_analysis.sensitivity_analysis is not None:
        sens = Path(cfg_analysis.sensitivity_analysis)
        if not sens.is_absolute():
            overlay["sensitivity_analysis"] = (bundle_root / sens).resolve()
    cfg_analysis_target = cfg_analysis.model_copy(update=overlay)

    # 3. Re-aim preflight at the target profile; extract the per-(sa_id, column) pairs.
    try:
        result = preflight_validate(
            cfg_system, cfg_analysis_target, cfg_hpc_system=target_hpc_profile
        )
        all_issues = result.errors + result.warnings
    except Exception:
        # Full preflight can trip on the reprex cfg's target-supplied (not-yet-fetched)
        # paths; fall back to the isolated per-row cap scan so problem-pair emission
        # stays robust — the cap scan is the reprex-specific signal that matters here.
        from hhemt.validation import _validate_per_sa_row_caps

        result = ValidationResult(context="reprex")
        _validate_per_sa_row_caps(cfg_analysis_target, target_hpc_profile, result)
        all_issues = result.errors + result.warnings

    problem_pairs = [i for i in all_issues if i.field.startswith(_ROW_ISSUE_PREFIX)]
    runnable = len(problem_pairs) == 0

    # 4. Per-field graduated experiment amendments.
    amendments = _emit_amendments(
        cfg_analysis, reprex_cfg, target_hpc_profile, problem_pairs
    )

    # 5. Informational zero-user-info scan (hard emit gate deferred — ADR-9 follow-up).
    leaks = _scan_zero_user_info(bundle_root)

    return ReprexResult(
        sif_reference_present=sif_reference_present,
        sif_verified=sif_verified,
        sif_signature_ok=sif_signature_ok,
        runnable=runnable,
        problem_pairs=problem_pairs,
        amendments=amendments,
        zero_user_info_leaks=leaks,
    )


def _emit_amendments(
    cfg_analysis, reprex_cfg, target_hpc_profile, problem_pairs: list[ValidationIssue]
) -> list[Amendment]:
    """Per-field graduated experiment amendments. VALIDATED where a deterministic
    ``PartitionSpec`` lookup pins the target value (a partition selector resolving to a
    declared target partition); ADVISORY-with-named-reason where no deterministic target
    mapping exists (a cap-exceeding resource whose experiment-appropriate value the
    reproducer must decide, or an HPC-execution field with no target-partition mapping).
    """
    from hhemt.config.reprex_taxonomy import all_field_bucket  # function-local: acyclicity

    amendments: list[Amendment] = []
    partitions = getattr(target_hpc_profile, "partitions", {}) or {}

    # (a) Partition selectors: validated iff the named target partition is declared.
    for field_name, target_val in (
        ("hpc_ensemble_partition", reprex_cfg.target_ensemble_partition),
        (
            "hpc_setup_and_analysis_processing_partition",
            reprex_cfg.target_setup_and_analysis_processing_partition
            or reprex_cfg.target_ensemble_partition,
        ),
    ):
        from_val = getattr(cfg_analysis, field_name, None)
        if target_val in partitions:
            amendments.append(
                Amendment(
                    field_name=field_name,
                    from_value=from_val,
                    to_value=target_val,
                    status="validated",
                    reason="target partition declared in hpc_system_config (deterministic lookup).",
                )
            )
        else:
            amendments.append(
                Amendment(
                    field_name=field_name,
                    from_value=from_val,
                    to_value=target_val,
                    status="advisory",
                    reason=(
                        f"target partition '{target_val}' is not declared in the target "
                        f"hpc_system_config; declare it or pick a listed partition."
                    ),
                )
            )

    # (b) Cap-exceeding resources -> advisory (the experiment-appropriate replacement is
    #     a reproducer decision, even though the cap itself is a deterministic value).
    for issue in problem_pairs:
        amendments.append(
            Amendment(
                field_name=issue.field,
                from_value=issue.current_value,
                to_value=None,
                status="advisory",
                reason=(issue.fix_hint or "exceeds a target partition cap; revise for your cluster."),
            )
        )

    # (c) HPC-execution fields with no deterministic target-partition mapping -> advisory.
    for field_name in ("execution_environment", "multi_sim_run_method"):
        try:
            bucket = all_field_bucket(field_name)
        except KeyError:
            continue
        if bucket != "hpc":
            continue
        amendments.append(
            Amendment(
                field_name=field_name,
                from_value=getattr(cfg_analysis, field_name, None),
                to_value=None,
                status="advisory",
                reason=(
                    "HPC-execution field with no deterministic target-partition mapping; "
                    "revise to match your cluster."
                ),
            )
        )

    return amendments
