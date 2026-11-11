"""analysis.reprex — bundle round-trip runnability validation + amendment emission.

``Bundle.reprex(reprex_config, target_hpc_profile) -> ReprexResult``:

1. Resolve + VERIFY the SIF **when the crate references one** (a container-run
   bundle): mandatory sha256 digest match (fail-closed — a mismatch raises). The
   toolkit does not PGP-sign SIFs and runs no signature check, so
   ``sif_signature_ok`` is always ``None``. A NATIVE-run bundle
   records no SIF entity in its crate; reprex then reports
   ``sif_reference_present=False`` and skips verification (a vacuous pass) — native
   runs are first-class (the primary sensitivity fixture is native).
2. Re-aim ``validation.preflight_validate(cfg_hpc_system=target)`` at the target HPC
   profile, overlaying the target partition selectors from ``reprex_config``.
3. Emit per-``(member_id, resource-column)`` problem pairs (``ValidationIssue`` shape) for
   sensitivity rows whose requested resources exceed the target partition caps.
4. Emit per-field graduated experiment amendments: **validated** where a deterministic
   ``PartitionSpec`` lookup pins the target value (the partition selectors resolve to a
   declared target partition); **advisory-with-named-reason** otherwise.
5. Run a consume-side **informational** zero-user-info scan. The HARD gate runs at emit time: the
   emit refuses to write a bundle whose text members hold producer user-info
   (``hhemt.bundle._reprex_gate.ZeroUserInfoScanner``). This scan runs the generic patterns and the
   private list only, because a consumer holds no producer configs, and it is reported
   (``zero_user_info_leaks``), never fatal.

Config->bundle acyclicity: any emit-time field/column bucketing uses a FUNCTION-LOCAL
import of ``hhemt.config.reprex_taxonomy`` (never module-top — see reprex_taxonomy's
package-scope import invariant).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from hhemt.exceptions import ProcessingError
from hhemt.validation import ValidationIssue

#: ``ValidationIssue.field`` prefix stamped by ``validation._validate_per_member_row_caps``.
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

    # False means ONLY "the crate carries no SIF entity". It does NOT mean "native run".
    # The comment here previously said `False => native-run bundle`, and that was wrong in
    # exactly the case that mattered: a CONTAINER bundle emitted before the ADR-19 (ii-a)
    # digest wire, or one whose SIF was a sandbox directory, or one whose setup-time digest
    # read failed, ALSO lands False. `sif_verified` tells those apart: a native bundle reports
    # True (nothing to verify) and a container bundle with no digest reports None (nothing to
    # verify WITH), which is the distinction this field alone cannot carry.
    sif_reference_present: bool
    # TRI-STATE, and the third value is the point:
    #   True  — a digest was present and MATCHED (`_verify_sif` fail-closed, so a mismatch raises)
    #   None  — NOTHING WAS CHECKED: a container bundle carrying no digest
    #   False — reserved; unreachable today, because a mismatch raises rather than returns
    # The old `bool` could not express None, so an unverifiable container bundle reported
    # `True` with the comment "vacuously True for a native bundle" — a pass over a check
    # that never ran, on a bundle that was not native. That is the same vacuity class this
    # thread has now found three times; here it is closed by making the state representable
    # rather than by a comment asking the reader to remember.
    sif_verified: bool | None
    # ONE REACHABLE VALUE, and saying so is the whole content of this comment:
    #   None  — ALWAYS. `_verify_sif` returns None on every success path and the
    #           no-reference arm sets None too, so no execution reaches any other value.
    #   True  — unreachable; no code path assigns it
    #   False — unreachable; no code path assigns it
    # The field is RETAINED at `bool | None` for schema stability -- a published table in
    # docs/how-to/reprex-roundtrip.md names it -- not because a second value can occur.
    # DIGEST verification is NOT what was retired, and the distinction is the point:
    # `_verify_sif` re-hashes the image and raises on a mismatch, fail-closed. Only
    # SIGNATURE checking is gone (ADR-2 as amended by ADR-21).
    sif_signature_ok: bool | None
    runnable: bool  # True => no sensitivity row exceeds a target partition cap
    problem_pairs: list[ValidationIssue] = field(default_factory=list)  # (member_id, column)
    amendments: list[Amendment] = field(default_factory=list)
    zero_user_info_leaks: list[str] = field(default_factory=list)  # informational (ADR-9)


def _bundle_is_container_mode(bundle_root: Path) -> bool:
    """Was the bundled analysis container-mode? Read from the bundle's own manifest.

    `emit_bundle` writes a `sif_manifests` list IFF
    `execution_environment == "container"` and refuses to emit at all without a built image, so
    the list's presence is a fact ABOUT THE BUNDLE rather than an inference about it. That
    is what lets `sif_reference_present=False` be split into "native, nothing to verify" and
    "container, nothing to verify WITH" — two states the old boolean collapsed into a pass.
    Absent/unreadable manifest reads False, which degrades to the prior behaviour.
    """
    manifest = bundle_root / "bundle_manifest.json"
    if not manifest.is_file():
        return False
    try:
        return bool(json.loads(manifest.read_text()).get("sif_manifests"))
    except (ValueError, OSError):
        return False


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


def _verify_sif(sif_root: Path, expected_sha256: str) -> tuple[bool, bool | None]:
    """Locate the fetched SIF under ``sif_root`` by the crate's DIGEST (the carried key) and
    re-hash it. A missing image or a digest mismatch raises ``ProcessingError`` (fail-closed).
    ``signature_ok`` is always ``None``: signing is not a toolkit mechanism (ADR-2 amended)."""
    from hhemt.sif.identity import find_by_sha256

    found = find_by_sha256(sif_root, expected_sha256)
    if found is None:
        raise ProcessingError(
            operation="reprex SIF verify",
            filepath=sif_root,
            reason=(
                f"no image under reprex_config.sif_root={sif_root} has a manifest recording sha256 "
                f"{expected_sha256}. Fetch the reference SIF + its .manifest.json (the crate's by-reference "
                "SoftwareApplication) to their identity path under sif_root first, or rebuild via hhemt build-sifs."
            ),
        )
    sif_path, _manifest = found
    h = hashlib.sha256()
    with sif_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != expected_sha256:
        raise ProcessingError(
            operation="reprex SIF verify",
            filepath=sif_path,
            reason=(f"sha256 mismatch: this is NOT the reference SIF (expected {expected_sha256}, got {digest})."),
        )
    return True, None


def _scan_zero_user_info(bundle_root: Path) -> list[str]:
    """Consume-side INFORMATIONAL scan for producer user-info leaks (ADR-9). Non-fatal:
    returns the leak list rather than raising. Producer user-info is refused at emit time by
    hhemt.bundle._reprex_gate.ZeroUserInfoScanner; this consume-side scan runs the generic
    patterns and the private list only, because a consumer holds no producer configs."""
    import os
    import warnings

    from hhemt.bundle._reprex_gate import _ENV_OVERRIDE, assert_bundle_zero_user_info

    # A scan that did not search the private list is not a clean result, and the leak list cannot say so:
    # putting the state in it would make `reprex()` report a leak this scan did not find. So the state is
    # a warning, worded as the emit's own warning is. A set, non-empty variable is not warned about here:
    # a usable carrier is searched, and an unusable one is diagnosed in the leak list by the except below.
    carrier = os.environ.get(_ENV_OVERRIDE)
    if not carrier:
        warnings.warn(
            f"zero-user-info check: ${_ENV_OVERRIDE} is {'not set' if carrier is None else 'set but empty'}, "
            "so the private list was not searched; only the generic patterns were.",
            UserWarning,
            stacklevel=2,
        )

    try:
        assert_bundle_zero_user_info(bundle_root)
        return []
    except ProcessingError as exc:
        return [exc.reason]


def _refuse_to_replace_unsafe_destination(dest: Path, zip_path: Path) -> None:
    """Raise ``ProcessingError``, before ``dest`` is deleted, when that would destroy more than an old extraction.

    ``extract_reprex_bundle`` replaces an existing ``dest``, and a caller can name any directory. Refused: a ``dest``
    that is, or contains, the zip being extracted (the deletion would remove the zip before it is opened, and every
    file beside it, and a filesystem root contains every zip, so a root is refused by this same rule); the user's home
    directory; and a directory that is, or contains, the current working directory. The default destination (a
    sibling of the zip named for its stem) can be none of the first kind, and is subject to the rest.
    """
    resolved = dest.resolve()
    reason = None
    if zip_path.resolve().is_relative_to(resolved):
        reason = "it is, or contains, the zip being extracted, so replacing it would delete the zip first"
    elif resolved == Path.home().resolve():
        reason = "it is the user's home directory"
    elif Path.cwd().resolve().is_relative_to(resolved):
        reason = "it is, or contains, the current working directory"
    if reason is not None:
        raise ProcessingError(
            "extract a reprex bundle",
            dest,
            f"refusing to replace the destination directory: {reason}. Name an empty or absent directory.",
        )


def extract_reprex_bundle(zip_path: Path, dest: Path | None = None) -> Path:
    """Extract an emitted bundle ``.zip`` to ``dest`` (default: a sibling directory, same stem) and return
    it. The reprex round-trip consumes a directory root (``Bundle.from_directory``),
    while ``emit_bundle`` yields a zip; the emit-side facades call this so a reprex
    bundle is directly consumable. An existing ``dest`` is replaced, except one it is unsafe to delete
    (see ``_refuse_to_replace_unsafe_destination``), which raises ``ProcessingError`` before anything is removed."""
    import shutil

    from hhemt.bundle._archive import extract_zip

    zip_path = Path(zip_path)
    dest = Path(dest) if dest is not None else zip_path.with_suffix("")  # strip the trailing ".zip"
    if dest.exists():
        _refuse_to_replace_unsafe_destination(dest, zip_path)
        # EXEMPT-DU: bundle-root
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    # Threaded extract rather than ZipFile.extractall: a bundle holds tens of thousands of
    # small entries and extractall creates them one after another, which is the cost that
    # bounds this call on a shared file system. The files and bytes are the same.
    extract_zip(zip_path, dest)
    return dest


def _redirect_analysis_dir(analysis_config_path: Path, run_outputs_dir: Path) -> None:
    """Point the reconstituted ``analysis_config.yaml`` at ``run_outputs_dir`` for its run outputs.

    ``reprex_config.scratch_dir`` is the target-local base under which a reproduction
    writes its run outputs. The reconstituted configuration names the bundle root as
    ``analysis_dir`` (the bundle's FORCED_DOT marker), so the run outputs land in the
    bundle; with ``scratch_dir`` set they land in ``{scratch_dir}/{bundle directory name}``.
    Every input path is already rebased onto the carried bundle, so only ``analysis_dir``
    changes.
    """
    import yaml

    cfg_dict = yaml.safe_load(Path(analysis_config_path).read_text())
    cfg_dict["analysis_dir"] = str(run_outputs_dir)
    Path(analysis_config_path).write_text(yaml.safe_dump(cfg_dict, sort_keys=False))


def reprex(bundle_root: Path, reprex_cfg, target_hpc_profile) -> ReprexResult:
    """Verify + validate runnability + emit amendments for a reprex bundle."""
    from hhemt.bundle._emit import (
        reconstitute_runnable_analysis_config,
        reconstitute_runnable_config,
    )
    from hhemt.config.analysis import analysis_config
    from hhemt.config.loaders import yaml_to_model
    from hhemt.config.system import system_config
    from hhemt.validation import ValidationResult, preflight_validate

    bundle_root = Path(bundle_root).resolve()

    # 1. SIF resolve + verify (container bundle) or native no-SIF path.
    sif_entity = _find_sif_entity(bundle_root)
    sif_container_mode = _bundle_is_container_mode(bundle_root)
    if sif_entity is not None:
        sif_reference_present = True
        sif_verified, sif_signature_ok = _verify_sif(Path(reprex_cfg.sif_root), sif_entity["sha256"])
    else:
        sif_reference_present = False
        sif_signature_ok = None
        # THE SPLIT. A native bundle has nothing to verify, so True is honest. A CONTAINER
        # bundle with no digest had nothing to verify WITH, which is a different fact and
        # must not report as a pass — it is exactly the state a pre-(ii-a) toolkit emits.
        sif_verified = None if sif_container_mode else True

    # 2. Load the bundle's configs and re-aim them at the target HPC profile. Both
    # sides use the SINGLE reconstitution helpers (reconstitute_runnable_config /
    # reconstitute_runnable_analysis_config), which rebase EVERY bundle-relative Path
    # — including sensitivity_analysis — onto the carried, self-contained bundle. The
    # former inline sensitivity_analysis rebase is folded into the analysis helper so
    # there is one rebase implementation.
    cfg_system = yaml_to_model(reconstitute_runnable_config(bundle_root), system_config)
    analysis_config_path = reconstitute_runnable_analysis_config(bundle_root)
    if reprex_cfg.scratch_dir is not None:
        _redirect_analysis_dir(analysis_config_path, Path(reprex_cfg.scratch_dir) / bundle_root.name)
    cfg_analysis = yaml_to_model(analysis_config_path, analysis_config)

    # Overlay only the target partition selectors (the HPC-revisable axis); the input
    # paths are already rebased by the reconstitution helper above.
    overlay: dict[str, Any] = {
        "hpc_ensemble_partition": reprex_cfg.target_ensemble_partition,
        "hpc_setup_and_analysis_processing_partition": (
            reprex_cfg.target_setup_and_analysis_processing_partition or reprex_cfg.target_ensemble_partition
        ),
    }
    cfg_analysis_target = cfg_analysis.model_copy(update=overlay)

    # 3. Re-aim preflight at the target profile; extract the per-(member_id, column) pairs.
    try:
        result = preflight_validate(cfg_system, cfg_analysis_target, cfg_hpc_system=target_hpc_profile)
        all_issues = result.errors + result.warnings
    except Exception:
        # Full preflight can trip on the reprex cfg's target-supplied (not-yet-fetched)
        # paths; fall back to the isolated per-row cap scan so problem-pair emission
        # stays robust — the cap scan is the reprex-specific signal that matters here.
        from hhemt.validation import _validate_per_member_row_caps

        result = ValidationResult(context="reprex")
        _validate_per_member_row_caps(cfg_analysis_target, target_hpc_profile, result)
        all_issues = result.errors + result.warnings

    problem_pairs = [i for i in all_issues if i.field.startswith(_ROW_ISSUE_PREFIX)]
    runnable = len(problem_pairs) == 0

    # 4. Per-field graduated experiment amendments.
    amendments = _emit_amendments(cfg_analysis, reprex_cfg, target_hpc_profile, problem_pairs)

    # 5. Informational zero-user-info scan; the emit itself already refuses a bundle with producer user-info.
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
            reprex_cfg.target_setup_and_analysis_processing_partition or reprex_cfg.target_ensemble_partition,
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
                    "HPC-execution field with no deterministic target-partition mapping; revise to match your cluster."
                ),
            )
        )

    return amendments


# ---- Reference-output comparison ---------------------------------------------------------------------------------

#: Where the emit carries the producer's ``analysis.test()`` summaries (``bundle/_emit.py::_copy_reference_outputs``).
REFERENCE_OUTPUTS_DIRNAME = "reference_outputs"

#: The per-scenario form of a stored summary under both ``reference_outputs/`` and a reproducer's
#: ``{analysis_dir}/_test/``. A bundle emitted before the summaries moved into the consolidated tree carries this form.
_REFERENCE_STORE_GLOB = "group_*/sims/*/processed/*_summary.*"

#: The consolidated-tree form: each ``group_i`` holds this tree and the tree holds only its ``summary`` groups.
_REFERENCE_TREE_GLOB = "group_*/analysis_datatree.zarr"

#: Where a ``summary`` group sits inside a consolidated tree (``{family}/{kind}/summary``).
_SUMMARY_GROUP_GLOB = "*/*/summary"


@dataclass(frozen=True)
class ReferenceStoreVerdict:
    """The verdict on one reference store.

    ``status`` is ``equal``, ``differs`` or ``missing_in_rerun`` (the bundle carries the store and the re-run produced
    no store at that path). ``variables`` maps each variable that differs, is absent from the re-run, or differs in
    shape to a short reason that names the largest absolute difference when there is one.
    """

    store: str
    status: Literal["equal", "differs", "missing_in_rerun"]
    variables: dict[str, str] = field(default_factory=dict)
    extra_variables: tuple[str, ...] = ()
    #: ``(variable, reason)`` for each variable present in both stores that was NOT compared, by rule.
    excluded: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ReferenceComparison:
    """The per-store verdicts of ``compare_reference_outputs``.

    ``extra_stores`` are summary stores the re-run produced that the bundle does not carry, and each store's
    ``extra_variables`` are variables the re-run holds that the reference does not. Both are REPORTED and neither fails
    the comparison: the reference is the producer's own subset, and a later toolkit may add a store or a variable
    without changing any value the reference records.
    """

    verdicts: tuple[ReferenceStoreVerdict, ...]
    extra_stores: tuple[str, ...]
    rtol: float
    atol: float

    @property
    def excluded(self) -> tuple[tuple[str, str, str], ...]:
        """``(store, variable, reason)`` for every variable the comparison left out, so a report says what it did not
        compare. An exclusion applies only to a variable present in BOTH stores: a missing store or variable is a
        failure whatever the rule, and a changed scientific variable is never excluded."""
        return tuple((v.store, name, reason) for v in self.verdicts for name, reason in v.excluded)

    @property
    def passed(self) -> bool:
        """True when the bundle carried at least one store and every one is ``equal``."""
        return bool(self.verdicts) and all(v.status == "equal" for v in self.verdicts)

    def summary(self) -> str:
        """One line per store that did not pass, naming the store and each variable, then a totals line."""
        lines = []
        for verdict in self.verdicts:
            if verdict.status == "missing_in_rerun":
                lines.append(f"{verdict.store}: not produced by the re-run")
            elif verdict.status == "differs":
                for name, reason in sorted(verdict.variables.items()):
                    lines.append(f"{verdict.store}: variable {name!r} {reason}")
        equal = sum(v.status == "equal" for v in self.verdicts)
        for store, name, reason in self.excluded:
            lines.append(f"not compared: {store}: variable {name!r} ({reason})")
        lines.append(
            f"{equal} of {len(self.verdicts)} reference stores equal (rtol={self.rtol}, atol={self.atol}); "
            f"{len(self.extra_stores)} store(s) in the re-run are not in the bundle"
        )
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        """Raise ``ProcessingError`` carrying ``summary()`` unless ``passed``."""
        if not self.passed:
            raise ProcessingError("compare re-run outputs with the bundle's reference outputs", None, self.summary())


def _open_reference_store(path: Path, group: str | None = None):
    """The store at ``path`` as a fully loaded Dataset: a zarr store through its index when it has one, else NetCDF.

    ``group`` opens one group of a zarr store, which is how a ``summary`` group of a consolidated tree is read.
    """
    import xarray as xr

    from hhemt.utils import open_store

    if path.is_dir():
        with open_store(path, kind="dataset", group=group, chunks=None) as ds:
            return ds.load()
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        return ds.load()


def _values_equal(reference, rerun, rtol: float, atol: float) -> tuple[bool, str]:
    """Whether two arrays hold the same values, NaN-aware, and a reason when they do not.

    Exact (``rtol == atol == 0``) uses equality with NaN and NaT equal to themselves, so a value that differs by one
    unit in the last place is a difference. Otherwise numeric arrays are compared with ``numpy.isclose`` (NaN equal to
    NaN); non-numeric arrays (text, times) are always compared exactly.
    """
    import numpy as np

    a, b = np.asarray(reference), np.asarray(rerun)
    if a.shape != b.shape:
        return False, f"differs in shape: reference {a.shape}, re-run {b.shape}"
    numeric = a.dtype.kind in "fc" and b.dtype.kind in "fiuc"
    if a.dtype.kind in "mM" and b.dtype.kind == a.dtype.kind:
        same = (a == b) | (np.isnat(a) & np.isnat(b))
    elif numeric and (rtol or atol):
        same = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    elif a.dtype.kind in "fc" or b.dtype.kind in "fc":
        same = (a == b) | (np.isnan(a.astype(float)) & np.isnan(b.astype(float)))
    else:
        same = a == b
    if bool(np.all(same)):
        return True, ""
    reason = f"differs at {int(np.size(same) - np.count_nonzero(same))} of {int(np.size(same))} values"
    if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
        with np.errstate(invalid="ignore"):
            gap = np.nanmax(np.abs(a.astype(float) - b.astype(float))[~np.asarray(same)])
        reason += f" (largest absolute difference {float(gap)!r})"
    return False, reason


def compare_reference_outputs(
    bundle_root: Path, analysis_dir: Path, *, rtol: float = 0.0, atol: float = 0.0
) -> ReferenceComparison:
    """Compare a reproducer's ``analysis.test()`` summaries with the producer's, carried in the bundle.

    Each ``summary`` group of ``reference_outputs/group_i/analysis_datatree.zarr`` of ``bundle_root`` is compared,
    variable by variable and coordinate by coordinate, with the same group of the tree at the same path under
    ``{analysis_dir}/_test/`` (the reproducer's own ``test()`` output), NaN-aware. A bundle emitted before the
    summaries lived in the consolidated tree carries ``reference_outputs/group_i/sims/{event_id}/processed/*_summary.*``
    stores instead, and each of those is compared with the store at the same path under ``_test/``. Which form a
    bundle holds is decided by which subtree is present, and a bundle holding both is compared in both.

    Tolerance. The default is EXACT (``rtol = atol = 0``), because the emit carries these stores "for per-arch
    within-family EXACT equality" (``_copy_reference_outputs``): the producer's ``test()`` and a reproducer's on the
    same CPU or GPU family and the same pinned solver build are expected to agree to the bit. A re-run on a DIFFERENT
    architecture or compiler need not agree to the bit, and no tolerance is wide enough to be right for every
    variable, so the caller who knows the architectures differ passes ``rtol`` and ``atol`` explicitly; they apply to
    floating-point variables only (text, integers and times stay exact).

    What counts as a failure. A store the bundle carries and the re-run did not produce; a variable the reference holds
    that the re-run lacks; a variable or coordinate whose values or shape differ. What does not: a store or variable
    only the re-run has (reported in ``extra_stores`` and ``extra_variables``).

    Raises ``ProcessingError`` when the bundle carries neither form (it was emitted with no producer ``test()``),
    because an absent reference cannot support a claim of reproduction.
    """
    bundle_root, analysis_dir = Path(bundle_root), Path(analysis_dir)
    ref_root = bundle_root / REFERENCE_OUTPUTS_DIRNAME
    per_scenario = sorted(ref_root.glob(_REFERENCE_STORE_GLOB)) if ref_root.is_dir() else []
    carried_trees = sorted(ref_root.glob(_REFERENCE_TREE_GLOB)) if ref_root.is_dir() else []
    if not per_scenario and not carried_trees:
        raise ProcessingError(
            "compare re-run outputs with the bundle's reference outputs",
            ref_root,
            "the bundle carries no reference_outputs/ summaries (it was emitted with no producer "
            "analysis.test()), so there is nothing to compare the re-run with",
        )
    test_root = analysis_dir / "_test"
    verdicts: list[ReferenceStoreVerdict] = []
    for ref_path in per_scenario:
        rel = ref_path.relative_to(ref_root).as_posix()
        rerun_path = test_root / rel
        if not rerun_path.exists():
            verdicts.append(ReferenceStoreVerdict(rel, "missing_in_rerun"))
            continue
        verdicts.append(_reference_verdict(rel, ref_path, rerun_path, None, rtol, atol))
    for ref_tree in carried_trees:
        tree_rel = ref_tree.relative_to(ref_root).as_posix()
        rerun_tree = test_root / tree_rel
        for summary in sorted(ref_tree.glob(_SUMMARY_GROUP_GLOB)):
            group = summary.relative_to(ref_tree).as_posix()
            rel = f"{tree_rel}/{group}"
            if not (rerun_tree / group / "zarr.json").is_file():
                verdicts.append(ReferenceStoreVerdict(rel, "missing_in_rerun"))
                continue
            verdicts.append(_reference_verdict(rel, ref_tree, rerun_tree, group, rtol, atol))
    carried = {v.store for v in verdicts}
    available: list[str] = []
    if test_root.is_dir():
        if per_scenario:
            available += [p.relative_to(test_root).as_posix() for p in test_root.glob(_REFERENCE_STORE_GLOB)]
        for ref_tree in carried_trees:
            tree_rel = ref_tree.relative_to(ref_root).as_posix()
            rerun_tree = test_root / tree_rel
            available += [
                f"{tree_rel}/{p.relative_to(rerun_tree).as_posix()}"
                for p in rerun_tree.glob(_SUMMARY_GROUP_GLOB)
                if (p / "zarr.json").is_file()
            ]
    extra_stores = tuple(s for s in sorted(available) if s not in carried)
    return ReferenceComparison(tuple(verdicts), extra_stores, rtol, atol)


_STAMP_REASON = "build-provenance stamp: records which build produced the store, not a scientific value"
_TIMER_REASON = "wall-clock timer of the performance summary: differs between any two runs of the same build"


def _build_stamp_names() -> frozenset[str]:
    """The per-event stamp coordinates, derived from the keys of ``provenance.producing_stamp`` (``hhemt_sha`` becomes
    ``hhemt_producing_sha``, and so on), so the list follows the stamp's own definition."""
    from hhemt.provenance import producing_stamp

    return frozenset("hhemt_producing_" + key.removeprefix("hhemt_") for key in producing_stamp())


def _is_performance_store(rel: str, group: str | None) -> bool:
    """Whether the store is a ``performance`` summary: the tree group ``{family}/performance/summary``, or a
    per-scenario ``*perf*_summary`` store. Measured on a real re-run: every floating-point variable of that group is a
    wall-clock timer (all but two zero-valued ones differ), and its integer and text variables are deterministic."""
    if group is not None:
        return group.rstrip("/").endswith("performance/summary")
    return "_perf_summary" in rel


def _reference_verdict(
    rel: str, ref_path: Path, rerun_path: Path, group: str | None, rtol: float, atol: float
) -> ReferenceStoreVerdict:
    """The verdict on one reference store, or on one ``group`` of the consolidated tree at ``ref_path``.

    ``rel`` names the store in the verdict. A variable of the reference is compared with the same-named variable of the
    re-run, values and coordinates alike, NaN-aware, and a variable only the re-run holds is reported as extra.
    """
    ref_ds = _open_reference_store(ref_path, group)
    new_ds = _open_reference_store(rerun_path, group)
    differing: dict[str, str] = {}
    excluded: list[tuple[str, str]] = []
    stamps = _build_stamp_names()
    performance = _is_performance_store(rel, group)
    for name in sorted(ref_ds.variables, key=str):
        if name not in new_ds.variables:
            differing[str(name)] = "is absent from the re-run"
            continue
        if str(name) in stamps:
            excluded.append((str(name), _STAMP_REASON))
            continue
        if performance and name in ref_ds.data_vars and ref_ds[name].dtype.kind == "f":
            excluded.append((str(name), _TIMER_REASON))
            continue
        same, reason = _values_equal(ref_ds[name].values, new_ds[name].values, rtol, atol)
        if not same:
            differing[str(name)] = reason
    extra = tuple(sorted(str(n) for n in new_ds.variables if n not in ref_ds.variables))
    return ReferenceStoreVerdict(
        rel,
        "differs" if differing else "equal",
        variables=differing,
        extra_variables=extra,
        excluded=tuple(excluded),
    )
