"""plan_sif_set: the deduplicated identity set N experiments need, or a refusal listing every offender."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from hhemt.exceptions import ConfigurationError
from hhemt.sif.identity import SifIdentity, derive_identity, manifest_path, resolve_sif
from hhemt.sif.pins import is_full_sha


@dataclass(frozen=True)
class ExperimentInputs:
    experiment_id: str
    cfg_system: object
    cfg_analysis: object
    cfg_hpc_target: object
    toolkit_pin_sha: str | None  # ExperimentBundle.toolkit_pin resolved to a sha, or None for raw triplets


@dataclass
class SifBuildPlan:
    entries: dict[str, SifIdentity] = field(default_factory=dict)  # key -> identity
    covers: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # key -> [(experiment_id, partition)]
    already_built: set[str] = field(default_factory=set)

    def table(self) -> str:  # the --dry-run instrument; the [Q312] counts are len(self.entries)
        rows = [
            f"{k}  {i.family:<18} {i.gpu_hardware or '-':<7} {i.triton_sha[:8]} {i.hhemt_sha[:8]}  "
            f"{'built' if k in self.already_built else 'TO BUILD':<8} covers={self.covers[k]}"
            for k, i in self.entries.items()
        ]
        n = len(self.entries)
        return "\n".join(rows) + f"\n{n} distinct identit{'y' if n == 1 else 'ies'}"


def _partitions(exp: ExperimentInputs) -> set[str]:
    from hhemt.bundle._emit import _matrix_required_partitions  # the lifted per-row enumeration (Spec 8)

    return set(_matrix_required_partitions(exp.cfg_analysis, exp.cfg_hpc_target))


def live_sentinels(sif_root: Path | str, key: str) -> list[Path]:
    """The sentinels that mean 'a producer for this identity exists right now': a QUEUED
    entry written by the driver before snakemake is invoked (Gotcha-46 shape) and a SUBMITTED
    entry written by the runner at job start."""
    root = Path(sif_root) / "_status"
    return [p for p in (root / "_queued" / f"{key}.json", root / "_submitted" / f"{key}.json") if p.exists()]


def plan_sif_set(
    experiments: list[ExperimentInputs],
    *,
    sif_root: Path,
    running_sha: str | None,
    recipes_dir: Path | None,
    toolkit_root: Path,
    force: bool = False,
) -> SifBuildPlan:
    offenders: list[str] = []
    if not is_full_sha(running_sha):
        raise ConfigurationError(
            field="toolkit",
            message=(
                "hhemt build-sifs must run from a git checkout (the builder IS the toolkit it bakes; "
                f"a wheel has no sha) — got {running_sha!r}."
            ),
        )
    if subprocess.run(
        ["git", "-C", str(toolkit_root), "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip():
        offenders.append(
            f"toolkit tree {toolkit_root} is DIRTY: a baked hhemt_sha would name a tree that does not exist"
        )
    plan = SifBuildPlan()
    for exp in experiments:
        if getattr(exp.cfg_analysis, "execution_environment", "native") != "container":
            offenders.append(
                f"{exp.experiment_id}: execution_environment != 'container' (nothing to build; refusing "
                "rather than warn-and-skip, which would hide a misconfigured experiment)"
            )
            continue
        if exp.toolkit_pin_sha and not str(running_sha).startswith(exp.toolkit_pin_sha[: len(str(running_sha))]):
            offenders.append(
                f"{exp.experiment_id}: toolkit_pin {exp.toolkit_pin_sha[:8]} != running {str(running_sha)[:8]}; "
                f"check out the pin and re-run: git -C {toolkit_root} checkout {exp.toolkit_pin_sha}"
            )
            continue
        for part in sorted(_partitions(exp)):
            ident = derive_identity(exp.cfg_hpc_target, exp.cfg_system, part, running_sha, recipes_dir=recipes_dir)
            plan.entries.setdefault(ident.key, ident)
            plan.covers.setdefault(ident.key, []).append((exp.experiment_id, part))
    if offenders:
        raise ConfigurationError(field="build-sifs", message="refusing to plan:\n  - " + "\n  - ".join(offenders))
    for key, ident in plan.entries.items():
        sif = resolve_sif(sif_root, ident)
        live = live_sentinels(sif_root, key)
        if live:
            raise ConfigurationError(
                field="build-sifs",
                message=f"identity {key} has a LIVE producer ({live[0]}); refusing a second one",
            )
        if not force and manifest_path(sif).exists() and sif.is_file():
            plan.already_built.add(key)
    return plan
