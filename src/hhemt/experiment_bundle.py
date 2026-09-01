"""Descriptor-driven experiment-bundle runner.

Mirrors the ``synthetic_experiment.py`` + ``config/synthetic_experiment.py`` pair
(Gotcha 67a): the FRAMEWORK lives in ``src/hhemt/`` so it ships in the wheel and is
importable without ``tests/`` or ``scripts/``; the estate holds experiment IDENTITY
(``experiment.yaml``) and a thin caller.

This module MUST NOT import from ``tests`` or ``scripts`` — a ``src -> tests`` or
``src -> scripts`` import breaks ``pip install -e .`` (the stipulation that forced the
synthetic-experiment matrix builder out of ``scripts/experiments/_matrix_builder.py``).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hhemt.config.experiment_bundle import ExperimentConfig
from hhemt.exceptions import ConfigurationError


@dataclass(frozen=True)
class OverrideReport:
    """One config-declared value that a CLI argument is overriding."""

    field: str
    config_value: str
    cli_value: str


def resolve_hpc_system_config(
    cluster: str,
    *,
    override: str | Path | None = None,
    bundle: ExperimentConfig | None = None,
    bundle_dir: str | Path | None = None,
) -> Path:
    """Resolve the operator's REAL ``hpc_system_config`` path for ``cluster``.

    Non-mutating: reads the estate config, never edits a tracked file.

    Precedence (highest first):
      1. explicit ``override`` (the ``--hpc-system-config`` CLI argument / argv[2]).
      2. the bundle's declared ``hpc_system_config[cluster]`` — estate-relative,
         resolved against ``$HHEMT_DEPLOYMENT_CONFIG`` (or, absent that, the bundle's
         grandparent, i.e. ``{estate}/experiments/{slug} -> {estate}``). Consulted ONLY
         when a ``bundle`` is supplied and declares this cluster.
      3. ``$HHEMT_HPC_SYSTEM_CONFIG``.
      4. ``$HHEMT_DEPLOYMENT_CONFIG/hpc/hpc_system_config_{cluster}.yaml``.

    Steps 1/3/4 are the precedence chain moved verbatim from the retired
    ``container_validation._resolve_hpc_system_config`` (D1 — already correct). Step 2
    is the additive bundle-declared source; with ``bundle=None`` the resolution is
    byte-identical to the retired script helper, which is why the demoted
    ``container_validation.build_case`` imports THIS function unchanged.
    """
    if override:
        path = Path(override).expanduser()
    elif bundle is not None and cluster in bundle.hpc_system_config:
        rel = bundle.hpc_system_config[cluster]
        declared = Path(rel).expanduser()
        if declared.is_absolute():
            path = declared
        else:
            estate = os.environ.get("HHEMT_DEPLOYMENT_CONFIG")
            if estate:
                estate_root = Path(estate).expanduser()
            elif bundle_dir is not None:
                # {estate}/experiments/{slug} -> {estate}
                estate_root = Path(bundle_dir).expanduser().resolve().parent.parent
            else:
                estate_root = Path.cwd()
            path = estate_root / rel
    elif os.environ.get("HHEMT_HPC_SYSTEM_CONFIG"):
        path = Path(os.environ["HHEMT_HPC_SYSTEM_CONFIG"]).expanduser()
    else:
        estate = os.environ.get("HHEMT_DEPLOYMENT_CONFIG")
        if not estate:
            raise ConfigurationError(
                field="hpc_system_config",
                message=(
                    f"No hpc_system_config source for cluster {cluster!r}: the bundle declares "
                    f"none and no fallback is set. Declare hpc_system_config[{cluster!r}] in "
                    "experiment.yaml, set $HHEMT_DEPLOYMENT_CONFIG to your compute-visible "
                    "deployment-config checkout, set $HHEMT_HPC_SYSTEM_CONFIG, or pass "
                    "--hpc-system-config."
                ),
                config_path=None,
            )
        path = Path(estate).expanduser() / "hpc" / f"hpc_system_config_{cluster}.yaml"
    if not path.is_file():
        raise ConfigurationError(
            field="hpc_system_config",
            message=(
                f"No hpc_system_config for cluster {cluster!r} at {path}. On the cluster, set "
                "$HHEMT_DEPLOYMENT_CONFIG to your compute-visible deployment-config checkout and "
                "`git pull` it, then reconstruct the per-cluster config (fill default_account and "
                "container.sif_path). Or set $HHEMT_HPC_SYSTEM_CONFIG, or pass --hpc-system-config."
            ),
            config_path=path,
        )
    return path.resolve()


def load_bundle(bundle_dir: str | Path) -> ExperimentConfig:
    """Load and validate `{bundle_dir}/experiment.yaml`.

    Raises ConfigurationError (CLI exit 2) on absence or schema violation.
    """
    import yaml

    manifest = Path(bundle_dir) / "experiment.yaml"
    if not manifest.is_file():
        raise ConfigurationError(
            field="experiment.yaml",
            message=f"No experiment.yaml in bundle directory {bundle_dir}.",
            config_path=manifest,
        )
    try:
        raw = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigurationError(field="experiment.yaml", message=f"unparseable YAML: {e}", config_path=manifest) from e
    try:
        return ExperimentConfig.model_validate(raw)
    except Exception as e:
        raise ConfigurationError(field="experiment.yaml", message=f"schema violation: {e}", config_path=manifest) from e


def resolve_def_recipe(bundle: ExperimentConfig, bundle_dir: str | Path) -> Path:
    """Resolve ``bundle.container.def_recipe`` to a concrete path, or raise.

    [Q161] ruled BUNDLE-relative the default and asked for a shared-container arm.
    Both live in ONE field because the VALUE declares its own root: a ``${VAR}``
    prefix means explicitly-rooted, anything else means bundle-relative. There is
    therefore no resolution ORDER and no cwd fallback -- the two arms cannot
    disagree about a value, because each value selects exactly one of them.

    Raises ConfigurationError when the bundle declares no container, when a
    ``${VAR}`` does not resolve, or when the resolved recipe is not on disk.
    """
    if bundle.container is None:
        raise ConfigurationError(
            field="container",
            message="bundle declares no container; there is no def_recipe to resolve.",
            config_path=Path(bundle_dir) / "experiment.yaml",
        )
    raw = bundle.container.def_recipe
    if raw.startswith("${"):
        expanded = os.path.expandvars(raw)
        if "${" in expanded:
            unresolved = sorted({"${" + tok.split("}")[0] + "}" for tok in expanded.split("${")[1:]})
            raise ConfigurationError(
                field="container.def_recipe",
                message=(
                    f"unresolved placeholder(s) {unresolved} in def_recipe {raw!r}. Export the "
                    "referenced environment variable(s) before running."
                ),
                config_path=Path(bundle_dir) / "experiment.yaml",
            )
        resolved = Path(expanded)
    else:
        resolved = Path(bundle_dir) / raw
    if not resolved.is_file():
        raise ConfigurationError(
            field="container.def_recipe",
            message=(
                f"def_recipe {raw!r} resolves to {resolved}, which does not exist. A bare "
                "relative value is BUNDLE-relative; use a ${VAR}-rooted value for a recipe "
                "shared across experiments."
            ),
            config_path=Path(bundle_dir) / "experiment.yaml",
        )
    return resolved


def resolve_container_defs(
    experiment_config: str | Path | None,
    explicit_defs: list[Path] | None,
) -> list[Path]:
    """Reconcile the descriptor's container recipe against an explicit ``--container-defs``.

    Sibling of ``resolve_overrides``/``_confirm_override_gate``, which already reconcile a
    CLI argument against the descriptor in this module and REFUSE rather than silently
    prefer either source. Same contract here, for the same reason: silently preferring one
    side is how a bundle ends up carrying a recipe nobody chose, and a bundle is the
    ten-year reproducibility artifact.

    - No descriptor -> the explicit list passes through UNCHANGED, so ``emit_bundle``'s
      hard refusal on a container-mode analysis with no recipe is preserved verbatim.
    - Descriptor, no flag -> the descriptor supplies.
    - Descriptor + agreeing flag -> accepted.
    - Descriptor + disagreeing flag -> ``ConfigurationError`` naming BOTH values.

    SINGLE-ARCH BY CONSTRUCTION, and deliberately so. ``ContainerRef.def_recipe`` is
    scalar, so a descriptor-driven bundle carries exactly one recipe. That matches the
    estate's own design -- every live hpc_system_config records `sif_paths_by_arch
    intentionally EMPTY: this bundle pins ONE partition and therefore one gpu_hardware`
    -- and all nine live descriptors declare exactly one def_recipe. ADR-19 multi-SIF
    (one .def per arch) remains available on the FLAG path: omit ``--experiment-config``
    and pass ``--container-defs`` per arch. The remedy below branches on that, because a
    multi-def operator cannot `correct the descriptor` -- the schema cannot express them.
    """
    if experiment_config is None:
        return list(explicit_defs or [])
    resolved = resolve_def_recipe(load_bundle(experiment_config), experiment_config)
    if explicit_defs and [Path(p).resolve() for p in explicit_defs] != [resolved.resolve()]:
        if len(explicit_defs) > 1:
            _remedy = (
                "This looks like an ADR-19 multi-SIF (one .def per arch) emit, which a "
                "descriptor cannot express: container.def_recipe is scalar. Omit "
                "--experiment-config and pass --container-defs per arch."
            )
        else:
            _remedy = "Drop the flag to use the descriptor, or correct the descriptor."
        raise ConfigurationError(
            field="container_defs",
            message=(
                f"--container-defs {[str(p) for p in explicit_defs]} disagrees with the "
                f"experiment descriptor's container.def_recipe, which resolves to "
                f"{resolved}. Refusing rather than silently preferring either source. "
                f"{_remedy}"
            ),
            config_path=Path(experiment_config) / "experiment.yaml",
        )
    return [resolved]


def expand_config_vars(cfg_path: str | Path, *, dest_dir: str | Path | None = None) -> Path:
    """Expand ``${VAR}`` env references in a config file, materialize a resolved copy, return its path.

    The toolkit config loader applies ``expanduser`` but NEVER ``expandvars`` — a
    ``${DATA_DIR}``-style placeholder reaches the preflight path-existence check
    (``validation.py::_validate_system_paths``) verbatim and fails
    ``"Path does not exist for DEM_fullres: ${DATA_DIR}/dem.tif"``. So expand the
    placeholders against ``os.environ`` here, before ``Toolkit.from_configs``.

    Materialization: the resolved copy is written to ``$SCRATCH_DIR/resolved_configs/``
    (a SHARED filesystem, so ``batch_job`` Snakemake rules dispatched to OTHER compute
    nodes can read it — node-local ``/tmp`` is invisible cross-node), falling back to
    the platform tempdir when ``$SCRATCH_DIR`` is unset (a purely-local dry-run). This
    is an ephemeral INPUT, not an analysis OUTPUT, so it is compatible with a
    ``dry_run`` that writes nothing to the analysis output tree.

    Lifts the estate driver's ``run_analysis_test.py::_resolve_config`` into the wheel
    (the same drift-free move Phase 5 made for ``resolve_hpc_system_config``), but
    raises ``ConfigurationError`` (CLI exit 2) instead of ``SystemExit``.
    """
    cfg_path = Path(cfg_path)
    resolved_text = os.path.expandvars(cfg_path.read_text(encoding="utf-8"))
    if "${" in resolved_text:
        unresolved = sorted({tok.split("}")[0] + "}" for tok in resolved_text.split("${")[1:]})
        raise ConfigurationError(
            field=cfg_path.name,
            message=(
                f"{cfg_path.name}: unresolved config placeholder(s) {unresolved}. "
                "Export the referenced environment variable(s) before running "
                "(e.g. DATA_DIR / PACKAGE_DIR / SCRATCH_DIR)."
            ),
            config_path=cfg_path,
        )
    if dest_dir is not None:
        root = Path(dest_dir)
    else:
        root = Path(os.environ.get("SCRATCH_DIR") or tempfile.gettempdir()) / "resolved_configs"
    root.mkdir(parents=True, exist_ok=True)
    resolved = root / f"resolved_{cfg_path.name}"
    resolved.write_text(resolved_text, encoding="utf-8")
    return resolved


def resolve_overrides(bundle: ExperimentConfig, cli_args: dict[str, object]) -> list[OverrideReport]:
    """Return every field where a CLI argument differs from the descriptor.

    An empty list means the CLI adds nothing the descriptor does not already say —
    the one-config path, no confirmation needed.

    Gateable-field mapping (R8): the CLI verb wires only ``--hpc-system-config`` as a
    real config override. ``dry_run``/``yes`` are controls, not overrides, so they are
    never reported here. An argument is an OVERRIDE only when the descriptor already
    DECLARES a value for it and the CLI value DIFFERS — supplying a value the descriptor
    omits fills a gap, it does not override.
    """
    reports: list[OverrideReport] = []
    cluster = cli_args.get("cluster")
    cli_hpc = cli_args.get("hpc_system_config_yaml")
    if cli_hpc is not None and cluster is not None:
        declared = bundle.hpc_system_config.get(str(cluster))
        if declared is not None and str(declared) != str(cli_hpc):
            reports.append(
                OverrideReport(
                    field=f"hpc_system_config[{cluster}]",
                    config_value=str(declared),
                    cli_value=str(cli_hpc),
                )
            )
    return reports


def format_override_gate(reports: list[OverrideReport]) -> str:
    """Render the side-by-side config-vs-CLI table for the confirmation prompt.

    R8 requires BOTH values be printed, not just the winner — a user who cannot see
    what they are overriding cannot consent to it.
    """
    fw = max([len("field")] + [len(r.field) for r in reports])
    cw = max([len("descriptor (config)")] + [len(r.config_value) for r in reports])
    vw = max([len("CLI override")] + [len(r.cli_value) for r in reports])
    lines = [
        "The following CLI arguments override values declared in experiment.yaml:",
        "",
        f"  {'field':<{fw}}  {'descriptor (config)':<{cw}}  {'CLI override':<{vw}}",
        f"  {'-' * fw}  {'-' * cw}  {'-' * vw}",
    ]
    for r in reports:
        lines.append(f"  {r.field:<{fw}}  {r.config_value:<{cw}}  {r.cli_value:<{vw}}")
    return "\n".join(lines)


def _confirm_override_gate(reports: list[OverrideReport], *, assume_yes: bool) -> None:
    """Enforce the R8 override contract for a non-empty override set.

    - ``assume_yes`` => accept without prompting.
    - non-TTY without ``assume_yes`` => REFUSE (raise) rather than silently preferring
      either source. Silently preferring the CLI is how a bare ``run_mode=gpu`` ran a
      CUDA-only build in CPU mode.
    - TTY => print the side-by-side table and require explicit confirmation.
    """
    import sys

    gate = format_override_gate(reports)
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise ConfigurationError(
            field="cli_overrides",
            message=(
                f"{gate}\n\nCLI arguments override the descriptor, but this is a "
                "non-interactive context and --yes was not given. Refusing rather than "
                "silently preferring the CLI. Re-run with --yes to accept these overrides."
            ),
            config_path=None,
        )
    print(gate)
    try:
        import typer

        confirmed = typer.confirm("Proceed with these overrides?")
    except Exception:
        confirmed = input("Proceed with these overrides? [y/N] ").strip().lower() in ("y", "yes")
    if not confirmed:
        raise ConfigurationError(
            field="cli_overrides",
            message="Override gate declined; aborting rather than proceeding against the descriptor.",
            config_path=None,
        )


def build_case_from_bundle(
    bundle: ExperimentConfig,
    bundle_dir: str | Path,
    cluster: str,
    *,
    hpc_system_config_yaml: str | Path | None = None,
):
    """Construct the analysis for `bundle` on `cluster`.

    Generalizes ``container_validation.py::build_case`` with NO hardcoded cluster map:
    the per-cluster knobs come from ``bundle.hpc_system_config[cluster]`` and the
    resolved ``hpc_system_config``'s ``PartitionSpec``, not from a module-level dict.

    Preserves the landed fail-fast guards verbatim: a ``default_account`` that is unset
    or still a ``{your-...}`` placeholder raises; a missing or placeholder
    ``container.sif_path`` raises when the bundle declares a container. Config
    resolution is NON-MUTATING — it reads the estate config, never edits a tracked file.
    """
    from hhemt.config.loaders import load_hpc_system_config
    from hhemt.toolkit import Toolkit

    bundle_dir = Path(bundle_dir)
    cfg_path = resolve_hpc_system_config(cluster, override=hpc_system_config_yaml, bundle=bundle, bundle_dir=bundle_dir)
    cfg_hpc = load_hpc_system_config(cfg_path)

    account = cfg_hpc.default_account or ""
    if (not account) or ("{your-" in account):
        raise ConfigurationError(
            field="default_account",
            message=(
                f"{cfg_path}: default_account is unset or still a placeholder ({account!r}). "
                "Set default_account to your real OLCF project / UVA allocation."
            ),
            config_path=cfg_path,
        )
    if bundle.container is not None:
        if cfg_hpc.container is None or "{your-" in (cfg_hpc.container.sif_path or ""):
            raise ConfigurationError(
                field="container.sif_path",
                message=(
                    f"{cfg_path}: container.sif_path is missing or still a placeholder, but the "
                    f"bundle declares a container ({bundle.container.def_recipe}). Set it to the "
                    "absolute on-cluster path of your transferred, signed SIF."
                ),
                config_path=cfg_path,
            )

    # Expand ${VAR} placeholders (${DATA_DIR}/${SCRATCH_DIR}/${PACKAGE_DIR}/${HHEMT_*})
    # in the two bundle-relative configs before from_configs — the loader does expanduser
    # but not expandvars, so an unexpanded placeholder fails preflight path-existence.
    # The hpc config path is already an absolute, env-resolved path (resolve_hpc_system_config),
    # and its contents are operator-filled real values, so it is passed through unmodified.
    return Toolkit.from_configs(
        system_config=expand_config_vars(bundle_dir / bundle.system_config),
        analysis_config=expand_config_vars(bundle_dir / bundle.analysis_config),
        hpc_system_config=cfg_path,
    )


def run_experiment(
    bundle_dir: str | Path,
    cluster: str,
    *,
    dry_run: bool = False,
    hpc_system_config_yaml: str | Path | None = None,
    assume_yes: bool = False,
    wait: bool | None = None,
    mode: str = "resume",
    override_wipe_nonempty: bool = False,
    **cli_overrides: object,
):
    """Load -> validate -> gate overrides -> build -> run.

    mode: 'resume' (default) picks up where the last invocation left off; 'fresh' wipes the
    analysis_dir first; 'overwrite' reruns existing scenarios without a full reset. The default
    matches ``Toolkit.run``'s own default so the two layers state one value rather than two.

    The override gate is the R8 contract: if `resolve_overrides` returns a non-empty
    list, print the side-by-side table and require explicit confirmation. A non-TTY
    invocation without `assume_yes` REFUSES rather than silently preferring either
    source — silently preferring the CLI is how a bare `run_mode=gpu` ran a CUDA-only
    build in CPU mode.
    """
    bundle_dir = Path(bundle_dir)
    bundle = load_bundle(bundle_dir)

    cli_args: dict[str, object] = {
        "cluster": cluster,
        "hpc_system_config_yaml": hpc_system_config_yaml,
        **cli_overrides,
    }
    reports = resolve_overrides(bundle, cli_args)
    if reports:
        _confirm_override_gate(reports, assume_yes=assume_yes)

    tk = build_case_from_bundle(bundle, bundle_dir, cluster, hpc_system_config_yaml=hpc_system_config_yaml)
    # Resolve the batch_job wait contract. A batch_job run launches a DETACHED tmux
    # orchestrator whose tmux-server lifetime is bounded by the node it runs on.
    # Inside an sbatch allocation ($SLURM_JOB_ID set) the tmux dies when the allocation
    # ends, so run-experiment MUST block to keep the allocation alive (mirrors
    # .test(wait_for_job_completion=True)). On a login node the tmux server persists
    # independently, so fire-and-forget is correct. --wait/--no-wait overrides.
    if wait is None:
        wait = (not dry_run) and ("SLURM_JOB_ID" in os.environ)
    # RESUME IS THE DEFAULT ([Q144]). This previously hardcoded mode="fresh", which was not a
    # considered choice -- it arrived with the runner in 34fd5904 and no message defends it -- and
    # which is DESTRUCTIVE rather than merely non-resuming: toolkit.py:309 maps mode=="fresh" to
    # from_scratch, and analysis.run() then fast_rmtree()s the whole analysis_dir. A requeue or a
    # re-run therefore discarded completed sims, status flags, hotstart checkpoints and consolidated
    # outputs, which is why experiments/norfolk/benchmarking/cpu_uva/submit_benchmarking_cpu_uva.sh
    # grew a REQUEUE GUARD that refuses rather than re-running. This line retires that workaround.
    #
    # `mode` moves FIVE flags at once (orchestration.py:381/389 + toolkit.py:309): the three
    # redo-completed-work flags, `pickup_where_leftoff` (whether an individual sim hotstarts from
    # config_NNNN.cfg), and the analysis_dir wipe. To decouple the sim-level half from the rest,
    # use analysis.run(override_pickup_where_leftoff=...), which exists for exactly that.
    return tk.run(
        mode=mode,
        dry_run=dry_run,
        wait_for_completion=wait,
        override_wipe_nonempty=override_wipe_nonempty,
    )
