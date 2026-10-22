"""Descriptor-driven experiment-bundle runner.

An experiment bundle is a directory whose ``experiment.yaml`` names the system config,
the analysis config, and a per-cluster ``hpc_system_config``. This module turns one
bundle into a run: load and validate the descriptor, refuse any CLI argument that would
silently override it, build the analysis, and run it.

It follows the same split as the ``synthetic_experiment.py`` +
``config/synthetic_experiment.py`` pair: the FRAMEWORK lives in ``src/hhemt/`` so it
ships in the wheel and is importable without ``tests/`` or ``scripts/``; the operator's
deployment-config repository holds experiment IDENTITY (``experiment.yaml``) and a thin
caller.

This module MUST NOT import from ``tests`` or ``scripts`` — a ``src -> tests`` or
``src -> scripts`` import breaks ``pip install -e .``.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hhemt.config.experiment_bundle import ExperimentConfig
from hhemt.exceptions import ConfigurationError
from hhemt.orchestration import RunMode


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

    Non-mutating: reads the deployment config, never edits a tracked file.

    Precedence (highest first):
      1. explicit ``override`` (``hhemt run-experiment --hpc-system-config``, or the
         second positional argument of ``container_validation.py``).
      2. the bundle's declared ``hpc_system_config[cluster]`` — deployment-config-relative,
         resolved against ``$HHEMT_DEPLOYMENT_CONFIG`` (or, absent that, the bundle's
         grandparent, i.e. ``{deployment_config}/experiments/{slug} ->
         {deployment_config}``). Consulted ONLY
         when a ``bundle`` is supplied and declares this cluster.
      3. ``$HHEMT_HPC_SYSTEM_CONFIG``.
      4. ``$HHEMT_DEPLOYMENT_CONFIG/hpc/hpc_system_config_{cluster}.yaml``.

    With ``bundle=None`` the resolution is steps 1, 3 and 4 alone.
    ``scripts/experiments/container_validation.py``'s ``build_case`` calls this function
    that way, with no bundle, so a change to any of those three steps changes that
    script's resolution too.
    """
    if override:
        path = Path(override).expanduser()
    elif bundle is not None and cluster in bundle.hpc_system_config:
        rel = bundle.hpc_system_config[cluster]
        declared = Path(rel).expanduser()
        if declared.is_absolute():
            path = declared
        else:
            deployment_config = os.environ.get("HHEMT_DEPLOYMENT_CONFIG")
            if deployment_config:
                deployment_config_root = Path(deployment_config).expanduser()
            elif bundle_dir is not None:
                # {deployment_config}/experiments/{slug} -> {deployment_config}
                deployment_config_root = Path(bundle_dir).expanduser().resolve().parent.parent
            else:
                deployment_config_root = Path.cwd()
            path = deployment_config_root / rel
    elif os.environ.get("HHEMT_HPC_SYSTEM_CONFIG"):
        path = Path(os.environ["HHEMT_HPC_SYSTEM_CONFIG"]).expanduser()
    else:
        deployment_config = os.environ.get("HHEMT_DEPLOYMENT_CONFIG")
        if not deployment_config:
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
        path = Path(deployment_config).expanduser() / "hpc" / f"hpc_system_config_{cluster}.yaml"
    if not path.is_file():
        raise ConfigurationError(
            field="hpc_system_config",
            message=(
                f"No hpc_system_config for cluster {cluster!r} at {path}. On the cluster, set "
                "$HHEMT_DEPLOYMENT_CONFIG to your compute-visible deployment-config checkout and "
                "`git pull` it, then reconstruct the per-cluster config (fill default_account). "
                "Or set $HHEMT_HPC_SYSTEM_CONFIG, or pass --hpc-system-config."
            ),
            config_path=path,
        )
    return path.resolve()


def bundle_experiment_id(bundle_dir: str | Path) -> str:
    """The experiment's identity is the bundle directory's NAME (by convention: an experiment
    descriptor carries no id field — `ExperimentConfig` is `extra="forbid"` — and the directory
    that holds `experiment.yaml` is what names it). Resolved first so a trailing slash or a
    relative `.` still yields the real directory name."""
    return Path(bundle_dir).resolve().name


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

    This lives in the wheel, like ``resolve_hpc_system_config`` above, so every
    deployment-config caller imports one copy instead of carrying its own. An unresolved
    placeholder raises ``ConfigurationError`` (CLI exit 2) rather than ``SystemExit``,
    because ``SystemExit`` is not an ``Exception`` and passes straight through a caller's
    ``except Exception``.
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

    Which arguments the override gate sees: the CLI verb wires only ``--hpc-system-config`` as a
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

    BOTH values are printed, not just the winner — a user who cannot see what they are
    overriding cannot consent to it.
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
    """Enforce the override contract for a non-empty override set: no CLI argument wins
    over the descriptor without the operator having seen both values and consented.

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
    ``bundle.hpc_system_config[cluster]`` names the per-cluster ``hpc_system_config``
    file, not a module-level dict. That resolved file is handed whole to
    ``Toolkit.from_configs``; this function itself reads only its ``default_account``
    and ``container`` fields, for the two guards below.

    Carries the same two fail-fast guards as that script, raising ``ConfigurationError``
    where the script raises ``ValueError``: a ``default_account`` that is unset or still a
    ``{your-...}`` placeholder raises; a missing or placeholder ``container.sif_root``
    raises when the bundle declares a container (the script checks it unconditionally). Config
    resolution is NON-MUTATING — it reads the deployment config, never edits a tracked file.
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
        if cfg_hpc.container is None or "{your-" in (cfg_hpc.container.sif_root or ""):
            raise ConfigurationError(
                field="container.sif_root",
                message=(
                    f"{cfg_path}: container.sif_root is missing or still a placeholder, but the "
                    "bundle declares a container. Set it to the directory `hhemt build-sifs` writes into."
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
    mode: RunMode = RunMode.resume,
    override_wipe_nonempty: bool = False,
    override_force_rerun: str | dict | None = None,
    override_clean_restart_member_ids: list[str] | None = None,
    debug: bool = False,
    **cli_overrides: object,
):
    """Load -> validate -> gate overrides -> build -> run.

    mode: a ``RunMode`` member. ``resume`` (default) picks up where the last invocation left
    off; ``fresh`` wipes the analysis_dir first. The default matches ``Toolkit.run``'s own
    default so the two layers state one value rather than two. This is a typed pass-through:
    the CLI refuses an unknown value at parse time and ``Toolkit.run`` refuses one at the
    reduction site, so no third check lives here.

    override_force_rerun: per-invocation override of the bundle analysis config's
    ``force_rerun`` (``"all"``, ``"none"``, or a subject dict), threaded untouched into
    ``Toolkit.run``. None, the default, reads the config field. This is the honest route to
    re-running completed scenarios without a wipe; it is a knob, not a mode.

    debug: the run's debug flag, threaded untouched into ``Toolkit.run``. Named here rather
    than left to ``**cli_overrides`` because it is a run argument, not a config override, and
    ``resolve_overrides`` would otherwise compare it against the bundle's configs. Today it does
    one thing: a coupled clean start asks the solver to keep the previous run's snapshot files.

    The override gate: if `resolve_overrides` returns a non-empty
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
    # RESUME IS THE DEFAULT; fresh must be asked for. Deliberately NOT fresh as the default:
    # fresh is DESTRUCTIVE rather than merely non-resuming. `Toolkit.run` maps mode=fresh to
    # from_scratch, and analysis.run() then deletes the whole analysis_dir with fast_rmtree --
    # completed sims, status flags, hotstart checkpoints and consolidated outputs. On a tree that
    # already holds work that delete is refused (`wipe_guard.assert_wipe_is_deliberate` raises
    # ConfigurationError) unless override_wipe_nonempty is passed, so under a fresh default a
    # requeue or a plain re-run of an existing analysis would fail instead of continuing.
    #
    # `mode` moves FIVE things at once (`orchestration.translate_mode`'s MODE_TRANSLATION plus
    # `Toolkit.run`'s from_scratch mapping): three redo-completed-work flags
    # (overwrite_system_inputs, overwrite_scenario_if_already_set_up,
    # rerun_swmm_hydro_if_outputs_exist), `pickup_where_leftoff` (whether an individual sim
    # hotstarts from config_NNNN.cfg), and the analysis_dir wipe. To decouple the sim-level half
    # from the rest, use analysis.run(override_pickup_where_leftoff=...), which exists for exactly
    # that.
    return tk.run(
        mode=mode,
        dry_run=dry_run,
        wait_for_completion=wait,
        override_wipe_nonempty=override_wipe_nonempty,
        override_force_rerun=override_force_rerun,
        override_clean_restart_member_ids=override_clean_restart_member_ids,
        debug=debug,
    )
