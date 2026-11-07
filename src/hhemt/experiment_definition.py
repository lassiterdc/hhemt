"""Descriptor-driven experiment-definition runner.

An experiment definition is a directory whose ``experiment.yaml`` names the system config,
the analysis config, and a per-cluster ``hpc_system_config``. The module turns a
definition into a run by loading and validating the descriptor, refusing any CLI argument that would
silently override the descriptor, building the analysis, and running the analysis.

The framework ships in the wheel and imports without ``tests/`` or ``scripts/``. The
operator's deployment-config repository holds experiment identity in ``experiment.yaml``
and a thin caller.
"""

from __future__ import annotations

# This module MUST NOT import from ``tests`` or ``scripts``: a ``src -> tests`` or
# ``src -> scripts`` import breaks ``pip install -e .``.
import hashlib
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hhemt.config.experiment_definition import ExperimentConfig
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
    definition: ExperimentConfig | None = None,
    definition_dir: str | Path | None = None,
) -> Path:
    """Resolve the operator's REAL path to ``hpc_system_config`` on ``cluster``.

    Non-mutating. Reads the deployment config and edits no tracked file.

    The function reads the sources below only, highest first:
      1. explicit ``override``, the ``--hpc-system-config`` CLI argument.
      2. the experiment definition's declared ``hpc_system_config[cluster]``. The path is absolute or relative to the
         definition directory itself, the folder holding ``experiment.yaml``, so the entry
         means the file beside the descriptor wherever the experiment definition sits, and no directory above
         the definition is searched. With no ``definition_dir``, a relative entry resolves against the
         working directory.

    With neither, the function raises ``ConfigurationError`` naming both. Environment variables are
    deliberately NOT consulted, because an HPC config chosen by the shell, and not by the command
    line or the experiment descriptor, is not recorded anywhere the run's record can see.
    """
    if override:
        path = Path(override).expanduser()
    elif definition is not None and cluster in definition.hpc_system_config:
        declared = Path(definition.hpc_system_config[cluster]).expanduser()
        if declared.is_absolute():
            path = declared
        else:
            # Relative to the experiment definition directory itself (never a parent of it); the working
            # directory only when the caller gives no definition_dir.
            root = Path(definition_dir).expanduser().resolve() if definition_dir is not None else Path.cwd()
            path = root / declared
    else:
        raise ConfigurationError(
            field="hpc_system_config",
            message=(
                f"No hpc_system_config source for cluster {cluster!r}: pass --hpc-system-config, "
                f"or declare hpc_system_config[{cluster!r}] in experiment.yaml."
            ),
            config_path=None,
        )
    if not path.is_file():
        raise ConfigurationError(
            field="hpc_system_config",
            message=(
                f"No hpc_system_config for cluster {cluster!r} at {path}. Reconstruct the per-cluster "
                "config there (fill default_account), or point --hpc-system-config or "
                f"hpc_system_config[{cluster!r}] in experiment.yaml at an existing one."
            ),
            config_path=path,
        )
    return path.resolve()


def experiment_definition_id(definition_dir: str | Path) -> str:
    """The experiment's identity is the NAME of the experiment definition directory. By convention, an experiment
    descriptor carries no id field, because `ExperimentConfig` is `extra="forbid"`, and the directory
    that holds `experiment.yaml` names the experiment. The path is resolved first so a trailing slash or a
    relative `.` still yields the real directory name."""
    return Path(definition_dir).resolve().name


def load_experiment_definition(definition_dir: str | Path) -> ExperimentConfig:
    """Load and validate `{definition_dir}/experiment.yaml`.

    Raises ConfigurationError on absence or schema violation, with CLI exit 2.
    """
    import yaml

    manifest = Path(definition_dir) / "experiment.yaml"
    if not manifest.is_file():
        raise ConfigurationError(
            field="experiment.yaml",
            message=f"No experiment.yaml in definition directory {definition_dir}.",
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
    """Expand ``${VAR}`` env references in a config file, materialize a resolved copy, and return the path of the copy.

    The toolkit config loader applies ``expanduser`` but NEVER ``expandvars``. A
    ``${DATA_DIR}``-style placeholder reaches the config-load path-existence check,
    ``config/base.py::cfgBaseModel._check_paths_exist``, verbatim. ``Toolkit.from_configs``
    reaches that check through ``load_system_config`` before any preflight runs, and the
    load raises ``ConfigurationError`` wrapping ``ValueError(f"File does not exist: {p}")``. So expand
    the placeholders against ``os.environ`` here, before ``Toolkit.from_configs``.

    The resolved copy is written to ``$SCRATCH_DIR/resolved_configs/``.
    The directory sits on a SHARED filesystem, so ``batch_job`` Snakemake rules dispatched to OTHER compute
    nodes can read the file, while node-local ``/tmp`` is invisible cross-node. The write falls back to
    the platform tempdir when ``$SCRATCH_DIR`` is unset, as in a purely-local dry-run. The copy
    is an ephemeral INPUT and no analysis OUTPUT, so the file is compatible with a
    ``dry_run`` that writes no file to the analysis output tree.

    The function lives in the wheel, like ``resolve_hpc_system_config`` above, so every
    deployment-config caller imports a single copy and keeps no copy of the function. An unresolved
    placeholder raises ``ConfigurationError``, with CLI exit 2, and never ``SystemExit``,
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


#: Sentinel returned by ``_default_rehydration_value`` for a token whose documented default is to
#: stay as the token, so "no default exists" (None) and "the default is the token" stay distinct.
_LEAVE_AS_TOKEN = object()


def _own_package_dir() -> Path:
    """The running install's ``hhemt`` package directory (present on a source checkout and an installed copy)."""
    import hhemt

    return Path(hhemt.__file__).resolve().parent


def _own_checkout_root() -> Path | None:
    """The running install's checkout root, or None when its package does not sit under ``src/``."""
    package_dir = _own_package_dir()
    return package_dir.parent.parent if package_dir.parent.name == "src" else None


def _nearest_bundle_root(member: Path) -> Path | None:
    """The nearest ancestor of ``member`` that holds ``bundle_manifest.json``, or None outside any bundle."""
    from hhemt.version_migration.constants import BUNDLE_MANIFEST_FILENAME

    for ancestor in member.resolve().parents:
        if (ancestor / BUNDLE_MANIFEST_FILENAME).is_file():
            return ancestor
    return None


def _default_rehydration_value(token: str, member: Path) -> object:
    """A closed token's documented default: a string, ``_LEAVE_AS_TOKEN``, or None when no default exists."""
    from hhemt.bundle import _placeholders as ph

    if token in (ph.TOKEN_ANALYSIS_DIR, ph.TOKEN_SYSTEM_DIR):
        root = _nearest_bundle_root(member)
        return None if root is None else str(root)
    if token == ph.TOKEN_TOOLKIT_DIR:
        checkout = _own_checkout_root()
        return None if checkout is None else str(checkout)
    if token == ph.TOKEN_PACKAGE_DIR:
        return str(_own_package_dir())
    if token == ph.TOKEN_PYTHON:
        return sys.executable
    if token == ph.TOKEN_HOME:
        return str(Path.home())
    return _LEAVE_AS_TOKEN


def rehydrate_placeholders(
    path: str | Path,
    values: Mapping[str, str | os.PathLike[str]] | None = None,
    *,
    dest_dir: str | Path | None = None,
) -> Path:
    """Put a consumer's own values back for the ``{hhemt_*}`` tokens in a bundle member, and return a resolved copy.

    An emitted bundle replaces the producer's directories, interpreter, home directory, login name, and
    SLURM account with the closed tokens that ``hhemt.bundle._placeholders.PLACEHOLDER_TOKENS`` lists. A reader
    that parses such a member as a path calls this first and reads the returned copy. The source member is
    never modified.

    ``values`` maps token, braces included as in ``"{hhemt_slurm_account}"``, to the consumer's value and
    wins over the default. Defaults, per token:

    * ``{hhemt_analysis_dir}`` and ``{hhemt_system_dir}``: the bundle root, taken as the nearest ancestor of
      ``path`` that holds ``bundle_manifest.json``. Outside any bundle there is no default.
    * ``{hhemt_toolkit_dir}``: the checkout root of this install, only when the package of the install
      sits under ``src/``.
      Otherwise there is no default and the token is refused until the caller supplies a value.
      ``{hhemt_package_dir}``: the ``hhemt`` package directory of this install.
    * ``{hhemt_python}``: ``sys.executable``. ``{hhemt_home}``: ``Path.home()``.
    * ``{hhemt_conda_env}``, ``{hhemt_user}``, ``{hhemt_slurm_account}`` and ``{hhemt_producer_path}``: left as
      the token, because the consumer has no value of the producer's type to give.

    Raises ``ConfigurationError``, with CLI exit 2, before writing any file, when the member holds a token that has
    neither a supplied value nor a default, or when ``values`` names a token outside the closed set. An unresolved
    token is refused and never left in place, because a directory token left in a path that is then opened
    produces a ``FileNotFoundError`` that names the token and not the missing value.

    The copy is ``rehydrated_{name}``. The prefix keeps the copy from ever being the source, even in the
    directory of the source. The copy lives under ``dest_dir`` when given. Without ``dest_dir``, the copy
    lives under ``$SCRATCH_DIR``, else the platform tempdir,
    in ``rehydrated_members/{digest of the source path}/``. The digest subdirectory is there because bundle
    members routinely share a file name, such as every sensitivity ``zarr.json``, and a shared directory would let a
    later call overwrite the copy of an earlier call while the caller of the earlier call still holds the path.

    Deliberately NOT ``expand_config_vars``. The latter runs ``os.path.expandvars``. The call expands every
    ``${...}`` and rewrites the shell references, such as ``${CONDA_PREFIX}``, that a bundled Snakefile legitimately
    carries. The function here matches only the closed token names, in one pass, so a substituted value is
    never scanned again.

    Caveat. The as-given and resolved spellings of a directory share one token, so a single value
    is unable to restore the spelling of a given occurrence. A
    percent-encoded path, ``%2Fscratch%2F...``, matched no placeholder row at emit time and is not touched
    here. The member is read whole, so the function suits text members.
    """
    from hhemt.bundle._placeholders import PLACEHOLDER_TOKENS

    member = Path(path)
    supplied = dict(values or {})
    unknown = sorted(set(supplied) - set(PLACEHOLDER_TOKENS))
    if unknown:
        raise ConfigurationError(
            field=member.name,
            message=(
                f"{member.name}: unknown placeholder token(s) {unknown}; the closed set is {list(PLACEHOLDER_TOKENS)}."
            ),
            config_path=member,
        )

    for token, supplied_value in supplied.items():
        if not os.fspath(supplied_value):
            raise ConfigurationError(
                field=member.name,
                message=f"{member.name}: the supplied value for {token} is empty.",
                config_path=member,
            )

    data = member.read_bytes()
    replacements: dict[bytes, bytes] = {}
    unresolved: list[str] = []
    for token in PLACEHOLDER_TOKENS:
        raw_token = token.encode()
        if raw_token not in data:
            continue
        value: object = supplied.get(token)
        if value is None:
            value = _default_rehydration_value(token, member)
        if value is None:
            unresolved.append(token)
        elif value is _LEAVE_AS_TOKEN:
            replacements[raw_token] = raw_token
        else:
            replacements[raw_token] = os.fspath(value).encode()  # type: ignore[arg-type]
    if unresolved:
        raise ConfigurationError(
            field=member.name,
            message=(
                f"{member.name}: unresolved placeholder token(s) {unresolved}. Pass values={{token: value}}, "
                "or rehydrate a member that sits inside a bundle (a directory holding bundle_manifest.json), "
                "whose root is the default for the two directory tokens."
            ),
            config_path=member,
        )

    if dest_dir is not None:
        root = Path(dest_dir)
    else:
        digest = hashlib.sha256(str(member.resolve()).encode()).hexdigest()[:12]
        root = Path(os.environ.get("SCRATCH_DIR") or tempfile.gettempdir()) / "rehydrated_members" / digest
    root.mkdir(parents=True, exist_ok=True)
    resolved = root / f"rehydrated_{member.name}"
    if replacements:
        pattern = re.compile(b"|".join(re.escape(raw) for raw in replacements))
        data = pattern.sub(lambda match: replacements[match.group(0)], data)
    resolved.write_bytes(data)
    return resolved


def resolve_overrides(definition: ExperimentConfig, cli_args: dict[str, object]) -> list[OverrideReport]:
    """Return every field where a CLI argument differs from the descriptor.

    An empty list means the CLI adds no value beyond the descriptor.
    The result is the one-config path, and no confirmation is needed.

    The override gate sees only some arguments. The CLI verb wires only ``--hpc-system-config`` as a
    real config override. ``dry_run`` and ``yes`` are controls and no overrides, so they are
    never reported here. An argument is an OVERRIDE only when the descriptor already
    DECLARES a value for the argument and the CLI value DIFFERS. Supplying a value the descriptor
    omits fills a gap and is no override.
    """
    reports: list[OverrideReport] = []
    cluster = cli_args.get("cluster")
    cli_hpc = cli_args.get("hpc_system_config_yaml")
    if cli_hpc is not None and cluster is not None:
        declared = definition.hpc_system_config.get(str(cluster))
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

    The gate prints BOTH values. A person who is unable to see the value under override
    is unable to consent to the override.
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


def build_case_from_definition(
    definition: ExperimentConfig,
    definition_dir: str | Path,
    cluster: str,
    *,
    hpc_system_config_yaml: str | Path | None = None,
):
    """Construct the analysis for `definition` on `cluster`.

    Generalizes ``container_validation.py::build_case`` with NO hardcoded cluster map.
    ``definition.hpc_system_config[cluster]`` names the per-cluster ``hpc_system_config``
    file and no module-level dict. The resolved file is handed whole to
    ``Toolkit.from_configs``, and the function reads only the ``default_account``
    and ``container`` fields of the file, for the guards below.

    Carries the same fail-fast guards as that script, raising ``ConfigurationError``
    where the script raises ``ValueError``. A ``default_account`` that is unset or still a
    ``{your-...}`` placeholder raises. A missing or placeholder ``container.sif_root``
    raises when the experiment definition declares a container, while the script checks the field
    unconditionally. Config
    resolution is NON-MUTATING, because the step reads the deployment config and edits no tracked file.
    """
    from hhemt.config.loaders import load_hpc_system_config
    from hhemt.toolkit import Toolkit

    definition_dir = Path(definition_dir)
    cfg_path = resolve_hpc_system_config(
        cluster, override=hpc_system_config_yaml, definition=definition, definition_dir=definition_dir
    )
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
    if definition.container is not None:
        if cfg_hpc.container is None or "{your-" in (cfg_hpc.container.sif_root or ""):
            raise ConfigurationError(
                field="container.sif_root",
                message=(
                    f"{cfg_path}: container.sif_root is missing or still a placeholder, but the "
                    "definition declares a container. Set it to the directory `hhemt build-sifs` writes into."
                ),
                config_path=cfg_path,
            )

    # Expand ${VAR} placeholders (${DATA_DIR}/${SCRATCH_DIR}/${PACKAGE_DIR}/${HHEMT_*})
    # in the two definition-relative configs before from_configs — the loader does expanduser
    # but not expandvars, so an unexpanded placeholder fails the config-load path-existence
    # check (cfgBaseModel._check_paths_exist) with a ConfigurationError, before any preflight.
    # The hpc config path is passed through unmodified: resolve_hpc_system_config took it from
    # the --hpc-system-config override or from the descriptor's hpc_system_config[cluster]
    # entry and returned it absolute, and its contents are operator-filled real values, so no
    # ${VAR} expansion is applied to it.
    return Toolkit.from_configs(
        system_config=expand_config_vars(definition_dir / definition.system_config),
        analysis_config=expand_config_vars(definition_dir / definition.analysis_config),
        hpc_system_config=cfg_path,
    )


_PYPI_PROJECT = "hhemt"
"""The package-index project whose source archive ``toolkit_pin.sdist_sha256`` names."""

_SDIST_HASH_CHUNK_BYTES = 1024 * 1024
"""Streaming read size, the case-manifest stipulation's 1 MiB rule: an archive is never
loaded whole."""

_PYPI_JSON_URL = "https://pypi.org/pypi/{project}/{version}/json"


def _fetch_sdist_chunks(version: str, chunk_size: int = _SDIST_HASH_CHUNK_BYTES):
    """Yield the bytes of ``hhemt``'s source archive for ``version`` from the package index, in chunks.

    Reads the index's JSON record for the version, takes its ``sdist`` file URL and
    streams that file in ``chunk_size`` reads. Any failure to reach the index or to find
    an sdist raises ``OSError``, and the caller converts that, ``http.client.HTTPException`` and
    ``ValueError`` into one refusal that names the pin. A seam: tests
    replace this function, so the digest comparison runs with no network.
    """
    import json
    import urllib.request

    json_url = _PYPI_JSON_URL.format(project=_PYPI_PROJECT, version=version)
    with urllib.request.urlopen(json_url, timeout=30) as resp:  # noqa: S310 - fixed https URL
        record = json.load(resp)
    files = record.get("urls", []) if isinstance(record, dict) else []
    sdist_url = next(
        (f["url"] for f in files if isinstance(f, dict) and f.get("packagetype") == "sdist" and "url" in f),
        None,
    )
    if sdist_url is None:
        raise OSError(f"the package index lists no source archive for {_PYPI_PROJECT} {version}")
    with urllib.request.urlopen(sdist_url, timeout=60) as resp:  # noqa: S310 - URL from the index record
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            yield chunk


def _verify_toolkit_pin_sdist(toolkit_pin, *, definition_dir: Path | None = None) -> None:
    """Refuse an experiment whose ``toolkit_pin.sdist_sha256`` is not the index's digest for ``version``.

    No-op when ``sdist_sha256`` is None. Otherwise fetches the source archive of
    ``toolkit_pin.version`` from the package index, hashes it in 1 MiB chunks and
    compares: a mismatch raises ``ConfigurationError`` naming both digests; an index
    that cannot be reached raises ``ConfigurationError`` naming the pin, because
    warn-and-skip would hide a misconfigured experiment (``sif/plan.py`` refuses the
    same way). The check attests that the declared digest names the index's file for
    ``version``. It does not compare the running toolkit with that file.
    """
    import hashlib
    import http.client

    declared = toolkit_pin.sdist_sha256
    if declared is None:
        return
    where = f"toolkit_pin (version {toolkit_pin.version!r}, sdist_sha256 {declared!r})"
    try:
        digest = hashlib.sha256()
        for chunk in _fetch_sdist_chunks(toolkit_pin.version):
            digest.update(chunk)
    except (OSError, http.client.HTTPException, ValueError) as exc:
        # URLError and HTTPError are OSError; a dropped connection raises HTTPException
        # (IncompleteRead); an index that answers with a page raises ValueError (JSONDecodeError).
        raise ConfigurationError(
            field="toolkit_pin.sdist_sha256",
            message=(
                f"{where}: the package index could not be reached or lists no source archive "
                f"for this version ({type(exc).__name__}: {exc}). The pin declares a digest, so "
                f"the run is refused rather than unverified."
            ),
            config_path=definition_dir,
        ) from exc
    measured = digest.hexdigest()
    if measured.lower() != declared.strip().lower():
        raise ConfigurationError(
            field="toolkit_pin.sdist_sha256",
            message=(
                f"{where}: the declared digest {declared!r} does not match the package index's "
                f"source archive for version {toolkit_pin.version!r}, whose sha256 is {measured!r}."
            ),
            config_path=definition_dir,
        )


def run_experiment(
    definition_dir: str | Path,
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

    mode : a ``RunMode`` member. ``resume``, the default, picks up where the last invocation left
    off, and ``fresh`` wipes the analysis_dir first. The default matches the default of ``Toolkit.run``,
    so both layers state a single value. The parameter is a typed pass-through.
    The CLI refuses an unknown value at parse time and ``Toolkit.run`` refuses an unknown value at the
    reduction site, so no third check lives here.

    override_force_rerun : per-invocation override of the ``force_rerun`` of the analysis config of the
    experiment definition,
    taking ``"all"``, ``"none"``, or a subject dict, and threaded untouched into
    ``Toolkit.run``. None, the default, reads the config field. The flag is the honest route to
    re-running completed scenarios without a wipe. The flag is a knob and no mode.

    debug : the run's debug flag, threaded untouched into ``Toolkit.run``. The flag is named here
    and left out of ``**cli_overrides``, because the flag is a run argument and no config override, so
    ``resolve_overrides`` would compare the flag against the configs of the experiment definition. Today the
    flag has one effect. A coupled clean start asks the solver to keep the snapshot files of the previous run.

    The override gate prints the side-by-side table and requires explicit confirmation when
    `resolve_overrides` returns a non-empty list. A non-TTY
    invocation without `assume_yes` REFUSES and avoids silently preferring either
    source. Silently preferring the CLI is how a bare `run_mode=gpu` ran a CUDA-only
    build in CPU mode.
    """
    definition_dir = Path(definition_dir)
    definition = load_experiment_definition(definition_dir)
    # The pin's digest is checked before anything is built: a pin that names the wrong
    # archive refuses the run at once rather than after a compile.
    _verify_toolkit_pin_sdist(definition.toolkit_pin, definition_dir=definition_dir)

    cli_args: dict[str, object] = {
        "cluster": cluster,
        "hpc_system_config_yaml": hpc_system_config_yaml,
        **cli_overrides,
    }
    reports = resolve_overrides(definition, cli_args)
    if reports:
        _confirm_override_gate(reports, assume_yes=assume_yes)

    tk = build_case_from_definition(definition, definition_dir, cluster, hpc_system_config_yaml=hpc_system_config_yaml)
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
