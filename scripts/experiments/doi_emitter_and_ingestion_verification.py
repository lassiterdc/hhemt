"""DOI emit + cross-hardware ingestion-verification driver ([Q8] proof vehicle, ADR-19).

Proves the USER IMPERATIVE ([Q16]): a cross-hardware sensitivity (CPU + a6000 + a100 in ONE
experiment) bundles as ONE artifact, deposits to a DOI host, ingests via ``from_doi``, and
runs each row against its OWN arch's SIF (multi-SIF, Option A / ADR-19 amended). It is the
CONTAINER + cross-hardware SUPERSET of ``container_validation.py`` (native, single-partition)
and the Phase-3 native ``test_doi_roundtrip_e2e.py`` (single-SIF-less round-trip).

FOUR entry points, split so the SECURE cross-machine flow keeps the Zenodo token on the
LOCAL machine and Rivanna only computes (token-free):

  build_case(...)       -> the durable, git-controlled analysis definition (container-mode +
                           swmm-enabled + the 8-row cross-hardware matrix). Mirrors
                           ``container_validation.build_case``.
  emit_bundle_only(...) -> RIVANNA, token-free: build + run (full matrix, SLURM) + render,
                           then ``emit_bundle(container_defs=[3 .defs])`` -> writes a
                           self-contained bundle ZIP and prints ``bundle_zip=<path>``. NO
                           deposit — emit needs a git checkout but NO token, so this stage
                           must never receive HHEMT_ZENODO_TOKEN. Run it under a git-backed
                           editable checkout (emit records the toolkit SHA + carries
                           ``git archive HEAD``); on the .git-less code sync, one-time
                           ``git init && git add -A && git commit`` first.
  deposit(zip, ...)     -> LOCAL only: read the SPDX license from the bundle's OWN crate and
                           publish the ZIP to Zenodo via ``_ZenodoTarget().publish`` (mints a
                           resolvable DOI; needs NO loaded analysis) -> prints ``data_doi=<doi>``.
                           The token lives here (``~/.config/hhemt/e2e.env`` -> HHEMT_ZENODO_TOKEN
                           / HHEMT_ZENODO_BASE_URL) and NEVER leaves; this module never reads a
                           credential value (``publishing._require_env`` reads it inside publish).
  verify(doi, ...)      -> RIVANNA, token-free: ``from_doi`` (builds 3 arch-keyed SIFs on
                           ingest) + ``test(execution_mode="slurm")`` + the two-layer PASS incl.
                           per-arch SIF routing. Needs only the non-secret HHEMT_ZENODO_BASE_URL
                           (pointed at the same host the deposit published to) — no credential.

The real per-cluster ``hpc_system_config`` (partition->hardware, account, apptainer module)
is resolved from the operator's PRIVATE deployment estate via ``_resolve_hpc_system_config``
($HHEMT_HPC_SYSTEM_CONFIG / $HHEMT_DEPLOYMENT_CONFIG / argv override) — NO git-tracked file is
edited per run, and the bundle carries zero user-specific info (ADR-9).

Run from the repo root so ``tests/`` is importable.
"""

from __future__ import annotations

import os
import sys as _sys
from pathlib import Path

# Initialize GDAL (rasterio/rioxarray) BEFORE the synthetic-model chain pulls in swmm.toolkit
# — importing the native swmm lib before GDAL inits corrupts GDAL's allocator (see
# container_validation.py). Must precede tests.fixtures.
import rioxarray  # noqa: F401  (import-order guard)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

# The 8-row cross-hardware matrix (2 a6000 GPU + 2 a100 GPU + 4 CPU), partition-as-axis,
# every row container-mode. ONE matrix -> ONE bundle -> 3 SIFs (Option A / ADR-19 amended).
_MATRIX = _REPO_ROOT / "tests/fixtures/doi_emitter_and_ingestion_verification/matrix_cross_hardware.csv"

# The 3 per-arch build recipes carried in the bundle. Each .def self-describes its arch via
# %labels org.hhemt.gpu_hardware (a100 / a6000 / none-for-CPU); from_doi builds one SIF each.
_CONTAINER_DEFS = [
    _REPO_ROOT / "containers/uva-cuda.def",        # a100  / AMPERE80
    _REPO_ROOT / "containers/uva-cuda-a6000.def",  # a6000 / AMPERE86
    _REPO_ROOT / "containers/uva-cpu.def",         # CPU   / no gpu_hardware label
]

# In-tree anonymized COPY-ME template — the operator reconstructs the real config in the
# estate from this (never the live config; the run edits ZERO git-tracked files).
_TEMPLATE = _REPO_ROOT / "test_data/norfolk_coastal_flooding/hpc_system_config_uva.yaml"

# The master ensemble partition is a GPU partition so the master participates in the
# GPU-target dedup (Gotcha 54); the per-row CSV overrides ensemble per row. setup/prepare/
# process/consolidate run on the CPU partition.
_MASTER_GPU_PARTITION = "gpu-a6000"
_CPU_PARTITION = "standard"

# PRODUCER SIF home — ABSOLUTE-referenced, DEDICATED, and OUTSIDE sif_cache_root() so the
# reproducer's from_doi (which builds content-addressed into sif_cache_root) can NEVER
# cache-hit these (FQ2 cache-isolation / no false-green build-on-ingest).
_PRODUCER_SIF_DIR = Path(
    os.environ.get(
        "HHEMT_Q8_PRODUCER_SIF_DIR",
        f"/scratch/{os.environ.get('USER', 'user')}/q8_producer_sifs",
    )
)


def _def_gpu_hardware(def_path: Path) -> str | None:
    """The .def's `org.hhemt.gpu_hardware` label value ("a100"/"a6000"), or None for the
    CPU/no-arch carrier. Mirrors bundle/_emit.py::_parse_def_labels + the target_arch
    derivation (_emit.py:869) so the producer's sif_paths_by_arch keys ("a100"/"a6000")
    match the gpu_hardware namespace resolve_gpu_target[0] returns at the SIM rung."""
    in_labels = False
    for line in Path(def_path).read_text().splitlines():
        if line.startswith("%"):
            in_labels = line.strip().split()[0] == "%labels"
            continue
        if in_labels and line.strip().startswith("org.hhemt.gpu_hardware"):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return None


def _resolve_hpc_system_config(override: str | None = None) -> Path:
    """Resolve the operator's REAL UVA ``hpc_system_config`` from the PRIVATE deployment estate.

    Precedence: explicit ``override`` > ``$HHEMT_HPC_SYSTEM_CONFIG`` >
    ``$HHEMT_DEPLOYMENT_CONFIG/hpc/hpc_system_config_uva.yaml``. The in-tree ``_TEMPLATE`` is
    the copy-me source, never the live config (mirrors ``container_validation`` — the public
    repo bakes in no estate path and no account).
    """
    if override:
        path = Path(override).expanduser()
    elif os.environ.get("HHEMT_HPC_SYSTEM_CONFIG"):
        path = Path(os.environ["HHEMT_HPC_SYSTEM_CONFIG"]).expanduser()
    else:
        estate = os.environ.get("HHEMT_DEPLOYMENT_CONFIG")
        if not estate:
            raise FileNotFoundError(
                "No hpc_system_config source: set $HHEMT_DEPLOYMENT_CONFIG to your "
                "compute-visible private deployment-config checkout (or set "
                "$HHEMT_HPC_SYSTEM_CONFIG, or pass the path as argv[1])."
            )
        path = Path(estate).expanduser() / "hpc" / "hpc_system_config_uva.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"No UVA hpc_system_config at {path}.\n"
            f"  Reconstruct it once from the in-tree template:\n"
            f"    cp {_TEMPLATE} {path}\n"
            f"  then fill default_account (your UVA allocation) and container.apptainer_module.\n"
            f"  Its partition keys must be REAL Rivanna partitions: gpu-a6000 (a6000), gpu "
            f"(a100 via --gres=gpu:a100:N), standard (CPU)."
        )
    return path.resolve()


def build_case(
    *,
    hpc_system_config_yaml: str | None = None,
    system_directory: str | None = None,
    start_from_scratch: bool = False,
):
    """Build the cross-hardware, container-mode, swmm-enabled sensitivity test case.

    The bundle's ``analysis_config`` carries ``multi_sim_run_method="local"`` (P1 — the
    dispatch-FAMILY label; under runtime ``execution_mode="slurm"`` it routes to the per-rule
    SLURM executor, ZERO login-node sims) and ``execution_environment="container"`` (so
    ``emit_bundle`` takes the container branch and carries the ``.def``s); the ``system_config``
    carries ``toggle_use_swmm_for_hydrology=True`` (P3). ``hpc_account`` is sourced from the
    resolved estate config's ``default_account`` (never a code default — [Q10]).
    """
    from hhemt.config.loaders import load_hpc_system_config
    from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case

    cfg_path = _resolve_hpc_system_config(hpc_system_config_yaml)
    cfg_hpc = load_hpc_system_config(cfg_path)
    account = cfg_hpc.default_account or ""
    if (not account) or ("{your-" in account):
        raise ValueError(
            f"{cfg_path}: default_account is unset or still a placeholder ({account!r}). "
            f"Set it to your real UVA allocation."
        )
    return retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="doi_emitter_and_ingestion_verification",
        sensitivity_csv=_MATRIX,
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        toggle_use_swmm_for_hydrology=True,  # P3 — coupled SWMM hydrology exercised in-SIF
        start_from_scratch=start_from_scratch,
        hpc_system_config_yaml=cfg_path,
        additional_system_configs=(
            {"system_directory": system_directory} if system_directory else {}
        ),
        additional_analysis_configs={
            "multi_sim_run_method": "local",  # P1 — dispatch-family; execution_mode pins locus
            "execution_environment": "container",  # base container-mode (CSV reaffirms per row)
            "hpc_account": account,
            "hpc_max_simultaneous_sims": 1000,
            "hpc_total_job_duration_min": 60,
            "hpc_gpus_per_node": 8,
            "hpc_time_min_per_sim": 30,
            "hpc_restart_times_simulate": 2,
            "hpc_restart_times_other": 2,
            "hpc_ensemble_partition": _MASTER_GPU_PARTITION,
            "hpc_setup_and_analysis_processing_partition": _CPU_PARTITION,
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": str(_MATRIX),
            "report": {
                "reporting_set": "benchmarking",
                "sensitivity": {
                    "independent_vars": ["n_devices"],
                    "dependent_var": "performance.Total",
                    "aggregation": "mean",
                    # partition-as-axis: group the cross-hardware rows by their partition so
                    # the report reads as a per-hardware comparison. Use the canonical
                    # dataframe column name `hpc.partition` (per _resolve_row_ensemble_partition,
                    # sensitivity_analysis.py) — the bare analysis-config field name
                    # `hpc_ensemble_partition` is NOT a resolvable benchmarking axis column.
                    "group_by_var": "hpc.partition",
                },
            },
        },
    )


def provision_producer_sifs(
    *,
    hpc_system_config_yaml: str | None = None,
) -> Path:
    """PRODUCER-side per-arch SIF provisioning (ADR-19). Batch-build all 3 bundle .defs into
    the dedicated _PRODUCER_SIF_DIR, then write a DERIVED hpc_system_config whose
    container.sif_paths_by_arch maps each GPU arch ("a100"/"a6000") to its SIF and
    container.sif_path names the CPU SIF. Returns the derived config path.

    Mirror of experiments._repoint_sif_paths (the reproducer analog). REQUIRED because the
    SIM rung (run_simulation.py:417-418) resolves each row's arch via
    resolve_gpu_target(cfg, partition)[0] and looks the SIF up in sif_paths_by_arch, falling
    back to sif_path for CPU rows — so without this every row would resolve to the single
    sif_path (an a100 SIF), running an AMPERE80 binary on a6000 devices and a GPU SIF on CPU
    nodes. All 3 are rebuilt FRESH from the bundle's OWN .defs (never the pre-existing UCX
    SIF, FQ3) so the producer validates the exact recipes the reproducer builds. The SIFs
    live in _PRODUCER_SIF_DIR (absolute), NEVER in sif_cache_root(), so from_doi cannot
    cache-hit them (FQ2).
    """
    from hhemt.container_build import build_sif
    from hhemt.utils import read_yaml, write_yaml

    cfg_path = _resolve_hpc_system_config(hpc_system_config_yaml)
    _hpc = read_yaml(cfg_path) or {}
    account = _hpc.get("default_account")
    apptainer_module = (_hpc.get("container") or {}).get("apptainer_module")
    if (not account) or ("{your-" in str(account)):
        raise ValueError(
            f"{cfg_path}: default_account is unset or a placeholder ({account!r}); the SIF "
            f"build sbatch needs a real UVA allocation."
        )
    _PRODUCER_SIF_DIR.mkdir(parents=True, exist_ok=True)

    # PARITY FIX (container-specialist q8b, 2026-07-17): build from the SAME clean git-archive
    # staging + %files rewrite the reproducer's from_doi uses, NOT the live 2.0 GB worktree.
    # `%files ../` over the live worktree dereferences (apptainer `cp -fLr`) and FATALs on the
    # first dangling symlink among the untracked build artifacts (test_data/norfolk_coastal_
    # flooding/triton/build*/input, tests/*/sims/*/build*; job 17069925), and also drags in
    # .venv/.git/caches. `git archive HEAD` is exactly the tracked set (0 tracked symlinks,
    # .venv gitignored), so this eliminates the whole class AND makes the producer validate the
    # EXACT build context the reproducer builds -- this function's stated contract. Reuses the
    # emit path's own tested helpers (bundle/_emit.py).
    from hhemt.bundle._emit import (
        SOURCE_TREE_RELPATH,
        _carry_source_tree,
        _rewrite_files_section,
    )

    _build_ctx = _PRODUCER_SIF_DIR / "_build_ctx"
    _build_ctx.mkdir(parents=True, exist_ok=True)
    _carry_source_tree(_build_ctx)  # git archive HEAD -> {_build_ctx}/hhemt_src (clean tree)

    sif_paths_by_arch: dict[str, str] = {}
    cpu_sif: Path | None = None
    for container_def in _CONTAINER_DEFS:
        # Land the rewritten .def beside the staged source so `cd {def.parent}` + the rewritten
        # `%files hhemt_src` resolve to the clean tree (mirrors _emit_container_build).
        staged_def = _build_ctx / container_def.name
        staged_def.write_text(
            _rewrite_files_section(container_def.read_text(), SOURCE_TREE_RELPATH)
        )
        built = build_sif(
            def_path=staged_def,
            sif_out=_PRODUCER_SIF_DIR / f"{container_def.stem}.sif",
            account=account,
            apptainer_module=apptainer_module,
            mode="batch",  # every bundle .def compiles in %post -> CPU-batch, never login-node
        )
        arch = _def_gpu_hardware(container_def)  # labels unchanged by the %files rewrite
        if arch:
            sif_paths_by_arch[arch] = str(built.resolve())
        else:
            cpu_sif = built.resolve()

    # OE-1: the producer has NO arch-coverage preflight (unlike from_doi's
    # _assert_container_arch_set_covers_matrix). Assert coverage HERE so a missing arch is a
    # loud failure, not a silent SIM-rung fall-through to the wrong-arch sif_path.
    import csv as _csv

    with open(_MATRIX) as _f:
        _parts = {r["hpc.partition"] for r in _csv.DictReader(_f)}
    _gpu_arch_by_partition = {"gpu": "a100", "gpu-a6000": "a6000"}  # standard -> CPU (sif_path)
    _need = {_gpu_arch_by_partition[p] for p in _parts if p in _gpu_arch_by_partition}
    _missing = _need - set(sif_paths_by_arch)
    if _missing:
        raise RuntimeError(
            f"producer SIF provisioning incomplete: matrix needs GPU arch(es) {sorted(_missing)} "
            f"but only built {sorted(sif_paths_by_arch)}. A missing arch would silently run the "
            f"wrong-arch sif_path at the SIM rung."
        )
    if cpu_sif is None:
        raise RuntimeError(
            "no CPU/no-arch .def among _CONTAINER_DEFS -> cannot set container.sif_path for the "
            "CPU rows / arch-agnostic process rung."
        )

    container = dict(_hpc.get("container") or {})
    container["sif_path"] = str(cpu_sif)                 # CPU rows + arch-agnostic process rung
    container["sif_paths_by_arch"] = sif_paths_by_arch   # {a100: ..., a6000: ...}
    _hpc["container"] = container
    derived = _PRODUCER_SIF_DIR / "hpc_system_config.producer.yaml"
    write_yaml(_hpc, derived)
    print(
        f"[provision] built {len(_CONTAINER_DEFS)} producer SIFs in {_PRODUCER_SIF_DIR}\n"
        f"[provision]   sif_paths_by_arch -> {sorted(sif_paths_by_arch)}; "
        f"sif_path -> {cpu_sif.name}\n"
        f"[provision]   derived config: {derived} (estate config unmodified)",
        flush=True,
    )
    return derived


def emit_bundle_only(
    *,
    hpc_system_config_yaml: str | None = None,
    system_directory: str | None = None,
    exclude_config: str | None = None,
) -> str:
    """RIVANNA / producer side, TOKEN-FREE: build + run (full matrix, SLURM) + render, then
    EMIT the self-contained bundle ZIP carrying the 3 per-arch ``.def``s. Returns (and prints
    ``bundle_zip=``) the ZIP path. NO deposit happens here.

    ``emit_bundle`` harvests ``render_report`` manifest sidecars, so the producer must RUN +
    render before bundling. Emit records the toolkit SHA and carries ``git archive HEAD``, so
    this MUST run under a git-backed editable checkout (on the .git-less code sync do a one-time
    ``git init && git add -A && git commit`` first). Emit performs ZERO network I/O and needs NO
    credential — the Zenodo token must never be present on this machine. The deposit is a
    SEPARATE, LOCAL-only stage (``deposit`` below) so the token stays local (totally secure).
    """
    from hhemt.bundle import emit_bundle

    # FQ1/FQ3 (R9): build the 3 per-arch SIFs (fresh, from the bundle's own .defs) and get a
    # DERIVED producer config carrying sif_paths_by_arch + the CPU sif_path. MUST precede
    # run() (OE-3: emit_bundle harvests a fully-green run's render sidecars). The estate
    # config is never edited; the derived config is what the run consumes.
    producer_cfg = provision_producer_sifs(hpc_system_config_yaml=hpc_system_config_yaml)

    tc = build_case(
        hpc_system_config_yaml=str(producer_cfg),
        system_directory=system_directory,
        start_from_scratch=True,
    )
    tc.analysis.run(from_scratch=True, execution_mode="slurm", wait_for_job_completion=True, verbose=True)
    tc.analysis.render_report()
    # [Q8] REQ-1 (R9b): generate the TRUNCATED test()-shaped reference the reproducer's own
    # analysis.test() will reproduce. Same n_reporting_timesteps => apples-to-apples exact
    # compare (a full-matrix reference is NOT comparable: test() truncates 13 of 181 weather
    # frames and the tracked vars are peaks). emit_bundle._copy_reference_outputs then carries
    # {analysis_dir}/_test/group_*/sims/*/processed/*_summary.* into reference_outputs/.
    tc.analysis.test(execution_mode="slurm", wait_for_job_completion=True, verbose=True)
    bundle_zip = emit_bundle(
        tc.analysis,
        exclude_config=Path(exclude_config).expanduser() if exclude_config else None,
        container_defs=list(_CONTAINER_DEFS),
    )
    print(f"bundle_zip={bundle_zip}")
    return str(bundle_zip)


def deposit(bundle_zip: str) -> str:
    """LOCAL side, TOKEN-BEARING: publish the emitted bundle ZIP to Zenodo, minting a
    resolvable DOI. Runs on the LOCAL machine ONLY — HHEMT_ZENODO_TOKEN lives here and never
    leaves. Returns (and prints ``data_doi=``) the minted DOI.

    Uses the low-level ``_ZenodoTarget().publish`` (NOT ``publish_analysis`` / the
    ``publish_reprex_bundle`` facade) because those need a LOADED analysis object to read the
    license sidecar from the analysis_dir, which does not exist locally for a Rivanna-run
    analysis. The license is read from the bundle's OWN crate instead. This function never reads
    the token value — ``publishing._require_env`` reads it inside ``publish``. Source the env
    file first: ``set -a; . ~/.config/hhemt/e2e.env; set +a``.
    """
    import tempfile
    import zipfile

    from hhemt.publishing import _read_license_from_sidecar, _ZenodoTarget

    zip_path = Path(bundle_zip).expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"bundle zip not found: {zip_path}")
    with tempfile.TemporaryDirectory(prefix="q8_deposit_") as _tmp:
        tmp = Path(_tmp)
        with zipfile.ZipFile(zip_path) as zf:
            crate_members = [n for n in zf.namelist() if n.endswith("ro-crate-metadata.json")]
            if not crate_members:
                raise RuntimeError(f"{zip_path} carries no ro-crate-metadata.json")
            crate_member = min(crate_members, key=lambda n: n.count("/"))  # shallowest = bundle-root crate
            zf.extract(crate_member, tmp)
        crate_dir = (tmp / crate_member).parent
        license_spdx = _read_license_from_sidecar(crate_dir)
        result = _ZenodoTarget().publish(
            deposit=[zip_path],
            license_spdx=license_spdx,
            software_doi=None,
            analysis_dir=crate_dir,  # title read + staging only; token comes from env inside publish
            creators=None,
        )
    doi = result.get("data_doi")
    if not doi:
        raise RuntimeError(f"deposit minted no DOI; publish returned {result}")
    print(f"data_doi={doi}")
    return doi


def verify(
    doi: str,
    *,
    hpc_system_config_yaml: str | None = None,
    target_dir: str | None = None,
    software_dir: str | None = None,
) -> bool:
    """Reproducer side: ingest by DOI (from_doi builds 3 arch-keyed SIFs), run under the
    per-rule SLURM executor, and adjudicate the two-layer PASS + per-arch SIF routing.

    Returns True iff every per-rule sim COMPLETEd AND each sim ran against its OWN arch's SIF
    (a6000 rows -> the AMPERE86 SIF, a100 rows -> the AMPERE80 SIF, CPU rows -> the CPU SIF).
    ``target_dir``/``software_dir`` MUST live OUTSIDE any bundle_root (from_doi rmtree's
    bundle_root on ingest).
    """
    from hhemt.container_build import sif_cache_root
    from hhemt.experiments import TRITON_SWMM_experiment

    cfg_path = _resolve_hpc_system_config(hpc_system_config_yaml)

    # FQ2 false-green guard (R9): from_doi must GENUINELY BUILD all 3 SIFs on ingest, not
    # cache-hit the producer's. The reproducer cache MUST be a fresh dir (set HHEMT_SIF_CACHE_DIR
    # before this stage; the runbook points it at an empty q8_reproducer_sif_cache). Assert it
    # starts empty of built SIFs, so a hit here is a FAIL rather than a silent false-green.
    _cache = sif_cache_root()
    _pre = sorted(_cache.glob("hhemt-*.sif")) if _cache.is_dir() else []
    if _pre:
        raise RuntimeError(
            f"FALSE-GREEN GUARD: reproducer SIF cache {_cache} already holds {len(_pre)} built "
            f"SIF(s) {[p.name for p in _pre]} pre-ingest. Point HHEMT_SIF_CACHE_DIR at a FRESH "
            f"dir so from_doi builds on ingest instead of cache-hitting."
        )

    # defect-8: from_doi extracts bundle_root UNDER target_dir, and a container-mode ingest
    # on a SLURM cluster submits a `sbatch --wait` SIF build that `cd`s into bundle_root from a
    # COMPUTE node. Rivanna /tmp is node-local, so a login-node mkdtemp default is invisible to
    # the builder. Resolve a SHARED-scratch target_dir here (env override > /scratch default) so
    # the `verify <DOI>` CLI path is safe without an explicit --target-dir. software_dir rides
    # for free: from_doi defaults it to {bundle_root}/software, under this shared target_dir.
    if target_dir is None:
        target_dir = os.environ.get("HHEMT_Q8_INGEST_DIR") or (
            f"/scratch/{os.environ.get('USER', 'user')}/q8_doi/q8_ingest"
        )
    exp = TRITON_SWMM_experiment.from_doi(
        doi=doi,
        host="zenodo",
        hpc_system_config_yaml=cfg_path,
        target_dir=Path(target_dir).expanduser() if target_dir else None,
        software_dir=Path(software_dir).expanduser() if software_dir else None,
    )

    # FQ2 (R9): prove the build-on-ingest actually happened (3 fresh content-addressed SIFs).
    _post = sorted(_cache.glob("hhemt-*.sif"))
    print(f"[verify] build-on-ingest: {len(_post)} SIF(s) freshly built in {_cache}: "
          f"{[p.name for p in _post]}", flush=True)
    if len(_post) < 3:
        raise RuntimeError(
            f"FALSE-GREEN GUARD: expected >=3 freshly-built SIFs in {_cache} after from_doi, "
            f"found {len(_post)}. Inspect the ingest log for 3 `[build-sif] cache MISS ... "
            f"-> building` lines and 3 COMPLETED `hhemt_sif_build` jobs in sacct."
        )
    result = exp.analysis.test(execution_mode="slurm", wait_for_job_completion=True, verbose=True)
    ok = _adjudicate_per_arch_pass(exp, result)
    print("OVERALL:", "cross-hardware per-arch routing PASS" if ok else "FAIL")
    return ok


def _read_combined_by_mode(analysis_obj, sim_dir: Path) -> dict:
    """Read every present mode's combined summary Dataset from ``sim_dir``, redirecting the
    analysis's simulation_directory (AnalysisPaths is a plain @dataclass, paths.py:51; the
    per-scenario processed/ path derives solely from simulation_directory, scenario.py:58/62).
    Scoped save/restore + _eda_mode_cache clear so neither the reference read nor the
    reproducer read poisons the other. Returns {mode: xr.Dataset}."""
    orig = analysis_obj.analysis_paths.simulation_directory
    analysis_obj.analysis_paths.simulation_directory = Path(sim_dir)
    analysis_obj.__dict__.pop("_eda_mode_cache", None)
    try:
        out = {}
        for mode in analysis_obj.process._MODE_CONFIG:
            try:
                out[mode] = analysis_obj.process._retrieve_combined_output(mode)
            except (FileNotFoundError, ValueError):
                continue
        return out
    finally:
        analysis_obj.analysis_paths.simulation_directory = orig
        analysis_obj.__dict__.pop("_eda_mode_cache", None)


def _compare_group_against_reference(sub_analysis, ref_sim_dir: Path) -> tuple[list, int, int]:
    """REQ-1: per (mode, tracked-var, event) EXACT equality of the reproducer's fresh
    summaries vs the carried producer reference for the SAME group (== same arch/compute
    config). Reuses cross_sim_identity.compare_variable_exact + TRACKED_VARS. Perf modes
    carry no TRACKED_VARS, so timing (expected to differ) is skipped automatically. Returns
    a list of human-readable mismatch strings (empty == bit-identical PASS)."""
    from hhemt.eda.cross_sim_identity import TRACKED_VARS, compare_variable_exact

    ref = _read_combined_by_mode(sub_analysis, ref_sim_dir)
    if not ref:
        # NOTE: returns the full 3-tuple — the caller unpacks (problems, n_cmp, n_signal).
        # A bare-list early return here would raise ValueError at the call site and turn a
        # clean "reference missing" diagnostic into a crash.
        return ([f"reference_outputs summaries ABSENT under {ref_sim_dir} — nothing to compare "
                 f"(did the producer run analysis.test() before emit so _copy_reference_outputs "
                 f"could carry them?)"], 0, 0)
    repro = _read_combined_by_mode(sub_analysis, sub_analysis.analysis_paths.simulation_directory)
    import numpy as np

    problems: list = []
    # defect (2026-07-21): a PASS that compared ZERO variables is textually identical
    # to a real PASS. That is how a half-blind TRACKED_VARS certified a [Q8] DoD --
    # two of its four names were cf_conventions attribute keys emitted nowhere, so
    # conduit capacity was never compared. Count the comparisons that carried SIGNAL
    # (at least one finite value on BOTH sides) and fail closed at zero.
    n_cmp = 0
    n_signal = 0
    for mode, ds_ref in ref.items():
        ds_cmp = repro.get(mode)
        if ds_cmp is None:
            problems.append(f"mode {mode}: reproducer produced no summary")
            continue
        for var in TRACKED_VARS:
            if var not in ds_ref.data_vars or var not in ds_cmp.data_vars:
                continue
            for e in ds_ref["event_iloc"].values:
                a = ds_ref[var].sel(event_iloc=e)
                b = ds_cmp[var].sel(event_iloc=e)
                res = compare_variable_exact(a, b)
                n_cmp += 1
                if bool(np.isfinite(a.values).any()) and bool(np.isfinite(b.values).any()):
                    n_signal += 1
                if not res["identical"]:
                    problems.append(
                        f"mode {mode} var {var} event_iloc {int(e)}: NOT bit-identical "
                        f"(max_abs_diff={res['max_abs_diff']:.6g}, dtype_match={res['dtype_match']}, "
                        f"coord_match={res['coord_match']})"
                    )
    if n_signal == 0:
        problems.append(
            f"VACUOUS: {n_cmp} comparison(s) performed, {n_signal} carried signal — no tracked "
            f"variable with finite data was present in BOTH the reference and the reproducer. "
            f"TRACKED_VARS={list(TRACKED_VARS)}; ref vars="
            f"{sorted({v for d in ref.values() for v in d.data_vars})}. This is NOT a reproduction "
            f"PASS."
        )
    return problems, n_cmp, n_signal


def _assert_sif_builds(expected: int = 3, since: str = "now-1day") -> bool:
    """REQ-2: N=3 hhemt_sif_build jobs COMPLETED (container_build.py:197 names the sbatch job).
    COLD-CACHE assertion: get_or_build_sif is content-addressed (container_build.py:101-107),
    so a WARM sif cache skips builds and this reports fewer than `expected` — run [Q8] with a
    fresh sif_cache_root. Returns True iff >= expected COMPLETED and 0 FAILED (or sacct absent)."""
    import subprocess

    try:
        out = subprocess.run(
            ["sacct", "--name", "hhemt_sif_build", "--starttime", since,
             "--noheader", "-P", "-o", "JobID,State"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception as exc:  # noqa: BLE001 — sacct unavailable is non-fatal for REQ-2
        print(f"  REQ-2: sacct unavailable ({exc}); skipping the {expected}-build assertion.")
        return True
    states = [ln.split("|")[-1].strip() for ln in out.splitlines()
              if ln.strip() and not ln.split("|")[0].endswith((".batch", ".extern"))]
    completed = sum(1 for s in states if s.startswith("COMPLETED"))
    failed = sum(1 for s in states if s.startswith(("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL")))
    print(f"  REQ-2: hhemt_sif_build -> {completed} COMPLETED, {failed} failed "
          f"since {since} (expect >= {expected} COMPLETED on a COLD cache; a warm "
          f"sif_cache_root skips builds). The window matters: without --starttime, a PRIOR "
          f"cycle's builds satisfy this run's threshold.")
    return failed == 0 and completed >= expected


def _adjudicate_per_arch_pass(exp, result) -> bool:
    """Two-layer PASS + per-arch SIF routing + REQ-1 per-arch reproduction + REQ-2 SIF builds.
    Each _test group must have completed, run on its arch-matched SIF, and its fresh summaries
    must be bit-identical to the carried producer reference for the SAME group. A cross-group
    (== cross-arch) comparison is NEVER made — CPU/a100/a6000 divergence is EXPECTED."""
    import tests.utils_for_testing as tst_ut

    subs = getattr(result, "subanalyses", None) or []
    if not subs:
        print("NO-DATA — analysis.test() produced no _test sub-analyses (run failed upstream)")
        return False
    all_ok = True
    for sub in subs:
        try:
            tst_ut.assert_analysis_workflow_completed_successfully(sub.analysis)
        except AssertionError as e:
            print(f"  sub {getattr(sub, 'name', '?')}: workflow INCOMPLETE — {e}")
            all_ok = False
    # Layer-2 per-arch routing is asserted from the per-rule sim logs (apptainer exec line)
    # by the [Q8] operator runbook's log-scan (namespace-agnostic: it names the SIF file,
    # hhemt-a6000-*.sif vs hhemt-a100-*.sif vs the CPU SIF). Reported here as the gate.
    print("  (per-arch SIF routing: adjudicate each sim log's `apptainer exec {sif}` Command "
          "line against its row's arch SIF per the [Q8] runbook two-layer PASS)")

    # REQ-1: per-arch within-family producer-vs-reproducer EXACT equality of the flat summaries.
    bundle_root = getattr(exp, "bundle_root", None) or exp.analysis.analysis_paths.analysis_dir
    ref_root = Path(bundle_root) / "reference_outputs"
    if not ref_root.is_dir():
        print(f"  REQ-1 FAIL: no reference_outputs/ under {bundle_root} — cannot verify per-arch "
              f"reproduction (the producer must run analysis.test() before emit so the truncated "
              f"reference is carried).")
        all_ok = False
    else:
        for sub in subs:
            group_name = sub.analysis.analysis_paths.analysis_dir.name  # "group_0", "group_1", ...
            ref_sim_dir = ref_root / group_name / "sims"
            problems, n_cmp, n_signal = _compare_group_against_reference(sub.analysis, ref_sim_dir)
            if problems:
                all_ok = False
                print(f"  REQ-1 {group_name}: per-arch within-family reproduction FAIL "
                      f"({n_cmp} comparisons, {n_signal} with signal):")
                for p in problems:
                    print(f"      - {p}")
            else:
                print(f"  REQ-1 {group_name}: per-arch within-family bit-identical PASS "
                      f"({n_cmp} comparisons, {n_signal} with signal)")

    # REQ-2: the 3 per-arch SIFs were built on ingest (cold-cache).
    if not _assert_sif_builds(expected=len(_CONTAINER_DEFS)):
        all_ok = False
    return all_ok


if __name__ == "__main__":
    # Secure cross-machine [Q8] flow (token stays LOCAL; Rivanna computes token-free):
    #   RIVANNA (producer, token-free) — emit provisions 3 fresh per-arch SIFs, runs the full
    #     matrix + a producer test() (the truncated REQ-1 reference), then bundles:
    #       export HHEMT_DEPLOYMENT_CONFIG=/scratch/$USER/hhemt_experiments/hhemt_projects
    #       export OPENBLAS_NUM_THREADS=1            # login-node thread-alloc guard
    #       # (optional) export HHEMT_Q8_PRODUCER_SIF_DIR=/scratch/$USER/q8_producer_sifs
    #       python -m scripts.experiments.doi_emitter_and_ingestion_verification emit-bundle
    #             -> provision_producer_sifs submits 3 blocking hhemt_sif_build sbatch jobs,
    #                writes hpc_system_config.producer.yaml, runs the 8-row matrix + test(),
    #                prints bundle_zip=<rivanna path>
    #   LOCAL (token-bearing):
    #       set -a; . ~/.config/hhemt/e2e.env; set +a
    #       python -m scripts.experiments.doi_emitter_and_ingestion_verification deposit <zip>
    #             -> prints data_doi=<doi>
    #   RIVANNA (reproducer, token-free) — from_doi GENUINELY builds 3 SIFs on ingest; a FRESH,
    #     EMPTY reproducer cache (distinct from the producer SIF dir) forecloses a cache-hit:
    #       export HHEMT_DEPLOYMENT_CONFIG=/scratch/$USER/hhemt_experiments/hhemt_projects
    #       export OPENBLAS_NUM_THREADS=1
    #       export HHEMT_SIF_CACHE_DIR=/scratch/$USER/q8_reproducer_sif_cache; rm -rf "$HHEMT_SIF_CACHE_DIR"
    #       export HHEMT_ZENODO_BASE_URL=<same host the deposit published to>   # non-secret
    #       python -m scripts.experiments.doi_emitter_and_ingestion_verification verify <DOI>
    stage = _sys.argv[1] if len(_sys.argv) > 1 else "emit-bundle"
    if stage == "emit-bundle":
        emit_bundle_only()
    elif stage == "deposit":
        if len(_sys.argv) < 3:
            raise SystemExit("deposit needs a bundle zip path: ... deposit <bundle_zip>")
        deposit(_sys.argv[2])
    elif stage == "verify":
        if len(_sys.argv) < 3:
            raise SystemExit("verify needs a DOI: ... verify <DOI>")
        _sys.exit(0 if verify(_sys.argv[2]) else 1)
    else:
        raise SystemExit(
            f"unknown stage {stage!r} (expected 'emit-bundle', 'deposit', or 'verify')"
        )
