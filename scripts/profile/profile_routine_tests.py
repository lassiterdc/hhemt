"""Durable, auto-discovering profiler for the TRITON-SWMM_toolkit routine test corpus.

Usage:
    python -m scripts.profile.profile_routine_tests

Refreshes $AGENTIC_WORKSPACE/library/knowledge/triton-swmm-toolkit/routine test profile results.md
from a clean checkout with zero required arguments. The output doc lives in the
agentic-workspace library so it is reachable from the Obsidian vault and indexed
by `knowledge MOC.md` Dataview queries; placing it under a project repo's tree
would be invisible to both. See --help for flags.

Architecture: orchestrator + pytest plugin + Snakemake harvest module + Markdown
emitter. The orchestrator spawns pytest subprocesses with an isolated env (per-pid
tmp root, six cache knobs), collects per-pass JSON artifacts, harvests Snakemake
metadata strictly post-hoc, and renders a deterministic Markdown doc.

Per-pass JSON convention (asv-borrowed pattern): each pass writes its own JSON
file under {tmp_root}/{rep_dir}/. The CLI surface is the contract; the JSON
schemas may evolve.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_default_output() -> Path:
    """Resolve DEFAULT_OUTPUT lazily so unit tests can import this module without AGENTIC_WORKSPACE set."""
    aw = os.environ.get("AGENTIC_WORKSPACE")
    if not aw:
        raise RuntimeError(
            "profile_routine_tests: $AGENTIC_WORKSPACE is not set. The profile-results "
            "doc must land in the agentic-workspace library to be reachable from the "
            "Obsidian vault and indexed by `knowledge MOC.md`. Export AGENTIC_WORKSPACE "
            "to your agentic-workspace repo root and retry, or pass --output explicitly."
        )
    return Path(aw) / "library/knowledge/triton-swmm-toolkit/routine test profile results.md"
DEFAULT_TOP_N = 20
DEFAULT_FINDINGS_TOP_K = 10
DEFAULT_REPETITIONS = 1
PLUGIN_ENV_VAR = "PROFILE_PLUGIN_JSON"


@dataclass
class CorpusEntry:
    nodeid: str
    status: str  # "collected" | "collection-failed"
    error: str | None = None


@dataclass
class PerTestRecord:
    nodeid: str
    setup_s: float = 0.0
    call_s: float = 0.0
    teardown_s: float = 0.0
    total_s: float = 0.0
    outcome: str = ""
    child_cpu_seconds: float = 0.0


@dataclass
class FixtureRecord:
    name: str
    scope: str
    duration_s: float
    consumer_nodeid: str


@dataclass
class SnakemakeRuleRecord:
    rule: str
    rule_normalized: str
    job_count: int
    total_s: float
    mean_s: float
    min_s: float
    max_s: float
    zero_duration_job_count: int
    test_origin: str


@dataclass
class SnakemakeDiagnosticsRecord:
    snakefiles_found: int = 0
    snakefiles_with_metadata: int = 0
    snakefiles_dry_run_only: int = 0
    snakefiles_zero_records: int = 0
    total_records: int = 0
    parser_warnings: list[str] = field(default_factory=list)


@dataclass
class RunArtifacts:
    """All intermediate data from a single repetition."""
    corpus: list[CorpusEntry]
    import_times: dict[str, float]
    conftest_import_times: dict[str, float]
    collection_total_s: float
    per_test: dict[str, PerTestRecord]
    fixtures: list[FixtureRecord]
    snakemake_rules: list[SnakemakeRuleRecord]
    snakemake_diagnostics: SnakemakeDiagnosticsRecord
    global_hot_functions: list[tuple[str, float, int]]
    pyspy_speedscope_path: Path | None
    cprofile_pstats_path: Path | None


def _build_isolation_env(tmp_root: Path) -> dict[str, str]:
    """Return env dict for the pytest subprocess that isolates all caches.

    Six knobs + ``-p no:cacheprovider`` (passed at the subprocess CLI, not via
    env) cover the documented cache surfaces:

    - ``XDG_CACHE_HOME``: redirects ``platformdirs.user_cache_dir`` and any other
      XDG cache consumers; isolates the toolkit's synthetic-model cache.
    - ``PYTHONDONTWRITEBYTECODE=1``: suppresses pyc *writes* (read-fallback to
      pre-existing pyc still applies; ``PYTHONPYCACHEPREFIX`` handles reads).
    - ``PYTHONPYCACHEPREFIX``: PEP 3147 — redirects bytecode reads AND writes to
      a per-pid root, so the ``AssertionRewritingHook`` pyc cache at
      ``<stem>.<cachetag>-pytest-<version>.pyc`` cannot warm-cache across runs.
    - ``HYPOTHESIS_DATABASE_FILE=:memory:``: belt-and-suspenders under ``CI=1``
      which already sets hypothesis's ``database=None``.
    - ``CI=1``: triggers hypothesis's ``ci`` profile.
    - ``TMPDIR``: per-pid tmp root for pytest's ``tmp_path`` lifecycle and any
      subprocess-launched filesystem operations.
    """
    env = os.environ.copy()
    env.update({
        "XDG_CACHE_HOME": str(tmp_root / "xdg_cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(tmp_root / "pycache"),
        "HYPOTHESIS_DATABASE_FILE": ":memory:",
        "CI": "1",
        "TMPDIR": str(tmp_root / "tmp"),
    })
    (tmp_root / "xdg_cache").mkdir(parents=True, exist_ok=True)
    (tmp_root / "pycache").mkdir(parents=True, exist_ok=True)
    (tmp_root / "tmp").mkdir(parents=True, exist_ok=True)
    return env


def _make_tmp_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix=f"triton_swmm_profile_{os.getpid()}_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


def _collect_corpus(env: dict[str, str]) -> list[CorpusEntry]:
    """Run pytest --collect-only and parse nodeids + collection-failed entries.

    pytest --collect-only -q emits one nodeid per line on stdout in the form
    ``<path>::<rest>``. Collection failures emit a multi-line ``=== ERRORS ===``
    block AND ``ERROR collecting <path>`` lines on stderr — we parse stderr
    separately for robustness in -q mode.

    Tolerates non-zero exit codes: pytest can exit 2 on partial-collection
    failures even when ``--continue-on-collection-errors`` is honored. The
    parser uses returned content, not exit code, as the source of truth.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-m", "not slow", "--continue-on-collection-errors",
         "-p", "no:cacheprovider"],
        env=env, cwd=str(REPO_ROOT),
        capture_output=True, text=True, check=False,
    )
    entries: list[CorpusEntry] = []
    err_re = re.compile(r"ERROR collecting (\S+)")
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        # Split on the FIRST "::" rather than constraining the rest via [^\s]+
        # — parametrize values can contain commas, brackets, and slashes that
        # the latter would mis-match.
        if "::" in stripped and stripped.split("::", 1)[0].endswith(".py"):
            entries.append(CorpusEntry(nodeid=stripped, status="collected"))
    for line in proc.stderr.splitlines():
        if m := err_re.search(line):
            entries.append(CorpusEntry(
                nodeid=m.group(1),
                status="collection-failed",
                error=line.strip(),
            ))
    return entries


def _measure_import_time(env: dict[str, str]) -> tuple[dict[str, float], dict[str, float], float]:
    """Run python -X importtime -m pytest --collect-only; return (per-package, conftest-only, total_s)."""
    importtime_log = Path(env["TMPDIR"]) / "importtime.log"
    t0 = time.perf_counter()
    with importtime_log.open("w") as f:
        subprocess.run(
            [sys.executable, "-X", "importtime", "-m", "pytest", "--collect-only",
             "-m", "not slow", "-p", "no:cacheprovider"],
            env=env, cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL, stderr=f, check=False,
        )
    elapsed = time.perf_counter() - t0
    per_package: dict[str, float] = {}
    conftest_only: dict[str, float] = {}
    row_re = re.compile(r"^import time:\s+(\d+) \|\s+(\d+) \|\s+(.+)$")
    for line in importtime_log.read_text().splitlines():
        if m := row_re.match(line):
            self_us, cum_us, pkg = m.groups()
            cum_s = int(cum_us) / 1_000_000.0
            per_package[pkg.strip()] = cum_s
            if "conftest" in pkg:
                conftest_only[pkg.strip()] = int(self_us) / 1_000_000.0
    return per_package, conftest_only, elapsed


def _run_pyspy_pass(env: dict[str, str], tmp_root: Path) -> tuple[Path, Path]:
    """Run py-spy record around pytest; return (speedscope_json, plugin_json)."""
    speedscope = tmp_root / "pyspy.json"
    plugin_out = tmp_root / "plugin_pyspy.json"
    env = {**env, PLUGIN_ENV_VAR: str(plugin_out)}
    subprocess.run(
        ["py-spy", "record", "--subprocesses", "--format", "speedscope", "-o", str(speedscope),
         "--", sys.executable, "-m", "pytest", "-m", "not slow",
         "-p", "no:cacheprovider",
         "-p", "scripts.profile._pytest_plugin",
         "--continue-on-collection-errors",
         "--durations=0", "--durations-min=0"],
        env=env, cwd=str(REPO_ROOT), check=False,
    )
    return speedscope, plugin_out


def _run_cprofile_pass(env: dict[str, str], tmp_root: Path) -> tuple[Path, Path]:
    """Run python -m cProfile around pytest; return (pstats_path, plugin_json)."""
    pstats_path = tmp_root / "profile.pstats"
    plugin_out = tmp_root / "plugin_cprofile.json"
    env = {**env, PLUGIN_ENV_VAR: str(plugin_out)}
    subprocess.run(
        [sys.executable, "-m", "cProfile", "-o", str(pstats_path),
         "-m", "pytest", "-m", "not slow",
         "-p", "no:cacheprovider",
         "-p", "scripts.profile._pytest_plugin",
         "--continue-on-collection-errors"],
        env=env, cwd=str(REPO_ROOT), check=False,
    )
    return pstats_path, plugin_out


def _read_plugin_json(path: Path) -> dict:
    if not path.exists():
        return {"per_test": {}, "fixtures": [], "collection": {}, "tmp_paths": []}
    return json.loads(path.read_text())


def _extract_hot_functions(pstats_path: Path, top_n: int) -> list[tuple[str, float, int]]:
    """Return global top-N functions by cumulative time.

    Per-test stratification is NOT produced: cProfile gives a single global Stats
    object covering the entire pytest session, and ``pstats.Stats`` does not
    expose the sampling-line correlation needed for post-hoc per-test attribution.
    Per-test wall-clock attribution flows through the plugin's
    ``pytest_runtest_logreport`` data; cProfile's role here is global hot-function
    identification only.

    Implementation note: ``stats.sort_stats('cumulative')`` mutates
    ``stats.fcn_list`` (the print-order list), NOT ``stats.stats`` (the raw
    dict). Iterating ``stats.stats.items()`` returns insertion order and is
    non-deterministic. ``stats.fcn_list`` is the correct iteration target.
    """
    import pstats
    if not pstats_path.exists() or pstats_path.stat().st_size == 0:
        return []
    stats = pstats.Stats(str(pstats_path))
    stats.sort_stats("cumulative")
    rows: list[tuple[str, float, int]] = []
    for func_key in (stats.fcn_list or [])[:top_n]:
        cc, nc, tt, ct, _callers = stats.stats[func_key]
        file, line, func = func_key
        rows.append((f"{Path(file).name}:{line}({func})", ct, nc))
    return rows


def _env_fingerprint() -> dict[str, str]:
    """Capture machine/env identity for the doc's fingerprint section.

    Hager-2010 stunt-defense surfaces: CPU model, CPU governor (#5
    machine-state camouflage), SMT-active + siblings (#6 affinity), thermal
    state if accessible. Reporting-discipline surfaces: filterwarnings
    contents and pytest-plugin list (config drift that would shift timing
    measurement without being visible in the fingerprint).
    """
    cpu_model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    cpu_count = os.cpu_count() or 0
    mem_total = ""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                mem_total = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        capture_output=True, text=True, check=False,
    ).stdout.strip()

    governor = ""
    try:
        governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        governor = "(unavailable)"
    smt_active = ""
    try:
        smt_active = Path("/sys/devices/system/cpu/smt/active").read_text().strip()
    except OSError:
        smt_active = "(unavailable)"
    siblings = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("siblings"):
                siblings = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    filterwarnings = ""
    try:
        import tomllib  # Python 3.11+
        with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
            ini = tomllib.load(fh).get("tool", {}).get("pytest", {}).get("ini_options", {})
            filterwarnings = "\n".join(ini.get("filterwarnings", []))
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError) as e:
        filterwarnings = f"(parse-failed: {type(e).__name__}: {e})"

    plugin_list = subprocess.run(
        [sys.executable, "-m", "pytest", "--version", "-V"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if not plugin_list:
        plugin_list = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()

    self_sha = ""
    try:
        self_sha = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
    except OSError:
        pass

    return {
        "cpu_model": cpu_model,
        "cpu_count": str(cpu_count),
        "mem_total": mem_total,
        "os": " ".join(platform.uname()),
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "(none)"),
        "git_sha": git_sha or "(no git)",
        "cpu_governor": governor,
        "smt_active": smt_active,
        "smt_siblings": siblings,
        "filterwarnings": filterwarnings,
        "pytest_plugins": plugin_list,
        "profile_script_sha": self_sha,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the TRITON-SWMM_toolkit routine test corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="Top-N hot functions (global)")
    parser.add_argument("--findings-top-k", type=int, default=DEFAULT_FINDINGS_TOP_K,
                        help="Top-K findings to surface")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS,
                        help="Profile runs (median across kept reps when >=2; first discarded as warmup when >=3)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output doc path (overwritten); default = $AGENTIC_WORKSPACE/library/knowledge/triton-swmm-toolkit/routine test profile results.md")
    parser.add_argument("--cprofile", dest="cprofile", action="store_true", default=True)
    parser.add_argument("--no-cprofile", dest="cprofile", action="store_false")
    parser.add_argument("--pyspy", dest="pyspy", action="store_true", default=True)
    parser.add_argument("--no-pyspy", dest="pyspy", action="store_false")
    parser.add_argument("--snakemake-harvest", dest="snakemake_harvest",
                        action="store_true", default=True)
    parser.add_argument("--no-snakemake-harvest", dest="snakemake_harvest",
                        action="store_false")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output is None:
        args.output = _resolve_default_output()
    from scripts.profile._snakemake_harvest import harvest as snakemake_harvest
    from scripts.profile._emitter import emit

    tmp_root = _make_tmp_root()
    print(
        f"[profile] PID={os.getpid()} output={args.output} repetitions={args.repetitions} "
        f"tmp_root={tmp_root}",
        flush=True,
    )

    runs: list[RunArtifacts] = []
    for rep in range(args.repetitions):
        per_rep_root = tmp_root / f"rep_{rep}"
        per_rep_root.mkdir()
        per_rep_env = _build_isolation_env(per_rep_root)

        tag = f"rep {rep + 1}/{args.repetitions}"
        print(f"[profile] {tag}: collecting corpus...", flush=True)
        corpus = _collect_corpus(per_rep_env)
        failed = sum(1 for e in corpus if e.status == "collection-failed")
        print(
            f"[profile] {tag}: collected {len(corpus)} nodeids ({failed} collection-failed)",
            flush=True,
        )

        print(f"[profile] {tag}: measuring import time...", flush=True)
        per_pkg_import, conftest_import, collection_total = _measure_import_time(per_rep_env)
        print(
            f"[profile] {tag}: import time {collection_total:.1f}s captured "
            f"({len(per_pkg_import)} packages, {len(conftest_import)} conftests)",
            flush=True,
        )

        pyspy_path: Path | None = None
        plugin_payload: dict = {}
        if args.pyspy:
            print(
                f"[profile] {tag}: py-spy pass starting "
                "(slowest stage; --subprocesses ptraces every fork)...",
                flush=True,
            )
            pyspy_path, plugin_json_path = _run_pyspy_pass(per_rep_env, per_rep_root)
            plugin_payload = _read_plugin_json(plugin_json_path)
            print(f"[profile] {tag}: py-spy pass complete", flush=True)

        cprofile_path: Path | None = None
        global_hot: list[tuple[str, float, int]] = []
        if args.cprofile:
            print(f"[profile] {tag}: cProfile pass starting...", flush=True)
            cprofile_path, cprofile_plugin_json = _run_cprofile_pass(per_rep_env, per_rep_root)
            cprofile_plugin_payload = _read_plugin_json(cprofile_plugin_json)
            if not plugin_payload:
                plugin_payload = cprofile_plugin_payload
            global_hot = _extract_hot_functions(cprofile_path, args.top_n)
            print(
                f"[profile] {tag}: cProfile pass complete; {len(global_hot)} hot fns",
                flush=True,
            )

        snakemake_rules: list[SnakemakeRuleRecord] = []
        snakemake_diag = SnakemakeDiagnosticsRecord()
        if args.snakemake_harvest:
            tmp_paths = plugin_payload.get("tmp_paths", [])
            print(
                f"[profile] {tag}: harvesting Snakemake metadata "
                f"from {len(tmp_paths)} tmp_paths...",
                flush=True,
            )
            for tmp_path_str in tmp_paths:
                tmp_path = Path(tmp_path_str)
                if not tmp_path.exists():
                    continue
                rules, diag = snakemake_harvest(tmp_path)
                for r in rules:
                    snakemake_rules.append(SnakemakeRuleRecord(
                        rule=r.rule,
                        rule_normalized=r.rule_normalized,
                        job_count=r.job_count,
                        total_s=r.total_s,
                        mean_s=r.mean_s,
                        min_s=r.min_s,
                        max_s=r.max_s,
                        zero_duration_job_count=r.zero_duration_job_count,
                        test_origin=r.test_origin,
                    ))
                snakemake_diag.snakefiles_found += diag.snakefiles_found
                snakemake_diag.snakefiles_with_metadata += diag.snakefiles_with_metadata
                snakemake_diag.snakefiles_dry_run_only += diag.snakefiles_dry_run_only
                snakemake_diag.snakefiles_zero_records += diag.snakefiles_zero_records
                snakemake_diag.total_records += diag.total_records
                snakemake_diag.parser_warnings.extend(diag.parser_warnings)
            print(
                f"[profile] {tag}: harvested {len(snakemake_rules)} rule rows "
                f"({snakemake_diag.snakefiles_with_metadata}/{snakemake_diag.snakefiles_found} "
                f"Snakefiles with metadata)",
                flush=True,
            )

        per_test: dict[str, PerTestRecord] = {}
        for nodeid, rec in plugin_payload.get("per_test", {}).items():
            setup_s = rec.get("setup_s", 0.0)
            call_s = rec.get("call_s", 0.0)
            teardown_s = rec.get("teardown_s", 0.0)
            per_test[nodeid] = PerTestRecord(
                nodeid=nodeid,
                setup_s=setup_s,
                call_s=call_s,
                teardown_s=teardown_s,
                total_s=setup_s + call_s + teardown_s,
                outcome=rec.get("outcome", ""),
                child_cpu_seconds=rec.get("child_cpu_seconds", 0.0),
            )

        fixtures = [
            FixtureRecord(
                name=f["name"], scope=f["scope"],
                duration_s=f["duration_s"], consumer_nodeid=f["consumer_nodeid"],
            )
            for f in plugin_payload.get("fixtures", [])
        ]

        runs.append(RunArtifacts(
            corpus=corpus,
            import_times=per_pkg_import,
            conftest_import_times=conftest_import,
            collection_total_s=collection_total,
            per_test=per_test,
            fixtures=fixtures,
            snakemake_rules=snakemake_rules,
            snakemake_diagnostics=snakemake_diag,
            global_hot_functions=global_hot,
            pyspy_speedscope_path=pyspy_path,
            cprofile_pstats_path=cprofile_path,
        ))

    # Discard warmup rep when N>=3 (pyperformance/asv convention; cold-cache
    # cost biases the median upward when measured N times).
    if args.repetitions >= 3:
        print(
            f"[profile] discarding rep 1 as warmup; emitting median across "
            f"reps 2..{args.repetitions}",
            flush=True,
        )
        runs = runs[1:]

    # Overhead-measurement mode (V-OVERHEAD-LEVEL): when no reps requested AND
    # all profile paths disabled, the run's only purpose is timing the fixed
    # orchestrator overhead. Do NOT write the canonical doc — that would
    # clobber a real prior profile-results output with empty-runs fallback.
    if args.repetitions == 0 and not (args.pyspy or args.cprofile or args.snakemake_harvest):
        print("[profile] overhead-measurement mode: skipping doc write.", flush=True)
        return 0

    print(
        f"[profile] emitting doc to {args.output} ({len(runs)} reps retained)...",
        flush=True,
    )
    emit(
        env_fingerprint=_env_fingerprint(),
        runs=runs,
        top_n=args.top_n,
        findings_top_k=args.findings_top_k,
        output_path=args.output,
    )
    print(f"[profile] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
