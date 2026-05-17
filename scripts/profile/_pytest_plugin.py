"""Pytest plugin loaded into the profile subprocess via -p scripts.profile._pytest_plugin.

Emits a JSON file at $PROFILE_PLUGIN_JSON with per-test, per-fixture, and
collection timings. Hooks: pytest_collectstart / pytest_collectreport
(Collector-granular file-level collection timing — see _pytest/hookspec.py;
pytest_itemcollected fires per-Item which is too granular), pytest_fixture_setup
(per-cache-miss fixture setup duration), pytest_runtest_logstart +
pytest_runtest_logreport (per-test setup/call/teardown durations + os.times()
child-CPU delta capture for cProfile-invisible-subprocess attribution), and
pytest_runtest_setup (post-fixture-resolution tmp_path capture for Snakemake
harvest discovery).

Determinism: only uses time.perf_counter and os.times in-memory; no per-event
fs writes. The sessionfinish hook flushes all accumulated state to the
$PROFILE_PLUGIN_JSON output path with sorted_keys=True for deterministic doc
rendering.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

PLUGIN_ENV_VAR = "PROFILE_PLUGIN_JSON"

_per_test: dict[str, dict] = {}
_fixtures: list[dict] = []
_collection: dict[str, float] = {}
_tmp_paths: set[str] = set()
_collect_start: dict[str, float] = {}
_phase_times_start: dict[tuple[str, str], tuple[float, ...]] = {}


@pytest.hookimpl(hookwrapper=True)
def pytest_collectstart(collector):
    _collect_start[collector.nodeid] = time.perf_counter()
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_collectreport(report):
    yield
    start = _collect_start.pop(report.nodeid, None)
    if start is not None:
        _collection[report.nodeid] = time.perf_counter() - start


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef, request):
    t0 = time.perf_counter()
    try:
        result = yield
    finally:
        duration = time.perf_counter() - t0
        _fixtures.append({
            "name": fixturedef.argname,
            "scope": fixturedef.scope,
            "duration_s": duration,
            "consumer_nodeid": getattr(request.node, "nodeid", ""),
        })
    return result


def pytest_runtest_logstart(nodeid, location):
    _phase_times_start[(nodeid, "setup")] = os.times()


def pytest_runtest_logreport(report: pytest.TestReport):
    rec = _per_test.setdefault(
        report.nodeid,
        {"setup_s": 0.0, "call_s": 0.0, "teardown_s": 0.0, "outcome": "", "child_cpu_seconds": 0.0},
    )
    rec[f"{report.when}_s"] = report.duration
    if report.when == "call":
        rec["outcome"] = report.outcome
    now = os.times()
    start = _phase_times_start.get((report.nodeid, report.when))
    if start is not None:
        # times tuple: (user, system, children_user, children_system, elapsed)
        child_delta = (now[2] - start[2]) + (now[3] - start[3])
        rec["child_cpu_seconds"] += max(0.0, child_delta)
    next_phase = {"setup": "call", "call": "teardown", "teardown": None}.get(report.when)
    if next_phase is not None:
        _phase_times_start[(report.nodeid, next_phase)] = now


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    # Capture tmp_path post-fixture-resolution: an autouse fixture would
    # participate in fixture resolution itself and miss tests that don't request
    # tmp_path; pytest_runtest_setup runs after setup completes for the item.
    try:
        result = yield
    finally:
        tmp = item.funcargs.get("tmp_path") if hasattr(item, "funcargs") else None
        if tmp is not None:
            _tmp_paths.add(str(tmp))
    return result


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get(PLUGIN_ENV_VAR)
    if not out:
        return
    Path(out).write_text(json.dumps({
        "per_test": _per_test,
        "fixtures": _fixtures,
        "collection": _collection,
        "tmp_paths": sorted(_tmp_paths),
    }, indent=2, sort_keys=True))
