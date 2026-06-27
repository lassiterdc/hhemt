"""Unit tests for the C2 provenance emitter (pure; log.py read-only)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from hhemt import provenance


class _Field:
    def __init__(self, v):
        self._v = v

    def get(self):
        return self._v


def _fake_analysis():
    log = SimpleNamespace(
        workflow_submission_time=_Field("2026-01-01T00:00:00"),
        workflow_submission_node=_Field("login1"),
    )
    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(analysis_id="a1"),
        # Real analysis exposes _get_enabled_model_types() (analysis.py:1431) encapsulating the
        # cfg_system.toggle_* fields; the emitter calls that method, so the fake exposes it too.
        # All-toggles-off → no enabled models (and df_sims.index=[] yields zero run-units anyway).
        _get_enabled_model_types=lambda: [],
        df_sims=SimpleNamespace(index=[]),
        log=log,
        _case_manifest=SimpleNamespace(case_name="norfolk", description="", manifest={"inputs/dem.tif": "ab" * 32}),
    )


def test_emit_returns_core_and_sidecar_strings(monkeypatch):
    monkeypatch.setattr(provenance, "_toolkit_git_sha", lambda: "deadbeef")
    core, sidecar = provenance.emit_provenance(_fake_analysis())
    assert isinstance(core, str) and isinstance(sidecar, str)
    json.loads(core)
    json.loads(sidecar)


def test_volatile_absent_from_core_present_in_sidecar(monkeypatch):
    # R3/R6: startTime/agent are in the sidecar full graph, stripped from the core.
    monkeypatch.setattr(provenance, "_toolkit_git_sha", lambda: "deadbeef")
    monkeypatch.setattr(provenance, "_iter_run_units", lambda a: [("", "0", "triton")])
    monkeypatch.setattr(provenance, "_output_ids", lambda *a, **k: ["sims/evt-0/processed/x.zarr"])
    a = _fake_analysis()
    a.df_sims.index = [0]
    core, sidecar = provenance.emit_provenance(a)
    assert "startTime" not in core
    # the sidecar's CreateAction carries the volatile field
    assert "startTime" in sidecar


def test_emit_is_deterministic_for_core(monkeypatch):
    # R10: the embedded core is byte-identical across two emits.
    monkeypatch.setattr(provenance, "_toolkit_git_sha", lambda: "deadbeef")
    c1, _ = provenance.emit_provenance(_fake_analysis())
    c2, _ = provenance.emit_provenance(_fake_analysis())
    assert c1 == c2


def test_default_code_repository_pins_homepage():
    # No-silent-fallback contract (D-URL): the RO-Crate codeRepository is sourced from the
    # installed package metadata (pyproject.toml [project.urls].homepage), pinned here so a
    # pyproject regression / org rename is caught at CI — never silently emitted into durable
    # provenance, and never thrown at hour-3 of an HPC consolidation. `lassiterdc` is the
    # sanctioned public org (in committed pyproject.toml; absent from the ADR-14 blocklist).
    assert provenance._default_code_repository() == "https://github.com/lassiterdc/hhemt"
