"""Regression test: with kaleido unavailable, the plotly-export path fails fast
with a developer-actionable preflight error rather than a cryptic Plotly stack.

kaleido is now a core dependency (see pyproject.toml); this test guards the
missing/corrupted-install UX — the `_check_static_backend_kaleido_available`
preflight is the fail-fast gate (no matplotlib fallback exists when
static_backend='plotly').
"""

from __future__ import annotations

import sys

from hhemt.validation import ValidationResult, _check_static_backend_kaleido_available


class _FakeInteractive:
    static_backend = "plotly"


class _FakeReport:
    interactive = _FakeInteractive()


def test_missing_kaleido_fails_fast_with_actionable_hint(monkeypatch):
    """kaleido absent + static_backend='plotly' -> a preflight ERROR whose
    fix_hint guides a reinstall (kaleido is core)."""
    monkeypatch.setitem(sys.modules, "kaleido", None)
    result = ValidationResult(context="test")
    _check_static_backend_kaleido_available(_FakeReport(), result)
    assert result.errors, "Expected a preflight error when kaleido is unavailable"
    assert any(
        "reinstall" in (issue.fix_hint or "").lower() or "pip install -e ." in (issue.fix_hint or "")
        for issue in result.errors
    ), "Expected a developer-actionable reinstall hint, not a cryptic Plotly error"


def test_matplotlib_backend_skips_kaleido_check(monkeypatch):
    """static_backend='matplotlib' -> no kaleido requirement, no error even when
    kaleido is absent."""
    monkeypatch.setitem(sys.modules, "kaleido", None)

    class _MplInteractive:
        static_backend = "matplotlib"

    class _MplReport:
        interactive = _MplInteractive()

    result = ValidationResult(context="test")
    _check_static_backend_kaleido_available(_MplReport(), result)
    assert not result.errors, "matplotlib backend must not require kaleido"
