"""Regression tests for the model-type-skip placeholder emit path.

Pre-fix, ``_emit_model_type_skip_placeholder`` persisted only via matplotlib
``fig.savefig(output_path)`` — which raises ``ValueError: Format 'html' is not
supported`` when the plotly report branch (default ``static_backend='plotly'``)
owns an ``.html`` ``output_path`` (the pure-TRITON conduit_flow smoke crash,
2026-07-26). The fix makes the placeholder format-aware: ``.html`` targets emit
a self-contained HTML document; every other suffix keeps the matplotlib image.
"""

from __future__ import annotations

from hhemt.report_renderers.per_sim_conduit_flow import (
    _emit_model_type_skip_placeholder,
)


def test_skip_placeholder_html_output_is_valid_html(tmp_path):
    """.html target emits a self-contained HTML doc carrying the message
    (pre-fix: ValueError: Format 'html' is not supported)."""
    out = tmp_path / "sub" / "conduit_flow__sa.serial_0_r1__evt.0.html"
    result = _emit_model_type_skip_placeholder(
        out, "conduit_flow not applicable for triton-only analyses", dpi=150
    )
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    text = out.read_text(encoding="utf-8")
    assert "conduit_flow not applicable for triton-only analyses" in text
    assert "<html" in text.lower()


def test_skip_placeholder_png_output_still_matplotlib(tmp_path):
    """Non-.html target keeps the matplotlib image (no regression)."""
    out = tmp_path / "conduit_flow__sa.x__evt.0.png"
    result = _emit_model_type_skip_placeholder(out, "skip message", dpi=100)
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes().startswith(b"\x89PNG")
