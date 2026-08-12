"""Regression tests for the model-type-skip placeholder emit path.

Pre-fix, ``_emit_model_type_skip_placeholder`` persisted only via matplotlib
``fig.savefig(output_path)`` — which raises ``ValueError: Format 'html' is not
supported`` when the plotly report branch (default ``static_backend='plotly'``)
owns an ``.html`` ``output_path`` (the pure-TRITON conduit_flow smoke crash,
2026-07-26). The fix makes the placeholder format-aware: ``.html`` targets emit
a self-contained HTML document; every other suffix keeps the matplotlib image.
"""

from __future__ import annotations

import difflib

import pytest

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


_GUARD_FIXTURE_BODY = (
    "<script>const e = React.createElement;\n"
    "class AbstractResults extends React.Component {\n"
    "    render() {\n"
    "        if (this.state.data.toggleLabels.size > 0) {\n"
    "            return null;\n"
    "        }\n"
    "    }\n"
    "}\n</script>"
)

_GUARD_FIXTURES = {
    # Both </body> shapes; see the module docstring for why CHARACTER granularity.
    "body_midline": "<html><body>" + _GUARD_FIXTURE_BODY + "</body></html>",
    "body_line_initial": "<html>\n<body>\n" + _GUARD_FIXTURE_BODY + "\n</body>\n</html>",
}


def _opcode_kinds(a: str, b: str) -> set[str]:
    """Distinct difflib opcode kinds between two strings, at CHARACTER
    granularity. Fixtures only -- do not scale this to a real 33 MB report."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return {op[0] for op in sm.get_opcodes()}


@pytest.mark.parametrize("shape", sorted(_GUARD_FIXTURES))
def test_render_guard_is_purely_additive(shape):
    """VMS-5: the guard surgery inserts and never replaces or deletes.

    Asserted as the INVARIANT ("no character is removed"), not as a satisfying
    position ("every line survives as a line"). The line-membership form is what
    failed first: `rfind("</body>")` lands mid-line in the compact fixture and
    splits `</script></body></html>`, so a line vanished as a unit while nothing
    was removed.
    """
    from hhemt.report_renderers._react_surgery import (
        _GUARD_RENDER_NEW,
        apply_post_process_surgery,
    )

    src = _GUARD_FIXTURES[shape]
    once = apply_post_process_surgery(src)

    # both halves of the guard landed
    assert _GUARD_RENDER_NEW in once
    assert "class ReportRenderGuard extends React.Component" in once

    kinds = _opcode_kinds(src, once)
    assert kinds <= {"equal", "insert"}, (
        f"surgery is not purely additive on {shape}: opcode kinds {sorted(kinds)}"
    )


@pytest.mark.parametrize("shape", sorted(_GUARD_FIXTURES))
def test_render_guard_is_idempotent(shape):
    """VMS-5: re-applying the surgery is a no-op on both </body> shapes."""
    from hhemt.report_renderers._react_surgery import apply_post_process_surgery

    once = apply_post_process_surgery(_GUARD_FIXTURES[shape])
    assert apply_post_process_surgery(once) == once


def test_render_guard_does_not_fire_without_the_anchor():
    """VMS-5 inertness: a document whose AbstractResults shape does not match
    receives neither the render patch nor an orphan <script> of guard defs.

    Deliberately NOT asserted as whole-document byte-identity: apply_post_process_surgery
    also runs steps 1-9, and step 7 legitimately injects the row-click delegate at
    </body> on any document that has one. Asserting byte-identity here would encode
    a satisfying position that an unrelated, correct surgery step breaks.
    """
    from hhemt.report_renderers._react_surgery import (
        _GUARD_RENDER_NEW,
        apply_post_process_surgery,
    )

    neutral = "<html><body><p>no react here</p></body></html>"
    out = apply_post_process_surgery(neutral)
    assert "ReportRenderGuard" not in out
    assert _GUARD_RENDER_NEW not in out
    assert "renderGuardedContent" not in out
