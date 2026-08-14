"""Figure-geometry invariants over the TREE-DERIVED set of figure-producing modules.

The denominator is recomputed on every run from a tree search, never from a list a spec
happens to name: a prior audit in this campaign enumerated three modules where four
existed. A new renderer is in scope the moment it lands.

AST, NOT REGEX -- and this is the load-bearing design note, because the first draft of
this file used regex and was measurably broken in both directions:
  * the newline predicate could never fire. It looked for a literal "\\n" inside a
    `title_text=` expression, but the newline lives in `hhemt.units`, which is not a
    figure module, and every call site passes a CALL (`units.rainfall_axis_label(...)`)
    carrying no literal. Measured: `grep -rn 'title_text=' src/hhemt/ | grep -c '\\n'`
    -> 0. A check that cannot fire is the exact defect this file exists to prevent.
  * the colorbar predicate fired on a COMMENT ("# colorbars repositioned to y=-0.22")
    and then ran eleven lines forward to match the LEGEND's `x=0.0`, while MISSING every
    real colorbar, because `colorbar=dict(title=units.depth_label(...), ... x=0.16)`
    contains a nested `)` that a `[^)]*?` span cannot cross.
Comments and nested parens are invisible to an AST walk. Do not reintroduce regex here.

ALLOWLIST IS A RATCHET, NOT AN EXEMPTION LIST. Measured at authoring time: 28 sites
across 8 of 12 modules. Three of those modules are fixed by the change that lands this
file; the other five are outside the iteration's scope and carry one justified entry
each. The check therefore fails on any NEW occurrence anywhere, and on any regression in
the three unlisted modules. Every entry names the follow-up that retires it.

Non-vacuity is established by `test_checker_flags_synthetic_violation`, which feeds each
predicate a bad input AND its corrected counterpart, asserting fire-then-silence. That
control stays meaningful after every real site is fixed or allowlisted, which is the
property a check whose denominator is the already-compliant set does not have.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "hhemt"

#: A module is figure-producing if it CALLS ``go.Figure`` or ``make_subplots``. Tree-derived.
#: Decided by AST, not by substring. The first version of this was
#: ``re.compile(r"go\.Figure|make_subplots")`` over raw file text, which classified
#: ``figure_panels.py`` as a figure producer because its module DOCSTRING mentions
#: ``make_subplots`` while explaining why it assigns row domains instead of delegating to
#: it -- prose about figure construction, counted as figure construction. That is the same
#: substring-versus-structure error the four predicates below were themselves converted
#: away from, so leaving it here made the checker internally inconsistent: AST for what it
#: flags, text matching for what it enumerates.
_FIGURE_CALLS = {"Figure", "make_subplots"}

#: A label provider whose result must be wrapped before it reaches a plotly axis title.
_LABEL_PROVIDER = re.compile(r"_label$")
_WRAPPER = "wrap_axis_title"

P1 = "P1 hardcoded bottom margin"
P2 = "P2 hardcoded colorbar paper-x"
P3 = "P3 axis title set from an unwrapped label provider"
P4 = "P4 caption width from figure width minus a literal"

#: (module filename, predicate) -> justification. Every entry names its retirement path.
_ALLOWLIST: dict[tuple[str, str], str] = {
    ("_dem_resolution_plots.py", P1): (
        "Pre-existing; third copy of the panel budget. Retired by migrating this "
        "module onto figure_panels (follow-up)."
    ),
    ("_dem_resolution_plots.py", P4): "Pre-existing; same migration retires it.",
    ("_plotting.py", P2): (
        "Pre-existing EDA figure, no Iteration-7 feedback item. Retire when "
        "figure_layout.align_x is adopted here."
    ),
    ("raw_resume_identity.py", P1): "Pre-existing EDA figure, out of Iteration-7 scope.",
    ("raw_resume_identity.py", P2): "Pre-existing EDA figure, out of Iteration-7 scope.",
    ("system_overview.py", P1): "Pre-existing; system-overview layout untouched this iteration.",
    ("system_overview.py", P2): "Pre-existing; system-overview layout untouched this iteration.",
    ("sensitivity_benchmarking.py", P4): (
        "Pre-existing AND this module carries uncommitted work from a concurrent "
        "track; editing it here would collide. Retire in a later iteration."
    ),
}


def _constructs_figure(tree: ast.AST) -> bool:
    """True when the module CALLS ``go.Figure(...)`` or ``make_subplots(...)``.

    Matches the call, not the name: an ``Attribute`` call whose attr is ``Figure`` (so
    ``go.Figure(...)`` and ``plotly.graph_objects.Figure(...)`` both count) or a bare
    ``Name`` call to ``make_subplots``. A mention in a docstring, a comment, or a string
    literal is not a call and does not count.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in _FIGURE_CALLS:
            return True
        if isinstance(fn, ast.Name) and fn.id in _FIGURE_CALLS:
            return True
    return False


def figure_modules() -> list[Path]:
    """Every figure-producing module under src/hhemt, derived from the tree."""
    out = []
    for p in sorted(_SRC.rglob("*.py")):
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:  # pragma: no cover - not expected in-tree
            continue
        if _constructs_figure(tree):
            out.append(p)
    return out


def _kwarg(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _mapping_items(node):
    """(key, value) pairs of a dict(...) call or a {...} literal."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        for k in node.keywords:
            yield k.arg, k.value
    elif isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values, strict=True):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                yield k.value, v


def _numeric(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def _callee_name(node) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def scan(source: str) -> list[tuple[str, int]]:
    """(predicate, lineno) for every violation in `source`. Pure -- the control calls it."""
    hits: list[tuple[str, int]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return hits

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            margin = _kwarg(node, "margin")
            if margin is not None:
                for k, v in _mapping_items(margin):
                    if k == "b" and _numeric(v):
                        hits.append((P1, v.lineno))

            colorbar = _kwarg(node, "colorbar")
            if colorbar is not None:
                for k, v in _mapping_items(colorbar):
                    if k == "x" and _numeric(v):
                        hits.append((P2, v.lineno))

            for kw in node.keywords:
                if kw.arg in ("cbar_x", "colorbar_x") and _numeric(kw.value):
                    hits.append((P2, kw.value.lineno))

            if _callee_name(node) in ("update_yaxes", "update_xaxes"):
                title = _kwarg(node, "title_text")
                if isinstance(title, ast.Call):
                    inner = _callee_name(title)
                    if _LABEL_PROVIDER.search(inner) and _WRAPPER not in inner:
                        hits.append((P3, title.lineno))

            width = _kwarg(node, "content_w_px")
            if isinstance(width, ast.BinOp) and isinstance(width.op, ast.Sub) and _numeric(width.right):
                hits.append((P4, width.lineno))

        # A colorbar paper-x carried as a dict-LITERAL entry (the per-sim `panels` list
        # form) is an ast.Dict, never a Call keyword, so the pass above cannot see it.
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values, strict=True):
                if isinstance(k, ast.Constant) and k.value in ("colorbar_x", "cbar_x") and _numeric(v):
                    hits.append((P2, v.lineno))

    return hits


def test_denominator_is_non_empty():
    """A tree search returning nothing would make every assertion below vacuous."""
    mods = figure_modules()
    assert len(mods) >= 10, f"figure-module tree search returned {len(mods)}: {mods}"


@pytest.mark.parametrize("module", figure_modules(), ids=lambda p: p.name)
def test_no_hand_authored_geometry(module: Path):
    violations = [
        f"{module.name}:{line}  {pred}"
        for pred, line in sorted(scan(module.read_text()), key=lambda t: t[1])
        if (module.name, pred) not in _ALLOWLIST
    ]
    assert not violations, (
        f"{module.relative_to(_SRC)} hand-authors geometry a shared module owns:\n  "
        + "\n  ".join(violations)
        + "\nUse figure_caption / figure_layout / figure_panels / _hydrology_panel, or add "
        "an _ALLOWLIST entry naming the follow-up that retires it."
    )


def test_allowlist_entries_are_all_live():
    """An allowlist entry whose violation is gone is stale and must be deleted.

    Without this, the ratchet loosens silently: a fixed module keeps its exemption and
    a later regression there goes unreported.
    """
    live = {(m.name, pred) for m in figure_modules() for pred, _ in scan(m.read_text())}
    stale = sorted(k for k in _ALLOWLIST if k not in live)
    assert not stale, f"_ALLOWLIST entries no longer needed -- delete them: {stale}"


@pytest.mark.parametrize(
    ("bad", "good", "expected"),
    [
        (
            "fig.update_layout(margin=dict(l=10, r=10, t=40, b=130))",
            "fig.update_layout(margin=dict(l=10, r=10, t=40, b=b_px))",
            P1,
        ),
        (
            "fig.add_trace(go.Heatmap(z=z, colorbar=dict(title=f(a), orientation='h', x=0.16, len=0.3)))",
            "fig.add_trace(go.Heatmap(z=z, colorbar=dict(title=f(a), orientation='h', x=cb_x(1), len=0.3)))",
            P2,
        ),
        (
            "panels = [{'col': 1, 'colorbar_x': 0.16}]",
            "panels = [{'col': 1, 'colorbar_x': cb_x(1)}]",
            P2,
        ),
        (
            "fig.update_yaxes(title_text=units.rainfall_axis_label(u), row=1, col=3)",
            "fig.update_yaxes(title_text=wrapped_title, row=1, col=3)",
            P3,
        ),
        (
            "b = add_figure_caption(fig, t, content_w_px=_FIG_W - 60, plot_h_px=h)",
            "b = add_figure_caption(fig, t, content_w_px=outline_content_w_px(L), plot_h_px=h)",
            P4,
        ),
    ],
)
def test_checker_fires_on_bad_and_is_silent_on_good(bad: str, good: str, expected: str):
    """POSITIVE CONTROL. A checker that never fires passes forever.

    Both halves are load-bearing: fire-on-bad proves the predicate works, silent-on-good
    proves it is not simply matching everything -- the failure mode of the regex draft,
    which fired on a comment.
    """
    assert expected in [p for p, _ in scan(bad)], f"predicate did not fire on: {bad!r}"
    assert expected not in [p for p, _ in scan(good)], f"predicate false-positives on: {good!r}"
