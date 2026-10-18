"""Post-process surgery applied to Snakemake-rendered report HTML.

Snakemake's `--report` engine produces a React-bundled `report.html`. Several
behaviors and visual elements are baked into that bundle and cannot be
addressed via the report-stylesheet alone (e.g., the JS click-handler that
opens figures, the category-sort comparator, the About menu item, the
navbar text). This module string-replaces those baked-in pieces.

Two entry points, one per rendered output form; the caller picks the one that
matches what it rendered (for example ``TRITONSWMM_analysis.render_report(format=...)``
calls the first for ``"html"`` and the second otherwise):
  - ``apply_post_process_surgery`` -- on the text of a single-file HTML report.
  - ``apply_post_process_surgery_to_zip`` -- on the ``report.html`` inside a zip
    report, which is then re-zipped. Without this, zip mode renders the
    eye-icon-hiding CSS but lacks the JS click-delegate that makes rows clickable,
    leaving figure tables with no clickable affordance (rows look interactive via
    CSS but click handlers are absent).

Re-applying the surgery to an already-surgeried report page leaves the page
unchanged. Most steps get this for free: each replaces an OLD literal that is gone
after the first pass (step 1 also checks for its OLD title literal before
replacing it). The steps that insert or would otherwise re-match (4c, 6, 6b and 7)
check for their NEW text first, and the card-name pass (8b) does not re-match a
humanized name because the humanizer strips the file extension its pattern
requires. The zip ARCHIVE is not byte-stable under re-application: the zip
function re-extracts and re-archives it, so every entry gets a new timestamp even
though the page inside is unchanged.
"""

from __future__ import annotations

import re

from hhemt.exceptions import ProcessingError
from hhemt.report_plot_ids import humanize_plot_id

# The category order used when the caller passes no category_order
# (Bundle.regenerate_report is one such caller), mirroring navbar_text's None default:
# a caller that passes nothing gets this fixed, module-defined order.
_DEFAULT_CATEGORY_ORDER = (
    "Workflow Status",
    "Errors and Warnings",
    "Key Results",
    "System Information",
    "Per Simulation Results",
)


def _order_js(category_order: tuple[str, ...] | list[str]) -> str:
    """Build the JS ORDER-dict literal from an ordered category list (1-indexed).

    Each name is written as a JSON string, so a name carrying ``"`` or ``\\`` cannot end
    its string early and leave the comparator this literal is spliced into unparseable
    (step 3 escapes the navbar text for the same reason). A name carrying neither
    character and no control character comes out byte-identical to plain
    double-quoting, and ``ensure_ascii=False`` leaves a non-ASCII name unescaped.
    """
    import json

    body = ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {i}" for i, k in enumerate(category_order, start=1))
    return "{" + body + "}"


_SHOW_CATEGORY_OLD = "this.setView({ navbarMode: mode, category: category, subcategory: subcategory })\n    }"

_SHOW_CATEGORY_NEW = (
    "this.setView({ navbarMode: mode, category: category, subcategory: subcategory });\n"
    "        setTimeout(function(){\n"
    '            var tbl = document.querySelector("table.table-auto");\n'
    "            if (!tbl) return;\n"
    '            var firstRow = tbl.querySelector("tbody tr");\n'
    "            if (!firstRow) return;\n"
    '            var actionDiv = firstRow.querySelector("td.text-right > div.inline-flex");\n'
    "            if (!actionDiv) return;\n"
    '            var firstBtn = actionDiv.querySelector("a, button");\n'
    "            if (firstBtn) firstBtn.click();\n"
    "        }, 80);\n"
    "    }"
)

_GUARD_RENDER_OLD = "    render() {\n        if (this.state.data.toggleLabels.size > 0) {"

_GUARD_RENDER_NEW = (
    "    render() {\n"
    "        try {\n"
    "            return e(\n"
    "                ReportRenderGuard,\n"
    "                { key: String(this.getCategory()), where: this.getCategory() },\n"
    "                this.renderGuardedContent()\n"
    "            );\n"
    "        } catch (err) {\n"
    "            return reportRenderGuardPanel(err, undefined);\n"
    "        }\n"
    "    }\n"
    "\n"
    "    renderGuardedContent() {\n"
    "        if (this.state.data.toggleLabels.size > 0) {"
)

_GUARD_DEFS = """
<script>
class ReportRenderGuard extends React.Component {
    constructor(props) {
        super(props);
        this.state = { error: null };
    }
    static getDerivedStateFromError(error) {
        return { error: error };
    }
    render() {
        if (this.state.error === null) {
            return this.props.children;
        }
        return reportRenderGuardPanel(this.state.error, this.props.where);
    }
}

function reportRenderGuardPanel(err, where) {
    // INLINE STYLES, NOT TAILWIND CLASSES. Snakemake ships a content-PURGED Tailwind
    // build: only classes its own templates used at ITS build time have CSS rules.
    // This panel is injected by post-process surgery, so the purge never saw it --
    // measured on a rendered report, `bg-red-50`, `text-red-700`, `border-red-700`,
    // `font-mono`, `text-sm` and `font-bold` each had ZERO CSS selector occurrences
    // and appeared exactly once in the document (here), while Snakemake's own
    // `inline-flex` / `text-right` had 5 and 18 rules. A class-styled panel therefore
    // renders entirely unstyled and inherits the ambient colour, so a reader sees what
    // looks like a blank page rather than an error. Any class-based fix has the same
    // defect regardless of which class is chosen; an inline style needs no stylesheet.
    // Applies to every future injected VISUAL surface -- the behaviour-only
    // injections (_SHOW_CATEGORY_NEW, _CLICK_DELEGATE) are unaffected.
    var _mono = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    return e(
        "div",
        { style: { margin: "0.5rem", padding: "0.75rem", borderRadius: "6px",
                   border: "2px solid #B11E1E", backgroundColor: "#FDECEC",
                   color: "#1A1A1A", fontSize: "13px" } },
        e("div", { style: { fontWeight: "700", color: "#B11E1E", marginBottom: "4px" } },
          "This results view could not be displayed."),
        e("div", { style: { marginBottom: "4px", color: "#1A1A1A" } },
          "Category: " + (where === undefined || where === null ? "(unknown)" : String(where))),
        e("div", { style: { fontFamily: _mono, fontSize: "12px", whiteSpace: "pre-wrap",
                            wordBreak: "break-all", color: "#1A1A1A",
                            backgroundColor: "#FFFFFF", padding: "6px",
                            borderRadius: "4px", border: "1px solid #E0B4B4" } },
          String((err && err.name) || "Error") + ": " + String((err && err.message) || err)),
        e("div", { style: { marginTop: "8px", fontSize: "12px", color: "#1A1A1A" } },
          "The rest of the report is unaffected \\u2014 use the sidebar to open another " +
          "category. Please copy the line above into a bug report.")
    );
}
</script>
"""


# Toggle-cell fallback (the per-sim `results[undefined].mime_type` throw).
#
# `AbstractResults.getData` promotes a label to a TOGGLE when it has exactly two values
# each occurring in exactly half the results. The heuristic's unstated precondition --
# stated in its own upstream comment, "a plot which is created twice for each sample,
# once with and once without legend" -- is that the two values are two RENDERINGS of one
# figure, so every entry carries both. In the combined report the `models` label splits
# 28/28 (`peak_flood_depth` pairs across both arms; `conduit_flow` is SWMM-dependent and
# cannot), and the two values are two DIFFERENT figures. `figure` stays in the entry key,
# so all 56 entries hold exactly ONE of the two toggle values and `entries.get(k).get(t)`
# returns `undefined` for 28 of them. That `undefined` reaches `ResultViewButton.render`
# -> `handleSelectedResult(undefined)` -> `results[undefined].mime_type` -> TypeError,
# during RENDER, which is why the guard panel appears on merely opening the category.
#
# Fixing the label emission was tried once already (the per-sim rule specs were changed
# to declare `figure` as their first label, which removed the same 50/50 accident from
# `figure`) and the collision reappeared on `models`; protecting `models` by reordering
# would put `figure` back in the toggle slice. This guards the LOOKUP instead, which
# closes the class rather than the instance. Behaviour when the cell IS
# populated is byte-identical; when it is not, the row opens its own figure -- which is
# the desired result for an unpaired entry. Idempotent because step 6b checks for each
# NEW form first and skips a lookup whose NEW form is present, so a second pass changes
# nothing. The OLD literals' absence is NOT what makes it safe: on a document carrying the
# toggle machinery, OLD absent with NEW absent raises the match guard's ProcessingError.
_TOGGLE_CELL_OLD = "            let entryPath = data.entries.get(arrayKey(entryLabels)).get(arrayKey(toggleLabels));"

_TOGGLE_CELL_NEW = (
    "            let _cell = data.entries.get(arrayKey(entryLabels));\n"
    "            let entryPath = _cell.get(arrayKey(toggleLabels));\n"
    "            if (entryPath === undefined) { entryPath = _cell.values().next().value; }"
)

_TOGGLE_CB_OLD = (
    "                let targetPath = _this.state.data.entries.get(arrayKey(entryLabels)).get(arrayKey(toggleLabels));"
)

_TOGGLE_CB_NEW = (
    "                let _cbCell = _this.state.data.entries.get(arrayKey(entryLabels));\n"
    "                let targetPath = _cbCell.get(arrayKey(toggleLabels));\n"
    "                if (targetPath === undefined) { targetPath = _cbCell.values().next().value; }"
)

# Toggle SUPPRESSION. The `models` label is promoted to a two-option radio pair by
# `AbstractResults.getData`, rendered as `MODELS` by Tailwind's `uppercase` on the label
# span (`toggle.js`) -- which is why a grep of the delivered report for "Models" returns
# only library source. The control is a no-op by construction: its two values are two
# DIFFERENT figure families, so selecting either re-opens the row's own figure via the
# lookup guard above.
#
# Removing the `models` LABEL was rejected on two grounds. The combined report needs it:
# it is the axis that campaign exists to expose, and inside one subcategory it separates
# pages carrying one arm from pages carrying two (bundle/combined_snakefile_generator.py's
# labels comment). And removing it would not close the class: `getLabels` orders labels by
# their first-seen key index and `getData` considers every label after the first
# (`labels.slice(1)`), so any remaining label with a 50/50 split is still promoted. `figure`
# is outside that slice wherever it is emitted as the first label (index 0) -- the per-sim
# rule specs and the combined harvest -- which is the move that removed it after its own
# 50/50 split (see the toggle-cell fallback note above `_TOGGLE_CELL_OLD`). The source-side
# benchmarking figures declare `independent_var` first, so there `figure` IS in the slice;
# it is not promoted only because its one constant value ("vs Total runtime") fails the
# exactly-two-values rule. Suppressing PROMOTION closes the class; removing one
# label moves it. One character, and the literal occurs exactly once in the delivered
# report (measured 2026-08-14: 1).
#
# Idempotent for the same reason as the lookup patches: step 6b skips this literal when its
# NEW form is present. On a document carrying the toggle machinery (it contains
# `arrayKey(`), an absent OLD with an absent NEW raises; a document without it never enters
# step 6b's loop, so this step leaves it unchanged.
_TOGGLE_SUPPRESS_OLD = "        if (toggleLabels.size > 1) {"
_TOGGLE_SUPPRESS_NEW = "        if (toggleLabels.size > 0) {"

#: TIER-1 FEATURE SENTINEL for the step-6b match guard. Answers "does this document
#: carry the toggle machinery step 6b patches?", which is the question that separates
#: a document legitimately WITHOUT the lookup (a fragment, such as the render-guard
#: test fixtures, that must pass through untouched) from one whose lookup literal has
#: DRIFTED (must fail loudly).
#:
#: `arrayKey` is chosen because it is a module-level HELPER that both lookup OLD literals
#: (`_TOGGLE_CELL_OLD`, `_TOGGLE_CB_OLD`) call twice, so it is a strict WEAKENING of them
#: along every axis that drifts -- indentation, a refactor of the `.get().get()` chain, or a
#: rename of `renderEntries` itself -- and it disappears only when the toggle feature is
#: gone upstream, which is exactly when inertness is correct. The sentinel and the
#: legitimate-absence condition are the same condition; that is what makes it a sentinel
#: rather than a heuristic.
#:
#: Measured 2026-08-13 -- 0 occurrences in all three fragment fixtures of
#: tests/test_report_renderer_skip_placeholder.py, 7 in each delivered real report. Do
#: NOT substitute `AbstractResults` (1 in the fixtures) or `toggleLabels` (1, at the
#: fixture body's `.size > 0` line): neither separates the classes.
#:
#: A second pass cannot flip tier 1 from absent to present, and so cannot raise on a
#: document the first pass left alone. The only module constants that carry the
#: sentinel, `_TOGGLE_CELL_NEW` and `_TOGGLE_CB_NEW`, are injected inside the tier-1
#: branch, so they reach only a document that already carried it. Every other module
#: constant this function injects (`_SHOW_CATEGORY_NEW`, `_GUARD_RENDER_NEW`,
#: `_GUARD_DEFS`, `_TOGGLE_SUPPRESS_NEW`, `_CLICK_DELEGATE`) does not contain it.
#: Not established: none of the four caller-supplied inputs that reach the document is
#: checked for the sentinel -- `navbar_text`, the `category_order` names, and the
#: `member_labels` and `event_labels` values that step 8b's `humanize_plot_id` writes into
#: figure card names.
_TOGGLE_FEATURE_SENTINEL = "arrayKey("


_CLICK_DELEGATE = """
<script>
(function(){
  function init(){
    document.addEventListener('click', function(e){
      if (e.target.closest('a, button, summary, input, select, label')) return;
      var tr = e.target.closest('tr');
      if (!tr) return;
      var actionDiv = tr.querySelector('td.text-right > div.inline-flex');
      if (!actionDiv) return;
      var firstBtn = actionDiv.querySelector('a, button');
      if (firstBtn) { e.preventDefault(); firstBtn.click(); }
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
</script>
"""


def apply_post_process_surgery(
    html_text: str,
    bundle_mode: bool = False,
    navbar_text: str | None = None,
    category_order: list[str] | None = None,
    member_labels: dict[str, str] | None = None,
    event_labels: dict[str, str] | None = None,
) -> str:
    """Apply all React-bundle post-process replacements and return modified text.

    Calling it again on its own output changes nothing; the module docstring
    states how each step achieves that.

    Every step, in the order applied (each step's in-body comment carries its
    reason):
      1. Browser-tab title "Snakemake Report" -> empty
      2. Drop the About menu item from the bundled JS (CSS-only hide infeasible)
      3. Replace bold "Snakemake" navbar span -> ``navbar_text`` (the report's
         upper-left brand text). When ``navbar_text`` is ``None``, the fixed literal
         "H&H Ensemble Modeling Toolkit" is used. The render facades source it from brand_theme.upper_left_text --
         the same config model that carries the report's colours, so one file
         rebrands the whole report -- and fall back to analysis_id when it is None.
      4. Patch category-sort comparator -> config-driven category order
         (``category_order``; None uses ``_DEFAULT_CATEGORY_ORDER``)
      4b. Rewrite the toggle-controls ``entries().map`` call through
         ``Array.from`` so a browser without ES2025 iterator helpers does not throw
      4c. Guard the results view's render so a render error shows an error panel
         instead of a blank page
      5. (retired) No sidebar category is injected on any path; the sidebar shows
         exactly the categories the report's figures declare
      6. Patch showCategory to auto-pop the first figure (setTimeout firstBtn.click)
      6b. On a document carrying the toggle machinery, make the two toggle-cell
         lookups fall back to the row's own figure and patch the toggle-promotion
         test (see the notes above ``_TOGGLE_CELL_OLD`` and ``_TOGGLE_SUPPRESS_OLD``)
      7. Inject row-click delegate at LAST `</body>` so clicks anywhere on a
         result row fire the hidden eye-icon button (paired with CSS in
         report.css that hides the eye-icon and styles rows as clickable)
      8. Force the App's initial ``content`` to ``"metadata"`` (the
         workflow_description landing page) rather than ``"rulegraph"`` (the
         DAG). The home-icon click handler already targets this view; the
         change makes the default-open match. Unconditional — applies to both
         source-side and bundle-side reports.
      8b. Rewrite each figure card's display name from its file stem to a readable
         label via ``report_plot_ids.humanize_plot_id``
      9. If ``bundle_mode=True``: drop the Workflow menu item, the Statistics
         menu item, and the "General" ``ListHeading`` from the bundled JS.
         Used by ``Bundle.regenerate_report`` because a bundle's regeneration
         Snakefile only describes plot rules + render_report — the Workflow
         and Statistics panels in a bundle-regenerated report describe only
         the regeneration DAG (no production-analysis DAG), so they carry no
         information about the analysis; and the "General" heading is empty once Workflow
         + Statistics + About are gone. About-drop (step 2) is unconditional
         and applies to source-side too. NOT applied to source-side reports
         where Workflow + Statistics describe real workflow content + runtime.

    Raises
    ------
    ProcessingError
        At step 6b, when the document contains ``arrayKey(`` (it carries the toggle
        machinery) but one of the three toggle OLD literals occurs other than exactly
        once while its NEW form is absent -- the bundled code has drifted.
    """
    # 1. Browser-tab title
    if "<title>Snakemake Report</title>" in html_text:
        html_text = html_text.replace("<title>Snakemake Report</title>", "<title></title>")

    # 2. Drop About menu item
    html_text = html_text.replace(
        'this.getMenuItem("About", "information-circle", this.props.app.showReportInfo),',
        "",
    )

    # 3. Navbar span text
    _navbar = "H&H Ensemble Modeling Toolkit" if navbar_text is None else navbar_text
    _navbar = _navbar.replace("\\", "\\\\").replace('"', '\\"')  # JS-literal safe
    html_text = html_text.replace(
        'e(\n                        "span",\n'
        '                        { className: "font-bold mx-1" },\n'
        '                        "Snakemake"\n                    )',
        'e(\n                        "span",\n'
        '                        { className: "font-bold mx-1" },\n'
        '                        "' + _navbar + '"\n                    )',
    )

    # 4. Category-sort comparator. category_order is the active reporting set's
    # sidebar order, so each named set ranks its own categories; None uses
    # _DEFAULT_CATEGORY_ORDER. Idempotent:
    # after the first pass the localeCompare literal is gone, so re-application is a
    # no-op.
    _order_js_literal = _order_js(category_order or _DEFAULT_CATEGORY_ORDER)
    _comparator = (
        f"(a, b) => {{const ORDER = {_order_js_literal}; "
        "return (ORDER[a] ?? 99) - (ORDER[b] ?? 99) || a.localeCompare(b);}"
    )
    html_text = html_text.replace("(a, b) => a.localeCompare(b)", _comparator)

    # 4b. Toggle-controls iterator-helper compat. AbstractResults.getToggleControls
    # calls `toggleLabels.entries().map(...)`. `Map.prototype.entries()` returns a
    # MapIterator, and `.map` on an ITERATOR is ES2025 Iterator Helpers (Chrome/Edge
    # 122+, Firefox 131+, Safari 18.4+). On an older engine this raises
    # `TypeError: toggleLabels.entries(...).map is not a function` inside React's
    # render, React unmounts the tree, and the category renders as a blank white page.
    # The branch is reached only when a non-first label has exactly two values each
    # occurring in exactly half the results. The coupled per-sim category once hit it on
    # `figure` (28 Conduit flow + 28 Peak flood depth = 56); the per-sim rule specs now
    # declare `figure` as their FIRST label, which the heuristic never considers, so that
    # trigger is gone at the source. This removes the engine dependency for any future
    # category that hits the same 50/50 accident. Idempotent: the old literal is gone
    # after the first pass.
    html_text = html_text.replace(
        "return toggleLabels.entries().map(function (entry) {",
        "return Array.from(toggleLabels.entries()).map(function (entry) {",
    )

    # 4c. Render guard (blank-page failure class). The bundled React is the
    # PRODUCTION build (`react.production.min` / `react-dom.production.min` are
    # each present exactly once in the rendered HTML), and production React
    # unmounts the ENTIRE tree on an uncaught render error -- which is why a throw
    # anywhere below the results view yields a white page rather than a broken
    # panel. Step 4b removes ONE known throw (the ES2025 iterator-helper call);
    # this bounds the CLASS.
    #
    # Two layers, both confined to AbstractResults, because each misses what the
    # other catches: the try/catch covers throws in the component's OWN render body
    # (getData / getToggleControls / getResultsTable / renderHeader / renderEntries,
    # all synchronous), and the ReportRenderGuard error boundary covers throws in
    # DESCENDANTS (ResultViewButton, Toggle, Button), which reconcile after the
    # parent's render() has returned. Subcategory and SearchResults both inherit
    # this render(), so one edit covers the category view and the search view.
    # A boundary at App level was rejected: it would swallow rulegraph/statistics/
    # metadata failures the user can currently see, and its fallback would replace
    # the navbar, leaving no way to navigate to a working category.
    #
    # Inert when nothing throws: the boundary returns `this.props.children`, so it
    # emits no DOM node and the rendered output is the original element by IDENTITY.
    # Measured on the delivered report: two `insert` opcodes, zero `replace`, zero
    # `delete`. `key` tracks the category so
    # an errored boundary cannot persist across a category switch.
    #
    # The defs are gated on the render patch having landed, so a report whose
    # AbstractResults shape has drifted upstream is left byte-identical rather than
    # receiving an orphan <script>. Idempotent on both halves.
    if _GUARD_RENDER_NEW not in html_text:
        html_text = html_text.replace(_GUARD_RENDER_OLD, _GUARD_RENDER_NEW, 1)
    if _GUARD_RENDER_NEW in html_text and "class ReportRenderGuard extends React.Component" not in html_text:
        _guard_body = html_text.rfind("</body>")
        if _guard_body != -1:
            html_text = html_text[:_guard_body] + _GUARD_DEFS + html_text[_guard_body:]

    # 5. (retired) No category is injected into the sidebar, on any render path.
    # Deliberately NOT a reserved empty "Simulation Health (placeholder)" slot: the sidebar
    # lists exactly the keys of the `var categories` global, so an injected key with no
    # figures renders as an entry that opens an empty page. This holds whatever bundle_mode
    # is.

    # 6. showCategory auto-pop (idempotent: check before injecting)
    if _SHOW_CATEGORY_NEW not in html_text:
        html_text = html_text.replace(_SHOW_CATEGORY_OLD, _SHOW_CATEGORY_NEW, 1)

    # 6b. Toggle-cell fallback; the defect and the fix are described above
    # _TOGGLE_CELL_OLD. Its place before step 7 is not an ordering dependency: the
    # row-click delegate contains none of this step's OLD or NEW literals nor the
    # sentinel, and step 7 checks only for its own text, so the two steps give the same
    # document in either order whenever this one does not raise.
    #
    # MATCH GUARD, not merely an idempotency guard. `if NEW not in html_text` alone makes a
    # drifted OLD literal a SILENT no-op: .replace returns the input unchanged, NEW stays
    # absent, and every later invocation repeats the no-op forever -- indistinguishable from
    # a report that never needed the fix. Assert OLD is present exactly once whenever NEW is
    # absent, and RAISE rather than warn: a warning is swallowed in Snakemake's rule log.
    # The raise reaches production: none of the four render callers (analysis.py,
    # sensitivity_analysis.py, bundle/__init__.py and bundle/combined_snakefile_generator.py)
    # catches it, so a drift fails the render with this message. The un-surgeried report
    # Snakemake already wrote is left on disk, and the render does not return it as done.
    # TIER-1 GATE. The match guard below is correct only on a document that actually
    # carries the toggle machinery. This function is deliberately fragment-tolerant --
    # the render-guard tests in tests/test_report_renderer_skip_placeholder.py pin that a
    # document whose AbstractResults shape does not match receives neither the patch
    # nor an error -- so an UNGATED match guard converts that tested
    # inertness into a hard failure for the same document class. The two contracts do not
    # conflict; they partition the document space, and this gate is the partition:
    #
    #   tier 1 absent                          -> inert          (fragment)
    #   tier 1 present, exact literal absent   -> drift -> raise (match guard)
    #
    # Absent the gate, the three fragment fixtures in that test module raise -- five red
    # tests, all of them correct about the contract they defend.
    if _TOGGLE_FEATURE_SENTINEL in html_text:
        for _old, _new, _what in (
            (_TOGGLE_CELL_OLD, _TOGGLE_CELL_NEW, "renderEntries"),
            (_TOGGLE_CB_OLD, _TOGGLE_CB_NEW, "toggleCallback"),
            (_TOGGLE_SUPPRESS_OLD, _TOGGLE_SUPPRESS_NEW, "getData toggle promotion"),
        ):
            if _new in html_text:
                continue
            _n = html_text.count(_old)
            if _n != 1:
                raise ProcessingError(
                    f"toggle-cell fallback: expected exactly 1 occurrence of the {_what} "
                    f"lookup literal, found {_n}. The document carries the toggle "
                    f"machinery ({_TOGGLE_FEATURE_SENTINEL!r} is present) but not this "
                    "literal, so the bundled AbstractResults shape has DRIFTED rather "
                    "than being absent; re-read the literal from the freshly rendered "
                    "analysis_report.html (renderEntries uses 12 leading spaces, "
                    "toggleCallback 16)."
                )
            html_text = html_text.replace(_old, _new, 1)

    # 7. Row-click delegate at LAST </body>
    if _CLICK_DELEGATE not in html_text:
        _last_body = html_text.rfind("</body>")
        if _last_body != -1:
            html_text = html_text[:_last_body] + _CLICK_DELEGATE + html_text[_last_body:]

    # 8. Force initial App view to "metadata" (workflow_description landing).
    # App.constructor's original logic: default "rulegraph", promote to
    # "metadata" only when the metadata global is non-empty. The metadata
    # global is empty under the regeneration path (snakemake --report does
    # not repopulate it from workflow_description.rst.j2 the same way it
    # does on a full report run), so the conditional never fires and the
    # default-open lands on the DAG. Force the constructor's default to
    # "metadata" — the metadata view itself is rendered from a different
    # data source than the metadata global, so its content still surfaces.
    html_text = html_text.replace(
        'this.content = "rulegraph";',
        'this.content = "metadata";',
    )

    # 8b. Card-name humanization. Snakemake names each figure card after its output-file
    # stem, which is the canonical plot id and cannot be renamed because the on-disk layout
    # depends on it, and report() takes no card-name argument, so a pass over the rendered
    # HTML is the only place the name can change. Every card name goes through
    # report_plot_ids.humanize_plot_id, which parses the canonical plot-id grammar
    # (segments joined by "__", "." within a segment), so a figure from a new renderer or
    # model gets a readable name with no per-figure entry here.
    # Only the "name" field is rewritten. The record KEY, "filename", data_uri filename and
    # category path-references keep the stem, so links and downloads still resolve. The
    # charset excludes "/" and ":" so a path or URL never matches, and the base64 figure
    # blob carries no plain `"name": "{stem}.{ext}"` fragment.
    def _humanize_card_name(m):
        return '"name": "' + humanize_plot_id(m.group(1), member_labels, event_labels) + '"'

    html_text = re.sub(r'"name": "([A-Za-z0-9_.]+\.(?:html|png|svg))"', _humanize_card_name, html_text)

    # 9. Bundle-mode: drop Workflow + Statistics menu items, drop "General"
    # ListHeading (which would otherwise be an empty heading after the
    # menu-item drops).
    if bundle_mode:
        html_text = html_text.replace(
            'this.getMenuItem("Workflow", "share", this.showWorkflow),',
            "",
        )
        html_text = html_text.replace(
            'this.getMenuItem("Statistics", "chart", this.showStatistics),',
            "",
        )
        html_text = html_text.replace(
            'return e(\n                ListHeading,\n                { text: "General" }\n            )',
            "return null",
        )

    return html_text


def apply_post_process_surgery_to_zip(
    zip_path,
    bundle_mode: bool = False,
    navbar_text: str | None = None,
    category_order: list[str] | None = None,
    member_labels: dict[str, str] | None = None,
    event_labels: dict[str, str] | None = None,
) -> None:
    """Apply post-process surgery to `analysis_report/report.html` inside a zip.

    Extracts the zip to a tempdir, modifies the inner HTML in place, then
    re-zips back to the original path (overwriting). Idempotent: re-running
    on a surgery'd zip does not double-inject (per ``apply_post_process_surgery``
    semantics).

    Parameters
    ----------
    zip_path : pathlib.Path
        Path to the Snakemake-rendered ``analysis_report.zip``.
    """
    import shutil
    import tempfile
    import zipfile
    from pathlib import Path

    zip_path = Path(zip_path)
    if not zip_path.exists():
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)
        # Find the report page by searching the extracted tree. Snakemake 9.15.0 writes
        # it at `{zip stem}/report.html`; the search takes the first `report.html`
        # anywhere, so a result file that is also named report.html could be picked
        # instead, depending on traversal order. With no match the zip is left untouched.
        candidates = list(tmpdir_path.rglob("report.html"))
        if not candidates:
            return
        inner_html = candidates[0]
        modified = apply_post_process_surgery(
            inner_html.read_text(),
            bundle_mode=bundle_mode,
            navbar_text=navbar_text,
            category_order=category_order,
            member_labels=member_labels,
            event_labels=event_labels,
        )
        inner_html.write_text(modified)
        # Write the new archive beside zip_path and only then move it over the original,
        # so a failure while zipping leaves the original archive intact rather than
        # half-written. root_dir is the extraction tree, so the archive keeps Snakemake's
        # top-level `{zip stem}/` directory.
        new_zip_no_ext = zip_path.with_suffix("")
        archive_tmp = Path(
            shutil.make_archive(
                str(new_zip_no_ext) + ".surgery_tmp",
                "zip",
                root_dir=str(tmpdir_path),
            )
        )
        shutil.move(str(archive_tmp), str(zip_path))
