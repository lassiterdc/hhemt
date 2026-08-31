"""Post-process surgery applied to Snakemake-rendered report HTML.

Snakemake's `--report` engine produces a React-bundled `report.html`. Several
behaviors and visual elements are baked into that bundle and cannot be
addressed via the report-stylesheet alone (e.g., the JS click-handler that
opens figures, the category-sort comparator, the About menu item, the
navbar text). This module string-replaces those baked-in pieces.

Applied to:
  - Single-file HTML output (`format="html"`) — directly on the rendered file.
  - Zip output (`format="zip"`) — on `analysis_report/report.html` inside the
    zip, then re-zipped. Without this, zip mode renders the eye-icon-hiding
    CSS but lacks the JS click-delegate that makes rows clickable, leaving
    figure tables with no clickable affordance (rows look interactive via
    CSS but click handlers are absent).

The replacements are idempotent: reapplying does not double-inject
(checks before each replace).
"""

from __future__ import annotations

import re

from hhemt.report_plot_ids import humanize_plot_id
from hhemt.exceptions import ProcessingError

# Historical default category order (used when no category_order is passed —
# byte-identical for non-passing callers, mirroring navbar_text's None default).
_DEFAULT_CATEGORY_ORDER = (
    "Workflow Status",
    "Errors and Warnings",
    "Key Results",
    "System Information",
    "Simulation Health (placeholder)",
    "Per Simulation Results",
)


def _order_js(category_order: tuple[str, ...] | list[str]) -> str:
    """Build the JS ORDER-dict literal from an ordered category list (1-indexed)."""
    body = ", ".join(f'"{k}": {i}' for i, k in enumerate(category_order, start=1))
    return "{" + body + "}"


_PLACEHOLDER_INJECT = ', "Simulation Health (placeholder)": {"Reserved": []}'

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

_GUARD_RENDER_OLD = (
    "    render() {\n"
    "        if (this.state.data.toggleLabels.size > 0) {"
)

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
    // measured on the delivered 150,109,331-byte report, `bg-red-50`, `text-red-700`,
    // `border-red-700`, `font-mono`, `text-sm` and `font-bold` each had ZERO CSS
    // selector occurrences and appeared exactly once in the document (here), while
    // Snakemake's own `inline-flex` / `text-right` had 5 and 18 rules. The panel
    // therefore rendered entirely unstyled and inherited ambient colour, which is why
    // it was triaged as a BLANK PAGE across three rounds. Any class-based fix has the
    // same defect regardless of which class is chosen; an inline style needs no
    // stylesheet. Applies to every future injected VISUAL surface -- the behaviour-only
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
# Fixing the label emission was tried once already (step 4b removed the same 50/50
# accident from `figure`) and the collision reappeared on `models`; protecting `models`
# by reordering would put `figure` back in the toggle slice. This guards the LOOKUP
# instead, which closes the class rather than the instance. Behaviour when the cell IS
# populated is byte-identical; when it is not, the row opens its own figure -- which is
# the desired result for an unpaired entry. Idempotent: the old literals are gone after
# the first pass.
_TOGGLE_CELL_OLD = (
    "            let entryPath = data.entries.get(arrayKey(entryLabels))"
    ".get(arrayKey(toggleLabels));"
)

_TOGGLE_CELL_NEW = (
    "            let _cell = data.entries.get(arrayKey(entryLabels));\n"
    "            let entryPath = _cell.get(arrayKey(toggleLabels));\n"
    "            if (entryPath === undefined) { entryPath = _cell.values().next().value; }"
)

_TOGGLE_CB_OLD = (
    "                let targetPath = _this.state.data.entries.get(arrayKey(entryLabels))"
    ".get(arrayKey(toggleLabels));"
)

_TOGGLE_CB_NEW = (
    "                let _cbCell = _this.state.data.entries.get(arrayKey(entryLabels));\n"
    "                let targetPath = _cbCell.get(arrayKey(toggleLabels));\n"
    "                if (targetPath === undefined) { targetPath = _cbCell.values().next().value; }"
)

#: TIER-1 FEATURE SENTINEL for the step-6b match guard. Answers "does this document
#: carry the toggle machinery step 6b patches?", which is the question that separates
#: a document legitimately WITHOUT the lookup (a fragment fixture; VMS-5 inertness)
#: from one whose lookup literal has DRIFTED (must fail loudly).
#:
#: `arrayKey` is chosen because it is a module-level HELPER that both OLD literals call
#: twice, so it is a strict WEAKENING of them along every axis that drifts -- indentation,
#: a refactor of the `.get().get()` chain, or a rename of `renderEntries` itself -- and it
#: disappears only when the toggle feature is gone upstream, which is exactly when
#: inertness is correct. The sentinel and the legitimate-absence condition are the same
#: condition; that is what makes it a sentinel rather than a heuristic.
#:
#: Measured 2026-08-13 -- 0 occurrences in all three VMS-5 fixtures, 7 in each delivered
#: real report. Do NOT substitute `AbstractResults` (1 in the fixtures) or `toggleLabels`
#: (1, at the fixture body's `.size > 0` line): neither separates the classes.
#:
#: Every constant this module INJECTS is sentinel-free (verified), so a second pass cannot
#: flip tier 1 from absent to present and raise on a document the first pass left alone.
# Toggle SUPPRESSION (I7-6). The `models` label is promoted to a two-option radio pair by
# `AbstractResults.getData`, rendered as `MODELS` by Tailwind's `uppercase` on the label
# span (`toggle.js`) -- which is why a grep of the delivered report for "Models" returns
# only library source. The control is a no-op by construction: its two values are two
# DIFFERENT figure families, so selecting either re-opens the row's own figure via the
# lookup guard above.
#
# Removing the `models` LABEL was rejected: `labels.slice(1)` would then leave `figure` as
# the sole candidate, and `figure` is the label whose 50/50 split caused this same defect
# before step 4b (see the note at the head of this block). Suppressing PROMOTION closes the
# class; removing one label moves it. One character, and the literal occurs exactly once in
# the delivered report (measured 2026-08-14: 1).
#
# Idempotent: after the first pass the OLD literal is gone, so a second pass is inert.
_TOGGLE_SUPPRESS_OLD = "        if (toggleLabels.size > 1) {"
_TOGGLE_SUPPRESS_NEW = "        if (toggleLabels.size > 0) {"

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

    Idempotent: each replace is conditional on the OLD pattern still being
    present. Calling twice on the same text does not double-inject.

    Replacements applied:
      1. Browser-tab title "Snakemake Report" -> empty
      2. Drop the About menu item from the bundled JS (CSS-only hide infeasible)
      3. Replace bold "Snakemake" navbar span -> ``navbar_text`` (the report's
         upper-left brand text). When ``navbar_text`` is ``None``, the historical
         literal "H&H Ensemble Modeling Toolkit" is used (byte-identical for non-passing
         callers). The facades source it from brand_theme.upper_left_text (ADR-7),
         defaulting to analysis_id.
      4. Patch category-sort comparator -> config-driven category order
         (``category_order``; None falls back to the historical default,
         byte-identical for non-passing callers)
      5. Inject "Simulation Health (placeholder)" entry into categories dict
      6. Patch showCategory to auto-pop the first figure (setTimeout firstBtn.click)
      7. Inject row-click delegate at LAST `</body>` so clicks anywhere on a
         result row fire the hidden eye-icon button (paired with CSS in
         report.css that hides the eye-icon and styles rows as clickable)
      8. Force the App's initial ``content`` to ``"metadata"`` (the
         workflow_description landing page) rather than ``"rulegraph"`` (the
         DAG). The home-icon click handler already targets this view; the
         change makes the default-open match. Unconditional — applies to both
         source-side and bundle-side reports.
      9. If ``bundle_mode=True``: drop the Workflow menu item, the Statistics
         menu item, and the "General" ``ListHeading`` from the bundled JS.
         Used by ``Bundle.regenerate_report`` because a bundle's regeneration
         Snakefile only describes plot rules + render_report — the Workflow
         and Statistics panels in a bundle-regenerated report describe only
         the regeneration DAG (no production-analysis DAG), which is "useless"
         per user feedback; and the "General" heading is empty once Workflow
         + Statistics + About are gone. About-drop (step 2) is unconditional
         and applies to source-side too. NOT applied to source-side reports
         where Workflow + Statistics describe real workflow content + runtime.
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

    # 4. Category-sort comparator. category_order is config-driven (ADR-5);
    # None falls back to the historical default (byte-identical for non-passing
    # callers). Idempotent: after the first pass the localeCompare literal is
    # gone, so re-application is a no-op.
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
    # occurring in exactly half the results -- today that is `figure` on the coupled
    # per-sim category (28 Conduit flow + 28 Peak flood depth = 56). VMS-2 removes that
    # trigger at the source; this removes the engine dependency for any future category
    # that hits the same 50/50 accident. Idempotent: the old literal is gone after the
    # first pass.
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
    # Measured on the delivered 33,221,667-byte report: two `insert` opcodes, zero
    # `replace`, zero `delete`, +1583 bytes (0.0048%). `key` tracks the category so
    # an errored boundary cannot persist across a category switch.
    #
    # The defs are gated on the render patch having landed, so a report whose
    # AbstractResults shape has drifted upstream is left byte-identical rather than
    # receiving an orphan <script>. Idempotent on both halves.
    if _GUARD_RENDER_NEW not in html_text:
        html_text = html_text.replace(_GUARD_RENDER_OLD, _GUARD_RENDER_NEW, 1)
    if (
        _GUARD_RENDER_NEW in html_text
        and "class ReportRenderGuard extends React.Component" not in html_text
    ):
        _guard_body = html_text.rfind("</body>")
        if _guard_body != -1:
            html_text = html_text[:_guard_body] + _GUARD_DEFS + html_text[_guard_body:]

    # 5. Placeholder category injection (idempotent: check before injecting).
    # F2 (v9): suppress the empty "Simulation Health (placeholder)" reserved slot in
    # bundle_mode (combined + single-bundle regenerated reports) — it is meaningless chrome
    # there, and in a cross-experiment report it is actively confusing. This mirrors the
    # bundle_mode chrome-strip of Workflow/Statistics/General below (step 9). Source-side
    # reports (bundle_mode=False) keep the reserved slot unchanged.
    if not bundle_mode and _PLACEHOLDER_INJECT[2:] not in html_text:
        html_text = html_text.replace(
            "var categories = {",
            "var categories = {" + _PLACEHOLDER_INJECT[2:] + ",",
            1,
        )

    # 6. showCategory auto-pop (idempotent: check before injecting)
    if _SHOW_CATEGORY_NEW not in html_text:
        html_text = html_text.replace(_SHOW_CATEGORY_OLD, _SHOW_CATEGORY_NEW, 1)

    # 6b. Toggle-cell fallback. Must run BEFORE the row-click delegate injection so a
    # delegated click can never reach an undefined result path. See _TOGGLE_CELL_OLD.
    #
    # MATCH GUARD, not merely an idempotency guard. `if NEW not in html_text` alone makes a
    # drifted OLD literal a SILENT no-op: .replace returns the input unchanged, NEW stays
    # absent, and every later invocation repeats the no-op forever -- indistinguishable from
    # a report that never needed the fix. Assert OLD is present exactly once whenever NEW is
    # absent, and RAISE rather than warn: a warning is swallowed in Snakemake's rule log.
    # TIER-1 GATE. The match guard below is correct only on a document that actually
    # carries the toggle machinery. This function is deliberately fragment-tolerant --
    # VMS-5 pins that a document whose AbstractResults shape does not match receives
    # neither the patch nor an error -- so an UNGATED match guard converts that tested
    # inertness into a hard failure for the same document class. The two contracts do not
    # conflict; they partition the document space, and this gate is the partition:
    #
    #   tier 1 absent                          -> inert          (VMS-5)
    #   tier 1 present, exact literal absent   -> drift -> raise (match guard)
    #
    # Absent the gate, the three VMS-5 fixtures raise -- five red tests, all of them
    # correct about the contract they defend.
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

    # 8b (iter-3): deterministic grammar-driven card-name humanization. Snakemake derives
    # the figure-card display name from the OUTPUT-FILENAME stem (= the canonical plot-id,
    # layout-relevant / un-renameable), and there is no report() kwarg for the card name,
    # so a post-render pass is the only lever. This replaces the hardcoded n_devices band-aid
    # with ONE deterministic transform over EVERY card name via the ADR-2 grammar humanizer
    # (report_plot_ids.humanize_plot_id) — model-fungible, never another per-figure hardcode.
    # The record KEY, "filename", data_uri filename, and category path-references are left
    # intact so links/downloads keep working; the charset excludes "/" and ":" so paths/URLs
    # never match, and the base64 figure blob carries no plain `"name": "<stem>.ext"` fragment.
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
        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)
        # Locate the inner report.html (Snakemake's zip layout is
        # `analysis_report/report.html` — the parent dir name matches the
        # zip's stem). Robust fallback: glob for the first report.html.
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
        # Re-zip. shutil.make_archive writes `<base>.zip` from `root_dir`.
        # Use a tempfile alongside zip_path then atomic-rename to avoid
        # leaving a half-written archive on errors.
        new_zip_no_ext = zip_path.with_suffix("")
        # make_archive returns the path it wrote; we then overwrite zip_path.
        # Snakemake's zip has a top-level dir named after the report stem;
        # preserve that by zipping the tmpdir's contents (root_dir=tmpdir).
        archive_tmp = Path(
            shutil.make_archive(
                str(new_zip_no_ext) + ".surgery_tmp",
                "zip",
                root_dir=str(tmpdir_path),
            )
        )
        shutil.move(str(archive_tmp), str(zip_path))
