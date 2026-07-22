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
