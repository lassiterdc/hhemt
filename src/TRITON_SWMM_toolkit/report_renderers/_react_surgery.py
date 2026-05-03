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

_CATEGORY_ORDER = {
    "Workflow Status": 1,
    "Errors and Warnings": 2,
    "Key Results": 3,
    "System Information": 4,
    "Simulation Health (placeholder)": 5,
    "Per Simulation Results": 6,
}

_ORDER_JS = "{" + ", ".join(f'"{k}": {v}' for k, v in _CATEGORY_ORDER.items()) + "}"

_PLACEHOLDER_INJECT = ', "Simulation Health (placeholder)": {"Reserved": []}'

_SHOW_CATEGORY_OLD = (
    'this.setView({ navbarMode: mode, category: category, subcategory: subcategory })\n'
    '    }'
)

_SHOW_CATEGORY_NEW = (
    'this.setView({ navbarMode: mode, category: category, subcategory: subcategory });\n'
    '        setTimeout(function(){\n'
    '            var tbl = document.querySelector("table.table-auto");\n'
    '            if (!tbl) return;\n'
    '            var firstRow = tbl.querySelector("tbody tr");\n'
    '            if (!firstRow) return;\n'
    '            var actionDiv = firstRow.querySelector("td.text-right > div.inline-flex");\n'
    '            if (!actionDiv) return;\n'
    '            var firstBtn = actionDiv.querySelector("a, button");\n'
    '            if (firstBtn) firstBtn.click();\n'
    '        }, 80);\n'
    '    }'
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


def apply_post_process_surgery(html_text: str) -> str:
    """Apply all React-bundle post-process replacements and return modified text.

    Idempotent: each replace is conditional on the OLD pattern still being
    present. Calling twice on the same text does not double-inject.

    Replacements applied:
      1. Browser-tab title "Snakemake Report" -> empty
      2. Drop the About menu item from the bundled JS (CSS-only hide infeasible)
      3. Replace bold "Snakemake" navbar span -> "TRITON-SWMM Toolkit"
      4. Patch category-sort comparator -> hardcoded category order
      5. Inject "Simulation Health (placeholder)" entry into categories dict
      6. Patch showCategory to auto-pop the first figure (setTimeout firstBtn.click)
      7. Inject row-click delegate at LAST `</body>` so clicks anywhere on a
         result row fire the hidden eye-icon button (paired with CSS in
         report.css that hides the eye-icon and styles rows as clickable)
    """
    # 1. Browser-tab title
    if "<title>Snakemake Report</title>" in html_text:
        html_text = html_text.replace(
            "<title>Snakemake Report</title>", "<title></title>"
        )

    # 2. Drop About menu item
    html_text = html_text.replace(
        'this.getMenuItem("About", "information-circle", this.props.app.showReportInfo),',
        "",
    )

    # 3. Navbar span text
    html_text = html_text.replace(
        'e(\n                        "span",\n                        { className: "font-bold mx-1" },\n                        "Snakemake"\n                    )',
        'e(\n                        "span",\n                        { className: "font-bold mx-1" },\n                        "TRITON-SWMM Toolkit"\n                    )',
    )

    # 4. Category-sort comparator
    html_text = html_text.replace(
        "(a, b) => a.localeCompare(b)",
        f"(a, b) => {{const ORDER = {_ORDER_JS}; return (ORDER[a] ?? 99) - (ORDER[b] ?? 99) || a.localeCompare(b);}}",
    )

    # 5. Placeholder category injection (idempotent: check before injecting)
    if _PLACEHOLDER_INJECT[2:] not in html_text:
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
            html_text = (
                html_text[:_last_body] + _CLICK_DELEGATE + html_text[_last_body:]
            )

    return html_text


def apply_post_process_surgery_to_zip(zip_path) -> None:
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
        modified = apply_post_process_surgery(inner_html.read_text())
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
