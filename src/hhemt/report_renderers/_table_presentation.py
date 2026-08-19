"""Engine-neutral header-presentation policy for every table the report emits.

The report renders tables through THREE mutually-incompatible engines:

  * Tabulator (CDN)      -- ``scenario_status_appendix`` via ``_tabulator_defaults``.
  * Plotly ``go.Table``  -- ``eda._config_diff``, ``eda._dem_resolution_plots``,
                            ``report_renderers.cross_experiment_intercomparison_maps``.
  * Static HTML + inline JS shim -- ``metadata``, ``errors_and_warnings``,
                            ``cross_experiment_errors_and_warnings``,
                            ``cross_experiment_compatibility``.

Those three cannot share an IMPLEMENTATION: a ``go.Table`` is a trace inside a figure,
not an HTML document, and the static pages carry their own inline shim. What they can
and must share is the POLICY -- how a header string is spelled, how wide it needs to be,
and where it sits in its cell. This module is the single declaration of that policy; no
call site writes an alignment token, a character-width constant, or an
underscore-stripping rule of its own.

Same inversion as ``figure_panels`` one layer down: the caller declares the header
LABELS and this module returns the geometry.
"""

from __future__ import annotations

#: Header alignment for EVERY table family. Iteration-12 item {2} asked for centred
#: headers on the metadata tables AND on the compute-config diff maps; naming it once
#: is what keeps the two families from drifting apart again.
HEADER_ALIGN: str = "center"

#: Width policy, in the units each engine accepts. Lifted verbatim from
#: ``_tabulator_defaults._estimate_column_width_px`` (iteration 9.1), which is the only
#: place in the report where a header-width rule was ever written down. The +CHROME
#: reserve is what guarantees the "small amount of space between text and the header
#: edge" iteration-12 item {1} asks for; on the Tabulator side it additionally covers the
#: sort-arrow padding and the filter-trigger button.
COL_WIDTH_MIN_PX: int = 90
COL_WIDTH_MAX_PX: int = 320
COL_WIDTH_PX_PER_CHAR: int = 9
COL_WIDTH_CHROME_PX: int = 56


def humanize_header(label: str) -> str:
    """Return the display spelling of a column header.

    Iteration-12 item {3}. ``n_resumes`` is ONE token to every line-breaking algorithm
    in every engine here, so no column width short of the full string prevents it from
    overflowing its cell -- the observed defect. Replacing the underscore with a space
    introduces the break opportunity that lets the header wrap instead of overflow.

    Applied to the DISPLAY string only. Field names, sort keys, filter ``data-col``
    indices and clipboard/TSV output are unchanged, so nothing downstream keys off the
    transformed spelling.
    """
    return str(label).replace("_", " ")


def header_width_px(label: str) -> int:
    """Fixed pixel width for a column, from its (humanized) header title.

    Header-title-length driven: body values are covered by horizontal scroll and by
    per-engine wrapping, and a body-driven rule would size every column to its widest
    cell. Clamped to ``[COL_WIDTH_MIN_PX, COL_WIDTH_MAX_PX]``.
    """
    raw = COL_WIDTH_PX_PER_CHAR * len(humanize_header(label)) + COL_WIDTH_CHROME_PX
    return max(COL_WIDTH_MIN_PX, min(COL_WIDTH_MAX_PX, raw))


def plotly_columnwidth(labels: list[str]) -> list[float]:
    """``go.Table(columnwidth=...)`` weights for the same policy.

    Plotly's ``columnwidth`` is a RELATIVE weight vector, not pixels: plotly normalizes
    it by its own sum and distributes the trace's domain width accordingly. The px policy
    is therefore expressed as a ratio here rather than converted -- the shape of the
    distribution is what carries over, and it is the shape that was missing (six of the
    eight ``go.Table`` call sites declare no ``columnwidth`` at all, so plotly divides
    the domain EQUALLY and a long header is guaranteed to clip beside a short one).
    """
    if not labels:
        return []
    return [float(header_width_px(label)) for label in labels]


def plotly_table_header(labels: list[str], **overrides: object) -> dict:
    """The ``go.Table(header=...)`` dict for a list of column labels.

    Applies the humanized spelling and the shared alignment. Every other header key
    (``fill_color``, ``font``, ``height``) stays the call site's business and is passed
    through ``overrides`` unchanged -- this module owns header GEOMETRY and SPELLING, not
    the palette, which is brand-theme-sourced elsewhere.
    """
    header: dict = {
        "values": [humanize_header(label) for label in labels],
        "align": HEADER_ALIGN,
    }
    header.update(overrides)
    return header


#: Iteration-12 item {12}. The capability set an interactive report table provides.
#:
#: There are TWO implementations of this set and there must be: the appendix table is a
#: Tabulator grid (`_tabulator_defaults`), and the static pages carry an inline
#: vanilla-JS shim. What the mandate forbids is two DECLARATIONS of what the set
#: contains, not two ways of drawing it -- so the set is named once, here, and each
#: implementation is tested against this tuple.
#:
#: The two implementations reach parity when every entry below is satisfied by both.
#: Measured at iteration 12: Tabulator satisfies 6/6; the static shim satisfies 3/6
#: (sort, per-column filter, column visibility) and is missing compound filter operators,
#: copy-to-clipboard, and persisted state reset.
#:
#: RETIREMENT PATH: when inline/vendored Tabulator lands (`js_mode="inline"` in
#: `_tabulator_defaults.build_html_document` is currently a warn-and-fall-back stub), the
#: static shim can be deleted and this tuple collapses to one implementation.
TABLE_CAPABILITIES: tuple[str, ...] = (
    "sort",                 # click a header to sort; a declared default sort on load
    "filter_per_column",    # one filter control per column, ANDed across columns
    "filter_compound",      # >1 criterion per column with an explicit AND/OR connector
    "column_visibility",    # per-column show/hide, with group-level show-all/hide-all
    "copy_table",           # copy the filtered rows x visible columns to the clipboard
    "reset_state",          # clear persisted table state and re-render
)
