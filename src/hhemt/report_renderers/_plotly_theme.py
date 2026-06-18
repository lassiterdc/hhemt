"""Project-level Plotly theme + colorway anchored to the Okabe-Ito CVD-safe palette.

Imported by every Plotly renderer in `report_renderers/` (Phases 2-5) so all 4
Plotly figures carry consistent typography, gridlines, and qualitative palette.
Plotly's default `'plotly'` template renders a gridded gray background that
reads as 'draft / informal' per the data-visualization specialist substrate
identity §III.9 — override on every figure via `fig.update_layout(template=...)`.

Tabulator widgets in Phases 6-7 reuse OKABE_ITO_COLORWAY for any color-coded
formatters (e.g., status-flag highlighting in scenario_status_appendix).
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Okabe-Ito CVD-safe 8-color palette — same anchor as
# SensitivityReportConfig.palette in config/report.py.
OKABE_ITO_COLORWAY: list[str] = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
]

# Project-scope template extending plotly_white. Registered at import time
# so renderers can reference by name: fig.update_layout(template="triton_journal").
_journal_template = pio.templates["plotly_white"]
_journal_template.layout.colorway = tuple(OKABE_ITO_COLORWAY)
_journal_template.layout.font = go.layout.Font(family="Helvetica, Arial, sans-serif", size=12)
_journal_template.layout.title.font = go.layout.title.Font(size=14)
pio.templates["triton_journal"] = _journal_template

TRITON_JOURNAL_TEMPLATE_NAME: str = "triton_journal"
