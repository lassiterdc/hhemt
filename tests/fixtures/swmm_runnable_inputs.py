"""Input models for the runnable-SWMM derivation tests: one plain runnable model and a fixed forcing fill.

Both the coder's and the reviewer's derivation tests take their input only through this module, so the
cut-over of the synthetic builder edits ONE fixture. The runnable model is the synthetic builder's model
(``swmm_model.build_model``) with the forcing set for a fixed window; the rain series are INLINE ``[TIMESERIES]``
rows, because a
``FILE`` row naming an absent ``.dat`` makes ``swmm_open`` fail with ``ERROR 361`` and every arm would then fail
for a reason other than the property it tests.

The model is built once per process and held as text, so a test pays for it at most once.
"""

from __future__ import annotations

import functools
import tempfile
from collections.abc import Mapping
from pathlib import Path

# The forcing one event carries. Same keys, and same shapes, as the mapping
# ``swmm_utils.create_swmm_inp_from_template`` fills: the seven ``[OPTIONS]`` values and two row blocks.
FIXED_WINDOW: Mapping[str, str] = {
    "START_DATE": "01/01/2020",
    "START_TIME": "00:00:00",
    "REPORT_START_DATE": "01/01/2020",
    "REPORT_START_TIME": "00:00:00",
    "END_DATE": "01/01/2020",
    "END_TIME": "00:30:00",
    "REPORT_STEP": "00:00:10",
}
RAIN_GAUGE = "RG_synth"
SOURCE_RAINGAGES = f"{RAIN_GAUGE} INTENSITY 00:10:00 1.0 TIMESERIES {RAIN_GAUGE}"
SOURCE_TIMESERIES = f"{RAIN_GAUGE} 01/01/2020 00:00 0.0"
# The series the matched outfall of record 0 reads in a role that keeps the outfall, and the outfall it names.
LEVEL_SERIES = "hhemt_water_level_0"
MATCHED_OUTFALL = "SYNTH_OUTFALL"

# A different event: other values for every owned key, so a rewrite that leaves any of them alone is visible.
EVENT_FILL: Mapping[str, str] = {
    "START_DATE": "03/04/2021",
    "START_TIME": "06:00:00",
    "REPORT_START_DATE": "03/04/2021",
    "REPORT_START_TIME": "06:00:00",
    "END_DATE": "03/04/2021",
    "END_TIME": "07:00:00",
    "REPORT_STEP": "00:00:30",
    "RAINGAGES": f"{RAIN_GAUGE} INTENSITY 00:20:00 1.0 TIMESERIES {RAIN_GAUGE}",
    "TIMESERIES": (
        f"{RAIN_GAUGE} 03/04/2021 06:00 0.0\n\n{RAIN_GAUGE} 03/04/2021 06:20 80.0\n\n"
        f"{RAIN_GAUGE} 03/04/2021 06:40 0.0\n\n{LEVEL_SERIES} 03/04/2021 06:00 0.2\n\n"
        f"{LEVEL_SERIES} 03/04/2021 07:00 0.9"
    ),
}

# Sections the synthetic model does not carry, appended to settle the three rows of the section table that SWMM
# 5.2.4's own reader does not decide: SYMBOLS (names a rain gauge), ADJUSTMENTS (one row names a subcatchment) and
# TEMPERATURE (a series the hydraulics role must still resolve). A second ``[TIMESERIES]`` block is deliberate:
# SWMM allows a section to recur and the module has to treat each occurrence.
EXTRA_SECTIONS = """
[PATTERNS]
npat  MONTHLY 1.0 1.0 1.0 1.0 1.0 1.0
npat  1.0 1.0 1.0 1.0 1.0 1.0

[TIMESERIES]
tmp_ts 01/01/2020 00:00 10.0
tmp_ts 01/01/2020 00:30 12.0

[TEMPERATURE]
TIMESERIES tmp_ts

[ADJUSTMENTS]
RAINFALL 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
N-PERV   S1 npat

[SYMBOLS]
RG_synth 10.0 10.0
"""


# The objects of the matched variable-water-level outfall the synthetic builder adds (the outfall, the conduit
# from `sewer_outflow`, and the series the outfall reads). The derivation tests state every property over the model
# with ONE outfall, so they take the model without these rows; the tests of the matched outfall take it whole.
_MATCHED_OBJECTS = {"synth_outfall", "c_out", "synth_level"}


def _without_matched_outfall(text: str) -> str:
    kept = []
    for line in text.splitlines(keepends=True):
        tokens = line.split(";")[0].split()
        if tokens and tokens[0].lower() in _MATCHED_OBJECTS:
            continue
        kept.append(line)
    return "".join(kept)


@functools.cache
def runnable_source_text(matched_outfall: bool = False) -> str:
    """The runnable model: the synthetic builder's model, with ``FIXED_WINDOW`` and one inline rain series.

    ``matched_outfall=True`` keeps the builder's matched variable-water-level outfall; the default drops its rows, so
    the model has the one outfall the derivation tests were written against.
    """
    from hhemt.swmm_runnable import rewrite_forcing
    from hhemt.synthetic_model import SyntheticModelParams
    from hhemt.synthetic_model.swmm_model import build_model

    with tempfile.TemporaryDirectory() as tmp:
        text = build_model(SyntheticModelParams(), Path(tmp)).read_text(encoding="utf-8")
    if not matched_outfall:
        text = _without_matched_outfall(text)
    return rewrite_forcing(text, {**FIXED_WINDOW, "RAINGAGES": SOURCE_RAINGAGES, "TIMESERIES": SOURCE_TIMESERIES})


def runnable_source_with_extras() -> str:
    """The runnable model plus :data:`EXTRA_SECTIONS`."""
    return runnable_source_text() + EXTRA_SECTIONS
