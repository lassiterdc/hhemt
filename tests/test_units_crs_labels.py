"""Horizontal axis labels must follow the unit the CRS itself declares.

EPSG 32147 (NAD83 / Virginia South) is metric and EPSG 2284 (the same zone in
US survey feet) is not; a label table kept by hand once called both feet.
"""

from __future__ import annotations

import pytest

from hhemt import units


@pytest.mark.parametrize(
    ("epsg", "label"),
    [(32147, "m"), (2284, "ft"), (None, "m"), (999999, "m")],
)
def test_crs_horizontal_unit_label(epsg: int | None, label: str) -> None:
    assert units._crs_horizontal_unit_label(epsg) == label


def test_axis_labels_carry_the_crs_unit() -> None:
    assert units.easting_axis_label(32147) == "Easting (m)"
    assert units.northing_axis_label(2284) == "Northing (ft)"
