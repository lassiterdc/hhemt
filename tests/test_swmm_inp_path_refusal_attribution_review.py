"""Which file a SWMM path-length refusal blames, and which operation it names.

``create_swmm_inp_from_template`` refuses an over-long FILE path and an over-long
binary output name. The refusal's message already contains the offending path in its
reason, so a test that searches the message for the path cannot tell whether the
exception's ``filepath`` attribute -- printed as the message's "File:" line -- names
that path or some other file, such as the input being written. These tests pin the
attribute itself, over both refusal kinds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hhemt.exceptions import ProcessingError
from hhemt.swmm_utils import create_swmm_inp_from_template
from tests.test_swmm_inp_file_path_length import LIMIT, _out_case, _path_of, _scenario


@pytest.fixture
def template(tmp_path: Path) -> Path:
    """A SWMM template holding only the two placeholders the function always fills."""
    t = tmp_path / "template.inp"
    t.write_text("[TIMESERIES]\n${TIMESERIES}\n\n[RAINGAGES]\n${RAINGAGES}\n")
    return t


@pytest.mark.parametrize("kind", ["file_path", "output_name"])
def test_a_path_length_refusal_names_the_offending_path_as_its_file(tmp_path, template, kind):
    """Property: the refusal's ``filepath`` is the over-long path and its ``operation``
    is ``create_swmm_inp_from_template``.

    Class: both refusal kinds -- a FILE path written into the input (here a rain gauge)
    and the binary output name SWMM is handed (here the full model's ``.out``) -- each
    at 260 bytes, with a short input name, so the input and the offending path differ.

    Why required: the "File:" line of the message is built from ``filepath``. Blaming
    the input being written sends the user to a file that is fine and whose name may be
    short, while the reason line names another; and ``operation`` is the one field that
    says which toolkit step refused.

    What kills it: passing ``filepath=Path(destination)`` (the input) or the template,
    or naming a different ``operation``. Measured: with both refusals rewritten to
    ``filepath=Path(destination)`` and ``operation="swmm_path_check"``, every test in
    ``tests/test_swmm_inp_file_path_length.py`` still passes, because each searches only
    the message text, which still carries the path through the reason.

    A second correct implementation under which it still passes: measuring each path
    while its FILE line is formatted, or selecting the output name through a lookup
    table, provided the raise names the measured path.
    """
    if kind == "file_path":
        offending = _path_of(tmp_path, LIMIT + 1, suffix=".dat")
        scenario = _scenario(tmp_path, rain=offending)
        destination = tmp_path / "out" / "model.inp"
    else:
        scen_paths, destination, offending = _out_case(tmp_path, "full", LIMIT + 1)
        scenario = _scenario(tmp_path, scen_paths=scen_paths)

    with pytest.raises(ProcessingError) as excinfo:
        create_swmm_inp_from_template(scenario, template, destination)

    assert excinfo.value.filepath == Path(offending), (excinfo.value.filepath, offending)
    assert excinfo.value.operation == "create_swmm_inp_from_template", excinfo.value.operation
