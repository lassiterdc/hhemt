"""Reviewer test for the T1c field of the per-step path-3 capture.

Chunk test: it guards ``TRITONSWMM_sim_post_processing._export_coupling_diagnostics`` in
``hhemt.process_simulation``. At the run's landing it is promoted into ``tests/test_coupling_diagnostics_ingest.py``
or deleted with its passing evidence kept in the run record. It carries no stored blob and no absolute count.

Every file is built here from the byte layout the solver's file format fixes (a 16-byte little-endian int32 header
``magic, version, value_width, n_items``, then records), with ``struct`` and the magic written as a literal, so a
wrong magic constant in production is not read back as right.
"""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import TRITONSWMM_sim_post_processing

STEP_MAGIC = 0x44505333  # "DPS3": swmm_node_path3_step.bin, item 1 of the solver's file format
NODE_NAMES = ["n1", "n2"]
RECORD_BYTES = 16 + 8 * len(NODE_NAMES)  # f64 simtime, f64 dt, f64[n]


N_NODES = len(NODE_NAMES)


def _header(*, magic=STEP_MAGIC, version=1, width=8, n_items=N_NODES) -> bytes:
    return struct.pack("<iiii", magic, version, width, n_items)


def _record(simtime: float) -> bytes:
    return struct.pack("<dd", simtime, 5.0) + struct.pack(f"<{len(NODE_NAMES)}d", 1.5, -2.5)


def _proc_with_run_output(tmp_path: Path, raw: bytes) -> TRITONSWMM_sim_post_processing:
    """A post-processing object for an analysis that asks for the per-step path-3 file, over a run that left ``raw``.

    The solver's sub-directory for a capture is not part of the interface, so the file sits one level down, where the
    toolkit finds it by name.
    """
    out_dir = tmp_path / "out_tritonswmm" / "swmm"
    out_dir.mkdir(parents=True)
    (out_dir / "swmm_node_path3_step.bin").write_bytes(raw)
    (out_dir / "swmm_node_diag.nodes").write_text("\n".join(NODE_NAMES) + "\n")
    proc = TRITONSWMM_sim_post_processing.__new__(TRITONSWMM_sim_post_processing)
    proc.scen_paths = SimpleNamespace(out_tritonswmm=tmp_path / "out_tritonswmm")
    proc._analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(
            toggle_coupling_diagnostics=False,
            swmm_diagnostic_links=None,
            swmm_diagnostic_node_path3_per_step=True,
        )
    )
    return proc


@pytest.mark.parametrize(
    ("raw", "refused"),
    [
        (_header() + _record(10.0) + _record(20.0), False),
        (b"\x00" * 16, True),
        (_header(magic=0x44504E33) + _record(10.0), True),
        (_header(version=2) + _record(10.0), True),
        (_header(width=4) + _record(10.0), True),
        (_header(n_items=5) + _record(10.0), True),
        (_header() + _record(10.0) + _record(20.0)[:-3], True),
    ],
    ids=[
        "valid-two-records",
        "header-of-zeros",
        "another-capture-magic",
        "version-2",
        "value-width-4",
        "width-differs-from-name-list",
        "ends-inside-a-record",
    ],
)
def test_the_per_step_path3_file_is_refused_unless_it_is_a_valid_file_of_its_kind(tmp_path, raw, refused) -> None:
    """Property: with the per-step path-3 field on, processing refuses a ``swmm_node_path3_step.bin`` that is not a
    well-formed file of its kind, and accepts a well-formed one, writing no store for either.

    Class: NEW CAPABILITY. The input class is every state a present file can be in, taken from the three things
    ``read_diag_header`` and the width and record-alignment checks refuse for the sibling captures: a header of
    another kind (wrong magic), a header of another format (version or value width), a header whose item count is not
    the name list's, and a body that ends inside a record. Plausible wrong implementation: a presence check that
    tests only that the file and the name list exist, which accepts all six invalid rows. Red under it, pasted in the
    reviewer's file.
    Reachability: the field description says the pin is part of the contract and no check covers it; a solver at
    another pin that reads the key and writes a different layout under the same file name, or a run whose write was
    cut short, leaves exactly these files, and the sibling captures are refused at the same point by their reader.
    Artifact: ``src/hhemt/process_simulation.py`` ``_export_coupling_diagnostics``, the
    ``expected_coupling_raw_captures`` loop, which never opens the file.
    Both-states anchor: the verdict turns on whether the file is well formed, which is a property of the bytes in the
    current tree and in any correct implementation; the first row is the control that a valid file passes, so a
    check that refuses everything also fails.
    Why required: the file is read later by a notebook and is gigabytes per event, so a bad file found there costs
    the run that wrote it; found here it costs nothing.
    Killed by: the presence-only check. Second correct implementation, materially different: read only the header
    and compare the byte size with the record width, without building a dataset; passes every row. A third: open the
    file through ``read_swmm_node_path3_step`` (a memory map, nothing copied) and close it; passes every row.
    The test asserts the refusal type and not its wording.
    Invocation: not isolatable from this tree; discrimination shown instead by the third implementation above,
    applied as a wrapper in a throwaway script, which greens every row.
    """
    proc = _proc_with_run_output(tmp_path, raw)

    if refused:
        with pytest.raises(ProcessingError):
            proc._export_coupling_diagnostics()
    else:
        proc._export_coupling_diagnostics()
    assert not list((tmp_path / "out_tritonswmm").glob("**/*.zarr"))
