"""Produce the standalone-SWMM reference pair ``hydraulics.rpt`` and ``hydraulics.out`` beside this script.

The coupled run that wrote ``../swmm_refactoring_reference/hydraulics.rpt`` kept no binary output, and that report
cannot be reproduced from the model file alone (its External Inflow came from the TRITON exchange). So the equivalence
tests of the binary-output reader use this pair instead: ONE stock SWMM run of the committed ``hydraulics.inp``,
which yields a report WITH both time-series sections and the binary output of the same run. Stock SWMM only, no TRITON,
no compile; it takes under a second on a laptop.

``swmm_run`` is not used: it prints no time series into the report. The loop below is the stepwise equivalent.

Usage: ``python make_reference.py`` (writes ``hydraulics.rpt`` and ``hydraulics.out`` here). The expected series
stores ``SWMM_node_tseries.zarr`` and ``SWMM_link_tseries.zarr`` here were then written by the report reader that
preceded the binary-output reader, from THIS ``hydraulics.rpt``, so all four files come from one run.
"""

from pathlib import Path

from swmm.toolkit import solver

HERE = Path(__file__).resolve().parent
INP = HERE.parent / "swmm_refactoring_reference" / "hydraulics.inp"


def main() -> None:
    rpt, out = HERE / "hydraulics.rpt", HERE / "hydraulics.out"
    solver.swmm_open(str(INP), str(rpt), str(out))
    solver.swmm_start(1)
    while solver.swmm_step() > 0:
        pass
    solver.swmm_end()
    solver.swmm_report()
    solver.swmm_close()


if __name__ == "__main__":
    main()
