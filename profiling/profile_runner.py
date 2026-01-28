"""Profile retrieve_SWMM_outputs_as_datasets using reference test inputs."""

from pathlib import Path

from TRITON_SWMM_toolkit.swmm_output_parser import retrieve_SWMM_outputs_as_datasets


def main() -> None:
    ref_dir = (
        Path(__file__).resolve().parents[1] / "test_data" / "swmm_refactoring_reference"
    )
    ref_inp = ref_dir / "hydraulics.inp"
    ref_rpt = ref_dir / "hydraulics.rpt"

    retrieve_SWMM_outputs_as_datasets(ref_inp, ref_rpt)


if __name__ == "__main__":
    main()
