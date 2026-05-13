"""Master-Snakefile byte-identity test (Plan Phase 2 Validation Plan item 4).

Asserts that the source-side ``SnakemakeWorkflowBuilder.generate_snakefile_content``
and ``SensitivityAnalysisWorkflowBuilder.generate_master_snakefile_content``
emit byte-identical Snakefile output to the captured pre-refactor goldens
when ``static_backend="matplotlib"`` (the pre-Phase-5 default behavior).

If byte-identity fails, the Plan Phase 2 refactor (VMS-5 module-level
helpers + VMS-6 plot-rule wrapper rewrites) introduced silent drift and
must be fixed before merge — NOT papered over by regenerating the
golden.

Note: VMS-6's static_backend kwarg is not yet threaded through
generate_*_content's signature; the methods' internal default of
"matplotlib" is what matches the goldens.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.test_case_catalog import Local_TestCases  # noqa: E402

from TRITON_SWMM_toolkit.sensitivity_analysis import (  # noqa: E402
    TRITONSWMM_sensitivity_analysis,
)
from TRITON_SWMM_toolkit.workflow import (  # noqa: E402
    SensitivityAnalysisWorkflowBuilder,
    SnakemakeWorkflowBuilder,
)

GOLDENS_DIR = Path(__file__).parent / "fixtures" / "golden_snakefiles"

# Pin sys.executable for byte-identity comparison: pytest's shebang resolves
# to `python3.11` whereas direct invocation resolves to `python`. The
# captured goldens use `python` (symlink-preserved); pin to match.
@pytest.fixture(autouse=True)
def _pin_sys_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = re.sub(r"/python3\.\d+$", "/python", sys.executable)
    monkeypatch.setattr(sys, "executable", pinned)


def _unified_diff_excerpt(want: str, got: str, max_lines: int = 80) -> str:
    import difflib

    diff = list(
        difflib.unified_diff(
            want.splitlines(keepends=True),
            got.splitlines(keepends=True),
            fromfile="golden",
            tofile="emitted",
            n=2,
        )
    )
    return "".join(diff[:max_lines])


def test_multi_sim_snakefile_byte_identity() -> None:
    """Source-side multi-sim Snakefile byte-identical to golden."""
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False, download_if_exists=False
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    got = builder.generate_snakefile_content()
    want = (GOLDENS_DIR / "multi_sim.Snakefile.golden").read_text()
    assert got == want, _unified_diff_excerpt(want, got)


def test_master_snakefile_byte_identity() -> None:
    """Source-side sensitivity-master Snakefile byte-identical to golden."""
    tc = Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=False, download_if_exists=False
    )
    sens = TRITONSWMM_sensitivity_analysis(tc.analysis)
    builder = SensitivityAnalysisWorkflowBuilder(sens)
    got = builder.generate_master_snakefile_content()
    want = (GOLDENS_DIR / "sensitivity_master.Snakefile.golden").read_text()
    assert got == want, _unified_diff_excerpt(want, got)
