"""Container-mode workflow-status detail: name the image, not a compile that never ran.

In container mode the setup rules skip both compiles because the SIF carries the solver
binary, so the setup phase's "compiled" detail used to report a step that never ran. These
two nodes pin both halves of the replacement: container mode names the image, and native
mode keeps the compile line the operator reads today.

Neither node compiles or runs a solver: both read ``get_workflow_status()``, which reads the
logs and the config.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.test_case_catalog import Local_TestCases  # noqa: E402

from hhemt.config.hpc_system import ContainerSpec  # noqa: E402

# A non-None hpc system config (all hpc_* selectors null), so the container block can be
# attached — the same fixture the container Snakefile guards use.
EXAMPLE_HPC_CONFIG = Path(__file__).parent / "fixtures" / "hpc_system_config_test.yaml"


def _synthetic_multi_sim_case():
    """A constructed synthetic case, with the target partition set so container mode validates."""
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    tc.analysis.cfg_analysis.hpc_ensemble_partition = "test_partition"
    return tc


def test_container_mode_status_detail_names_the_image_not_a_compile():
    """Property: on a container-mode analysis the setup phase's ``compiled`` detail names the
    container image and never claims a compile.

    Class: REGRESSION -- red before the fix, where the detail read
    ``✓ TRITON-SWMM compiled`` on a mode whose setup rules compile nothing.

    Why it is required: an operator reading status on a container run is told a build step
    completed that the workflow skipped by design; the same label is the audit trail for a
    native run, so leaving it in place for both modes makes the two runs indistinguishable.

    What kills it: restoring a literal compile label on the container branch, or dropping the
    image name from the detail.

    A second correct implementation that still passes: naming the image by its full path, by
    its filename, or by any identifier derived from the container spec, so long as the word
    "compiled" does not appear and the image is named.
    """
    tc = _synthetic_multi_sim_case()
    tc.analysis.cfg_analysis.execution_environment = "container"
    tc.analysis.cfg_hpc_system.container = ContainerSpec(sif_root="/opt/sifs")

    status = tc.analysis.get_workflow_status()
    detail = status.setup.details["compiled"]

    assert "compiled" not in detail.lower(), detail
    assert "container image" in detail, detail
    assert "/opt/sifs" in detail or detail.endswith(".sif (no in-workflow compile)"), detail


def test_native_mode_status_detail_still_carries_the_compile_line():
    """The must-not-fire control: native mode keeps the compile label, so the container
    replacement is not a silent removal of the line every native report carries.

    Class: REGRESSION on the other side of the same branch. Red under an implementation that
    replaces the label unconditionally.
    """
    tc = _synthetic_multi_sim_case()
    assert tc.analysis.cfg_analysis.execution_environment == "native"

    status = tc.analysis.get_workflow_status()
    detail = status.setup.details["compiled"]

    assert detail.endswith("TRITON-SWMM compiled"), detail
    assert "container image" not in detail, detail


def test_container_mode_status_detail_does_not_raise_without_a_container_block():
    """Property: a container-mode analysis whose hpc config declares NO container block still
    returns a status detail, naming the container image rather than a compile.

    Class: REGRESSION -- red before the fix, with ``AttributeError: 'NoneType' object has no
    attribute 'sif_root'``: the old fallback re-read the very expression that raised.

    Why it is required: preflight REPORTS a missing container block as an error rather than
    making the state impossible, and nothing validates before a status read, so this state is
    reachable through every status surface. A status read that raises on a config preflight
    merely objects to is worse than one that names where it looked.

    What kills it: re-reading the spec outside the guard, or dropping the spec-independent
    fallback text.

    A second correct implementation that still passes: naming the cluster's container settings
    by any phrase, or naming the sif_root when one is present, so long as it does not raise,
    mentions the container image, and does not claim a compile.
    """
    tc = _synthetic_multi_sim_case()
    tc.analysis.cfg_analysis.execution_environment = "container"
    tc.analysis.cfg_hpc_system.container = None

    detail = tc.analysis.get_workflow_status().setup.details["compiled"]

    assert isinstance(detail, str)
    assert "container image" in detail, detail
    assert "compiled" not in detail.lower().replace("(no in-workflow compile)", ""), detail
    assert "container settings" in detail, detail
