"""The second solver tier of the open-water outfall tests, and the one call that provisions it.

`tests/test_synth_open_water_outfall_review.py` runs most of its arms on a solver that is not the synthetic pin
(`TRITON_PIN`): the tip of a pushed work branch of the solver that reads both `swmm_driven_outfalls` and
`swmm_dem_rim_exempt_nodes`. Those arms point their system config at `OUTFALL_SOLVER_SHA` and at
`outfall_software_directory()`, a software directory beside (never inside) the shared slug `_software` tier.

WHY THIS IS A MODULE OF ITS OWN. Under the suite harness a chunk never provisions a shared solver tier
(`provision_borrower` raises when `HHEMT_SUITE_ROLE` is `chunk` and the tier is absent), and the harness's warm
step runs `tests/test_synth_00_compile_models.py` outside a chunk. The warm step therefore provisioned only the
standard tier, and each test of the outfall file failed in a chunk with `refusing to provision ... from a suite
chunk: the tier is ABSENT`. The tests and the warm step must name ONE SHA and ONE directory, or the tier the warm
step provisions is not the tier the chunk looks for. This module is that single statement; the test file's
fixtures and `tests/test_synth_00_compile_models.py::test_provision_the_open_water_outfall_solver_tier` both
consume it.

Provisioning clones a solver source tree and so belongs on Rivanna scratch, never on a laptop.
"""

from __future__ import annotations

from pathlib import Path

#: The tip of the solver's pushed work branch `solver-work/open-water-outfall-driven-stage`: it reads both
#: `swmm_driven_outfalls` and `swmm_dem_rim_exempt_nodes`.
OUTFALL_SOLVER_SHA = "0f4d1039094219776c4f0919a9474fcf3af17622"

#: The label of the software directory the work-branch arms build in.
OUTFALL_TIER_LABEL = "work_branch"


def outfall_software_directory(label: str = OUTFALL_TIER_LABEL) -> Path:
    """A software directory of the outfall tests' own, beside (never inside) the shared slug `_software` tier."""
    from tests.fixtures import worktree_slug
    from tests.fixtures._triton_source_cache import slug_runs_root

    return slug_runs_root(worktree_slug()) / f"_software_open_water_outfall_{label}" / "triton"


def provision_outfall_tier() -> Path:
    """Provision the work-branch solver tier at `OUTFALL_SOLVER_SHA` and return its directory.

    The call is `provision_borrower` with the tier's directory and pin, so it inherits that function's contract:
    a healthy tier is returned untouched, an absent or unhealthy tier is created (from a session that is not a
    suite chunk), and a suite chunk raises instead of creating it. A standalone session of the outfall tests
    reaches the same call through `retrieve_synth_TRITON_SWMM_test_case`, which provisions the directory named in
    `TRITONSWMM_software_directory` at the pin named in `TRITONSWMM_branch_key`.
    """
    from tests.fixtures import _triton_source_cache

    return _triton_source_cache.provision_borrower(outfall_software_directory(), pin=OUTFALL_SOLVER_SHA)
