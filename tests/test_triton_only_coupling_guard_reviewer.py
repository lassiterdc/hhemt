"""Reviewer tests: the coupled cfg generator's two coupled-only validations skip a TRITON-only system.

``_generate_TRITON_SWMM_cfg`` runs for every system, but it carries two validations that exist
only for a coupled run: the coupling-node check (``require_triton_couples_the_listed_nodes``) and
the DEM-rim check (``require_dem_rim_capable_triton``, run when ``swmm_node_rim_from_dem`` is on).
The coupling check is skipped for a system whose coupled model is off. These tests hold the
second check to the same rule. They build the scenario on the shared fakes of
``tests.test_scenario_boundary_guards`` and call the real cfg generator; none compiles or runs a
solver.

The two build identities below are written here as literals, not imported, so the module's own
sets are checked against independently written values.
"""

from __future__ import annotations

import pytest

#: A TRITON container commit that is NOT in the set of commits known to read
#: ``swmm_node_rim_from_dem`` (it is one of the two pins the solver matrix measured failing), so
#: the DEM-rim check cannot show this build reads the key.
_RIM_UNPROVABLE_PIN = "b3820a448f304b3f732f4b6fac5564adf86ac333"

#: A curated TRITON container commit that reads both ``swmm_node_rim_from_dem`` and
#: ``swmm_coupling_nodes``, so neither check refuses it for either kind of system.
_RIM_AND_LIST_PIN = "a146fb6ed7c22790fb9a15ff854c446db4519ad2"


def _generate_cfg(tmp_path, monkeypatch, *, coupled, rim_from_dem, environment, pin):
    """Run the real coupled cfg generator on a fake system; return the cfg path it wrote.

    ``coupled`` is the system's ``toggle_tritonswmm_model``. In container mode the build is the
    commit ``pin`` (recorded and configured alike); in native mode there is no clone to judge, so
    a build that must be shown to read the key cannot be.
    """
    from tests.test_scenario_boundary_guards import _scenario_for_coupled_cfg

    scen, cfg = _scenario_for_coupled_cfg(
        tmp_path,
        monkeypatch,
        rim_from_dem=rim_from_dem,
        environment=environment,
        software_dir=None,
        branch_key=pin if environment == "container" else None,
        recorded_sha=pin if environment == "container" else None,
    )
    scen._system.cfg_system.toggle_tritonswmm_model = coupled
    scen._generate_TRITON_SWMM_cfg()
    return cfg


@pytest.mark.parametrize("environment", ["container", "native"])
def test_a_triton_only_system_is_not_refused_for_a_dem_rim_key_it_never_reads(tmp_path, monkeypatch, environment):
    """Property: a system with the coupled model off finishes cfg generation with the DEM-rim option on.

    Class: TRITON-only systems (``toggle_tritonswmm_model`` off) with ``swmm_node_rim_from_dem``
    on and a build that cannot be shown to read the key, in container mode (a commit outside the
    known set) and in native mode (no clone to judge). Reachability: the option is a field of
    the analysis config that a TRITON-only analysis can set, and this generator runs for such a
    system whatever the option, so a TRITON-only analysis with the option on reaches the check;
    this test constructs exactly that input. Both states: the assertion turns on whether cfg
    generation raises for the input, a property of the code before and after the change, not on
    any message wording. Why: the key is read only by the coupled solver's cfg, so a system that
    never runs the coupled solver must not be stopped over whether its build reads the key. What
    kills it: running the DEM-rim check whatever the toggle, which raises ``ConfigurationError``
    here. Second correct implementation: judging the rim check through a flag passed in from
    ``prepare_scenario``; the assertions read only whether generation completed and the cfg
    exists. Invocation: the scenario is built in ``tmp_path`` and the test calls the generator
    directly, so the tree the run resolves from does not decide it.
    """
    cfg = _generate_cfg(
        tmp_path, monkeypatch, coupled=False, rim_from_dem=True, environment=environment, pin=_RIM_UNPROVABLE_PIN
    )

    assert cfg.is_file()


@pytest.mark.parametrize("environment", ["container", "native"])
def test_a_coupled_system_is_still_refused_for_a_dem_rim_key_its_build_cannot_be_shown_to_read(
    tmp_path, monkeypatch, environment
):
    """Property: with the coupled model on, the DEM-rim option is still refused on an unprovable build.

    Class: coupled systems with ``swmm_node_rim_from_dem`` on and a build that cannot be shown to
    read the key, in container mode and in native mode. Why: an older build ignores the key and
    runs with SWMM's rims while the configuration says DEM rims, with no error; the check is the
    only thing that reports it, and the fix for a TRITON-only system must not weaken it for a
    coupled one. What kills it: dropping the rim check for every system, or catching its error.
    In container mode that removal leaves the coupling check, which raises ``FileNotFoundError``
    on the unwritten list, and in native mode nothing raises, so the ``ConfigurationError`` that
    names the DEM-rim option is not satisfied by any other check. Second correct implementation:
    moving the rim check into ``prepare_scenario`` under the toggle; the assertion reads only the
    exception type and the option's name in its message. Held before and after the change: this
    test is green on the code as it stands, so it is a characterization of what must not move.
    """
    from hhemt.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="swmm_node_rim_from_dem is True"):
        _generate_cfg(
            tmp_path, monkeypatch, coupled=True, rim_from_dem=True, environment=environment, pin=_RIM_UNPROVABLE_PIN
        )


def test_the_dem_rim_cfg_a_triton_only_system_writes_is_the_one_a_coupled_system_writes(tmp_path, monkeypatch):
    """Property: skipping the rim check for a TRITON-only system changes no byte of the cfg it writes.

    Class: the same analysis with ``swmm_node_rim_from_dem`` on, prepared once with the coupled
    model on and once with it off, on a curated build that reads the key and the list (so
    neither check refuses either run). Why: the change is a validation change only; the cfg a
    system gets must not depend on whether the rim check ran, and a TRITON-only system on a
    capable build wrote the key's line before the change and must still write it. What kills it:
    appending the key's line only for a coupled system, or skipping the generator for a
    TRITON-only one. Second correct implementation: passing the toggle in from
    ``prepare_scenario``; the assertions compare the two files' lines with each run's own
    directory name replaced, and count the key's line. Held before and after the change: this
    test is green on the code as it stands, so it is a characterization of what must not move.
    """
    on_dir = tmp_path / "on"
    off_dir = tmp_path / "off"
    on_dir.mkdir()
    off_dir.mkdir()
    cfg_on = _generate_cfg(
        on_dir, monkeypatch, coupled=True, rim_from_dem=True, environment="container", pin=_RIM_AND_LIST_PIN
    )
    cfg_off = _generate_cfg(
        off_dir, monkeypatch, coupled=False, rim_from_dem=True, environment="container", pin=_RIM_AND_LIST_PIN
    )

    on_lines = cfg_on.read_text().replace(str(on_dir), "ROOT").splitlines()
    off_lines = cfg_off.read_text().replace(str(off_dir), "ROOT").splitlines()
    assert on_lines == off_lines
    assert off_lines.count("swmm_node_rim_from_dem=1") == 1
