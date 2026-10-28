"""Independent tests for the pin-aware quality-snapshot rule that the TRITON re-pin to ``bd18215`` introduced.

Problem under test. A coupled TRITON-SWMM resume from a SWMM snapshot restarts the node and link
water-quality state at every TRITON build before ``bd18215``, and carries it at ``bd18215``. The
toolkit therefore keys two decisions on the coupled build's recorded sha: the preparation-time
default of ``swmm_snapshot_disable`` (replay versus snapshot route) and the processing-time rule
that withholds a snapshot-resumed event's water-quality statistics. These tests pin the two
decisions from the outside: through the real preparation path and the real status rule, never
through a restatement of the helper they read.

The two build identities come from the defect registry's constants, not from the module under test,
so the separate set in `hhemt.scenario` is checked against an independently written literal; the
registry's own tests assert that literal equals the synthetic tier's pin.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hhemt.model_defects import SHA_CURATED_MAIN_PIN, SHA_MAIN_QUALITY_SNAPSHOT_PIN

#: The fork `main` commit whose snapshot carries node and link water-quality state (`oldQual` count 7 in
#: `external/swmm/src/solver/snapshot.c`; measured with `git show {sha}:... | grep -c oldQual`).
_C11 = SHA_MAIN_QUALITY_SNAPSHOT_PIN
#: The curated fork `main` pin whose snapshot does NOT (`oldQual` count 0); still the estate family-C pin.
_OLD = SHA_CURATED_MAIN_PIN
#: A well-formed full sha that no registry or set names.
_UNLISTED = "deadbeef" * 5

_DISABLE_LINE = "swmm_snapshot_disable=1"
_TWO = (("TSS", "MG/L"), ("Coliform_ABC", "#/L"))


def _inp(pollutants=_TWO, ignore_quality=False) -> str:
    """A minimal hydraulics `.inp` whose quality routing is decided by `[POLLUTANTS]` and `IGNORE_QUALITY`."""
    rows = "\n".join(f"{name}  {unit}  0 0 0 0 NO * 0 0" for name, unit in pollutants)
    section = (
        f"[POLLUTANTS]\n;;Name Units Crain Cgw Crdii Kdecay SnowOnly CoPollutant CoFrac Cinit\n{rows}\n"
        if pollutants
        else ""
    )
    quality = "YES" if ignore_quality else "NO"
    return f"[OPTIONS]\nFLOW_UNITS CMS\nIGNORE_QUALITY {quality}\n\n{section}\n[JUNCTIONS]\nN1 0 3\n"


def _log(producing=None, head=None):
    """A system-log stand-in exposing the two sha fields the way `LogField` does (`.get()`)."""
    return SimpleNamespace(
        tritonswmm_producing_sha=SimpleNamespace(get=lambda: producing),
        triton_head_sha=SimpleNamespace(get=lambda: head),
    )


def _expected_build(producing, head, environment):
    """The build the run executes, by the documented mode rule, written without the code under test.

    Native: the compiled coupled target's sha, else the clone head. Container: the image label only,
    because container setup never writes the producing sha, so one in the log is a stale native compile's.
    """
    if environment == "container":
        return head
    return producing or head


# --------------------------------------------------------------------------- A3 resolution (arm a)
_INP_KINDS = {
    "routed": dict(pollutants=_TWO, ignore_quality=False),
    "ignored": dict(pollutants=_TWO, ignore_quality=True),
    "pollutant_free": dict(pollutants=(), ignore_quality=False),
}
_BUILDS = {"carrying": _C11, "older_pin": _OLD, "unlisted": _UNLISTED, "unrecorded": None}


@pytest.mark.parametrize("inp_kind", list(_INP_KINDS))
@pytest.mark.parametrize("build_name", list(_BUILDS))
def test_unset_selector_replays_only_a_quality_routed_model_at_a_build_that_does_not_carry_the_state(
    tmp_path, inp_kind, build_name
):
    """Property: unset resolves to True only for a quality-routed model at a build that does not carry the state.

    Class: every (hydraulics model kind) x (build identity) pair in the 3 x 4 grid below, where the kinds are
    quality-routed, `IGNORE_QUALITY YES` with pollutants declared, and pollutant-free, and the builds are the
    carrying pin, the older curated pin, an unlisted full sha and no recorded build. Why it is required: an
    unset field that kept forcing replay at the carrying build would leave every quality-routed run replaying
    for the whole life of the pin, and one that stopped forcing it elsewhere would resume quality state that
    is lost. Reachability: the grid's cells are exactly the inputs `_generate_TRITON_SWMM_cfg` can hand the
    resolver (a recorded sha, an unlisted sha, or none), and the `.inp` kinds are the three the quality
    predicate separates. Both-states anchor: the assertion turns on the returned bool for one `.inp` across
    builds, a property of the old and the new code alike. Invocation isolation: the grid varies the build and
    the `.inp` independently, so a rule keyed on only one of them is red in some cell. What kills it: treating
    the carrying build as always-replay, or any unlisted build as carrying. Second correct implementation:
    deciding the build verdict at the call site and passing a bool, which these assertions cannot see.
    """
    from hhemt.scenario import resolve_swmm_snapshot_disable

    inp = tmp_path / "m.inp"
    inp.write_text(_inp(**_INP_KINDS[inp_kind]))
    build = _BUILDS[build_name]
    expected = inp_kind == "routed" and build_name != "carrying"
    assert resolve_swmm_snapshot_disable(None, inp, build_sha=build) is expected


@pytest.mark.parametrize("build_name", list(_BUILDS))
@pytest.mark.parametrize("explicit", [True, False])
def test_an_explicit_selector_wins_at_every_build(tmp_path, build_name, explicit):
    """Property: an explicit True or False is returned unchanged at every build.

    Class: both explicit values x four build identities, on a quality-routed model (the one kind where the
    unset default differs from either explicit value at some build). Why required: the carrying build must
    not override a user who asked for replay, and an older build must not override one who asked for the
    snapshot route. What kills it: consulting the build before the explicit value. Both-states anchor: the
    returned bool equals the configured one in both the old and the new code. Reachability: the field is a
    user-settable `bool | None`. Invocation isolation: the eight cells differ in both inputs. Second correct
    implementation: short-circuiting on `configured is not None` before reading any build, which this passes.
    """
    from hhemt.scenario import resolve_swmm_snapshot_disable

    inp = tmp_path / "m.inp"
    inp.write_text(_inp())
    assert resolve_swmm_snapshot_disable(explicit, inp, build_sha=_BUILDS[build_name]) is explicit


# --------------------------------------------------------------------------- preparation (A3) and processing (A2) agree
def _prepared_cfg(tmp_path, monkeypatch, *, producing, head, environment):
    """The coupled TRITON cfg text the real generator writes for an unset-field, quality-routed scenario."""
    from tests.test_scenario_boundary_guards import _scenario_for_coupled_cfg

    tmp_path.mkdir(parents=True, exist_ok=True)
    # A container run's image is built from TRITONSWMM_branch_key and the image preflight holds
    # the image's recorded commit to it, so the configured pin is the recorded build there.
    scen, cfg = _scenario_for_coupled_cfg(
        tmp_path,
        monkeypatch,
        rim_from_dem=False,
        environment=environment,
        branch_key=head if environment == "container" else None,
        recorded_sha=None,
    )
    scen._analysis.cfg_analysis.swmm_snapshot_disable = None
    scen._system.log = _log(producing, head)
    (tmp_path / "hydraulics.inp").write_text(_inp())
    scen._generate_TRITON_SWMM_cfg()
    return cfg.read_text()


def _summary(n_events=1):
    """A coupled summary store: the statistics the rule withholds, one it never touches, and the pollutant dim."""
    pollutants = [p for p, _ in _TWO]
    shape = (n_events, 1, len(pollutants))
    return xr.Dataset(
        {
            "pollutant_max_conc": (("event_iloc", "node_id", "pollutant"), np.ones(shape)),
            "outfall_total_load": (("event_iloc", "node_id", "pollutant"), np.full(shape, 3.0)),
            "outfall_total_vol_10e6_ltr": (("event_iloc", "node_id"), np.full((n_events, 1), 4.0)),
            "qual_routing_external_inflow": (("event_iloc", "pollutant"), np.full((n_events, len(pollutants)), 2.0)),
        },
        coords={"event_iloc": list(range(n_events)), "node_id": ["OUT1"], "pollutant": pollutants},
    )


def _node_store():
    """A per-scenario node store: no `event_iloc` dimension, so the status is a scalar coordinate."""
    pollutants = [p for p, _ in _TWO]
    return xr.Dataset(
        {
            "pollutant_max_conc": (("node_id", "pollutant"), np.ones((1, len(pollutants)))),
            "outfall_total_load": (("node_id", "pollutant"), np.full((1, len(pollutants)), 3.0)),
        },
        coords={"node_id": ["OUT1"], "pollutant": pollutants},
    )


def _link_store():
    """A per-scenario link store: carries `link_total_load`, the link-side statistic the rule withholds."""
    pollutants = [p for p, _ in _TWO]
    return xr.Dataset(
        {"link_total_load": (("link_id", "pollutant"), np.full((1, len(pollutants)), 5.0))},
        coords={"link_id": ["L1"], "pollutant": pollutants},
    )


def _apply_rule(
    tmp_path, monkeypatch, ds, *, rows, producing=None, head=None, environment="native", n_resumes=None, cfg_text=""
):
    """Run the real `_apply_quality_stats_rule` on `ds` with the real `_quality_stats_status`, on a stand-in object.

    `rows` is the resume-event ledger content (None for no ledger file). The stand-in carries the same
    attributes the real post-processing object reads: `_analysis.cfg_analysis`, `_system.log`, `_scenario`,
    `log.n_resumes` and `scen_paths.triton_swmm_cfg`.
    """
    from hhemt import resume_events
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    tmp_path.mkdir(parents=True, exist_ok=True)
    log_path = tmp_path / "run.log"
    ledger = resume_events.ledger_path_for(log_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = tmp_path / "triton.cfg"
    cfg.write_text(cfg_text)
    monkeypatch.setattr("hhemt.run_simulation.model_logfile_for", lambda analysis, event_iloc, model_type: log_path)
    fake = SimpleNamespace(
        _analysis=SimpleNamespace(cfg_analysis=SimpleNamespace(execution_environment=environment)),
        _system=SimpleNamespace(log=_log(producing, head)),
        _scenario=SimpleNamespace(event_iloc=0),
        log=SimpleNamespace(n_resumes=None if n_resumes is None else SimpleNamespace(get=lambda: n_resumes)),
        scen_paths=SimpleNamespace(triton_swmm_cfg=cfg),
    )
    fake._quality_stats_status = lambda model_type: TRITONSWMM_sim_post_processing._quality_stats_status(
        fake, model_type
    )
    return TRITONSWMM_sim_post_processing._apply_quality_stats_rule(fake, ds, "tritonswmm")


_SHA_FIELD_VALUES = (None, _C11, _OLD)


@pytest.mark.parametrize("environment", ["native", "container"])
@pytest.mark.parametrize("producing", _SHA_FIELD_VALUES, ids=["prod=none", "prod=C11", "prod=OLD"])
@pytest.mark.parametrize("head", _SHA_FIELD_VALUES, ids=["head=none", "head=C11", "head=OLD"])
def test_preparation_and_processing_read_the_same_build_by_the_same_mode_rule(
    tmp_path, monkeypatch, environment, producing, head
):
    """Property: for one recorded system log, the cfg the preparation path writes forces replay exactly when the
    processing path withholds a snapshot-resumed event's statistics, and both follow the mode rule.

    Class: the full 2 x 3 x 3 grid of (execution environment) x (producing sha field) x (head sha field),
    each field one of unrecorded, the carrying pin, or the older pin. The grid contains the stale-native-compile
    cell in container mode (producing = carrying pin, image label = older pin) and its mirror (native, producing
    = older pin, head = carrying pin), which are the two cells where a wrong precedence changes the answer.
    Why it is required: preparation and processing run in different jobs from the same log; if they ever read
    different builds, a run is prepared for the snapshot route and its statistics withheld as if replayed (or the
    reverse), and the field's documented meaning is false for that run. Reachability: a system directory compiled
    natively once and later used in container mode holds both fields, and the stale producing sha is written by
    exactly that history (`system.py` writes it only from a native compile, `setup_workflow.py` never clears it);
    artifact: the grid's container cells with a recorded producing sha. Both-states anchor: the assertion turns
    on whether the cfg carries the disable line and on the NaN-ness of the withheld statistic, and on the status
    constants imported from the module, never on wording. Invocation isolation: the expected build is computed
    by `_expected_build` from the grid's inputs, not from the helper under test, and the two production paths are
    run separately (the real `_generate_TRITON_SWMM_cfg` and the real `_apply_quality_stats_rule`). What kills
    it: reading the producing sha first in container mode, swapping the native precedence, or passing a
    different mode value at one of the two call sites. Second correct implementation: a single helper both
    sites share, or two sites that each inline the rule; this reads outcomes only.
    """
    from hhemt.process_simulation import QUALITY_STATS_SNAPSHOT, QUALITY_STATS_WITHHELD

    build = _expected_build(producing, head, environment)
    carried = build == _C11

    cfg_text = _prepared_cfg(tmp_path / "prep", monkeypatch, producing=producing, head=head, environment=environment)
    assert (_DISABLE_LINE in cfg_text.splitlines()) is (not carried), (environment, producing, head, build)

    out = _apply_rule(
        tmp_path / "proc",
        monkeypatch,
        _summary(),
        rows=[{"path": "snapshot"}],
        producing=producing,
        head=head,
        environment=environment,
    )
    assert bool(np.isnan(out["pollutant_max_conc"]).all()) is (not carried), (environment, producing, head, build)
    assert out["pollutant_stats_status"].values.tolist() == [
        QUALITY_STATS_SNAPSHOT if carried else QUALITY_STATS_WITHHELD
    ]


# --------------------------------------------------------------------------- A2 (arm c)
_STORES = {"summary": _summary, "node_store": _node_store, "link_store": _link_store}
_WITHHELD_VARIABLES = {
    "summary": ("pollutant_max_conc", "outfall_total_load", "qual_routing_external_inflow"),
    "node_store": ("pollutant_max_conc", "outfall_total_load"),
    "link_store": ("link_total_load",),
}


@pytest.mark.parametrize("store_name", list(_STORES))
@pytest.mark.parametrize("build_name", ["carrying", "older_pin"])
@pytest.mark.parametrize("route", ["snapshot", "replay", "ledgerless_resumed", "never_resumed"])
def test_statistics_are_kept_or_withheld_by_route_and_build_in_every_store_kind(
    tmp_path, monkeypatch, store_name, build_name, route
):
    """Property: only a snapshot-route event at a build that does not carry the state has its statistics NaN-ed, and
    the status coordinate records the route actually taken, in the summary and in both per-scenario stores.

    Class: store kind (coupled summary, per-scenario node store, per-scenario link store) x build (carrying pin,
    older pin) x route (a `snapshot` ledger row, a `replay` row, no ledger with at least one resume and no disable
    line, no resume at all). Why it is required: the same rule feeds three write sites, and a status or a
    withholding applied at one but not the others publishes NaN beside a status that says `computed`, or the
    reverse. Reachability: `_export_TRITON_summary`, the node-summary and the link-summary writers each call the
    rule on the stores built here; artifact: `process_simulation.py` call sites of `_apply_quality_stats_rule`.
    Both-states anchor: NaN-ness of the named statistics, and equality of the status with the module constant.
    Invocation isolation: the store kinds differ in dimensions (`event_iloc` present or absent), so a rule that
    only handles the summary shape is red on the scalar-status stores. What kills it: returning the computed or
    snapshot status for a snapshot row at the older pin, withholding at the carrying pin, or returning the
    snapshot status for a replay row. Second correct implementation: deciding the status once per event and
    caching it; the assertions read the datasets only.
    """
    from hhemt.process_simulation import (
        QUALITY_STATS_COMPUTED,
        QUALITY_STATS_REPLAY,
        QUALITY_STATS_SNAPSHOT,
        QUALITY_STATS_WITHHELD,
    )

    carried = build_name == "carrying"
    kwargs = {
        "snapshot": dict(rows=[{"path": "snapshot"}]),
        "replay": dict(rows=[{"path": "replay"}, {"path": "replay"}]),
        "ledgerless_resumed": dict(rows=None, n_resumes=2, cfg_text="a=1\n"),
        "never_resumed": dict(rows=None, n_resumes=0, cfg_text="a=1\n"),
    }[route]
    out = _apply_rule(
        tmp_path, monkeypatch, _STORES[store_name](), producing=_C11 if carried else _OLD, head=None, **kwargs
    )
    withheld = (route in ("snapshot", "ledgerless_resumed")) and not carried
    expected_status = {
        ("snapshot", True): QUALITY_STATS_SNAPSHOT,
        ("snapshot", False): QUALITY_STATS_WITHHELD,
        ("replay", True): QUALITY_STATS_REPLAY,
        ("replay", False): QUALITY_STATS_REPLAY,
        ("ledgerless_resumed", True): QUALITY_STATS_COMPUTED,
        ("ledgerless_resumed", False): QUALITY_STATS_WITHHELD,
        ("never_resumed", True): QUALITY_STATS_COMPUTED,
        ("never_resumed", False): QUALITY_STATS_COMPUTED,
    }[(route, carried)]
    status = out["pollutant_stats_status"].values.tolist()
    assert status == ([expected_status] if "event_iloc" in out.dims else expected_status)
    for name in _WITHHELD_VARIABLES[store_name]:
        assert bool(np.isnan(out[name]).all()) is withheld, (store_name, name)
    if store_name == "summary":
        assert (out["outfall_total_vol_10e6_ltr"] == 4.0).all(), "a hydraulic quantity is never withheld"


# --------------------------------------------------------------------------- build-sha reader (arm d, signature)
@pytest.mark.parametrize("environment", ["native", "container"])
def test_the_build_reader_is_tolerant_of_an_absent_unreadable_or_empty_log(environment):
    """Property: `coupled_build_sha` returns None, and never raises, for every log shape a run can present.

    Class: no log at all, a log lacking both fields, a log whose field `.get()` raises, and a log whose fields
    hold empty strings. Why it is required: the preparation and processing stand-ins in this suite, bundles and
    legacy system logs all present such logs, and a raise there would abort preparation or processing for a
    reason unrelated to the run. Reachability: `getattr(self._system, "log", None)` passes None for a system with
    no log; an unreadable field is what a corrupt persisted log gives. Both-states anchor: the assertion is on
    the returned value and the absence of an exception, true of any version of the helper. Invocation isolation:
    four distinct failure shapes, in both modes. What kills it: dropping the `try`/`except`, or the `is not None`
    guard on the field. Second correct implementation: a helper that reads through `getattr(log, name, None)`
    with a different guard structure; the assertions read the return only.
    """
    from hhemt.scenario import coupled_build_sha

    def _raises():
        raise RuntimeError("unreadable field")

    shapes = [
        None,
        SimpleNamespace(),
        SimpleNamespace(
            tritonswmm_producing_sha=SimpleNamespace(get=_raises), triton_head_sha=SimpleNamespace(get=_raises)
        ),
        _log("", ""),
    ]
    for log in shapes:
        assert coupled_build_sha(log, execution_environment=environment) is None


def test_an_unreadable_producing_field_falls_through_to_the_head_field_natively():
    """Property: a native log whose producing field cannot be read still yields the head field's sha.

    Class: native logs where the first-ranked field raises on read. Why it is required: falling to None would
    force replay at a build whose head sha names it correctly, and the safe direction does not excuse losing
    a readable value. Reachability: a persisted log with one corrupt field is the case. Both-states anchor:
    the returned sha. Invocation isolation: the head field is the carrying pin so the answer is distinct from
    None. What kills it: wrapping the whole loop, not each field, in the `try`. Second correct implementation:
    reading both fields up front and choosing.
    """
    from hhemt.scenario import coupled_build_sha

    def _raises():
        raise RuntimeError("unreadable field")

    log = SimpleNamespace(
        tritonswmm_producing_sha=SimpleNamespace(get=_raises), triton_head_sha=SimpleNamespace(get=lambda: _C11)
    )
    assert coupled_build_sha(log, execution_environment="native") == _C11


def test_omitting_the_execution_environment_raises_instead_of_taking_native_precedence():
    """Property: the build reader has no default for the execution environment, so omitting it raises `TypeError`.

    Class: every call site of `coupled_build_sha`, present and future. Why it is required: a default would let a
    new call site silently take native precedence in a container run, which is the stale-sha answer the
    mode rule exists to prevent (the keyword-required parameter is deliberate). Reachability: any new caller
    that forgets the argument; artifact: the two production call sites both pass it. Both-states anchor: the
    parameter has no default in the post-change code, a property a pre-change tree lacks only because the
    function does not exist. Invocation isolation: both a bare call and the signature are checked, so a default
    hidden behind `*args` is also caught. What kills it: giving the parameter any default. Second correct
    implementation: a keyword-only or a required positional parameter; both raise `TypeError` when omitted.
    """
    from hhemt.scenario import coupled_build_sha

    with pytest.raises(TypeError):
        coupled_build_sha(_log(_C11, _C11))
    param = inspect.signature(coupled_build_sha).parameters["execution_environment"]
    assert param.default is inspect.Parameter.empty


# --------------------------------------------------------------------------- exact membership (arm e)
_PRE_C11_PREFIXES = (
    # no snapshot payload at all
    "15eb18a5",
    "3a832f7d",
    "b3820a44",
    "9db367dd",
    "5d2ad1e8",
    "21e666d6",
    "a38338b0",
    # snapshot payload without the quality fields (`oldQual` count 0 in `snapshot.c`)
    "658a7a37",
    "01e95a76",
    "e53c2fa0",
    "031ea42b",
    "5fadf167",
    "23b75781",
    "c7b2fd1d",
    "cf357e4b",
    "320b3313",
    "1b797e6d",
    "65e7f681",
    "681d809a",
    "d4565182",
    "852f10c5",
    "2ecb1cff",
    "a146fb6e",
)


def _registered_full_shas():
    """Every full 40-hex sha the registry module binds to a `SHA_*` constant, by constant name."""
    from hhemt import model_defects

    return {
        name: value
        for name, value in vars(model_defects).items()
        if name.startswith("SHA_") and isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)
    }


@pytest.mark.parametrize("prefix", _PRE_C11_PREFIXES)
def test_no_build_before_the_carrying_pin_is_listed_as_carrying_the_state(prefix):
    """Property: no registered TRITON sha that predates the carrying pin is a member of the carrying set.

    Class: the 23 registered shas measured as lacking the quality fields in their snapshot (7 with no snapshot
    payload at all, 16 whose `snapshot.c` has no `oldQual` line), each resolved from the registry module's own
    `SHA_*` constants by 8-character prefix. Why it is required: the error that matters is the false-carrying
    one, which publishes statistics from a snapshot that restarted concentrations and link loads (the coupled
    snapshot before `bd18215` excludes node and link quality state), and a set that grew a pre-pin sha would do
    it silently. Reachability: each is a pin a user or an image can hold; the registry exists because they do.
    Both-states anchor: the membership verdict. Invocation isolation: the shas are drawn from the registry, not
    from the sibling tests in `test_swmm_water_quality_outputs.py`, which probe only the older curated pin and
    one invented sha. What kills it: adding any listed pre-pin sha to the set. Second correct implementation: a
    set built from a derived ancestry rule over a clone; the assertion reads the verdict only.
    """
    from hhemt.scenario import snapshot_carries_quality_state

    matches = [v for v in _registered_full_shas().values() if v.startswith(prefix)]
    assert len(matches) == 1, f"instrument control: {prefix} must resolve to exactly one registry constant"
    assert snapshot_carries_quality_state(matches[0]) is False
    assert snapshot_carries_quality_state(matches[0].upper()) is False


def test_membership_is_the_exact_full_sha_case_folded_and_never_a_prefix():
    """Property: the carrying pin counts as a full sha in any case, and neither its abbreviation nor a
    near-miss nor an empty value counts.

    Class: the full 40-character pin (lower and upper case), its 7- and 12-character prefixes, the pin with its
    last character altered, the pin with a trailing character appended, an empty string and None. Why it is
    required: every recorder writes a full sha, so an abbreviated value is evidence of a hand-edited or foreign
    record, and counting it would extend the carrying claim to a build nobody measured. Reachability: a log
    edited by hand, a bundle, or an image label from another tool. Both-states anchor: the verdict per input.
    Invocation isolation: positive and negative inputs in one test so a constant True or constant False is red.
    What kills it: prefix matching, dropping the case fold, or returning True for a falsy input. Second correct
    implementation: comparing through a normalising function; the assertions read verdicts.
    """
    from hhemt.scenario import SWMM_QUALITY_SNAPSHOT_SHAS, snapshot_carries_quality_state

    assert _C11 in SWMM_QUALITY_SNAPSHOT_SHAS
    assert snapshot_carries_quality_state(_C11) is True
    assert snapshot_carries_quality_state(_C11.upper()) is True
    for not_a_member in (_C11[:7], _C11[:12], _C11[:-1] + "9", _C11 + "0", "", None):
        assert snapshot_carries_quality_state(not_a_member) is False, not_a_member


# --------------------------------------------------------------------------- docs agree with the status constants
def test_every_computed_status_constant_is_listed_verbatim_in_the_output_data_model():
    """Property: each `computed`-family `QUALITY_STATS_*` constant appears, verbatim and backticked, in the
    status-coordinate row of the output data model.

    Class: every `QUALITY_STATS_*` constant in `process_simulation` whose text starts with `computed`.
    Why it is required: readers of a processed tree learn the permitted values of `pollutant_stats_status` from
    that row, and a value the code can write that the row omits is an undocumented output state. Reachability:
    the row is the only enumeration of the coordinate's values in the docs; artifact: the row read below.
    Both-states anchor: containment of the module constant's own text, which follows any rewording of it.
    Invocation isolation: the constants are discovered from the module, so a status added later is covered.
    What kills it: adding a `computed` status without listing it, or rewording a constant without the row.
    Second correct implementation: generating the row from the constants; this reads the final file.
    """
    import hhemt.process_simulation as ps

    constants = {n: v for n, v in vars(ps).items() if n.startswith("QUALITY_STATS_") and isinstance(v, str)}
    computed = {n: v for n, v in constants.items() if v.startswith("computed")}
    assert len(computed) >= 3, ("instrument control: the module exposes the three computed statuses", sorted(computed))
    doc = Path(__file__).resolve().parents[1] / "docs" / "reference" / "output-data-model.md"
    row = next(line for line in doc.read_text().splitlines() if line.startswith("| `pollutant_stats_status`"))
    for name, text in computed.items():
        assert f"`{text}`" in row, (name, text)
