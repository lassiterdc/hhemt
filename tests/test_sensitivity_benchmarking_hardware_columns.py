"""The hardware COLUMN key is the hardware token, and the baseline anchors on that same key.

Work chunk ``2a.3``. Before it, `_hardware_family` WAS the axis: every non-GPU group shared
the one ``cpu`` column and inherited ONE anchor. On a master whose CPU members span two
silicon models that is not a rendering nicety — the figure divided 24 EPYC runs by the one
Xeon serial measurement and drew the cross-hardware ratio as parallel efficiency, with no
exception and no log line.

Each test is differential rather than absolute: it asserts the new rule's answer AND, on the
same frame, that the pre-change rule gives a DIFFERENT one. That second assertion is what
makes the test kill the plausible wrong implementation (`_hardware_family` used as the
column key) instead of merely passing under the right one, and it keeps the mutant visible
to a future editor who deletes the token map because "the family rule was fine".

Reachability: every input below is an in-memory frame or a stub member; nothing reads the
tree, the solver, or a fixture that compiles. Invocation is not isolatable from this
worktree (the discriminating change is to what the code DOES on these inputs, not to how it
is invoked), so each test discharges B4 condition 4 through its second arm.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hhemt.report_renderers import sensitivity_benchmarking as sb

#: Two CPU models and one GPU token under ONE group value — the arrangement that was
#: mis-drawn. `e*` are EPYC and start at two devices; `x1` is the single Xeon serial
#: reference at one device; `g*` are one GPU arm at one and two devices.
_EPYC = "AMD EPYC 9454 48-Core Processor"
_XEON = "Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz"
_COLUMN_BY_MEMBER = {"e2": _EPYC, "e4": _EPYC, "x1": _XEON, "g1": "a6000", "g2": "a6000"}


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(sa_id="e2", group_value="standard", n_devices=2, wallclock_s=50.0),
            dict(sa_id="e4", group_value="standard", n_devices=4, wallclock_s=30.0),
            dict(sa_id="x1", group_value="standard", n_devices=1, wallclock_s=100.0),
            dict(sa_id="g1", group_value="gpu (a6000)", n_devices=1, wallclock_s=20.0),
            dict(sa_id="g2", group_value="gpu (a6000)", n_devices=2, wallclock_s=12.0),
        ]
    )


def _baselines(df: pd.DataFrame, token_by_group):
    return sb._resolve_family_baselines(
        df,
        t_col="wallclock_s",
        indep_col="n_devices",
        group_col="group_value",
        token_by_group=token_by_group,
    )


def test_a_group_spanning_two_cpu_tokens_is_drawn_in_two_columns():
    """Property: one group value carrying two hardware tokens becomes two columns.

    Why required: the column facet, the per-column row filters and the scaling anchors all
    read one group value, so a heterogeneous group is ONE series in ONE column however the
    column set is computed. Splitting the group is what makes the token expressible at all.

    What kills it: a correct implementation that keys the scaling dicts on a
    ``(token, group_value)`` TUPLE instead of splitting the label still passes — the
    assertion reads the SET of column keys, which both forms produce.

    Second correct implementation: a correct implementation that splits inside
    `_column_key` rather than in `_attach_hardware_columns` also passes, because this test
    reads the mapping's VALUES, not which function produced them.
    """
    df = _frame()
    by_group = sb._attach_hardware_columns(df, _COLUMN_BY_MEMBER)

    assert set(by_group.values()) == {_EPYC, _XEON, "gpu (a6000)"}
    # The split is what makes the map single-valued; assert it happened rather than trusting
    # the values, which two groups could in principle carry while sharing one label.
    assert len(set(df["group_value"])) == 3

    # DIFFERENTIAL: the pre-change rule collapses the two CPU tokens onto one column.
    legacy = {sb._hardware_family(str(g)) for g in df["group_value"].unique() if not str(g).startswith("gpu")}
    assert legacy == {"cpu"}, "the family rule must be the mutant this test kills, and it must still collapse"


def test_a_heterogeneous_cpu_group_anchors_each_token_on_its_own_one_device_run():
    """Property: the anchor partition is the column partition, not the family.

    Why required: an anchor resolved as one hardware and a panel drawn as another produces a
    within-hardware reference presented as a cross-hardware ratio. The Xeon serial run is the
    ONLY one-device run on this frame, so a family-keyed anchor hands 100.0 to the EPYC group
    as well.

    What kills it: any implementation that keys `_resolve_family_baselines` on
    `_hardware_family` — differential assertion below.

    Second correct implementation: resolving the anchor from the token column with
    ``df.groupby(token_col)`` instead of the group-value map produces the same numbers and
    passes, because the assertion reads the mapping by TOKEN, not by the key used to build it.
    """
    df = _frame()
    by_group = sb._attach_hardware_columns(df, _COLUMN_BY_MEMBER)
    baselines = _baselines(df, by_group)

    column_of = {gv: sb._column_key(gv, by_group) for gv in df["group_value"].unique()}
    xeons = [gv for gv, col in column_of.items() if col == _XEON]
    epycs = [gv for gv, col in column_of.items() if col == _EPYC]
    assert len(xeons) == 1 and len(epycs) == 1, f"expected one group per CPU token; got {column_of}"

    assert baselines[xeons[0]] == pytest.approx(100.0)
    assert epycs[0] not in baselines, "the EPYC column has no one-device run, so it must have no anchor"

    # DIFFERENTIAL: without the token map the same frame folds both CPU models into ONE
    # group value, so the family rule hands that one group the Xeon serial anchor (100.0)
    # and the EPYC rows are divided by it. That single number IS the borrow this test kills.
    merged = _baselines(_frame(), None)
    assert merged.get("standard") == pytest.approx(100.0), (
        "the family rule must anchor the whole CPU group on the serial Xeon run; that is the borrow this test kills"
    )
    assert set(merged) == {"standard", "gpu (a6000)"}, (
        f"the family rule must NOT split the CPU group; got {sorted(merged)}"
    )


def test_a_column_with_no_one_device_run_borrows_no_other_columns_anchor():
    """Property: a column whose smallest width exceeds one device resolves NO anchor.

    Why required: ``S(N) = t(1) / t(N)`` is a claim about scaling from one device. Anchoring
    at the smallest width available prints ``S = 1.0`` at, say, two devices and asserts that
    two devices are the reference — a fabricated within-hardware statement in place of an
    honest absence.

    What kills it: restoring the old ``min(indep_col)`` anchor — a one-line change, and the
    mutant this test exists to catch.

    Second correct implementation: skipping the column in the caller
    (`_compute_scaling_series`) instead of inside `_resolve_family_baselines` also passes,
    because the assertion is that no mapping entry names the EPYC group.
    """
    df = _frame()
    by_group = sb._attach_hardware_columns(df, _COLUMN_BY_MEMBER)
    baselines = _baselines(df, by_group)

    epyc_groups = [
        gv for gv, col in ((g, sb._column_key(g, by_group)) for g in df["group_value"].unique()) if col == _EPYC
    ]
    assert epyc_groups, "fixture no longer produces an EPYC column; the assertion below would be vacuous"
    for gv in epyc_groups:
        assert gv not in baselines

    # The GPU column DOES have a one-device run, so the rule must keep it: a rule that
    # dropped every anchor would also satisfy the assertion above.
    assert baselines["gpu (a6000)"] == pytest.approx(20.0)


class _Spec:
    def __init__(self, hardware=None, gpu_hardware=None):
        self.hardware = hardware
        self.gpu_hardware = gpu_hardware


class _Hpc:
    def __init__(self, spec):
        self.partitions = {"p": spec}


class _CfgAnalysis:
    hpc_ensemble_partition = "p"


class _Paths:
    def __init__(self, d):
        self.analysis_dir = d


class _Sub:
    def __init__(self, spec, d):
        self.cfg_hpc_system = _Hpc(spec)
        self.cfg_analysis = _CfgAnalysis()
        self.analysis_paths = _Paths(d)


def test_a_gpu_member_is_identified_from_configuration_alone(tmp_path):
    """Property: a GPU partition's member resolves its token from `gpu_hardware`, no raw output.

    Why required: the CPU arm of the resolution order reads the TRITON run header, which
    `clear_raw` legitimately purges. A GPU member measured that way becomes unidentifiable
    exactly when the raw tier is gone, and the refusal gate then takes down a report whose
    hardware was declared in configuration all along — the a6000 and a100 regeneration this
    package feeds runs on trees whose raw tier may be cleared.

    What kills it: dropping the GPU arm and leaving the CPU-only order in place — the
    differential assertion below (no log files on disk, no resolved token).

    Second correct implementation: reading `gpu_hardware` through a partition-name lookup
    rather than through the declared spec produces the same token and passes; the assertion
    reads the resolved token, not the path it took.

    Class: REGRESSION — red on the pre-change tree, where `_resolve_member_hardware` returns
    no token for a GPU member with no run headers.
    """
    sub = _Sub(_Spec(gpu_hardware="a6000"), tmp_path)
    ident = sb._resolve_member_hardware(sub)

    assert ident.token == "a6000"
    assert ident.source == "declared"
    # No disagreement is invented against a CPU name that was never read.
    assert ident.disagreement is False

    # DIFFERENTIAL: the same member with the GPU declaration removed has NO identity, which
    # is what the pre-change order returned for every GPU member without raw output.
    assert sb._resolve_member_hardware(_Sub(_Spec(), tmp_path)).token is None
