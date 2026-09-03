"""The member-identity contract: the resolver's states, its readers, and the string rendered.

THREE tiers, and each catches what the others cannot.

  1. The resolver's own state table. The module-stem name promises it.
  2. A READER table: one scenario_status.csv, every production reader of it, asserting both
     that each reader produces the member id and that the readers AGREE with one another.
     Agreement is the property a per-site assertion does not state, and it is what catches a
     reader whose prefix normalisation is wrong rather than whose column name is.
  3. One END-TO-END arm through the string the rendered report prints. It is the only
     assertion in this file that the column repair alone does not turn green, which is what
     makes it a witness rather than a restatement of the repair.

The CSV is written inside each test body, never in a yield fixture: this project sets
tmp_path_retention_policy = "failed", which keys off the call phase only, so a teardown-phase
failure would discard the artifact needed to debug it.
"""

import numpy as np
import pytest
import xarray as xr

from hhemt.eda._config_diff import _load_subs, _n_resumes_by_member_id
from hhemt.member_identity import (
    ACCEPTED_MEMBER_ID_COLUMNS,
    member_id_from,
    member_id_from_mapping,
    resolve_member_id_column,
)
from hhemt.report_plot_ids import humanize_plot_id, member_labels_from_status
from hhemt.report_renderers.metadata import _read_scenario_status

_MEMBER = "serial_6_r1"
_STATUS_TAIL_H = "event_iloc,model_type,scenario_directory,n_resumes,run_mode,n_gpus,n_mpi_procs,n_omp_threads,n_nodes"
_STATUS_TAIL_R = "3,triton,/runs/event_index.3,2,serial,1,1,1,1"


def _write_member_status_csv(directory, column, value=_MEMBER):
    """Write one scenario_status.csv whose identity column is `column` and value is `value`."""
    (directory / "scenario_status.csv").write_text(f"{column},{_STATUS_TAIL_H}\n{value},{_STATUS_TAIL_R}\n")
    return directory


def _members_via_read_scenario_status(directory):
    out, _path = _read_scenario_status(directory)
    return {key[0] for key in out}


def _members_via_member_labels(directory):
    return set(member_labels_from_status(directory))


def _members_via_n_resumes(directory):
    return set(_n_resumes_by_member_id(directory))


#: Every reader of scenario_status.csv that reads an identity-column VALUE. Other readers
#: of the same file exist (event labels, bundle role, the appendix's reserved-field set);
#: none of them reads a member id, so none belongs in this table.
_STATUS_READERS = [
    pytest.param(_members_via_read_scenario_status, id="metadata._read_scenario_status"),
    pytest.param(_members_via_member_labels, id="report_plot_ids.member_labels_from_status"),
    pytest.param(_members_via_n_resumes, id="_config_diff._n_resumes_by_member_id"),
]


@pytest.mark.parametrize(
    "columns, expected",
    [
        (["member_id", "event_iloc"], "member_id"),
        (["sa_id", "event_iloc"], "sa_id"),
        # BOTH present: the canonical name must win, or an artifact mid-migration would
        # resolve to the spelling being retired.
        (["sa_id", "member_id", "event_iloc"], "member_id"),
        # No member axis at all. None is a LEGITIMATE answer and is why this resolver
        # returns rather than raises; raising would redden every non-sensitivity run.
        (["event_iloc"], None),
    ],
)
def test_resolve_member_id_column_states(columns, expected):
    assert resolve_member_id_column(columns) == expected


def test_no_member_axis_is_distinguishable_from_a_blank_identity():
    """The distinction the pre-repair code could not express.

    Both cases yield "" as a VALUE. Only the resolved COLUMN separates them, which is why
    the resolver returns the column name rather than the value.
    """
    no_axis = resolve_member_id_column(["event_iloc"])
    blank_id = resolve_member_id_column(["member_id", "event_iloc"])
    assert no_axis is None and blank_id == "member_id"
    assert member_id_from({}, no_axis) == member_id_from({}, blank_id) == ""


@pytest.mark.parametrize("column", ACCEPTED_MEMBER_ID_COLUMNS)
def test_member_id_from_mapping_reads_every_accepted_spelling(column):
    assert member_id_from_mapping({column: _MEMBER}, "fallback") == _MEMBER


def test_member_id_from_mapping_returns_the_caller_default_when_absent():
    """The .get(key, default) form. The default is what made the miss invisible."""
    assert member_id_from_mapping({"other": 1}, "fallback") == "fallback"


@pytest.mark.parametrize("reader", _STATUS_READERS)
@pytest.mark.parametrize(
    "column",
    [
        # A-violating: the spelling the producer emits today. Measured pre-fix, every
        # reader produces an empty result on this arm.
        "member_id",
        # B-satisfying, DIFFERENTLY POSITIONED rather than arm A repaired: the legacy
        # on-disk spelling must keep working, which is what catches a repair that merely
        # swaps the literal. Measured pre-fix, every reader passes this arm.
        "sa_id",
    ],
)
def test_scenario_status_readers_produce_the_member_identity(tmp_path, reader, column):
    _write_member_status_csv(tmp_path, column)
    produced = reader(tmp_path)
    assert produced == {_MEMBER}, (
        f"identity column {column!r} carries {_MEMBER!r}, but this reader produced "
        f"{produced!r}. An empty member slot silently collapses every row onto one key."
    )


_N_RESUMES = 7


def _member_node_group(member_id):
    """Mirrors the consolidation writer at sensitivity_analysis.py:1566,
    `node_name = f"{self.member_prefix}{member_id}"`.

    This is the ONE producer construction this file restates rather than calls, because
    `member_prefix` is an instance attribute (sensitivity_analysis.py:280) with no
    importable constant. The restatement is made falsifiable by the first assertion in
    the test below, which checks that the real consumer decodes this node name back to
    `member_id` -- so a prefix change reddens there instead of silently making this arm
    test nothing.
    """
    return f"/member_{member_id}"


def _build_member_tree(root, member_id, *, with_attrs):
    """A producer-shaped sensitivity_datatree.zarr plus its scenario_status.csv row."""
    ds = xr.Dataset(
        {"max_wlevel_m": (("event_iloc", "y", "x"), np.ones((1, 2, 2), dtype="f4"))},
        coords={"event_iloc": [0], "y": [0, 1], "x": [0, 1]},
    )
    node = _member_node_group(member_id)
    nodes = {f"{node}/tritonswmm/triton": ds}
    if with_attrs:
        nodes[node] = xr.Dataset(attrs={"member_id": str(member_id)})
    xr.DataTree.from_dict(nodes).to_zarr(root / "sensitivity_datatree.zarr", consolidated=False)
    (root / "scenario_status.csv").write_text(
        f"member_id,{_STATUS_TAIL_H}\n{member_id},0,triton,/runs/e0,{_N_RESUMES},serial,1,1,1,1\n"
    )


@pytest.mark.parametrize("with_attrs", [True, False], ids=["attrs-present", "attrs-absent"])
@pytest.mark.parametrize("member_id", ["serial_6_r1", "member_2"], ids=["bare", "member-prefixed"])
def test_n_resumes_round_trips_through_the_load_subs_join(tmp_path, member_id, with_attrs):
    """`_n_resumes_by_member_id`'s key must be the one `_load_subs` looks up.

    This REPLACES a withdrawn cross-reader agreement assertion. That arm was withdrawn on
    CORRECTNESS grounds, not because it was red: agreement holds only because every current
    consumer happens to key on the verbatim member_id, so it is a corollary of today's
    consumer set rather than an invariant. It also cannot localize -- it reports that N
    readers differ and its message has to guess which is wrong, and it guessed the only
    reader doing anything deliberate.

    The invariant asserted here instead is the relation the proxy stood in for: each reader
    produces the key ITS OWN consumer looks up. The consumer key is obtained by CALLING
    `_load_subs`, never by restating its derivation, so a failure names one pair.

    `member_2` is a LEGAL member id -- the charset is ^[A-Za-z0-9_.]+$ at
    sensitivity_analysis.py:2013 -- and it is the violating arm. `serial_6_r1` is the
    differently-positioned satisfying arm: it must pass in both states, so a repair that
    over-corrected would redden it. Both attrs branches are covered because `_load_subs`
    reads `attrs["member_id"]` when present and decodes the node path when absent, and the
    two must agree.
    """
    _build_member_tree(tmp_path, member_id, with_attrs=with_attrs)
    subs = _load_subs(tmp_path)
    assert set(subs) == {
        member_id
    }, f"fixture/consumer coupling broken: _load_subs keyed {sorted(subs)!r} for member {member_id!r}."
    assert subs[member_id]["n_resumes"] == _N_RESUMES, (
        f"scenario_status.csv records n_resumes={_N_RESUMES} for member {member_id!r}, but "
        f"_load_subs attached {subs[member_id]['n_resumes']!r}."
    )


@pytest.mark.parametrize("column", ["member_id", "sa_id"])
def test_rendered_figure_name_carries_the_resolved_member_label(tmp_path, column):
    """The string the report prints, end to end.

    member_labels_from_status feeds humanize_plot_id, whose output is injected as the
    figure name by report_renderers/_react_surgery.py and as the bundle facet label by
    bundle/combined_snakefile_generator.py. Both halves of the chain must be correct for
    the reader to see a compute-config label, so this arm stays red while either is broken.
    """
    _write_member_status_csv(tmp_path, column)
    labels = member_labels_from_status(tmp_path)
    label = labels.get(_MEMBER)
    assert label is not None, f"identity column {column!r} produced no label map: {labels!r}"
    rendered = humanize_plot_id(f"eda_cross_sim_identity__member.{_MEMBER}__evt.0", member_labels=labels)
    assert label in rendered, (
        f"the resolved label {label!r} does not appear in the rendered figure name "
        f"{rendered!r}; the reader is shown a raw or corrupted member id instead."
    )
