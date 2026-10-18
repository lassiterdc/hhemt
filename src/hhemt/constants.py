"""Named values that several modules must spell identically, kept in one place.

A literal that two modules each write out drifts the moment one of them is edited,
and the mismatch usually shows only at run time: a status flag one rule writes and
another never finds, or a figure a sweep deletes that a guard meant to spare. This
module holds those values once. It carries the layout-version re-exports, the
Norfolk example-case file names, the column headers the SWMM report parser emits,
the ``analysis.test()`` defaults and the test and case directory names, the Globus
endpoint identifiers for UVA and Frontier, the registry of analysis subtrees that
figure-deleting sweeps spare, and the ``_status/`` flag-name builders together with
their inverse parser.
"""

import re

# Re-exported from version_migration.constants so callers can import the layout
# version from here (provenance.py does). This is not the source of truth:
# scripts/check_layout_version.py reads src/hhemt/version_migration/constants.py
# directly, so a version bump is made there.
from hhemt.version_migration.constants import (  # noqa: F401
    LAYOUT_VERSION,
    MINIMUM_SUPPORTED_VERSION,
)

APP_NAME = "hhemt"
NORFOLK_EX = "norfolk_coastal_flooding"
NORFOLK_ANALYSIS_CONFIG = "template_analysis_config.yaml"
NORFOLK_SYSTEM_CONFIG = "template_system_config.yaml"
NORFOLK_CASE_CONFIG = "case.yaml"

#: The EDA subdirectory name under an analysis's ``plots/``. Its only reader in
#: src/hhemt is UNREGENERABLE_ANALYSIS_SUBTREES below, through which
#: utils.select_regenerable_figures decides what the figure-deleting callers spare. Most
#: figures here come from analysis.eda(), a non-Snakemake in-process facade, and nothing
#: rebuilds those after deletion. The exception is a figure the active reporting set's own
#: rules write under plots/eda/ (compute-sensitivity's ``plots/eda/config_diff_maps``, for
#: example): its report rule rebuilds it, so a caller that passes ``report_rule_stems``
#: still deletes it.
EDA_PLOTS_SUBDIR = "eda"

#: Analysis-dir-relative subtrees produced OUTSIDE the Snakemake graph, which a
#: deletion sweep spares. What lives here is rebuilt by no rule, with one exception: a
#: figure under plots/eda/ that the active reporting set's own rules write is rebuilt
#: by its report rule, so utils.select_regenerable_figures still selects it when the
#: caller passes ``report_rule_stems`` (from utils.report_rule_figure_stems). This
#: names the subtrees; the selector applies that carve-out. The selector and
#: provenance.collect_plot_stamps consume the whole set. bundle/_emit.py's
#: undeclared-figure prune reaches it through the same selector, rooted at plots/ so
#: only plots/eda applies, and passes no report-rule stems, so it spares every figure
#: there.
#:
#: `eda_local` is PRECAUTIONARY. No sweep in this codebase reaches it today: every
#: deletion walk is rooted at `{analysis_dir}/plots` and eda_local/ is a SIBLING of
#: plots/, emitted at {root}/eda_local by eda/_local_surface.py. Its presence here is
#: NOT evidence that any loop ever threatened it. The entry becomes operative the
#: moment a caller points the helper at a root above plots/, which is how such a
#: sweep would naturally be written; until then it is conditionally inactive rather
#: than dead, and the analysis-root test in
#: tests/test_reprocess_rebuild_invalidation.py is what holds it to that.
UNREGENERABLE_ANALYSIS_SUBTREES: tuple[str, ...] = (
    f"plots/{EDA_PLOTS_SUBDIR}",
    "eda_local",
)

# POST PROCESSING

LST_COL_HEADERS_NODE_FLOOD_SUMMARY = [
    "node_id",
    "hours_flooded",
    "max_flow_cms",
    "time_of_max_flood_d_hr_mn",
    "tot_flooded_vol_10e6_ltr",
    "max_ponded_depth_m",
]
LST_COL_HEADERS_NODE_FLOW_SUMMARY = [
    "node_id",
    "type",
    "max_lateral_inflow_cms",
    "max_total_inflow_cms",
    "time_of_max_flow_d_hr_mn",
    "lateral_inflow_vol_10e6_ltr",
    "total_inflow_vol_10e6_ltr",
    "flow_balance_error_percent",
]
LST_COL_HEADERS_LINK_FLOW_SUMMARY = [
    "link_id",
    "type",
    "max_flow_cms",
    "time_of_max_flow_d_hr_mn",
    "max_velocity_mps",
    "max_over_full_flow",
    "max_over_full_depth",
]

TEST_SYSTEM_DIRNAME = "tests"
CASE_SYSTEM_DIRNAME = "cases"
TEST_N_REPORTING_TSTEPS_PER_SIM = 12
TEST_TRITON_REPORTING_TIMESTEP_S = 10

# Globus endpoint identifiers
# UUIDs are stable public identifiers for Globus collections.
# Find them at app.globus.org > Collections > search by name.
# Per-user paths (usernames, experiment dirs) belong in configs/transfers/ YAML,
# not here — see configs/transfers/template_transfer.yaml.
UVA_GLOBUS_COLLECTION_NAME = "UVA Standard Security Storage"
UVA_GLOBUS_COLLECTION_UUID = "af187d15-768f-4449-8670-d00e1eb1ce6a"
UVA_GLOBUS_SCRATCH_BASE = "/scratch/{username}"  # expand with os.getenv("USER")

FRONTIER_GLOBUS_COLLECTION_NAME = "OLCF DTN (Globus 5)"
FRONTIER_GLOBUS_COLLECTION_UUID = "36d521b3-c182-4071-b7d5-91db5d380d42"
# Substitute {project} with your OLCF allocation and {username} with your OLCF user.
FRONTIER_GLOBUS_SCRATCH_BASE = "/lustre/orion/{project}/scratch/{username}"
FRONTIER_GLOBUS_PROJECT_BASE = "/lustre/orion/{project}/proj-shared"

# System-name-to-endpoint mapping for PostRunTransferConfig.
# Keys are system names ("uva", "frontier") used by PostRunTransferConfig.
# Values are (source_uuid, scratch_base, needs_data_access, session_domain) tuples.
# needs_data_access: whether the endpoint requires a data_access dependent
#   scope at auth time.  UVA (Globus 5 mapped collection) does; OLCF DTN does not.
# session_domain: identity domain required by the endpoint's access policy.
#   OLCF requires sso.ccs.ornl.gov; UVA has no domain restriction (None).
GLOBUS_SYSTEM_ENDPOINTS: dict[str, tuple[str, str, bool, str | None]] = {
    "uva": (UVA_GLOBUS_COLLECTION_UUID, UVA_GLOBUS_SCRATCH_BASE, True, None),
    "frontier": (
        FRONTIER_GLOBUS_COLLECTION_UUID,
        FRONTIER_GLOBUS_SCRATCH_BASE,
        False,
        "sso.ccs.ornl.gov",
    ),
}


# ============================================================================
# Status-flag builders — single source of truth for `_status/*.flag` paths.
#
# These builders cover the sensitivity-workflow flag shapes and the member input
# fingerprint. They are not yet the only spelling: other sites still write flag
# names inline (for example sensitivity_analysis.py's per-member e_consolidate
# unlink and analysis.py's c_run_ glob), so a change to a flag name must also be
# made at those sites.
#
# Naming convention: descriptive function names communicate purpose at the
# call site (the on-disk `c_run_*` / `d_process_*` / `e_consolidate_*`
# string outputs are unchanged — those are the persistent contract).
# Per the accepted-decision wildcard-charset stipulation, member_id and
# event_id must match `^[A-Za-z0-9_.]+$`; callers are responsible for
# validating at CSV/config load time. The `_validate_id_fragment` helper
# below provides a runtime fast-fail at the builder call site for
# path-fragment-unsafe values (forbidden: `/`, `\`, `.flag`, whitespace).
# ============================================================================

STATUS_DIR_NAME: str = "_status"


def _validate_id_fragment(name: str, value: str) -> None:
    """Reject path-fragment-unsafe values in flag-name builder inputs.

    member_id / event_id end up baked into Snakemake rule names and on-disk
    flag file paths. Forbidden characters can corrupt rule names or paths.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string; got {value!r}")
    for ch in ("/", "\\", ".flag", " ", "\t", "\n"):
        if ch in value:
            raise ValueError(f"{name}={value!r} contains forbidden fragment {ch!r}")


def sim_run_flag_per_member(model_type: str, member_id: str, event_id: str) -> str:
    """Per-member per-event simulation completion flag (sensitivity workflow).

    member_id and event_id must match `^[A-Za-z0-9_.]+$` per the
    accepted-decision wildcard-charset stipulation; this builder fast-fails
    on path-fragment-unsafe inputs via `_validate_id_fragment`.
    """
    _validate_id_fragment("member_id", member_id)
    _validate_id_fragment("event_id", event_id)
    return f"{STATUS_DIR_NAME}/c_run_{model_type}_member-{member_id}_evt-{event_id}_complete.flag"


def process_timeseries_flag_per_member(model_type: str, member_id: str, event_id: str) -> str:
    """Per-member per-event process_timeseries completion flag (sensitivity workflow).

    Same wildcard-charset contract as `sim_run_flag_per_member`.
    """
    _validate_id_fragment("member_id", member_id)
    _validate_id_fragment("event_id", event_id)
    return f"{STATUS_DIR_NAME}/d_process_{model_type}_member-{member_id}_evt-{event_id}_complete.flag"


def consolidate_analysis_flag(member_id: str) -> str:
    """Per-member consolidate completion flag (sensitivity workflow)."""
    _validate_id_fragment("member_id", member_id)
    return f"{STATUS_DIR_NAME}/e_consolidate_member-{member_id}_complete.flag"


def consolidate_experiment_flag() -> str:
    """Experiment-level consolidate completion flag (sensitivity workflow)."""
    return f"{STATUS_DIR_NAME}/f_consolidate_experiment_complete.flag"


def member_inputs_fingerprint_flag(member_id: str) -> str:
    """Per-member input fingerprint file used as an mtime-trigger sentinel for per-member rules."""
    _validate_id_fragment("member_id", member_id)
    return f"{STATUS_DIR_NAME}/member-{member_id}_inputs.json"


# ============================================================================
# Flag-name GRAMMAR, READ in exactly one place.
#
# The builders above MINT `_status/` flag names; this is their inverse, and it is the
# only place in the tree that PARSES a model type out of a flag name. Two shapes are
# minted, and both carry the model in the same position:
#
#   plain analysis:        c_run_{model}_evt-{event_id}_complete.flag
#   sensitivity member:    c_run_{model}_member-{member_id}_evt-{event_id}_complete.flag
#   (and the same two shapes under the d_process_ family)
#
# The DELIMITERS carry the parse, not the id charsets: `_member-` and `_evt-` are
# literal, and the model group is `[a-z]+` terminated by `_` (the three model types are
# `triton`, `tritonswmm`, `swmm`). The member_id charset `^[A-Za-z0-9_.]+$` IS enforced,
# but at CSV/XLSX load by sensitivity_analysis._retrieve_df_setup, NOT by
# _validate_id_fragment above (which forbids only `/`, `\`, `.flag` and whitespace), so
# a builder can be handed a `-`-bearing id; and the event slug from
# scenario.compute_event_id_slug is unconstrained (it joins raw weather-indexer values).
# The id and slug groups are therefore PERMISSIVE. A previous inline parser split on
# `_evt-` alone and read `tritonswmm_member-0` as the model of every member-shaped flag,
# so no sensitivity member's c_run_/d_process_ flag was ever deletable by a force.
# ============================================================================

#: The flag families whose names carry a model segment. e_consolidate_* /
#: f_consolidate_experiment / a_setup / *_inputs.json carry none.
_MODEL_BEARING_FLAG_PREFIXES: tuple[str, ...] = ("c_run_", "d_process_")

FLAG_NAME_WITH_MODEL_RE = re.compile(r"^(?:c_run|d_process)_(?P<model>[a-z]+)_(?:member-.+?_)?evt-.+_complete\.flag$")


def model_type_from_flag_name(name: str) -> str | None:
    """Return the model type a `_status/` flag BASENAME was minted for.

    Inverse of `sim_run_flag_per_member` / `process_timeseries_flag_per_member` and of
    the plain-shape `c_run_{model}_evt-…` / `d_process_{model}_evt-…` names the multisim
    generator emits. Basename only — the builders return `_status/…` paths, callers pass
    `Path.name`; a path component is a caller bug and is refused loudly.

    Returns None ONLY when `name` does not start with a model-bearing family prefix (the
    consolidate / setup / fingerprint families carry no model segment). Raises ValueError
    when `name` starts with `c_run_` / `d_process_` and does not fit the grammar, so a
    consumer deciding whether to DELETE the flag can never fail open on a model-bearing
    name it could not attribute.
    """
    if "/" in name:
        raise AssertionError(f"model_type_from_flag_name takes a basename, got a path: {name!r}")
    m = FLAG_NAME_WITH_MODEL_RE.match(name)
    if m:
        return m.group("model")
    if name.startswith(_MODEL_BEARING_FLAG_PREFIXES):
        raise ValueError(
            f"flag name {name!r} starts with a model-bearing family prefix but does not fit "
            f"the flag-name grammar {FLAG_NAME_WITH_MODEL_RE.pattern!r}"
        )
    return None
