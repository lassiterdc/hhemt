"""Consolidate one analysis's per-scenario summaries into a hierarchical DataTree zarr.

Each scenario's processing step writes small per-model summary stores; nothing
analysis-level exists until they are joined. This module concatenates them along
``event_iloc`` into ``analysis_datatree.zarr``, records where the members' variable sets
differ, and stamps provenance and resume evidence on the tree root.
"""

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from hhemt.cf_conventions import (
    apply_cf_attributes,
    apply_global_attributes,
)
from hhemt.constants import LST_SWMM_SYSTEM_SUMMARY_VARIABLES
from hhemt.exceptions import ProcessingError
from hhemt.member_identity import resolve_member_id_column
from hhemt.scenario import TRITONSWMM_scenario
from hhemt.summary_paths import SUMMARY_CLEARING_FIELD, parse_summary_clearing_record, summary_clearing_record
from hhemt.utils import (
    CONSOLIDATION_STAMP_ATTRS,
    EXPERIMENT_TREE_NAME,
    append_zarr,
    current_datetime_string,
    deferred_consolidation,
    get_file_size_MiB,
    open_store,
    return_dic_zarr_encodings,
    update_zarr_root_attrs,
    zarr_root_attrs,
    zarr_store_has_index,
)

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis


#: Name of the per-``event_iloc`` COORDINATE carrying the across-MEMBERS variable-set
#: finding, stamped immediately before the member concat below. Same shape as the
#: column-set record (``PERF_COLUMN_SET_COORD``, from
#: ``process_simulation.stamp_perf_column_set``) and the producing-sha coordinates
#: ``_retrieve_combined_output`` normalizes: a DIMENSIONED
#: ``event_iloc`` coord. A per-member ``attrs`` stamp is the natural-looking alternative
#: and it is exactly wrong -- the concat's ``combine_attrs="drop_conflicts"`` deletes any
#: attr whose value DISAGREES across members, which is precisely the condition being
#: recorded, so the record would vanish in the only case it exists for.
VARIABLE_SET_COORD: str = "variable_set_across_members"

#: The two ``_MODE_CONFIG`` modes whose variable set is the TRITON coupled-timer column
#: set, and therefore the only two at which a differing variable set can be attributed to
#: a differing solver build.
#:
#: THIS IS NOT A DETECTION ALLOWLIST and must not become one. The heterogeneity DETECTION
#: is deliberately general across all eight modes -- the join NaN-fills identically at
#: every one of them, and because the guard records rather than raises, a false positive
#: costs one coordinate string. What is NOT general is the CAUSAL attribution: at
#: ``swmm_only_node``, ``tritonswmm_swmm_link`` or ``tritonswmm_triton`` a differing
#: variable set has nothing to do with solver builds or timer columns, so a confident
#: solver-build sentence there is a wrong explanation stamped onto a published artifact in
#: the voice of a diagnostic. This tuple selects a CLAUSE OF PROSE and nothing else, which
#: is why a two-name mirror list is acceptable here where a detection allowlist would not
#: be. ``test_the_performance_mode_names_are_real_mode_config_keys`` fails if either name
#: stops being a live mode, which is the drift this mirror could otherwise suffer silently.
_PERFORMANCE_MODES: frozenset[str] = frozenset({"tritonswmm_performance", "triton_only_performance"})


def _heterogeneity_cause(mode: str) -> str:
    """The cause sentence for a short-member finding, licensed by the mode.

    Returned rather than interpolated inline so the performance and non-performance
    wordings sit beside each other and neither can be edited without seeing the other.
    """
    if mode in _PERFORMANCE_MODES:
        return "Most likely these members were produced by different solver builds emitting different column sets."
    return (
        "The cause is mode-specific and is not diagnosed here -- for this mode the live "
        "candidates are differing model configuration between the members or a differing "
        "parse of a member's model outputs, NOT a differing solver build."
    )


def describe_member_variable_sets(
    labels: list[str],
    variable_sets: list[object],
    *,
    mode: str,
) -> list[str]:
    """Per-member finding strings for the objects about to be joined at the member concat.

    One string per member, positionally aligned with ``labels``. A SHORT member names what
    IT lacks; a COMPLETE member in a mixed population says so and reports the population
    count, so a reader who opens ANY member learns whether the population was mixed rather
    than only a reader who happens to open a short one. A homogeneous population yields the
    same uniform string on every member.
    """
    from hhemt.process_simulation import (  # function-local: keeps the import graph acyclic
        diagnose_name_set_heterogeneity,
        format_missing_names,
    )

    union, short = diagnose_name_set_heterogeneity(labels, variable_sets)
    n = len(labels)
    if not short:
        return [f"uniform: all {n} member(s) carry the same {len(union)} variable(s) for mode {mode!r}"] * n
    missing_any = sorted({m for ms in short.values() for m in ms})
    out: list[str] = []
    for label in labels:
        if label in short:
            out.append(
                f"HETEROGENEOUS: this member (event_iloc {label}) LACKS variable(s) "
                f"{format_missing_names(short[label])} for mode {mode!r}; {len(short)} of {n} "
                "member(s) are short of the analysis-wide union. The join fills those "
                "variables with NaN for this member, so any cross-member statistic over "
                f"them is computed on a subset of the members. {_heterogeneity_cause(mode)}"
            )
        else:
            out.append(
                f"HETEROGENEOUS: this member (event_iloc {label}) carries the full variable "
                f"set for mode {mode!r}, but {len(short)} of {n} member(s) are short of it "
                f"(missing across the analysis: {format_missing_names(missing_any)}). "
                "Cross-member statistics over those variables are computed on a subset."
            )
    return out


def _assert_store_index_current(store, operation: str) -> None:
    """Raise ``ProcessingError`` if the store just written does not carry a current consolidated-metadata index.

    The end-of-step run of ``analysis_validation.check_consolidated_metadata_current``: the store was published
    by ``write_datatree_zarr`` a moment ago, so a difference here is a defect in the write path and not a
    legacy store, and it must stop the step rather than reach a reader that trusts the index.
    """
    from hhemt.analysis_validation import check_consolidated_metadata_current
    from hhemt.exceptions import ProcessingError

    verdict = check_consolidated_metadata_current(store)
    if not verdict.passed:
        raise ProcessingError(operation=operation, filepath=str(store), reason=verdict.summary)


class TRITONSWMM_analysis_post_processing:
    """Analysis-level consolidation for one TRITONSWMM_analysis.

    Reads each scenario's summary store (and, opt-in, its SWMM timeseries stores),
    concatenates them along ``event_iloc``, and writes ``analysis_datatree.zarr``.
    """

    # Maps consolidation mode to: (scenario_path_attr, spatial_coords)
    _MODE_CONFIG = {
        "tritonswmm_triton": (
            "output_tritonswmm_triton_summary",
            ["x", "y"],
        ),
        "tritonswmm_swmm_node": (
            "output_tritonswmm_node_summary",
            "node_id",
        ),
        "tritonswmm_swmm_link": (
            "output_tritonswmm_link_summary",
            "link_id",
        ),
        "triton_only": (
            "output_triton_only_summary",
            ["x", "y"],
        ),
        "triton_only_performance": (
            "output_triton_only_performance_summary",
            None,
        ),
        "tritonswmm_performance": (
            "output_tritonswmm_performance_summary",
            None,
        ),
        "swmm_only_node": (
            "output_swmm_only_node_summary",
            "node_id",
        ),
        "swmm_only_link": (
            "output_swmm_only_link_summary",
            "link_id",
        ),
        # The standalone SWMM system summary shares the node summary's stem and selects the twelve system reductions
        # from it (`_STEM_VARIABLE_SELECTION`); it has no spatial dimension.
        "swmm_only_system": (
            "output_swmm_only_node_summary",
            None,
        ),
    }

    # Mode -> (how the mode picks variables from its per-scenario stem, the variable names). A stem that feeds two
    # groups (the standalone SWMM node summary store holds the system reductions beside the node variables, D19) is
    # split so that each variable lands in exactly one group: ``"include"`` keeps only the named variables and
    # ``"exclude"`` drops them. A mode absent here passes every variable of its stem through.
    _STEM_VARIABLE_SELECTION = {
        "swmm_only_system": ("include", tuple(LST_SWMM_SYSTEM_SUMMARY_VARIABLES)),
        "swmm_only_node": ("exclude", tuple(LST_SWMM_SYSTEM_SUMMARY_VARIABLES)),
    }

    # Every group of the consolidated tree is `{family}/{kind}` and holds TWO branches, `summary` (one
    # statistic per event and entity, REBUILDABLE from the per-scenario summary stores and rewritten per group
    # in place) and `timeseries` (APPEND-ONLY, one scenario at a time). The summary modes write the first.
    _MODE_TO_TREE_PATH = {
        "tritonswmm_triton": "tritonswmm/triton/summary",
        "tritonswmm_swmm_node": "tritonswmm/swmm_node/summary",
        "tritonswmm_swmm_link": "tritonswmm/swmm_link/summary",
        "tritonswmm_performance": "tritonswmm/performance/summary",
        "triton_only": "triton_only/triton/summary",
        "triton_only_performance": "triton_only/performance/summary",
        "swmm_only_node": "swmm_only/swmm_node/summary",
        "swmm_only_link": "swmm_only/swmm_link/summary",
        "swmm_only_system": "swmm_only/system/summary",
    }

    # Mode -> the per-scenario summary stem the mode reads. Two modes can read one stem (the standalone SWMM node
    # store holds the system reductions beside the node variables), and a stem's record in the model log names every
    # `summary` group it fed and the variables each group received.
    _MODE_STEM = {
        "tritonswmm_triton": "TRITONSWMM_TRITON_summary",
        "tritonswmm_swmm_node": "TRITONSWMM_SWMM_node_summary",
        "tritonswmm_swmm_link": "TRITONSWMM_SWMM_link_summary",
        "tritonswmm_performance": "TRITONSWMM_perf_summary",
        "triton_only": "TRITON_only_summary",
        "triton_only_performance": "TRITON_only_perf_summary",
        "swmm_only_node": "SWMM_only_node_summary",
        "swmm_only_system": "SWMM_only_node_summary",
        "swmm_only_link": "SWMM_only_link_summary",
    }

    # The coupled node summary mode keeps the coupling-node flag in its `timeseries` group alone once its summary
    # branch is append-only, so a variable is never stored in two groups of one tree.
    _COUPLING_FLAG = "is_coupling_node"

    # Per-scenario TIMESERIES modes (opt-in via analysis_config.toggle_consolidate_timeseries).
    # Maps mode -> the scenario-path attr for the per-scenario series store: the SWMM node and link stores and the
    # gridded TRITON store (`timestep`, `y`, `x`) of EVERY model family that writes series: the coupled model,
    # the SWMM-only model (node and link) and the TRITON-only model (gridded). Each lands in its own `timeseries`
    # group of analysis_datatree.zarr on that group's own `elapsed_s` axis. The gridded store is the large
    # one -- about 1 GB uncompressed per scenario per kept variable on a Norfolk grid -- which is why its slice is
    # written and read back in blocks and why the keep-list prune (`timeseries_keep_triton_gridded`) runs
    # before it is consolidated.
    _TIMESERIES_MODE_CONFIG = {
        "tritonswmm_swmm_node_ts": "output_tritonswmm_node_time_series",
        "tritonswmm_swmm_link_ts": "output_tritonswmm_link_time_series",
        "tritonswmm_triton_ts": "output_tritonswmm_triton_timeseries",
        "swmm_only_swmm_node_ts": "output_swmm_only_node_time_series",
        "swmm_only_swmm_link_ts": "output_swmm_only_link_time_series",
        "swmm_only_system_ts": "output_swmm_only_system_time_series",
        "triton_only_triton_ts": "output_triton_only_timeseries",
    }

    _TIMESERIES_MODE_TO_TREE_PATH = {
        "tritonswmm_swmm_node_ts": "tritonswmm/swmm_node/timeseries",
        "tritonswmm_swmm_link_ts": "tritonswmm/swmm_link/timeseries",
        "tritonswmm_triton_ts": "tritonswmm/triton/timeseries",
        "swmm_only_swmm_node_ts": "swmm_only/swmm_node/timeseries",
        "swmm_only_swmm_link_ts": "swmm_only/swmm_link/timeseries",
        "swmm_only_system_ts": "swmm_only/system/timeseries",
        "triton_only_triton_ts": "triton_only/triton/timeseries",
    }

    # Mode -> the model whose per-scenario log records that mode's clearing. A model has one log of its own
    # (`log_{model}.json`), so the cleared and pending fields of the three tables below are read from, and written
    # to, the log of the model that wrote the store: the coupled model's log for the coupled modes, the standalone
    # SWMM model's for the SWMM-only modes, the standalone TRITON model's for the TRITON-only mode.
    _TIMESERIES_MODE_MODEL = {
        "tritonswmm_swmm_node_ts": "tritonswmm",
        "tritonswmm_swmm_link_ts": "tritonswmm",
        "tritonswmm_triton_ts": "tritonswmm",
        "swmm_only_swmm_node_ts": "swmm",
        "swmm_only_swmm_link_ts": "swmm",
        "swmm_only_system_ts": "swmm",
        "triton_only_triton_ts": "triton",
    }

    # Mode -> (the log bool that records the store's clearing as done, the log list that records it as pending).
    _TIMESERIES_CLEARED_FIELDS = {
        "tritonswmm_swmm_node_ts": (
            "consolidated_node_timeseries_cleared",
            "consolidated_node_timeseries_clearing_pending",
        ),
        "tritonswmm_swmm_link_ts": (
            "consolidated_link_timeseries_cleared",
            "consolidated_link_timeseries_clearing_pending",
        ),
        "tritonswmm_triton_ts": (
            "consolidated_triton_timeseries_cleared",
            "consolidated_triton_timeseries_clearing_pending",
        ),
        "swmm_only_swmm_node_ts": (
            "consolidated_node_timeseries_cleared",
            "consolidated_node_timeseries_clearing_pending",
        ),
        "swmm_only_swmm_link_ts": (
            "consolidated_link_timeseries_cleared",
            "consolidated_link_timeseries_clearing_pending",
        ),
        "swmm_only_system_ts": (
            "consolidated_system_timeseries_cleared",
            "consolidated_system_timeseries_clearing_pending",
        ),
        "triton_only_triton_ts": (
            "consolidated_triton_timeseries_cleared",
            "consolidated_triton_timeseries_clearing_pending",
        ),
    }

    # Bytes one write chunk of a series group, and one read-back block, is sized to. 32 MB is the developer-
    # facing chunk the shape fixes; a Norfolk gridded step is 537 x 551 float32 = 1.18 MB, so the block is 27 steps.
    _TIMESERIES_BLOCK_BYTES = 32_000_000

    def __init__(self, analysis: "TRITONSWMM_analysis") -> None:
        self._analysis = analysis

    # 2: the SWMM node/link time series left the datatree for analysis_timeseries.zarr (elapsed_s axis), the
    # coupling-node flag is always present in a coupled node store, and the flux and clearing toggles join the
    # fingerprint. Every tree consolidated at version 1 rebuilds once.
    # 3: every group holds a `summary` and a `timeseries` branch inside analysis_datatree.zarr, and
    # analysis_timeseries.zarr does not exist. Every tree consolidated at version 2 rebuilds its summary branches once
    # and its series branches not at all.
    CONSOLIDATION_VERSION = 3

    def _consolidation_inputs_fingerprint(self) -> str:
        """Hash of every input that determines the SHAPE of the consolidated tree.

        The consolidate guard rebuilds when this differs from the stamped value, so a
        consolidation-affecting config change invalidates an otherwise-complete tree
        with no operator action and no artifact deletion.

        Evaluated against ``self._analysis.cfg_analysis`` — which, because every consolidate
        Snakemake rule runs ``consolidate_workflow`` as a separate process that reads a persisted
        analysis config, is the config that SUBPROCESS loaded from disk, i.e. the same object
        the toggle-gated loops below read. That is why this predicate works where a
        driver-side kwarg cannot: the driver's in-memory config never reaches here.

        MAINTENANCE CONTRACT: add a field here whenever a config value starts gating
        WHAT LANDS in the tree. Do NOT add values that affect only bytes
        (``compression_level``) or provenance (git sha, timestamps) — those change on
        every run and would force a spurious rebuild every time.

        ONE TERM HERE IS NOT A CONFIG VALUE, AND THAT IS DELIBERATE. ``scenario_ids``
        hashes the SET of scenarios the tree is keyed by. Without it the guard is
        asymmetric: adding a scenario leaves every config field unchanged, so a
        complete-and-stamped tree is reused and the new scenario silently never lands.
        A count would not close it either — remove one scenario and add another and the
        count is identical. Do NOT remove this term as a contract violation; the
        contract is about what gates WHAT LANDS, and the scenario set gates it directly.
        """
        import hashlib
        import json

        from hhemt.scenario import compute_event_id_slug

        payload = {
            "consolidation_version": self.CONSOLIDATION_VERSION,
            "toggle_consolidate_timeseries": bool(
                getattr(self._analysis.cfg_analysis, "toggle_consolidate_timeseries", False)
            ),
            "toggle_coupling_exchange_flow": bool(
                getattr(self._analysis.cfg_analysis, "toggle_coupling_exchange_flow", False)
            ),
            "toggle_clear_scenario_timeseries_after_consolidation": bool(
                getattr(self._analysis.cfg_analysis, "toggle_clear_scenario_timeseries_after_consolidation", False)
            ),
            "timeseries_keep": self._timeseries_keep_for_fingerprint(),
            "enabled_model_types": sorted(self._analysis._get_enabled_model_types()),
            "scenario_ids": sorted(
                compute_event_id_slug(self._analysis._retrieve_weather_indexer_using_integer_index(ei))
                for ei in self._analysis.df_sims.index
            ),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _timeseries_keep_for_fingerprint(self) -> dict[str, list[str]]:
        """The RESOLVED keep-lists, per kind and enabled family, sorted so a reordered list does not rebuild.

        The keep-lists gate which variables land in the ``timeseries`` branches, so what they RESOLVE to (not the
        raw field text: ``None`` and ``"all"`` are one selection) joins the fingerprint.
        """
        from hhemt.timeseries_selection import FAMILIES_BY_KIND, keep_setting, resolve_keep

        enabled = set(self._analysis._get_enabled_model_types())
        family_of = {"tritonswmm": "tritonswmm", "swmm": "swmm_only", "triton": "triton_only"}
        families = {family_of[m] for m in enabled if m in family_of}
        out: dict[str, list[str]] = {}
        for kind, kind_families in FAMILIES_BY_KIND.items():
            value = keep_setting(self._analysis.cfg_analysis, kind)
            for family in kind_families:
                if family in families:
                    out[f"{kind}/{family}"] = sorted(resolve_keep(kind, family, value))
        return out

    def consolidate_to_datatree(
        self,
        compression_level: int = 5,
        verbose: bool = False,
        *,
        df_status=None,
    ) -> Path:
        """Assemble per-scenario summaries directly into a hierarchical DataTree zarr.

        No intermediate master-level per-mode flat zarrs are produced. The
        DataTree IS the canonical master-level artifact; per-scenario summaries
        are its only inputs.

        This method has no overwrite parameter. An existing tree is reused only when it
        is present, its log marks it complete, and both the inputs fingerprint and the
        consolidation build stamp still match; otherwise it is rebuilt. A forced rerun is
        requested through ``override_force_rerun`` on ``TRITONSWMM_analysis.run`` or
        ``submit_workflow``.
        """
        fname_out = self._analysis.analysis_paths.analysis_datatree_zarr
        if fname_out is None:
            raise ValueError("analysis_datatree_zarr path is not configured on AnalysisPaths.")

        # .exists() alone is an unreliable completion signal — a
        # present-but-corrupt zarr (a write that crashed mid-stream) .exists()
        # as True. Align with open_datatree()'s canonical signal: "already
        # consolidated" iff it exists AND datatree_consolidation_complete is True
        # (set only on a successful full write below). Present-but-incomplete
        # falls through to a clean rebuild.
        self._analysis._refresh_log()
        _log_complete = (
            hasattr(self._analysis.log, "datatree_consolidation_complete")
            and self._analysis.log.datatree_consolidation_complete.get() is True
        )
        # Staleness, distinct from the corruption signal above. A tree can be
        # intact AND complete and still have been built from different inputs than
        # the ones now on disk (the 2026-07-20 case: toggle_consolidate_timeseries
        # flipped ON in every persisted per-sub config, consolidation re-ran, output
        # silently unchanged). An ABSENT stamp is treated as stale, not as a match —
        # every tree consolidated before this stamp existed rebuilds exactly once.
        _inputs_fingerprint = self._consolidation_inputs_fingerprint()
        _stamped_fingerprint = (
            self._analysis.log.consolidation_inputs_fingerprint.get()
            if hasattr(self._analysis.log, "consolidation_inputs_fingerprint")
            else None
        )
        _inputs_match = _stamped_fingerprint == _inputs_fingerprint
        # Third reuse input, alongside _log_complete and _inputs_match. The reuse
        # early-return is the FIRST branch below, so this term must appear in BOTH
        # conditions: a disjunct added only to the rebuild branch would be unreachable.
        from hhemt.provenance import store_build_mismatch

        _stamped_build = (
            self._analysis.log.consolidation_build_stamp.get()
            if hasattr(self._analysis.log, "consolidation_build_stamp")
            else None
        )
        _build_why = store_build_mismatch(fname_out, _stamped_build)
        _build_mismatch = _build_why is not None
        # Fourth reuse input: the STORE's own root stamp. The summary branches are written in place and the root
        # attributes last (``_write_summary_branches``), and the reprocess delete removes them first, so a store
        # whose root carries no stamp, or another build's, is one a write or a delete was interrupted in. It is
        # read as incomplete however the log reads, and every summary group is then rewritten.
        _root_attrs = zarr_root_attrs(fname_out) if fname_out.exists() else {}
        _stamp_match = (
            _root_attrs.get("consolidation_version") == self.CONSOLIDATION_VERSION
            and _root_attrs.get("consolidation_inputs_fingerprint") == _inputs_fingerprint
        )
        if fname_out.exists() and _log_complete and _inputs_match and not _build_mismatch and not _stamp_match:
            if verbose:
                print(f"DataTree zarr present at {fname_out} but its root carries no current stamp — rewriting groups.")
        if fname_out.exists() and _log_complete and _inputs_match and not _build_mismatch and _stamp_match:
            if verbose:
                print(f"DataTree zarr already present at {fname_out} and log complete. Not overwriting.")
            return fname_out
        if fname_out.exists() and _log_complete and (not _inputs_match or _build_mismatch):
            # NO PRE-DELETE, and for a stronger reason than before: this store holds the `timeseries` branches,
            # which with the clearing switch on exist nowhere else, so nothing here may remove or replace the
            # store as a whole. The summary groups are rewritten one by one in place (_write_summary_branches).
            # A reason this tier also owns: _retrieve_combined_output below raises FileNotFoundError when a
            # summary is absent, and when this method is called from the master's member-ensure loop that
            # exception is swallowed by allow_incomplete=True and the member is skipped -- so a pre-delete here
            # would destroy the member's store AND drop it from the master tree in one pass, silently.
            if verbose:
                _why = (
                    _build_why
                    if _build_mismatch
                    else (
                        "consolidation inputs changed "
                        f"(stamped={_stamped_fingerprint!r}, current={_inputs_fingerprint!r})"
                    )
                )
                print(f"DataTree zarr present at {fname_out} but {_why} — rebuilding.")
        if fname_out.exists() and not _log_complete:
            # NO PRE-DELETE -- see the sibling arm above.
            if verbose:
                print(f"DataTree zarr present at {fname_out} but log incomplete — rebuilding (treating as corrupt).")

        start_time = time.time()
        tree_dict: dict[str, xr.Dataset] = {}
        for mode, tree_path in self._MODE_TO_TREE_PATH.items():
            scen_path_attr = self._MODE_CONFIG[mode][0]
            first_scen = TRITONSWMM_scenario(self._analysis.df_sims.index[0], self._analysis)
            if getattr(first_scen.scen_paths, scen_path_attr) is None:
                continue
            ds = self._retrieve_combined_output(mode)
            if mode in self._STEM_VARIABLE_SELECTION and not ds.data_vars:
                # A mode that selects variables from a shared stem has nothing to write when no scenario's stem holds
                # them (a stem written before the system reductions existed); an empty group would only mislead.
                continue
            apply_cf_attributes(ds, mode)
            tree_dict[tree_path] = ds

        # Per-scenario time series (opt-in, default OFF) are NOT built into this in-memory tree: they are appended
        # one scenario at a time to the `timeseries` branches of the same store, inside the write block of
        # `_write_summary_branches` below. Rebuilding the summaries therefore never touches a series, which with
        # the clearing toggle on exists nowhere else. A missing per-scenario store raises FileNotFoundError there
        # exactly as the summary path does, which the master's member-ensure loop already tolerates under
        # allow_incomplete=True.

        tree_dict["/"] = xr.Dataset(
            attrs={
                "analysis_id": str(self._analysis.cfg_analysis.analysis_id),
                "output_creation_date": current_datetime_string(),
                "consolidation_version": self.CONSOLIDATION_VERSION,
                "consolidation_inputs_fingerprint": _inputs_fingerprint,
            }
        )
        tree = xr.DataTree.from_dict(tree_dict)
        apply_global_attributes(tree, analysis_id=str(self._analysis.cfg_analysis.analysis_id))

        from hhemt.cf_conventions import apply_producing_stamp, apply_provenance_core
        from hhemt.provenance import emit_provenance

        # BOTH halves, from one pass. `emitted_attrs` is what makes the crate's descriptors
        # STORE-SOURCED; omitting it here leaves the crate on the static base map and therefore
        # override-blind, which publishes the static base-map label for a mode whose store stamps a
        # different `long_name`. This is the only production call threading
        # `emitted_attrs` into the analysis/member-tier crate; the experiment tier's own
        # call is in sensitivity_analysis.py.
        _emitted_vars, _emitted_attrs = emitted_vars_and_attrs(_ds for _p, _ds in tree_dict.items() if _p != "/")
        _core_json, _graph_json = emit_provenance(
            self._analysis, emitted_vars=_emitted_vars, emitted_attrs=_emitted_attrs
        )
        apply_provenance_core(tree, core_json_str=_core_json)

        # Re-derive the scalar producing-stamp fast-path on the
        # root from the per-event coordinates that rode up on the assembled mode
        # datasets. Set only when uniform across ALL events (else absent + a
        # divergent breadcrumb); the per-event coordinate stays authoritative.
        _sha_vals: list[str] = []
        _semver_vals: list[str] = []
        for _mode_path, _mode_ds in tree_dict.items():
            if _mode_path == "/":
                continue
            if "hhemt_producing_sha" in _mode_ds.coords:
                _sha_vals.extend(str(v) for v in _mode_ds["hhemt_producing_sha"].values.tolist())
            if "hhemt_producing_version" in _mode_ds.coords:
                _semver_vals.extend(str(v) for v in _mode_ds["hhemt_producing_version"].values.tolist())
        apply_producing_stamp(tree, _sha_vals, _semver_vals)

        _stamp_triton_provenance(tree, self._analysis)
        _stamp_coupled_resume_evidence(tree, self._analysis, df_status=df_status)

        self._write_summary_branches(tree, fname_out, compression_level=compression_level, verbose=verbose)
        _assert_store_index_current(fname_out, "consolidate_to_datatree")

        from hhemt.metadata import write_rocrate_sidecar

        write_rocrate_sidecar(self._analysis.analysis_paths.analysis_dir, graph_json=_graph_json)

        # ARM REFRESH AT THE WRITE BOUNDARY. Three other sites write this file: eda(), publish()
        # (publishing._deposit_set), and the sensitivity master's consolidate_sensitivity_datatree,
        # which writes it when the directory runs as a master. A regular analysis that
        # CONSOLIDATED AND STOPPED runs none of them, so without this write it left
        # cfg_analysis.yaml asserting the PRIOR arm while the newest store beside it
        # was the current arm's. Written here, at the boundary that just wrote the store,
        # the file describes the store it sits next to.
        # UNCONDITIONAL and idempotent: no read of the on-disk arm is performed, because a
        # divergence test would have to parse and trust the value this write replaces, and
        # would decline to fire on a malformed file -- the state most in need of repair.
        # Perturbs no scheduling: this file is a declared input of NO emitted Snakemake
        # rule, and the mtime-preserving rewrite primitive is migration-only.
        # DOES NOT cover readers for which this file is the ONLY record of the arm: an
        # archived tree, a bundle emitted from a root whose file was already stale, and a
        # migration still cannot detect that a directory's arm was changed back.
        import yaml as _yaml

        (self._analysis.analysis_paths.analysis_dir / "cfg_analysis.yaml").write_text(
            _yaml.safe_dump(self._analysis.cfg_analysis.model_dump(mode="json"))
        )

        self._analysis._refresh_log()
        if hasattr(self._analysis.log, "datatree_consolidation_complete"):
            self._analysis.log.datatree_consolidation_complete.set(True)
        if hasattr(self._analysis.log, "consolidation_version"):
            self._analysis.log.consolidation_version.set(self.CONSOLIDATION_VERSION)
        if hasattr(self._analysis.log, "consolidation_inputs_fingerprint"):
            self._analysis.log.consolidation_inputs_fingerprint.set(_inputs_fingerprint)
        # The convergent operand for the build gate: stamp the CONSOLIDATION build that
        # just wrote this store, minted inline per producing_stamp()'s rule that a stamp
        # records the code that actually ran the stage. Placed beside the fingerprint set
        # deliberately: both are written by the stage they gate and read back by that same
        # stage's reuse check, which is what lets each converge. Deliberately NOT the tree
        # root's hhemt_producing_sha: that value is frozen at PROCESSING time, so comparing
        # it against the running consolidation build mismatches on every call -- each
        # rebuild re-derives the same processing-stage sha, and the next consolidation
        # rebuilds again, indefinitely.
        if hasattr(self._analysis.log, "consolidation_build_stamp"):
            from hhemt.provenance import _write_consolidate_manifest
            from hhemt.provenance import producing_stamp as _producing_stamp

            _stamp = _producing_stamp()
            self._analysis.log.consolidation_build_stamp.set(str(_stamp.get("hhemt_sha") or ""))
            _write_consolidate_manifest(self._analysis.analysis_paths.analysis_dir, _stamp)
        elapsed_s = time.time() - start_time
        self._analysis.log.add_sim_processing_entry(fname_out, get_file_size_MiB(fname_out), elapsed_s, True)

        # Write the analysis-level DU sentinel. Compare-and-write semantics in
        # du_sentinels.write_du_sentinel preserve mtime on idempotent re-runs,
        # so consumer rules that declare _du.json as input: are not cascade-rerun.
        from hhemt.du_sentinels import _infer_scope, sum_child_sentinels

        # DERIVED, never hardcoded. `scope` here is a BEHAVIOURAL SWITCH, not a label:
        # sum_child_sentinels (du_sentinels.py) tests `scope == "analysis"` to add
        # _EPHEMERAL_TOP_LEVEL_DIRS = {"_test"} to its skip set, so the value changes the
        # computed total and not merely the recorded string.
        #
        # There are three production call sites -- consolidate_workflow.main (the runner,
        # both the master/multisim path and the per-member `--member-id` path), the
        # per-member loop inside sensitivity_analysis's consolidate_sensitivity_datatree
        # (which calls this for every member unconditionally), and
        # TRITONSWMM_analysis._consolidate_analysis_outputs. A member's own
        # `analysis_paths.analysis_dir` IS its member root, so a literal "analysis"
        # mislabels every member root any of them touches; the site where this derivation
        # does the real work is that per-member loop, which runs for every member each
        # time the master consolidate runs.
        #
        # The per-member DU-sentinel write in consolidate_workflow.main (a scope="member" sentinel,
        # written when `--member-id` is passed) corrected the label on the
        # per-member RULE path -- but it fires as a rule SIDE-EFFECT. Deriving here
        # removes the dependency on any rule firing again.
        _du_scope_dir = self._analysis.analysis_paths.analysis_dir
        sum_child_sentinels(
            _du_scope_dir,
            scope=_infer_scope(_du_scope_dir, _du_scope_dir),
            child_scope_dirs=["members", "sims"],
        )

        if verbose:
            print(f"Wrote DataTree zarr to {fname_out} in {elapsed_s:.1f}s")
        return fname_out

    def _write_summary_branches(
        self,
        tree: "xr.DataTree",
        store: Path,
        *,
        compression_level: int = 5,
        verbose: bool = False,
    ) -> None:
        """Write the ``summary`` groups of ``tree`` into ``store`` IN PLACE, then the series, the clearing, the root.

        The store is never replaced as a whole (``write_datatree_zarr``'s publish through a rename-aside is not
        used for it), because it also holds the ``timeseries`` branches, which are append-only and which with the
        clearing switch on are the only copy of the series. One ``deferred_consolidation`` block covers the whole
        write, so the zarr index is rebuilt once at the end, and a kill inside it leaves a store with no index
        that the next consolidation reads as incomplete. Order inside the block: (1) the root stamp is removed, so
        an interrupted write is recognisable; (2) each summary group is written by ``_write_summary_group``, which
        rewrites the group whole while every scenario still holds its own summary store and otherwise plans each
        event; (3) a ``summary`` group of a mode the analysis no longer produces is removed unless a scenario's
        record names it; (4) the ``timeseries`` branches are appended when the toggle is on; (5) with the clearing
        switch on, each scenario's per-scenario summary stores are deleted once every slice reads back; (6) the root
        attributes, stamp included, are written last. A scenario recorded as cleared that has left the analysis
        refuses before anything is written, because its slice is the only copy and a whole-group rewrite would drop it.
        """
        from hhemt.du_sentinels import delete_and_account

        at_layout_30 = self._groups_holding_layout_30_arrays(Path(store))
        if at_layout_30:
            from hhemt.version_migration.constants import LAYOUT_VERSION

            raise ProcessingError(
                "write the summary branches of the consolidated tree",
                str(store),
                f"{at_layout_30[0]} holds a summary directly at its layout-30 path (and {len(at_layout_30) - 1} other "
                f"group(s) like it); layout {LAYOUT_VERSION} stores it at {{family}}/{{kind}}/summary. Writing a "
                "`summary` group inside it would leave a hybrid tree, so migrate the tree (hhemt migrate) first.",
            )
        self._assert_no_cleared_event_dropped(Path(store), tree)
        written: set[str] = set()
        clear = bool(
            getattr(self._analysis.cfg_analysis, "toggle_clear_scenario_timeseries_after_consolidation", False)
        )
        with deferred_consolidation(store):
            update_zarr_root_attrs(store, remove=CONSOLIDATION_STAMP_ATTRS)
            for path, node in tree.subtree_with_keys:
                if not node.has_data:
                    continue
                group = path.lstrip("/")
                self._write_summary_group(Path(store), group, node.dataset, compression_level=compression_level)
                written.add(group)
            for stale in sorted(set(self._MODE_TO_TREE_PATH.values()) - written):
                stale_dir = Path(store) / stale
                if stale_dir.exists() and not self._group_named_by_a_record(stale):
                    delete_and_account(
                        [stale_dir], scope_dir=self._analysis.analysis_paths.analysis_dir, scope="analysis"
                    )
            if getattr(self._analysis.cfg_analysis, "toggle_consolidate_timeseries", False):
                self.consolidate_timeseries(verbose=verbose)
            if clear:
                self._clear_summary_stores(Path(store), drop_flag=self._exclude_coupling_flag())
            update_zarr_root_attrs(store, dict(tree.attrs))

    # ------------------------------------------------------------------------------------------------
    # The `summary` branches: group writes, the per-scenario clearing record, and the branch read
    # ------------------------------------------------------------------------------------------------

    @staticmethod
    def _mode_model(mode: str) -> str:
        """The model whose per-scenario log records ``mode``'s summary stem: the model that wrote the store."""
        if mode.startswith("tritonswmm"):
            return "tritonswmm"
        if mode.startswith("triton"):
            return "triton"
        return "swmm"

    def _mode_of_group(self, group: str) -> str | None:
        """The summary mode whose tree path ends ``group`` (a master's member prefix is ignored), or None."""
        for mode, tree_path in self._MODE_TO_TREE_PATH.items():
            if group == tree_path or group.endswith("/" + tree_path):
                return mode
        return None

    def _event_sim_folder(self, event_iloc: int) -> Path:
        """``sims/{event_id}`` of the event, built from the slug alone so no scenario is constructed."""
        return Path(self._analysis.analysis_paths.simulation_directory) / self._event_slug(int(event_iloc))

    def _summary_records(self, mode: str) -> dict[int, list[str]]:
        """Each event whose model log records ``mode``'s group as cleared, with the variable names the group received.

        Read straight from ``sims/{event_id}/log_{model}.json``, so it is cheap and builds no scenario. An event
        whose record does not name this mode's stem and group is not in the result, which is what keeps the two
        groups fed by one stem apart.
        """
        model = self._mode_model(mode)
        stem = self._MODE_STEM[mode]
        group = self._MODE_TO_TREE_PATH[mode]
        found: dict[int, list[str]] = {}
        for event_iloc in self._analysis.df_sims.index:
            record = summary_clearing_record(self._event_sim_folder(event_iloc), model)
            names = None if record is None else record["stems"].get(stem, {}).get(group)
            if names is not None:
                found[int(event_iloc)] = [str(n) for n in names]
        return found

    def _group_named_by_a_record(self, group: str) -> bool:
        """Whether any scenario's record names ``group``."""
        mode = self._mode_of_group(group)
        return mode is not None and bool(self._summary_records(mode))

    def _exclude_coupling_flag(self) -> bool:
        """Whether the coupled node summary leaves out the coupling-node flag.

        True when the clearing switch is on, and also when any scenario already records the node summary group as
        cleared: the group is then append-only, and turning the switch off later must not bring the flag into it as
        a new array beside the copy the node ``timeseries`` group already holds. With neither, nothing changes.
        """
        if bool(getattr(self._analysis.cfg_analysis, "toggle_clear_scenario_timeseries_after_consolidation", False)):
            return True
        return bool(self._summary_records("tritonswmm_swmm_node"))

    def _summary_tree_location(self) -> tuple[Path, str]:
        """The store holding this analysis's ``summary`` branches and the group prefix inside it.

        The analysis's own tree and no prefix, except for a sensitivity member whose log records its tree as moved
        (``pending`` or ``done``): its groups are then under ``{analysis_id}/`` of the master's
        ``experiment_datatree.zarr``, two directories above the member directory.
        """
        analysis = self._analysis
        analysis._refresh_log()
        moved = getattr(analysis.log, "member_tree_moved_to_master", None)
        if moved is not None and moved.get() in ("pending", "done"):
            member_dir = Path(analysis.analysis_paths.analysis_dir)
            return member_dir.parent.parent / EXPERIMENT_TREE_NAME, f"{analysis.cfg_analysis.analysis_id}/"
        store = analysis.analysis_paths.analysis_datatree_zarr
        if store is None:
            raise ValueError("analysis_datatree_zarr path is not configured on AnalysisPaths.")
        return Path(store), ""

    @staticmethod
    def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
        """Whether two arrays hold the same values, a missing value (NaN or NaT) matching a missing value."""
        if a.shape != b.shape:
            return False
        if a.dtype.kind in "fc" and b.dtype.kind in "fc":
            return bool(np.array_equal(a, b, equal_nan=True))
        if a.dtype.kind in "mM" and b.dtype.kind in "mM":
            gone_a, gone_b = np.isnat(a), np.isnat(b)
            return bool(np.array_equal(gone_a, gone_b) and np.array_equal(a[~gone_a], b[~gone_b]))
        return bool(np.array_equal(a, b))

    #: Smallest character width stored for a string coordinate along ``event_iloc``. The two per-event findings are
    #: sentences that grow with the variables they name, so they get far more room than an identifier or a stamp.
    _EVENT_STRING_WIDTH = {VARIABLE_SET_COORD: 2048, "perf_column_set_across_allocations": 2048}
    _EVENT_STRING_WIDTH_DEFAULT = 255

    @staticmethod
    def _prepare_summary_group(ds: xr.Dataset, compression_level: int) -> tuple[xr.Dataset, dict]:
        """``ds`` and the encoding that store a ``summary`` group one event per chunk, ready for ``to_zarr``.

        Chunk length 1 along ``event_iloc`` for every array that carries it, so an append or a region write of one
        event rewrites no chunk file that holds another event. A string coordinate along ``event_iloc`` is stored
        fixed-width at no less than ``_EVENT_STRING_WIDTH`` characters and compressed, because a fixed-width array
        keeps the width of the events present when it was created and a later event with a longer value would be cut
        short (a variable-length string is not used, since dask cannot size it when a reader opens the tree with
        automatic chunks).
        """
        prepared = ds.chunk({d: (1 if d == "event_iloc" else ds.sizes[d]) for d in ds.dims})
        encoding = return_dic_zarr_encodings(prepared, compression_level)
        import zarr

        compressor = zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=compression_level,
            shuffle=zarr.codecs.BloscShuffle.shuffle,  # type: ignore
        )
        for name in prepared.coords:
            coord = prepared[name]
            if coord.dtype.kind == "U" and "event_iloc" in coord.dims:
                floor = TRITONSWMM_analysis_post_processing._EVENT_STRING_WIDTH.get(
                    str(name), TRITONSWMM_analysis_post_processing._EVENT_STRING_WIDTH_DEFAULT
                )
                width = max(int(coord.dtype.itemsize // 4), floor)
                encoding[name] = {"dtype": f"<U{width}", "compressors": compressor}
        for name, var in prepared.variables.items():
            if "event_iloc" in var.dims:
                shape = tuple(1 if d == "event_iloc" else n for d, n in zip(var.dims, var.shape, strict=True))
                encoding.setdefault(name, {})["chunks"] = shape
        return prepared, encoding

    @staticmethod
    def _fit_strings(store: Path, group: str, piece: xr.Dataset) -> xr.Dataset:
        """``piece`` with each string coordinate widened to the stored array's width, refused when it needs more."""
        import zarr

        root = zarr.open_group(store, mode="r")
        fitted = {}
        for name in piece.coords:
            coord = piece[name]
            if coord.dtype.kind != "U" or "event_iloc" not in coord.dims or f"{group}/{name}" not in root:
                continue
            width = int(root[f"{group}/{name}"].dtype.itemsize // 4)
            longest = int(coord.dtype.itemsize // 4)
            if longest > width:
                raise ProcessingError(
                    "append a scenario to a summary group",
                    store,
                    f"the value of coordinate {name} in {group} needs {longest} characters and the stored array holds "
                    f"{width}; nothing was written",
                )
            fitted[name] = (coord.dims, coord.values.astype(f"<U{width}"))
        return piece.assign_coords(fitted)

    def _write_summary_group(
        self,
        store: Path,
        group: str,
        ds: xr.Dataset,
        *,
        compression_level: int = 5,
        whole: bool = False,
    ) -> None:
        """Write one ``summary`` group, whole while every event still has its own store and per event otherwise.

        Whole means ``to_zarr(group=..., mode="w")``, which replaces that group only and recomputes the
        variable-set record over every event. Once any event records its summary stores as cleared, the group is
        the only copy of that event and is never replaced: each event is kept, verified, rewritten in place or
        appended (``_write_summary_group_by_event``). ``whole`` forces the whole write for a caller whose source
        still holds every event, such as a sensitivity master copying a member tree that has not been removed.
        """
        mode = self._mode_of_group(group)
        cleared = {} if whole or mode is None else self._summary_records(mode)
        if not cleared:
            prepared, encoding = self._prepare_summary_group(ds, compression_level)
            with deferred_consolidation(store):
                append_zarr(prepared, store, group=group, consolidate=False, mode="w", encoding=encoding)
            return
        self._write_summary_group_by_event(store, group, ds, cleared, compression_level)

    def _group_event_ids(self, store: Path, group: str) -> list[str]:
        """The ``event_id`` values of the stored group, or an empty list when it does not exist or has none."""
        if self._timeseries_group_state(store, group) is None:
            return []
        stored = xr.open_zarr(store, group=group, consolidated=zarr_store_has_index(store), chunks=None)
        return [str(v) for v in stored["event_id"].values] if "event_id" in stored.coords else []

    def _assert_no_cleared_event_dropped(self, store: Path, tree: "xr.DataTree") -> None:
        """Refuse, before anything is written, a stored ``summary`` event that is no longer part of the analysis.

        A scenario recorded as cleared is refused by name, because its slice is the only copy and a group rewrite
        would drop it. Any other stored event that has left the analysis is refused too when its group is
        append-only, because an append-only group has no way to drop a row.
        """
        if not Path(store).is_dir():
            return
        current = {self._event_slug(ei) for ei in self._analysis.df_sims.index}
        for path, node in tree.subtree_with_keys:
            group = path.lstrip("/")
            mode = self._mode_of_group(group)
            if not node.has_data or mode is None:
                continue
            stale = [ident for ident in self._group_event_ids(store, group) if ident not in current]
            if not stale:
                continue
            model, stem = self._mode_model(mode), self._MODE_STEM[mode]
            sim_dir = Path(self._analysis.analysis_paths.simulation_directory)
            for ident in stale:
                record = summary_clearing_record(sim_dir / ident, model)
                if record is not None and group in record["stems"].get(stem, {}):
                    raise ProcessingError(
                        "consolidate the summary branches",
                        store,
                        f"scenario {ident} is recorded as cleared in {group} but is no longer part of the analysis; "
                        "its slice there is the only copy and a cleared scenario cannot be dropped from a group. "
                        "Nothing was written.",
                    )
            if self._summary_records(mode):
                raise ProcessingError(
                    "consolidate the summary branches",
                    store,
                    f"{group} is append-only because scenarios were cleared from it, and it holds "
                    f"{', '.join(stale)}, which {'is' if len(stale) == 1 else 'are'} no longer part of the analysis; "
                    "an append-only group cannot drop an event. Nothing was written.",
                )

    def _ensure_group_variables(self, store: Path, group: str, piece: xr.Dataset, compression_level: int) -> None:
        """Create in ``group`` every variable of ``piece`` the group lacks, NaN over the events it already holds.

        Appending an event that carries a variable the group does not hold would raise on the mismatched
        ``event_iloc`` sizes (measured), so the new array is created first at the group's current length.
        """
        state = self._timeseries_group_state(store, group)
        if state is None:
            return
        missing = [v for v in piece.data_vars if v not in state[0]]
        if not missing:
            return
        n_events = len(state[1])
        arrays = {}
        for name in missing:
            da = piece[name]
            shape = tuple(n_events if d == "event_iloc" else da.sizes[d] for d in da.dims)
            if da.dtype.kind in "mM":
                fill = np.full(shape, np.datetime64("NaT"), dtype=da.dtype)
            else:
                fill = np.full(shape, np.nan, dtype=da.dtype if da.dtype.kind == "f" else "float64")
            arrays[name] = (da.dims, fill)
        grown = xr.Dataset(arrays)
        encoding = return_dic_zarr_encodings(grown, compression_level)
        for name, da in grown.data_vars.items():
            shape = tuple(1 if d == "event_iloc" else n for d, n in zip(da.dims, da.shape, strict=True))
            encoding.setdefault(name, {})["chunks"] = shape
        with deferred_consolidation(store):
            append_zarr(grown, store, group=group, consolidate=False, mode="a", encoding=encoding)

    def _summary_piece(self, ds: xr.Dataset, event_iloc: int) -> xr.Dataset:
        """One event of the combined ``summary`` Dataset, loaded and stripped of the encodings it was read with."""
        piece = ds.sel(event_iloc=[int(event_iloc)]).load()
        for variable in piece.variables.values():
            variable.encoding = {}
        return piece

    def _write_summary_group_by_event(
        self,
        store: Path,
        group: str,
        ds: xr.Dataset,
        cleared: dict[int, list[str]],
        compression_level: int,
    ) -> None:
        """Plan and perform the write of an append-only ``summary`` group, one event at a time.

        An event recorded cleared is ``kept`` as it stands (and must be there with every recorded variable, else
        the write refuses naming it). Any other event is ``verified`` when its stored slice already equals the
        combined Dataset's, ``rewrite`` in place when it differs, and ``append`` when it is new; a variable the
        event carries and the group lacks is created first, NaN over the events already stored, and a variable
        the event lacks is NaN at it, exactly as the whole-group join fills. Every written slice is read back.
        """
        slug_of = self._event_slug
        state = self._timeseries_group_state(store, group)
        for event_iloc in sorted(cleared):
            held = state is not None and event_iloc in state[1]
            missing = (
                sorted(set(cleared[event_iloc]) - set(state[0])) if state is not None else list(cleared[event_iloc])
            )
            if not held or missing:
                raise ProcessingError(
                    "keep a cleared scenario's summary slice",
                    store,
                    f"scenario {slug_of(event_iloc)} (event_iloc {event_iloc}) records its summary stores as cleared, "
                    f"but {group} does not hold its slice with every recorded variable (missing {missing}, event "
                    f"present: {held}); re-running the simulation is the only remedy. Nothing was written for it.",
                )
        for event_iloc in (int(v) for v in ds["event_iloc"].values):
            if event_iloc in cleared:
                continue
            piece = self._summary_piece(ds, event_iloc)
            state = self._timeseries_group_state(store, group)
            present = state is not None and event_iloc in state[1]
            if present and self._stored_slice_equals(store, group, piece, event_iloc):
                continue
            self._ensure_group_variables(store, group, piece, compression_level)
            prepared = self._fit_strings(store, group, piece)
            with deferred_consolidation(store):
                if present:
                    position = self._timeseries_group_state(store, group)[1].index(event_iloc)
                    append_zarr(
                        prepared.drop_vars([c for c in prepared.coords if "event_iloc" not in prepared[c].dims]),
                        store,
                        group=group,
                        consolidate=False,
                        mode="r+",
                        region={"event_iloc": slice(position, position + 1)},
                    )
                else:
                    append_zarr(prepared, store, group=group, consolidate=False, mode="a", append_dim="event_iloc")
            if not self._stored_slice_equals(store, group, piece, event_iloc):
                raise ProcessingError(
                    "read back a summary slice",
                    store,
                    f"the slice of scenario {slug_of(event_iloc)} (event_iloc {event_iloc}) in {group} does not equal "
                    "the data it was written from; its own summary stores were kept",
                )

    def _event_summary_dataset(self, mode: str, event_iloc: int, *, drop_flag: bool) -> xr.Dataset:
        """One scenario's own summary store for ``mode``, opened lazily, after its completeness checks.

        Raises ``FileNotFoundError`` when the store is absent, or present without a successful processing-log record
        (a write that did not return may have left a fill-filled partial), so every handler that already catches
        that type for a missing summary keeps catching it.
        """
        scen = TRITONSWMM_scenario(event_iloc, self._analysis)
        summary_file = getattr(scen.scen_paths, self._MODE_CONFIG[mode][0])
        if summary_file is None:
            raise ValueError(
                f"Summary file path is None for mode '{mode}' and event_iloc={event_iloc}. "
                f"Check that the appropriate model types are enabled in system config."
            )
        if not summary_file.exists():
            raise FileNotFoundError(
                f"Summary file not found: {summary_file}. "
                f"Run timeseries processing with summary creation before consolidating."
            )
        # COMPLETENESS, not just existence. A kill inside write_zarr leaves a store that exists, opens and returns
        # fill for the chunks that never landed (measured: zarr writes the declared shape before the chunk bytes and
        # an absent chunk reads as fill_value). The per-model processing-log record is the only signal that attests
        # RETURN rather than PRESENCE, because add_sim_processing_entry runs after write_zarr returns. The error type
        # is FileNotFoundError on purpose: four enclosing handlers (eda.cross_sim_identity, eda.compute_sensitivity,
        # and the sensitivity consolidation) catch exactly (FileNotFoundError, ValueError) and skip with a warning,
        # and a distinct type would turn that skip into an abort.
        entry = scen.get_log(self._mode_model(mode)).processing_log.outputs.get(summary_file.name)
        if entry is None or not entry.success:
            raise FileNotFoundError(
                f"Summary store present but incomplete: {summary_file}. "
                "It carries no successful processing-log record, so its write did "
                "not return and it may be a fill-filled partial. Re-run timeseries "
                "processing for this scenario."
            )
        open_kwargs = {"chunks": "auto", "engine": self._open_engine(), "decode_timedelta": False}
        if open_kwargs["engine"] == "zarr":
            open_kwargs["consolidated"] = zarr_store_has_index(summary_file)
        ds = self._select_stem_variables(mode, xr.open_dataset(summary_file, **open_kwargs))
        if drop_flag and mode == "tritonswmm_swmm_node" and self._COUPLING_FLAG in ds.data_vars:
            ds = ds.drop_vars(self._COUPLING_FLAG)
        return ds

    def _read_cleared_summary(self, mode: str, event_iloc: int, names: list[str]) -> xr.Dataset:
        """A cleared event's slice of ``mode``'s ``summary`` group, restricted to the variables the record names.

        The slice carries the event's own coordinates (its position, its identifier and its producing stamps) and is
        loaded, so the Dataset does not stay tied to a store that a later append resizes. Its values are the ones the
        read-back proved equal to the deleted per-scenario store's. Raises ``FileNotFoundError`` naming the scenario
        and the group when the tree, the group, the event or a recorded variable is absent: the slice is then the
        only copy and the caller must not substitute anything for it.
        """
        store, prefix = self._summary_tree_location()
        group = prefix + self._MODE_TO_TREE_PATH[mode]
        slug = self._event_slug(event_iloc)
        if not Path(store).is_dir():
            raise FileNotFoundError(
                f"Scenario {slug} (event_iloc {event_iloc}) records its {mode} summary store as cleared, and the tree "
                f"that holds its only copy is absent: {store} (analysis {self._analysis.cfg_analysis.analysis_id})."
            )
        state = self._timeseries_group_state(store, group)
        missing = sorted(set(names) - set(state[0])) if state is not None else list(names)
        if state is None or event_iloc not in state[1] or missing:
            raise FileNotFoundError(
                f"Scenario {slug} (event_iloc {event_iloc}) records its {mode} summary store as cleared, but {group} "
                f"of {store} does not hold its slice (variables missing {missing}, group present: {state is not None})."
            )
        stored = xr.open_zarr(
            store, group=group, consolidated=zarr_store_has_index(store), chunks=None, decode_timedelta=False
        )
        positions = [int(v) for v in stored["event_iloc"].values]
        piece = stored.isel(event_iloc=[positions.index(event_iloc)])[list(names)].load()
        # The identifier is assigned for every event after the join, as it is for an event read from its own store.
        piece = piece.drop_vars("event_id") if "event_id" in piece.coords else piece
        return piece

    def _summary_slice_matches(self, store: Path, group: str, own: xr.Dataset, event_iloc: int) -> bool:
        """Whether the stored slice of ``event_iloc`` equals the scenario's own summary, nothing else standing in it.

        Every variable of ``own`` must read back equal to the stored one (a missing value matching a missing
        value), and every other data variable of the group must be entirely missing at the event, which is what the
        join leaves where the scenario lacks a variable another scenario has.
        """
        state = self._timeseries_group_state(store, group)
        if state is None or event_iloc not in state[1]:
            return False
        stored = xr.open_zarr(
            store, group=group, consolidated=zarr_store_has_index(store), chunks=None, decode_timedelta=False
        )
        read = stored.isel(event_iloc=[state[1].index(event_iloc)]).load()
        labels = set(own.coords) | {"event_id", VARIABLE_SET_COORD, "hhemt_producing_sha", "hhemt_producing_version"}
        for name in read.data_vars:
            if name not in own.data_vars:
                if name in labels:
                    continue
                if not bool(read[name].isnull().all()):
                    return False
                continue
            want, got = own[name].load(), read[name]
            if set(want.dims) != set(got.dims):
                return False
            want = want.transpose(*got.dims)
            if not self._arrays_equal(np.asarray(got.values), np.asarray(want.values)):
                return False
        return all(name in read.data_vars for name in own.data_vars)

    def _clear_summary_stores(self, store: Path, *, drop_flag: bool) -> None:
        """Delete each scenario's per-scenario summary stores once every slice of the scenario reads back.

        Per scenario and per model whose summary stores the analysis defines: a model whose log already records the
        clearing as ``pending`` is finished from the group and never compared against its half-deleted stores; a
        model with no record has each mode's slice read back from the store (a slice that does not match keeps the
        stores and raises naming the scenario), then the record is written as ``pending`` with each stem's groups and
        variable names, the stores are deleted through the deletion tool, and the record is set to ``done``. Peak disk
        use therefore ends near one copy of the summaries.
        """
        from hhemt.du_sentinels import delete_and_account

        first = TRITONSWMM_scenario(self._analysis.df_sims.index[0], self._analysis)
        by_model: dict[str, list[str]] = {}
        for mode in self._MODE_TO_TREE_PATH:
            if getattr(first.scen_paths, self._MODE_CONFIG[mode][0]) is not None:
                by_model.setdefault(self._mode_model(mode), []).append(mode)
        for event_iloc in (int(i) for i in self._analysis.df_sims.index):
            scen = TRITONSWMM_scenario(event_iloc, self._analysis)
            for model, modes in by_model.items():
                log = scen.get_log(model)
                existing = getattr(log, SUMMARY_CLEARING_FIELD, None)
                record = parse_summary_clearing_record(None if existing is None else existing.get())
                if record is not None and record["state"] == "pending":
                    self._finish_pending_summary_clearing(scen, log, model, record, event_iloc, store)
                    record = parse_summary_clearing_record(getattr(log, SUMMARY_CLEARING_FIELD).get())
                # A partial re-process leaves a record that names only some of the model's stems; a stem it does not
                # name whose store is back is read back and cleared like any other, and joins the record.
                held = {} if record is None else record["stems"]
                modes = [m for m in modes if self._MODE_STEM[m] not in held]
                if not modes:
                    continue
                stems: dict[str, dict[str, list[str]]] = {}
                for mode in modes:
                    own = self._event_summary_dataset(mode, event_iloc, drop_flag=drop_flag)
                    group = self._MODE_TO_TREE_PATH[mode]
                    if not self._summary_slice_matches(store, group, own, event_iloc):
                        raise ProcessingError(
                            "read back the consolidated summaries",
                            store,
                            f"the slice of scenario {self._event_slug(event_iloc)} (event_iloc {event_iloc}) in "
                            f"{group} does not equal the data it was written from; its own summary stores were kept",
                        )
                    stems.setdefault(self._MODE_STEM[mode], {})[group] = sorted(own.data_vars)
                self._log_field(log, SUMMARY_CLEARING_FIELD).set({"state": "pending", "stems": {**held, **stems}})
                paths = [self._stem_path(scen, stem) for stem in stems]
                delete_and_account(
                    [p for p in paths if p is not None and p.exists()],
                    scope_dir=scen.scen_paths.sim_folder,
                    scope="scenario",
                )
                self._log_field(log, SUMMARY_CLEARING_FIELD).set({"state": "done", "stems": {**held, **stems}})

    def _stem_path(self, scen: TRITONSWMM_scenario, stem: str) -> Path | None:
        """The scenario's path of the per-scenario summary store named ``stem``."""
        for mode, mode_stem in self._MODE_STEM.items():
            if mode_stem == stem:
                return getattr(scen.scen_paths, self._MODE_CONFIG[mode][0])
        return None

    def _finish_pending_summary_clearing(
        self,
        scen: TRITONSWMM_scenario,
        log,
        model: str,
        record: dict,
        event_iloc: int,
        store: Path,
    ) -> None:
        """Finish a summary clearing recorded as pending, from the branch and never from the stores.

        A pending record means the slices were already read back equal, so whatever is left of the stores (a zarr
        directory with some chunk files removed still exists) is neither compared nor rewritten from: a comparison
        against a partly deleted store fails, and the failure would be answered by overwriting a verified slice with
        a truncated one. Each recorded group is confirmed to hold the event with every recorded variable, the rest of
        the stores is deleted, and the record is set to ``done``. A group missing the event or a variable refuses
        naming the scenario and the group, since the deleted store was then the only copy.
        """
        from hhemt.du_sentinels import delete_and_account

        for groups in record["stems"].values():
            for group, names in groups.items():
                state = self._timeseries_group_state(store, group)
                missing = sorted(set(names) - set(state[0])) if state is not None else list(names)
                if state is None or event_iloc not in state[1] or missing:
                    raise ProcessingError(
                        "finish a recorded clearing of summary stores",
                        store,
                        f"scenario {self._event_slug(event_iloc)} (event_iloc {event_iloc}) records its {model} "
                        f"summary stores as being cleared, but {group} does not hold its slice with every recorded "
                        f"variable (missing {missing}, event present: {state is not None and event_iloc in state[1]}); "
                        "nothing was deleted or marked done",
                    )
        paths = [self._stem_path(scen, stem) for stem in record["stems"]]
        delete_and_account(
            [p for p in paths if p is not None and p.exists()], scope_dir=scen.scen_paths.sim_folder, scope="scenario"
        )
        self._log_field(log, SUMMARY_CLEARING_FIELD).set({"state": "done", "stems": record["stems"]})

    @staticmethod
    def _groups_holding_layout_30_arrays(store: Path) -> list[str]:
        """Each ``{family}/{kind}`` group of ``store`` that holds data arrays directly (the layout-30 shape), sorted."""
        import json

        found: list[str] = []
        for family in ("tritonswmm", "triton_only", "swmm_only"):
            for kind in ("triton", "swmm_node", "swmm_link", "performance"):
                group = Path(store) / family / kind
                if not group.is_dir():
                    continue
                for child in group.iterdir():
                    doc = child / "zarr.json"
                    if not (child.is_dir() and doc.is_file()):
                        continue
                    try:
                        is_array = json.loads(doc.read_text()).get("node_type") == "array"
                    except (OSError, ValueError):
                        continue
                    if is_array:
                        found.append(f"{family}/{kind}")
                        break
        return sorted(found)

    @staticmethod
    def assert_summary_branch_layout(tree: "xr.DataTree", source: Path | str) -> None:
        """Raise ``ProcessingError`` when ``tree`` holds a summary at a layout-30 path, naming the layout version.

        Layout 30 stored each summary directly at ``{family}/{kind}``; layout 31 stores it at
        ``{family}/{kind}/summary``. A tree with a data-bearing ``{family}/{kind}`` node and no ``summary`` child
        was written by a toolkit older than layout 31 and must be migrated, not grafted.
        """
        from hhemt.version_migration.constants import LAYOUT_VERSION

        for family in ("tritonswmm", "triton_only", "swmm_only"):
            if family not in tree.children:
                continue
            for kind, node in tree[family].children.items():
                if node.has_data and "summary" not in node.children:
                    raise ProcessingError(
                        "read a member's consolidated tree",
                        str(source),
                        f"{family}/{kind} holds a summary at its layout-30 path; layout {LAYOUT_VERSION} stores it "
                        f"at {family}/{kind}/summary. Migrate the tree (hhemt migrate) before grafting it.",
                    )

    def open_datatree(self) -> "xr.DataTree":
        """Open the consolidated hierarchical DataTree zarr lazily.

        The canonical signal that the DataTree is present and complete is
        `log.datatree_consolidation_complete`. File existence is a weaker
        signal (a corrupt-but-on-disk zarr would `.exists()` as True);
        the log marker is set only on successful write completion in
        `consolidate_to_datatree()`.
        """
        path = self._analysis.analysis_paths.analysis_datatree_zarr
        if path is None:
            raise ValueError("analysis_datatree_zarr path is not configured on AnalysisPaths.")
        # Render-phase readers may hold a long-lived in-memory log constructed
        # before a concurrent consolidate job set the flag. Reload from disk so
        # the gate reflects on-disk truth, not a possibly-stale in-memory value.
        # Mirrors the producer (consolidate_to_datatree's _refresh_log before .set) and
        # the per-member _refresh_log in sensitivity_analysis's build_sensitivity_datatree.
        # The lost-update WRITE race is already closed (compute-on-read
        # rollups landed); this closes the residual READ-staleness window.
        self._analysis._refresh_log()
        consolidated = (
            hasattr(self._analysis.log, "datatree_consolidation_complete")
            and self._analysis.log.datatree_consolidation_complete.get() is True
        )
        if not consolidated:
            raise ValueError(
                "DataTree zarr not present (log.datatree_consolidation_complete is "
                "False or unset). Run consolidate_to_datatree() first."
            )
        return open_store(path, chunks="auto")

    # ------------------------------------------------------------------------------------------------
    # Consolidated per-scenario time series: the `timeseries` branches of analysis_datatree.zarr
    # ------------------------------------------------------------------------------------------------

    def _timeseries_axis(self) -> tuple[dict[int, "pd.Timestamp"], float, np.ndarray]:
        """Each event's simulation start, the reporting step, and the ``elapsed_s`` axis shared by every event.

        The axis runs from one reporting step to the longest simulation duration among ALL events in ``df_sims``
        (each duration is the analysis's own event window, ``_resolve_event_window``, not the length of whatever
        series happens to be on disk), and it is fixed here, before the first append. An append whose
        non-append dimension differs from the store's raises, so every scenario is reindexed to this axis.
        """
        step = float(self._analysis.cfg_analysis.TRITON_reporting_timestep_s)
        starts: dict[int, pd.Timestamp] = {}
        longest_s = 0.0
        for event_iloc in self._analysis.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self._analysis)
            start, end = scen._resolve_event_window()
            starts[int(event_iloc)] = pd.Timestamp(start)
            longest_s = max(longest_s, (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds())
        n_steps = int(round(longest_s / step))
        if n_steps < 1:
            raise ProcessingError(
                "consolidate timeseries",
                self._analysis.analysis_paths.analysis_datatree_zarr,
                f"the longest event window ({longest_s} s) is shorter than one reporting step ({step} s)",
            )
        values = step * np.arange(1, n_steps + 1)
        axis = values.astype("int64") if float(step).is_integer() else values
        return starts, step, axis

    @staticmethod
    def _timeseries_group_state(store: Path, group: str) -> tuple[list[str], list[int]] | None:
        """The group's data-variable names and its ``event_iloc`` values, or None when the group does not exist.

        Opened WITHOUT the consolidated index, because inside ``deferred_consolidation`` the store has none.
        """
        group_dir = Path(store) / group
        if not ((group_dir / "zarr.json").is_file() or (group_dir / ".zgroup").is_file()):
            return None
        stored = xr.open_zarr(store, group=group, consolidated=zarr_store_has_index(store), chunks=None)
        return sorted(stored.data_vars), [int(v) for v in stored["event_iloc"].values]

    def _timeseries_slice(
        self,
        scen: TRITONSWMM_scenario,
        ts_mode: str,
        event_iloc: int,
        start: "pd.Timestamp",
        step: float,
        axis: np.ndarray,
    ) -> xr.Dataset:
        """One scenario's contribution to the consolidated store: time-variant variables and the flag, on ``elapsed_s``.

        The appended Dataset holds the per-scenario store's variables that carry ``date_time`` and, for the COUPLED
        node store, ``is_coupling_node``, and no other variable: the static report variables reach the analysis
        level through the summary tree, so a scenario written before a static variable existed does not differ from
        a later one. Every data variable gets ``event_iloc`` (``expand_dims``), the flag included, because a variable
        without it is silently overwritten by each append (a per-node 0 then 10 reads back as 10 for both events).
        A coupled node store lacking the flag (written by a toolkit older than the coupling outputs) gets it
        derived from the scenario's own ``coupling_nodes.txt``. The SWMM-only node store has no coupling and
        therefore no flag: nothing is kept for it beyond the time-variant variables and nothing is derived.
        """
        if ts_mode.endswith("_triton_ts"):
            return self._gridded_timeseries_slice(scen, ts_mode, event_iloc, start, step, axis)
        path = getattr(scen.scen_paths, self._TIMESERIES_MODE_CONFIG[ts_mode])
        is_node = ts_mode.endswith("_node_ts")
        is_system = ts_mode.endswith("_system_ts")
        coupled = ts_mode.startswith("tritonswmm")
        open_kwargs = {"chunks": "auto", "engine": self._open_engine(), "decode_timedelta": False}
        if open_kwargs["engine"] == "zarr":
            open_kwargs["consolidated"] = zarr_store_has_index(path)
        full = xr.open_dataset(path, **open_kwargs)
        keep = [v for v in full.data_vars if "date_time" in full[v].dims]
        if is_node and coupled and "is_coupling_node" in full.data_vars:
            keep.append("is_coupling_node")
        ds = full[keep].load()
        slug = self._event_slug(event_iloc)
        if is_node and coupled and "is_coupling_node" not in ds.data_vars:
            from hhemt.scenario import coupling_nodes_file_for

            nodes_file = coupling_nodes_file_for(scen.scen_paths.swmm_hydraulics_inp)
            if not nodes_file.is_file():
                raise ProcessingError(
                    "derive is_coupling_node for the consolidated time series",
                    nodes_file,
                    f"scenario {slug} (event_iloc {event_iloc}) has no coupling-node list to derive the flag from",
                )
            listed = {ln.strip() for ln in nodes_file.read_text().splitlines() if ln.strip()}
            flag = np.array([str(n) in listed for n in ds["node_id"].values], dtype="int8")
            ds["is_coupling_node"] = ("node_id", flag)
        seconds = (pd.to_datetime(ds["date_time"].values) - start).total_seconds().to_numpy()
        index = np.rint(seconds / step).astype("int64")
        if not np.allclose(seconds, index * step, atol=1e-6) or index.min() < 1 or index.max() > len(axis):
            raise ProcessingError(
                "place a scenario on the shared elapsed_s axis",
                path,
                f"scenario {slug} (event_iloc {event_iloc}) has report times that are not whole multiples of the "
                f"{step} s reporting step after its start {start}, or that run past the longest event "
                f"({len(axis)} steps); first/last elapsed seconds {seconds.min()}/{seconds.max()}",
            )
        ds = ds.assign_coords(elapsed_s=("date_time", axis[index - 1])).swap_dims(date_time="elapsed_s")
        ds = ds.drop_vars("date_time").reindex(elapsed_s=axis)
        cf_family = "tritonswmm_swmm" if coupled else "swmm_only"
        if is_system:
            apply_cf_attributes(ds, "swmm_only_system")
        else:
            apply_cf_attributes(ds, f"{cf_family}_node" if is_node else f"{cf_family}_link")
        ds = ds.expand_dims(event_iloc=[int(event_iloc)])
        return ds.assign_coords(
            event_id=("event_iloc", [slug]),
            event_start_datetime=("event_iloc", np.array([start.to_datetime64()], dtype="datetime64[ns]")),
        )

    @staticmethod
    def _gridded_axis(axis: np.ndarray) -> np.ndarray:
        """The ``elapsed_s`` axis of the gridded group: the shared axis with the instant 0 put in front.

        Rather than ``axis`` itself, because TRITON may print its initial state at ``timestep`` 0 and may not (open
        item O14, which the first real run closes): an axis that starts at one step would refuse a store that
        starts at 0, and an axis fixed from the first store written would refuse the next one if the two differ. The
        0 row is NaN for a store that has no initial state, which is a missing value and not a made-up one.
        """
        return np.concatenate([np.zeros(1, dtype=axis.dtype), axis])

    def _gridded_timeseries_slice(
        self,
        scen: TRITONSWMM_scenario,
        ts_mode: str,
        event_iloc: int,
        start: "pd.Timestamp",
        step: float,
        axis: np.ndarray,
    ) -> xr.Dataset:
        """One scenario's gridded contribution to its family's ``triton/timeseries`` group, LAZY and chunked by block.

        The coupled model writes ``tritonswmm/triton/timeseries`` and the TRITON-only model writes
        ``triton_only/triton/timeseries``; the two stores have the same shape and take the same path here.

        Every variable carrying ``timestep`` is kept (the keep-list prune has already left the store holding only
        the selected ones), ``timestep`` in minutes becomes integer ``elapsed_s`` seconds at the TRITON reporting
        step, and the event is NaN-padded to the group's axis. The Dataset is never loaded: its chunks are
        ``(event_iloc 1, elapsed_s k, every spatial dim whole)`` with ``k`` sized so one chunk is about
        ``_TIMESERIES_BLOCK_BYTES``, so the write and the read-back hold a few blocks and not the 1 GB slice.
        The source encodings are dropped: a source chunk shape that differs from ``k`` would make ``to_zarr`` raise
        on a chunk overlap, and an append may not restate the encoding of a variable the group already holds.
        """
        path = getattr(scen.scen_paths, self._TIMESERIES_MODE_CONFIG[ts_mode])
        open_kwargs = {"chunks": {}, "engine": self._open_engine(), "decode_timedelta": False}
        if open_kwargs["engine"] == "zarr":
            open_kwargs["consolidated"] = zarr_store_has_index(path)
        full = xr.open_dataset(path, **open_kwargs)
        ds = full[[v for v in full.data_vars if "timestep" in full[v].dims]]
        slug = self._event_slug(event_iloc)
        grid_axis = self._gridded_axis(axis)
        seconds = np.asarray(ds["timestep"].values, dtype="float64") * 60.0
        index = np.rint(seconds / step).astype("int64")
        if not np.allclose(seconds, index * step, atol=1e-6) or index.min() < 0 or index.max() > len(grid_axis) - 1:
            raise ProcessingError(
                "place a scenario's gridded series on the shared elapsed_s axis",
                path,
                f"scenario {slug} (event_iloc {event_iloc}) has timestep times that are not whole multiples of the "
                f"{step} s reporting step, or that run past the longest event ({len(grid_axis) - 1} steps after "
                f"the initial instant); first/last elapsed seconds {seconds.min()}/{seconds.max()}",
            )
        bytes_per_step = max(
            da.dtype.itemsize * int(np.prod([n for d, n in da.sizes.items() if d != "timestep"]))
            for da in ds.data_vars.values()
        )
        block = int(max(1, min(len(grid_axis), round(self._TIMESERIES_BLOCK_BYTES / bytes_per_step))))
        ds = ds.assign_coords(elapsed_s=("timestep", grid_axis[index])).swap_dims(timestep="elapsed_s")
        ds = ds.drop_vars("timestep").reindex(elapsed_s=grid_axis)
        ds = ds.chunk({d: (block if d == "elapsed_s" else -1) for d in ds.dims})
        for variable in ds.variables.values():
            variable.encoding = {}
        ds = ds.expand_dims(event_iloc=[int(event_iloc)])
        return ds.assign_coords(
            event_id=("event_iloc", [slug]),
            event_start_datetime=("event_iloc", np.array([start.to_datetime64()], dtype="datetime64[ns]")),
        )

    def _event_slug(self, event_iloc: int) -> str:
        from hhemt.scenario import compute_event_id_slug

        return compute_event_id_slug(self._analysis._retrieve_weather_indexer_using_integer_index(event_iloc))

    @staticmethod
    def _stored_slice_equals(store: Path, group: str, expected: xr.Dataset, event_iloc: int) -> bool:
        """Read the event's slice back from the store and compare it NaN-aware with ``expected``, every data variable.

        The comparison is of VALUES read from disk, BLOCK BY BLOCK along ``elapsed_s`` (blocks of about
        ``_TIMESERIES_BLOCK_BYTES``), so a gridded slice of about 1 GB is compared with a few blocks resident and
        never as a whole; a SWMM slice is one block. Shape metadata and chunk presence do not qualify: an append
        persists the new shape before its chunks, and chunk presence is inexact where empty chunks are not written.
        """
        stored = xr.open_zarr(store, group=group, consolidated=zarr_store_has_index(store), chunks=None)
        positions = [int(v) for v in stored["event_iloc"].values]
        if event_iloc not in positions:
            return False
        read = stored.isel(event_iloc=[positions.index(event_iloc)])
        # A coordinate that a gridded group stores without the coordinates attribute reads back as a data variable.
        if (set(expected.data_vars) - set(read.data_vars)) or (
            set(read.data_vars) - set(expected.data_vars) - set(expected.coords)
        ):
            return False
        block_bytes = TRITONSWMM_analysis_post_processing._TIMESERIES_BLOCK_BYTES
        for name in expected.data_vars:
            want = expected[name]
            got = read[name]
            if got.dims != want.dims:
                want = want.transpose(*got.dims)
            if got.shape != want.shape:
                return False
            if "elapsed_s" in got.dims:
                n = got.sizes["elapsed_s"]
                per_step = got.dtype.itemsize * max(1, got.size // max(1, n))
                width = max(1, block_bytes // per_step)
                blocks = [{"elapsed_s": slice(i, min(i + width, n))} for i in range(0, n, width)]
            else:
                blocks = [{}]
            for block in blocks:
                a, b = np.asarray(got.isel(block).values), np.asarray(want.isel(block).values)
                if not TRITONSWMM_analysis_post_processing._arrays_equal(a, b):
                    return False
        return True

    def _plan_timeseries_write(
        self,
        scen: TRITONSWMM_scenario,
        ts_mode: str,
        group: str,
        event_iloc: int,
        start: "pd.Timestamp",
        step: float,
        axis: np.ndarray,
        cleared_already: bool,
    ) -> tuple[str, xr.Dataset | None, int | None]:
        """Decide how this scenario's slice gets into ``group``; the caller performs the write.

        Returns ``(action, dataset, position)``. ``"verified"``: the slice is already in the store and passed its
        read-back, nothing to write; the dataset is returned (None only when the scenario's own store is gone and
        its clearing is recorded) because the clearing record names its variables.
        ``"rewrite"``: the event is in the store without a passing read-back, so it is
        rewritten in place at ``position`` from the scenario's own store (refused when that store was already
        cleared). ``"append"``: the group exists and gains the event along ``event_iloc``. ``"create"``: the group
        does not exist yet. An unconditional re-append would duplicate the event (measured ``[0, 1, 1]``). The writes
        themselves stay in the caller, inside its ``deferred_consolidation`` block.
        """
        store = self._analysis.analysis_paths.analysis_datatree_zarr
        path = getattr(scen.scen_paths, self._TIMESERIES_MODE_CONFIG[ts_mode])
        state = self._timeseries_group_state(store, group)
        present = state is not None and event_iloc in state[1]
        slug = self._event_slug(event_iloc)
        if not path.exists():
            if present and cleared_already:
                return "verified", None, None
            if present:
                raise ProcessingError(
                    "consolidate timeseries",
                    path,
                    f"scenario {slug} (event_iloc {event_iloc}) is in {store.name} but its own store is gone and the "
                    "log records no verified clearing, so the stored slice cannot be checked or rewritten",
                )
            raise FileNotFoundError(
                f"Timeseries file not found: {path}. Run timeseries processing before consolidating."
            )
        ds = self._timeseries_slice(scen, ts_mode, event_iloc, start, step, axis)
        if present:
            if self._stored_slice_equals(store, group, ds, event_iloc):
                return "verified", ds, None
            return "rewrite", ds, state[1].index(event_iloc)
        if state is not None:
            missing = sorted(set(state[0]) - set(ds.data_vars))
            extra = sorted(set(ds.data_vars) - set(state[0]))
            if missing or extra:
                raise ProcessingError(
                    "append a scenario to the consolidated time series",
                    store,
                    f"scenario {slug} (event_iloc {event_iloc}) differs from {group} in its variables: "
                    f"missing {missing}, not in the store {extra}; nothing was written",
                )
            return "append", ds, None
        return "create", ds, None

    def consolidate_timeseries(self, verbose: bool = False) -> Path:
        """Append each scenario's SWMM node, SWMM link and gridded TRITON series to the ``timeseries`` branches.

        Every model family that writes series is consolidated the same way: the coupled model's three stores,
        the SWMM-only model's node and link stores and the TRITON-only model's gridded store, each into its own
        ``{family}/{kind}/timeseries`` group, the family decided by which per-scenario paths the analysis defines.
        Written through ``append_zarr(..., append_dim="event_iloc")`` inside ``deferred_consolidation(store)``, so
        the store's index is rebuilt once, at the end. With ``toggle_clear_scenario_timeseries_after_consolidation``
        on, a scenario's own series stores of one MODEL are deleted as soon as every one of that model's slices has
        been read back from the store and found equal, so peak disk use stays near one copy of the series and a
        failing read-back keeps the stores of its own model (and every model processed after it) while the stores
        already verified stay cleared. A clearing is recorded in the model's own log as pending BEFORE the delete
        and as done after it, and a scenario found with a pending record is finished from the store's slice
        rather than re-verified against its half-deleted source. The ``timeseries`` branches are APPEND-ONLY: no
        consolidation rebuild (``_write_summary_branches`` rewrites ``summary`` groups only) and no reprocess
        delete (``delete_reprocess_zarr_runner`` keeps every ``timeseries`` branch) removes or rewrites one,
        because with clearing on it holds the only copy.
        """
        store = self._analysis.analysis_paths.analysis_datatree_zarr
        if store is None:
            raise ValueError("analysis_datatree_zarr path is not configured on AnalysisPaths.")
        clear = bool(
            getattr(self._analysis.cfg_analysis, "toggle_clear_scenario_timeseries_after_consolidation", False)
        )
        first = TRITONSWMM_scenario(self._analysis.df_sims.index[0], self._analysis)
        modes = [
            (ts_mode, self._TIMESERIES_MODE_TO_TREE_PATH[ts_mode])
            for ts_mode, attr in self._TIMESERIES_MODE_CONFIG.items()
            if getattr(first.scen_paths, attr) is not None
        ]
        if not modes:
            return store
        by_model: dict[str, list[tuple[str, str]]] = {}
        for ts_mode, group in modes:
            by_model.setdefault(self._TIMESERIES_MODE_MODEL[ts_mode], []).append((ts_mode, group))
        starts, step, axis = self._timeseries_axis()
        with deferred_consolidation(store):
            for event_iloc in self._analysis.df_sims.index:
                event_iloc = int(event_iloc)
                scen = TRITONSWMM_scenario(event_iloc, self._analysis)
                for model_type, model_modes in by_model.items():
                    log = scen.get_log(model_type)
                    verified: dict[str, list[str]] = {}
                    for ts_mode, group in model_modes:
                        cleared_name, pending_name = self._TIMESERIES_CLEARED_FIELDS[ts_mode]
                        cleared_field = getattr(log, cleared_name, None)
                        cleared_already = bool(cleared_field is not None and cleared_field.get())
                        pending_field = getattr(log, pending_name, None)
                        pending = list(pending_field.get() or []) if pending_field is not None else []
                        if pending and not cleared_already:
                            self._finish_pending_clearing(scen, log, ts_mode, group, event_iloc, pending)
                            continue
                        if pending:
                            pending_field.clear()
                        action, ds, position = self._plan_timeseries_write(
                            scen, ts_mode, group, event_iloc, starts[event_iloc], step, axis, cleared_already
                        )
                        if action == "rewrite":
                            append_zarr(
                                ds.drop_vars([c for c in ds.coords if "event_iloc" not in ds[c].dims]),
                                store,
                                group=group,
                                consolidate=False,
                                mode="r+",
                                region={"event_iloc": slice(position, position + 1)},
                            )
                        elif action == "append":
                            append_zarr(ds, store, group=group, consolidate=False, mode="a", append_dim="event_iloc")
                        elif action == "create":
                            append_zarr(ds, store, group=group, consolidate=False, mode="a")
                        ok = action == "verified" or self._stored_slice_equals(store, group, ds, event_iloc)
                        if not ok:
                            raise ProcessingError(
                                "read back the consolidated time series",
                                store,
                                f"the slice of scenario {self._event_slug(event_iloc)} (event_iloc {event_iloc}) in "
                                f"{group} does not equal the data it was written from; its own store was kept",
                            )
                        if ds is not None:
                            verified[ts_mode] = sorted(ds.data_vars)
                    if clear and verified:
                        self._clear_scenario_timeseries(scen, log, verified)
        return store

    @staticmethod
    def _log_field(log, name: str):
        """The model log's LogField ``name``, created empty (and bound to the log) when the log lacks it."""
        from hhemt.log import LogField

        if getattr(log, name) is None:
            setattr(log, name, LogField())
            log.model_post_init(None)
        return getattr(log, name)

    def _clear_scenario_timeseries(self, scen: TRITONSWMM_scenario, log, verified: dict[str, list[str]]) -> None:
        """Delete the scenario's series stores, recording the clearing as pending first and done last.

        Only called once every slice of the scenario has passed its read-back. ``verified`` maps each mode to the
        data-variable names its slice was verified with. The pending record carries those names, because after a
        kill part-way through the delete the source can no longer say which variables its slice held. The deletion
        goes through the deletion tool so the scenario's disk-use sentinel is adjusted once.
        """
        from hhemt.du_sentinels import delete_and_account

        for ts_mode, names in verified.items():
            self._log_field(log, self._TIMESERIES_CLEARED_FIELDS[ts_mode][1]).set(list(names))
        paths = [getattr(scen.scen_paths, self._TIMESERIES_MODE_CONFIG[m]) for m in verified]
        delete_and_account(
            [p for p in paths if p is not None and p.exists()], scope_dir=scen.scen_paths.sim_folder, scope="scenario"
        )
        for ts_mode in verified:
            cleared_name, pending_name = self._TIMESERIES_CLEARED_FIELDS[ts_mode]
            self._log_field(log, cleared_name).set(True)
            getattr(log, pending_name).clear()

    def _finish_pending_clearing(
        self,
        scen: TRITONSWMM_scenario,
        log,
        ts_mode: str,
        group: str,
        event_iloc: int,
        names: list[str],
    ) -> None:
        """Finish a clearing the log records as pending, from the store's slice and never from the source.

        The pending record means this slice was already read back equal, so the source (whatever is left of it, and
        a zarr directory with some chunk files removed still exists) is neither compared nor rewritten from: a
        comparison against a partly deleted source fails, and the failure would be answered by overwriting a
        verified slice with a truncated one. The slice is confirmed present in its group with every recorded
        variable, the rest of the source is deleted (an absent path is not an error), and the clearing is recorded
        done. A group missing the event or a recorded variable refuses naming the scenario, since then the deleted
        source was the only copy.
        """
        from hhemt.du_sentinels import delete_and_account

        store = self._analysis.analysis_paths.analysis_datatree_zarr
        state = self._timeseries_group_state(store, group)
        missing = sorted(set(names) - set(state[0])) if state is not None else list(names)
        if state is None or event_iloc not in state[1] or missing:
            raise ProcessingError(
                "finish a recorded clearing of a time-series store",
                store,
                f"scenario {self._event_slug(event_iloc)} (event_iloc {event_iloc}) records its {ts_mode} store as "
                f"being cleared, but {group} does not hold its slice with every recorded variable "
                f"(missing {missing}, event present: {state is not None and event_iloc in state[1]}); nothing was "
                "deleted or marked done",
            )
        path = getattr(scen.scen_paths, self._TIMESERIES_MODE_CONFIG[ts_mode])
        if path is not None and path.exists():
            delete_and_account([path], scope_dir=scen.scen_paths.sim_folder, scope="scenario")
        cleared_name, pending_name = self._TIMESERIES_CLEARED_FIELDS[ts_mode]
        self._log_field(log, cleared_name).set(True)
        getattr(log, pending_name).clear()

    def _retrieve_combined_timeseries(self, ts_mode: str) -> xr.Dataset:
        """Load per-scenario TIMESERIES zarrs and concatenate them along event_iloc.

        Mirrors _retrieve_combined_output but reads the timeseries scenario-path attr
        (_TIMESERIES_MODE_CONFIG) rather than the summary attr, and keeps each scenario's
        `time` dimension. Concat is outer-join on time (scenarios of different weather
        events have different time axes; within a clean-vs-resume PAIR the axes match, so
        the over-time diff has no NaN in the overlap). Raises FileNotFoundError on a missing
        file so the master consolidate loop's allow_incomplete=True skip applies uniformly.
        """
        scen_attr = self._TIMESERIES_MODE_CONFIG[ts_mode]
        lst_ds = []
        for event_iloc in self._analysis.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self._analysis)
            ts_file = getattr(scen.scen_paths, scen_attr)
            if ts_file is None:
                raise ValueError(f"Timeseries path is None for ts_mode '{ts_mode}' and event_iloc={event_iloc}.")
            if not ts_file.exists():
                raise FileNotFoundError(
                    f"Timeseries file not found: {ts_file}. Run timeseries processing before consolidating."
                )
            open_kwargs = {"chunks": "auto", "engine": self._open_engine(), "decode_timedelta": False}
            if open_kwargs["engine"] == "zarr":
                open_kwargs["consolidated"] = zarr_store_has_index(ts_file)
            # Stamp the CURRENT loop position as event_iloc, for the reason given in
            # _retrieve_combined_output's loader. A per-scenario timeseries store carries no
            # event_iloc today, so this adds an index coordinate equal to the positions the
            # concat already produced and changes no data value; it keeps a store that ever
            # does carry one from publishing a stale position.
            lst_ds.append(xr.open_dataset(ts_file, **open_kwargs).assign_coords(event_iloc=[int(event_iloc)]))
        ds_ts = xr.concat(lst_ds, dim="event_iloc", combine_attrs="drop_conflicts", join="outer")

        from hhemt.scenario import compute_event_id_slug

        event_ids = [
            compute_event_id_slug(self._analysis._retrieve_weather_indexer_using_integer_index(ei))
            for ei in self._analysis.df_sims.index
        ]
        return ds_ts.assign_coords(event_id=("event_iloc", event_ids))

    def _select_stem_variables(self, mode: str, ds: xr.Dataset) -> xr.Dataset:
        """The part of one scenario's summary stem that ``mode`` consolidates (``_STEM_VARIABLE_SELECTION``).

        A mode with no entry keeps every variable. ``"include"`` keeps the listed names the stem holds, and
        ``"exclude"`` drops them, so two modes reading one stem never both carry a variable.
        """
        selection = self._STEM_VARIABLE_SELECTION.get(mode)
        if selection is None:
            return ds
        how, names = selection
        if how == "include":
            # Subsetting by name keeps only the coordinates the chosen variables use, so a node coordinate of the
            # stem does not travel into a group that has no node dimension.
            return ds[[v for v in names if v in ds.data_vars]]
        return ds.drop_vars([v for v in names if v in ds.data_vars])

    def _retrieve_combined_output(self, mode: str, *, event_ilocs: "list[int] | None" = None) -> xr.Dataset:
        """
        Load pre-created summary files for each scenario and concatenate them.

        A scenario whose model log records its summary stores as cleared contributes its slice of the
        ``summary`` group instead (``_read_cleared_summary``), restricted to the variables the record names, so the
        result is the same Dataset whether or not the stores were cleared.

        Parameters
        ----------
        mode : str
            One of the nine keys in _MODE_CONFIG:
            "tritonswmm_triton", "tritonswmm_swmm_node", "tritonswmm_swmm_link",
            "tritonswmm_performance", "triton_only", "triton_only_performance",
            "swmm_only_node", "swmm_only_link", "swmm_only_system". An unknown mode raises ``ValueError``
            listing the valid keys.
        event_ilocs : list of int, optional
            Read only these scenarios, in this order. Default: every scenario of the analysis.
        """
        if mode not in self._MODE_CONFIG:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {list(self._MODE_CONFIG.keys())}")

        events = [int(i) for i in (self._analysis.df_sims.index if event_ilocs is None else event_ilocs)]
        cleared = self._summary_records(mode)
        drop_flag = mode == "tritonswmm_swmm_node" and self._exclude_coupling_flag()

        lst_ds = []
        for event_iloc in events:
            if event_iloc in cleared:
                lst_ds.append(self._read_cleared_summary(mode, event_iloc, cleared[event_iloc]))
                continue
            ds = self._event_summary_dataset(mode, event_iloc, drop_flag=drop_flag)
            # event_iloc is POSITIONAL. A summary not reprocessed after an add/remove
            # still carries the position it was stamped with, so concat on the embedded
            # value duplicates a coordinate (measured [0, 2, 2]). Re-stamp from the
            # current loop position, as event_id below already is. In-memory only: the
            # per-scenario store on disk is not rewritten.
            ds = ds.assign_coords(event_iloc=[int(event_iloc)])
            # Normalize the producing-stamp coordinates before concat so a mix of
            # stamped (LAYOUT_VERSION 17+) and unstamped (legacy) summaries concats
            # cleanly. The "unknown" sentinel here exists ONLY in the in-memory
            # concatenated tree — it is never written back to the legacy flat
            # summary, so no historical provenance is fabricated.
            if "hhemt_producing_sha" not in ds.coords:
                ds = ds.assign_coords(hhemt_producing_sha=("event_iloc", ["unknown"]))
            if "hhemt_producing_version" not in ds.coords:
                ds = ds.assign_coords(hhemt_producing_version=("event_iloc", ["unknown"]))
            lst_ds.append(ds)

        # Column-set record normalization, mirroring the producing-stamp normalization in
        # the loop above, and for the same reason. The record is the per-scenario
        # PERF_COLUMN_SET_COORD that process_simulation stamps on the performance timeseries
        # store and that rides the summary reduction into the summary read here -- the FIRST
        # of the two heterogeneity-detection sites, covering one sim's allocations; the
        # second is the member join below.
        # A summary written before the column-set guard shipped carries no column-set coord,
        # and a population mixing stamped and unstamped members must concat cleanly. The
        # sentinel exists ONLY in the in-memory concatenated tree -- it is never written
        # back to the legacy flat summary, so no verdict is fabricated for a member whose
        # raw performance{N}.txt inputs are gone.
        from hhemt.process_simulation import (  # function-local: keeps the import graph acyclic
            NAME_SET_UNKNOWN,
            PERF_COLUMN_SET_COORD,
        )

        if any(PERF_COLUMN_SET_COORD in d.coords for d in lst_ds):
            lst_ds = [
                d
                if PERF_COLUMN_SET_COORD in d.coords
                else d.assign_coords({PERF_COLUMN_SET_COORD: ("event_iloc", [NAME_SET_UNKNOWN])})
                for d in lst_ds
            ]

        # VARIABLE-SET HETEROGENEITY DETECTION -- the second detection site, across members,
        # immediately before the join.
        #
        # WHY HERE: `lst_ds` is still a LIST of Datasets each carrying its own variable
        # set. The concat one line below takes no `join=` and therefore defaults to
        # `join="outer"` -- it does NOT raise, it NaN-fills -- and after it runs, a
        # variable a member never emitted is indistinguishable from one it emitted as
        # NaN. This is the last frame in which the question is answerable.
        #
        # DELIVER, DO NOT STOP, and the reason is measured rather than preferred. All
        # three callers of `consolidate_to_datatree` were surveyed: `analysis.py` does not
        # catch; `sensitivity_analysis.py` catches only (FileNotFoundError, ValueError) and
        # routes a caught one to a print; `consolidate_workflow.py` catches Exception and
        # routes it to `logger.error` plus a non-zero return. So on the raise path the
        # finding reaches a RUNTIME LOG and no tree is written -- and a runtime log is
        # excluded in terms from the published surfaces this record has to reach. Raising
        # here would therefore destroy the artifact that is the only thing able to carry
        # the finding. The record is made on the non-raise path.
        #
        # NOT SCOPED TO THE PERFORMANCE MODES. The property is the same at every mode this
        # method serves, the detection costs one pass over already-open metadata, and the
        # consequence of a false positive is one extra coordinate string rather than a
        # failure -- an asymmetry that is only available because this records instead of
        # raising.
        _labels = [str(ei) for ei in events]
        _findings = describe_member_variable_sets(_labels, [set(d.data_vars) for d in lst_ds], mode=mode)
        lst_ds = [
            d.assign_coords({VARIABLE_SET_COORD: ("event_iloc", [f])}) for d, f in zip(lst_ds, _findings, strict=True)
        ]

        ds_combined_outputs = xr.concat(lst_ds, dim="event_iloc", combine_attrs="drop_conflicts")

        from hhemt.scenario import compute_event_id_slug

        event_ids = [
            compute_event_id_slug(self._analysis._retrieve_weather_indexer_using_integer_index(ei)) for ei in events
        ]
        ds_combined_outputs = ds_combined_outputs.assign_coords(event_id=("event_iloc", event_ids))
        return ds_combined_outputs

    def _chunk_for_writing(
        self,
        ds_combined_outputs: xr.Dataset,
        spatial_coords: list[str] | str | None,
        spatial_coord_size: int = 65536,  # 256x256 for x,y coords
        verbose: bool = True,
        max_mem_usage_MiB: int | None = None,
    ):
        """
        Compute optimal chunk sizes for writing xarray datasets to disk.

        This is a wrapper around utils.compute_optimal_chunks() that provides
        the memory budget from analysis configuration, with an optional override
        for testing at specific memory budgets.

        Parameters
        ----------
        ds_combined_outputs : xr.Dataset
            Dataset to compute chunks for
        spatial_coords : List[str] | str | None
            Spatial coordinate names (e.g., ['x', 'y'] or 'node_id')
        spatial_coord_size : int
            Target total cells per spatial chunk (default 65536 = 256^2)
        verbose : bool
            Print chunk information if True
        max_mem_usage_MiB : int | None
            Memory budget override in MiB. If None, reads from
            cfg_analysis.process_output_target_chunksize_mb.

        Returns
        -------
        dict or "auto"
            Chunk specification for each dimension
        """
        from hhemt.utils import compute_optimal_chunks

        if max_mem_usage_MiB is None:
            max_mem_usage_MiB = self._analysis.cfg_analysis.process_output_target_chunksize_mb

        return compute_optimal_chunks(
            ds=ds_combined_outputs,
            spatial_coords=spatial_coords,
            max_mem_usage_MiB=max_mem_usage_MiB,
            spatial_coord_size=spatial_coord_size,
            verbose=verbose,
        )

    def _open_engine(self):
        processed_out_type = self._analysis.cfg_analysis.target_processed_output_type
        if processed_out_type == "zarr":
            return "zarr"
        elif processed_out_type == "nc":
            return "h5netcdf"

    def _already_written(self, f_out) -> bool:
        """
        Checks log file to determine whether the file was written successfully
        """
        proc_log = self._analysis.log.processing_log.outputs
        already_written = False
        if f_out.name in proc_log.keys():
            if proc_log[f_out.name].success is True:
                already_written = True
        return already_written


#: The four descriptor keys ``metadata.build_analysis_crate._descriptor`` projects. Declared here
#: so the mapping this module hands it carries exactly the keys it reads and no others; a wider
#: mapping would be silently truncated there and a narrower one would publish a `None`.
_CRATE_DESCRIPTOR_KEYS: tuple[str, ...] = ("long_name", "units", "standard_name", "cell_methods")


def emitted_vars_and_attrs(node_datasets) -> tuple[set[str], dict[str, dict[str, str | None]]]:
    """Return ``(emitted_vars, emitted_attrs)`` for the crate, from ONE pass over the tree's nodes.

    THIS EXISTS SO THE TWO HALVES CANNOT DIVERGE IN MEMBERSHIP, which is the whole reason it is a
    helper rather than a second comprehension at each call site. ``build_analysis_crate``
    advertises the names in ``emitted_vars`` and describes each one from ``emitted_attrs``, so a
    name present in the first and absent from the second is advertised with a STATIC descriptor
    while reading as store-sourced. Both callers previously built only the first half, each with
    its own comprehension over its own tree handle, and both now derive both halves here.

    WHY THIS LIVES IN THIS MODULE AND NOT IN ``metadata.py``. That module deliberately carries no
    xarray dependency -- its ``emitted_attrs`` parameter is documented as "a precomputed MAPPING
    rather than the datasets themselves" for exactly that reason -- and this function reads
    ``ds.data_vars`` and ``da.attrs``. ``sensitivity_analysis.py`` already imports its sibling
    stamping helpers from here, so this is the module both callers can reach.

    THE CONFLICT POLICY, and it is a DECIDED choice rather than a fallthrough. A flat
    ``{name: attrs}`` mapping cannot express one name described two ways, and a tree MAY carry one
    name in two nodes: ``max_velocity`` is the measured case, carried by both
    ``tritonswmm_swmm_link`` and ``swmm_only_link``. Where two nodes agree, the agreed descriptor
    is used. Where they DISAGREE, the name is OMITTED from the mapping so ``_descriptor`` falls
    back to its static entry -- because there is no single correct description, and publishing
    either one would be a false claim about the other node. That keeps this change MONOTONE in the
    sense ``_descriptor``'s own contract states: it can only replace a static descriptor with a
    store descriptor that actually exists AND is unambiguous. Measured on the live tables, no
    variable's override diverges from its base entry (``max_velocity`` is now the SWMM conduit
    in the base and in both link modes, which are byte-identical to it), so the disagree branch is
    reachable only by a future heterogeneous tree and is inert on the current corpus. An unstamped
    node contributes nothing: ``_descriptor``'s predicate is a non-empty ``long_name``, so a node carrying none
    leaves the name on its static entry regardless of what this returns.
    """
    emitted_vars: set[str] = set()
    attrs_by_var: dict[str, dict[str, str | None]] = {}
    conflicted: set[str] = set()
    for ds in node_datasets:
        for name, da in ds.data_vars.items():
            var = str(name)
            emitted_vars.add(var)
            if var in conflicted:
                continue
            desc = {k: da.attrs.get(k) for k in _CRATE_DESCRIPTOR_KEYS}
            if not desc.get("long_name"):
                continue
            prior = attrs_by_var.get(var)
            if prior is None:
                attrs_by_var[var] = desc
            elif prior != desc:
                conflicted.add(var)
                del attrs_by_var[var]
    return emitted_vars, attrs_by_var


def _stamp_triton_provenance(tree: "xr.DataTree", analysis) -> None:
    """Stamp the producing-TRITON provenance onto a consolidated-tree ROOT as plain attrs.

    Carries ``triton_producing_sha``, plus the per-target TRITON and standalone-SWMM
    producing records stamped below — each captured at COMPILE time onto the system log (a
    DIFFERENT process on HPC; see ``system.py::_capture_tritonswmm_provenance``) — onto the
    consolidated tree root. Kept
    as PLAIN root attrs (NOT ``metadata._EMBEDDED_PROV_KEYS``) so an additive provenance
    stamp does not churn the embedded-core byte-identity golden fixtures and needs no
    LAYOUT_VERSION bump. Shared by both consolidation sites
    (``consolidate_to_datatree`` here + ``consolidate_sensitivity_datatree`` in
    ``sensitivity_analysis.py``). Graceful-absent: when the system log is missing or
    unstamped (a pre-provenance build, or a reconstituted reprex bundle), the attrs are
    simply omitted and ``check_coupled_resume_validity`` treats their absence as
    INDETERMINATE — never a false pre-fix warn. Never raises.
    """
    _sys = getattr(analysis, "_system", None)
    _sys_log = getattr(_sys, "log", None)
    if _sys_log is None:
        return
    try:
        _sys_log.refresh()  # pick up the compile-process write in the cross-process case
    except Exception:
        pass
    try:
        _sha = _sys_log.triton_head_sha.get()
    except Exception:
        return
    # ONE CLONE-LEVEL ATTR. `triton_producing_sha` is retained deliberately and is the only
    # CLONE-level TRITON sha this stamps; the per-target build shas are stamped by the loop
    # below. It is the registry's input, it is what the per-arm provenance table
    # prints per arm, and under a split pin it is what makes the per-arm solver difference
    # legible in the report. The two per-defect booleans that were stamped beside it are
    # retired — see log.py. An OLD tree carrying them still reads correctly, because nothing
    # consults them any more; a NEW tree read by an OLD toolkit shows them absent, which that
    # toolkit already treats as INDETERMINATE rather than as a false warn.
    if _sha is not None:
        tree.attrs["triton_producing_sha"] = str(_sha)
    else:
        # LOUD-ABSENT when THIS log says a TRITON target was built (native: a compile flag)
        # or an image was set up (container: no compile flag is ever set, but setup records
        # sif_sha256 before the label sha). A recorded build or image with no recorded sha is
        # the lost-stamp signature, and this line is the only record of what consolidation
        # read. Both unset (a reprex bundle, a legacy log) stay silent. Everything, the print
        # included, sits inside the try: this function never raises.
        try:
            _built = [
                _f
                for _f in (
                    "compilation_tritonswmm_cpu_successful",
                    "compilation_tritonswmm_gpu_successful",
                    "compilation_triton_cpu_successful",
                    "compilation_triton_gpu_successful",
                )
                if getattr(_sys_log, _f).get() is True
            ]
            if _sys_log.sif_sha256.get():
                _built.append("sif_sha256")
            if _built:
                print(
                    f"[Provenance] WARNING: {_sys_log.logfile} records a TRITON build or image "
                    f"({', '.join(_built)}) but no triton_head_sha; this consolidated tree is NOT "
                    f"stamped with triton_producing_sha. last_writer={_sys_log.last_writer}",
                    flush=True,
                )
        except Exception:
            pass

    # The clone HEAD's committer date (`YYYYMMDD`), same carrier and same graceful-absent rule
    # as the sha above, stamped as a PLAIN root attr beside it and, like it, kept out of
    # `metadata._EMBEDDED_PROV_KEYS`. Independent of the sha: a log carrying a sha and no date
    # (a toolkit that captured the sha before the date existed) stamps the sha alone.
    try:
        _commit_date = _sys_log.triton_head_commit_date.get()
    except Exception:
        _commit_date = None
    if _commit_date is not None:
        tree.attrs["triton_producing_commit_date"] = str(_commit_date)

    # Per-target TRITON and standalone-SWMM provenance, same carrier and same
    # graceful-absent rule as the clone-level sha above: captured at compile time onto the
    # system log by the invocation that ACTUALLY built each target, stamped here as
    # PLAIN root attrs so no _EMBEDDED_PROV_KEYS change and therefore no byte-identity golden
    # regeneration and no LAYOUT_VERSION bump. Each field is stamped independently --
    # container mode supplies a version and no sha, and a half-present record is the
    # honest one. Absence reads as INDETERMINATE downstream, never as agreement.
    for _attr, _field in (
        ("tritonswmm_producing_sha", "tritonswmm_producing_sha"),
        ("triton_only_producing_sha", "triton_only_producing_sha"),
        ("standalone_swmm_producing_version", "standalone_swmm_producing_version"),
        ("standalone_swmm_producing_sha", "standalone_swmm_producing_sha"),
    ):
        try:
            _val = getattr(_sys_log, _field).get()
        except Exception:
            continue
        if _val is not None:
            tree.attrs[_attr] = str(_val)


def _parse_replay_t(text: str, marker: str) -> "float | None":
    """Parse the numeric ``t=`` value from the LAST replay marker in a model log.

    The model log is ``"w"``-truncated per exec (``run_simulation_runner`` reopens it in
    ``"w"`` mode on every exec), so this is the LAST
    resume's replay boundary — the vertical-line marker the over-time figure needs.
    Best-effort: returns None when the marker is absent or unparseable.
    """
    _m = None
    for _m in re.finditer(rf"{re.escape(marker)}\s*([-+0-9.eE]+)", text):
        pass  # keep the last match (last exec's replay)
    if _m is None:
        return None
    try:
        return float(_m.group(1))
    except (ValueError, IndexError):
        return None


def _stamp_coupled_resume_evidence(tree: "xr.DataTree", analysis, *, df_status=None) -> None:
    """Stamp per-sub coupled-resume replay evidence onto the consolidated ROOT.

    Captured at CONSOLIDATION time (logs still live, before any raw-output purge) as a PLAIN root attr
    ``coupled_resume_replay_evidence`` = JSON ``{sub_id: {resumed, completed, resume_mechanism, ...}}`` over
    each tritonswmm resume-candidate sim. Makes the resume-validity evidence DURABLE (survives the
    ``"w"``-mode last-exec log being cleared/purged) and bundle-portable, so a downstream combine
    / reprex consumer can assert genuine replay without the raw logs. Mirrors
    ``_stamp_triton_provenance``: a plain root attr (NOT an ``_EMBEDDED_PROV_KEYS`` member) so no
    LAYOUT_VERSION bump / no change to the provenance-core key set the tests pin. Best-effort;
    never raises.
    """
    import json

    from hhemt.analysis_validation import (
        _TRITON_CHECKPOINT_READ_MARKER,
        _TRITON_COMPLETION_MARKER,
        _TRITON_REPLAY_MARKER,
        _TRITON_SNAPSHOT_RESTORE_MARKER,
        _iter_members_or_self,
        resume_mechanism_from_log,
    )
    from hhemt.run_simulation import model_logfile_for

    try:
        import pandas as pd

        df = df_status if df_status is not None else getattr(analysis, "df_status", None)
        if df is None or not {"model_type", "n_resumes", "event_iloc"}.issubset(getattr(df, "columns", [])):
            return
        n_res = pd.to_numeric(df["n_resumes"], errors="coerce").fillna(0)
        cands = df[(df["model_type"] == "tritonswmm") & (n_res >= 1)]
        if len(cands) == 0:
            return
        subs = {(str(k) if k is not None else None): v for k, v in _iter_members_or_self(analysis)}
        evidence: dict[str, dict[str, bool]] = {}
        _id_col = resolve_member_id_column(getattr(cands, "columns", []))
        for _, row in cands.iterrows():
            _sa = row.get(_id_col) if _id_col else None
            sub = subs.get(str(_sa) if _sa is not None else None)
            if sub is None:
                continue
            try:
                text = model_logfile_for(sub, int(row["event_iloc"]), "tritonswmm").read_text(errors="replace")
            except Exception:  # noqa: BLE001 — log unreadable at consolidation: skip this sub, best-effort
                continue
            key = str(_sa) if _sa is not None else str(row.get("scenario_directory", ""))
            # WHICH MECHANISM, not a replay boolean. TRITON restores the coupled SWMM state
            # either from a full-precision per-checkpoint SNAPSHOT (the fast path) or by
            # REPLAYING the recorded exchange history (the fallback, retained for checkpoints
            # written before snapshots existed). Both are correct; only "neither" is the
            # truncation defect. This stamp is the LOAD-BEARING half of the resume-validity
            # machinery -- it has no registry gate and always runs -- so whatever it records
            # here is the most any downstream consumer can ever know once the "w"-mode log is
            # purged. Recording a replay boolean would have made every snapshot-restored sim
            # indistinguishable from the defect.
            _mech = resume_mechanism_from_log(text)
            _replay_t = _parse_replay_t(text, _TRITON_REPLAY_MARKER)
            _snapshot_t = _parse_replay_t(text, _TRITON_SNAPSHOT_RESTORE_MARKER)
            # The positioned resume time, from WHICHEVER marker fired. Both markers carry the
            # identical "...to t=" shape, so one parser serves both.
            _resume_t = _snapshot_t if _mech == "snapshot" else _replay_t
            # Schedule VERIFICATION. The resume schedule is now CONFIGURED, so the
            # parsed resume time is a check against it, not the sole source of truth. Units differ:
            # resume_t is TRITON sim-time SECONDS while schedule entries are checkpoint INDICES,
            # so the last requested boundary in seconds is schedule[-1] * TRITON_reporting_timestep_s.
            # The realized kill is a lower-bounded lag of the request, so the honored predicate is
            # resume_t >= that product. COUPLED-ONLY: a TRITON-only sim (no coupled SWMM) emits
            # neither marker, so resume_t is None and the comparison is None here (its schedule
            # verification rests on df_status n_resumes == len(schedule); see
            # analysis_validation.check_resume_schedule_honored).
            _c = getattr(sub, "cfg_analysis", None)
            _sched = getattr(_c, "resume_interruption_schedule", None)
            _interval = getattr(_c, "TRITON_reporting_timestep_s", None)
            _expected_t = float(_sched[-1]) * float(_interval) if _sched and _interval is not None else None
            _matches = (_resume_t >= _expected_t) if (_resume_t is not None and _expected_t is not None) else None
            _replay_matches = (
                (_replay_t >= _expected_t) if (_replay_t is not None and _expected_t is not None) else None
            )
            evidence[key] = {
                # v2 discriminator. Its ABSENCE is what tells a reader a record is v1 --
                # written by a toolkit that looked for the replay marker and nothing else --
                # so `resume_mechanism_from_stamp` can report INDETERMINATE for a v1
                # `replayed: false` instead of asserting the defect it never measured.
                "stamp_schema": 2,
                "resumed": _TRITON_CHECKPOINT_READ_MARKER in text,
                "completed": _TRITON_COMPLETION_MARKER in text,
                "resume_mechanism": _mech,
                "resume_t": _resume_t,
                "resume_matches_schedule": _matches,
                # RETAINED, and truthful in both worlds: each names ONE marker and says
                # whether that marker fired. They are kept so a v1-shaped reader anywhere
                # (a bundle consumer, a downstream notebook) keeps getting a correct answer
                # to the question it asks. The v1 READING of `replayed: false` as "defective"
                # is what is retired, not the field.
                "replayed": _mech == "replay",
                "restored_from_snapshot": _mech == "snapshot",
                "replay_t": _replay_t,
                "expected_replay_t": _expected_t,
                "replay_matches_schedule": _replay_matches,
            }
        if evidence:
            tree.attrs["coupled_resume_replay_evidence"] = json.dumps(evidence, sort_keys=True)
    except Exception:  # noqa: BLE001 — durable-evidence stamp is best-effort; never block consolidation
        return


def ds_memory_req_MiB(ds):
    """Return the in-memory size of ``ds`` in MiB (``ds.nbytes`` / 2**20)."""
    return ds.nbytes / 1024**2


def make_sure_ds_are_compatible_for_concatenation(ds_ref, ds_comp, lst_common_dims=("x", "y")):
    """Return the dimension and coordinate-value mismatches between two datasets as warning text.

    An empty string means compatible. Most mismatches come back as warning lines, but two
    raise instead: a common dimension whose two coordinates have different lengths (other
    than a length of 1) raises ValueError, and a dataset not opened from a file (no
    ``encoding["source"]``) raises KeyError.
    """
    all_problems = ""
    problems = check_matching_dimensions(ds_ref, ds_comp)
    matching_dim_problems = check_for_matching_dim_values(ds_ref, ds_comp, lst_common_dims)
    all_problems += problems + matching_dim_problems
    # print(all_problems)
    return all_problems


def check_matching_dimensions(ds_ref, ds_comp):
    """Return one warning line per dimension present in one dataset and absent from the other.

    Names the files from each dataset's ``encoding["source"]``, so both must have been
    opened from a file; a dataset built in memory raises KeyError.
    """
    problems = ""
    lst_common_dims = []
    f_ref = ds_ref.encoding["source"]
    f_comp = ds_comp.encoding["source"]
    for dim in ds_ref.dims:
        if dim not in ds_comp.dims:
            problems += f"| WARNING: {dim} in {f_ref} but not in {f_comp} |\n"
        else:
            lst_common_dims.append(dim)
            # print(problems)
    for dim in ds_comp.dims:
        if dim not in ds_ref.dims:
            problems += f"| WARNING: {dim} in {f_comp} but not in {f_ref} |\n"
    # print(problems)
    return problems


def check_for_matching_dim_values(ds_ref, ds_comp, lst_common_dims=("x", "y")):
    """Return one warning line per dimension in ``lst_common_dims`` whose coordinate values differ.

    Compares elementwise. Coordinates of unequal length raise ValueError in the
    subtraction, except that a length-1 coordinate is broadcast against the other and
    the difference is reported as a warning line.
    """
    problems = ""
    f_ref = ds_ref.encoding["source"]
    f_comp = ds_comp.encoding["source"]
    for dim in lst_common_dims:
        ar_dif = ds_ref[dim].values - ds_comp[dim].values
        n_diff = ((ar_dif) != 0).sum()
        if n_diff > 0:
            problems += f"| WARNING: {dim} values are not all equal in {f_ref} and {f_comp} |\n"
    # print(problems)
    return problems


def check_da_for_na(da):
    """Return True if ``da`` contains any NaN."""
    nan_mask = da.isnull()
    any_nans = bool(nan_mask.any().values)
    return any_nans


def return_lst_dic_of_unique_storm_idxs(ds):
    """Return one ``{coord: value}`` selector per unique storm in ``ds``.

    The storm coordinates are every coordinate except x, y, model, simtype, link_id and
    node_id. The probe variable is the first of max_flood_depth, max_flow and
    total_inflow_vol present; a dataset carrying none of them raises
    UnboundLocalError.
    """
    lst_coords = []
    for coord in ds.coords:
        if coord not in [
            "x",
            "y",
            "model",
            "simtype",
            "link_id",
            "node_id",
        ]:  # and (len(ds_triton[coord].values)>1):
            lst_coords.append(coord)
    # Read the unique non-spatial coordinate combinations -- one per storm -- off a single
    # probe element, selected by POSITION 1 along x and y, link_id, or node_id, of the
    # first of max_flood_depth / max_flow / total_inflow_vol the dataset carries.
    if "max_flood_depth" in ds.data_vars:
        datavar = "max_flood_depth"
        idx_loc = dict(x=1, y=1)
    elif "max_flow" in ds.data_vars:
        datavar = "max_flow"
        idx_loc = dict(link_id=1)
    elif "total_inflow_vol" in ds.data_vars:
        datavar = "total_inflow_vol"
        idx_loc = dict(node_id=1)
    if "x" in ds.coords and "y" in ds.coords:
        idx_storms = ds.isel(idx_loc)[datavar].to_dataframe().reset_index().set_index(lst_coords).index.unique()
    else:
        idx_storms = ds.isel(idx_loc)[datavar].to_dataframe().reset_index().set_index(lst_coords).index.unique()
    idx_names = idx_storms.names
    lst_dic_storm_sel = []
    for idx in idx_storms:
        dic_sel = dict()
        for i, name in enumerate(idx_names):
            dic_sel[name] = idx[i]
        lst_dic_storm_sel.append(dic_sel)
    return lst_dic_storm_sel
