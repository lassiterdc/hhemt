import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

from hhemt.cf_conventions import (
    apply_cf_attributes,
    apply_global_attributes,
)
from hhemt.scenario import TRITONSWMM_scenario
from hhemt.utils import (
    current_datetime_string,
    get_file_size_MiB,
    write_datatree_zarr,
)

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis


class TRITONSWMM_analysis_post_processing:
    # Maps consolidation mode to: (scenario_path_attr, analysis_path_attr, spatial_coords)
    _MODE_CONFIG = {
        "tritonswmm_triton": (
            "output_tritonswmm_triton_summary",
            "output_tritonswmm_triton_summary",
            ["x", "y"],
        ),
        "tritonswmm_swmm_node": (
            "output_tritonswmm_node_summary",
            "output_tritonswmm_node_summary",
            "node_id",
        ),
        "tritonswmm_swmm_link": (
            "output_tritonswmm_link_summary",
            "output_tritonswmm_link_summary",
            "link_id",
        ),
        "triton_only": (
            "output_triton_only_summary",
            "output_triton_only_summary",
            ["x", "y"],
        ),
        "triton_only_performance": (
            "output_triton_only_performance_summary",
            "output_triton_only_performance_summary",
            None,
        ),
        "tritonswmm_performance": (
            "output_tritonswmm_performance_summary",
            "output_tritonswmm_performance_summary",
            None,
        ),
        "swmm_only_node": (
            "output_swmm_only_node_summary",
            "output_swmm_only_node_summary",
            "node_id",
        ),
        "swmm_only_link": (
            "output_swmm_only_link_summary",
            "output_swmm_only_link_summary",
            "link_id",
        ),
    }

    _MODE_TO_TREE_PATH = {
        "tritonswmm_triton": "tritonswmm/triton",
        "tritonswmm_swmm_node": "tritonswmm/swmm_node",
        "tritonswmm_swmm_link": "tritonswmm/swmm_link",
        "tritonswmm_performance": "tritonswmm/performance",
        "triton_only": "triton_only/triton",
        "triton_only_performance": "triton_only/performance",
        "swmm_only_node": "swmm_only/swmm_node",
        "swmm_only_link": "swmm_only/swmm_link",
    }

    # Per-scenario TIMESERIES modes (opt-in via analysis_config.toggle_consolidate_timeseries).
    # Maps mode -> the scenario-path attr for the per-scenario timeseries zarr. SWMM node
    # (wlevel(t)) + SWMM link (flow(t)) only: the TRITON gridded timeseries is ~24x larger
    # per scenario and is NOT needed for the clean-vs-resume over-time figure, so it is
    # deliberately excluded here (adding it would be GB-scale on a fine grid).
    _TIMESERIES_MODE_CONFIG = {
        "tritonswmm_swmm_node_ts": "output_tritonswmm_node_time_series",
        "tritonswmm_swmm_link_ts": "output_tritonswmm_link_time_series",
    }

    _TIMESERIES_MODE_TO_TREE_PATH = {
        "tritonswmm_swmm_node_ts": "tritonswmm/swmm_node_timeseries",
        "tritonswmm_swmm_link_ts": "tritonswmm/swmm_link_timeseries",
    }

    def __init__(self, analysis: "TRITONSWMM_analysis") -> None:
        self._analysis = analysis

    def to_datatree(self) -> "xr.DataTree":
        tree_dict: dict[str, xr.Dataset] = {}
        for mode, tree_path in self._MODE_TO_TREE_PATH.items():
            analysis_path_attr = self._MODE_CONFIG[mode][1]
            f = getattr(self._analysis.analysis_paths, analysis_path_attr)
            if f is None or not f.exists():
                continue
            tree_dict[tree_path] = self._open(f)
        tree_dict["/"] = xr.Dataset(attrs={"analysis_id": str(self._analysis.cfg_analysis.analysis_id)})
        return xr.DataTree.from_dict(tree_dict)

    CONSOLIDATION_VERSION = 1

    def consolidate_to_datatree(
        self,
        compression_level: int = 5,
        verbose: bool = False,
    ) -> Path:
        """Assemble per-scenario summaries directly into a hierarchical DataTree zarr.

        Per Option B (render_bundle plan, 2026-05-05): no intermediate
        master-level per-mode flat zarrs are produced. The DataTree IS the
        canonical master-level artifact; per-scenario summaries are its
        only inputs.

        Per cleanup-rerun-delete-redesign Phase 3, the legacy
        ``overwrite_if_already_created`` parameter is retired; force-rerun
        capability arrives in Phase 4 via ``override_force_rerun``.
        """
        fname_out = self._analysis.analysis_paths.analysis_datatree_zarr
        if fname_out is None:
            raise ValueError("analysis_datatree_zarr path is not configured on AnalysisPaths.")

        # Per D5: .exists() alone is an unreliable completion signal — a
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
        if fname_out.exists() and _log_complete:
            if verbose:
                print(f"DataTree zarr already present at {fname_out} and log complete. Not overwriting.")
            return fname_out
        if fname_out.exists() and not _log_complete:
            from hhemt.utils import fast_rmtree

            fast_rmtree(fname_out, analysis_dir=self._analysis.analysis_paths.analysis_dir)
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
            apply_cf_attributes(ds, mode)
            tree_dict[tree_path] = ds

        # Per-scenario timeseries nodes (opt-in). Default OFF. When enabled, each SWMM
        # node/link timeseries is concatenated along event_iloc exactly as the summaries
        # are, and grafts up to the sensitivity master for free via
        # build_sensitivity_datatree's subtree_with_keys copy. A missing timeseries file
        # raises FileNotFoundError here, which the master consolidate loop already tolerates
        # under allow_incomplete=True (whole-sub skip), mirroring the summary path.
        if getattr(self._analysis.cfg_analysis, "toggle_consolidate_timeseries", False):
            for ts_mode, ts_tree_path in self._TIMESERIES_MODE_TO_TREE_PATH.items():
                ts_scen_attr = self._TIMESERIES_MODE_CONFIG[ts_mode]
                first_scen = TRITONSWMM_scenario(self._analysis.df_sims.index[0], self._analysis)
                if getattr(first_scen.scen_paths, ts_scen_attr) is None:
                    continue
                tree_dict[ts_tree_path] = self._retrieve_combined_timeseries(ts_mode)

        tree_dict["/"] = xr.Dataset(
            attrs={
                "analysis_id": str(self._analysis.cfg_analysis.analysis_id),
                "output_creation_date": current_datetime_string(),
                "consolidation_version": self.CONSOLIDATION_VERSION,
            }
        )
        tree = xr.DataTree.from_dict(tree_dict)
        apply_global_attributes(tree, analysis_id=str(self._analysis.cfg_analysis.analysis_id))

        from hhemt.cf_conventions import apply_producing_stamp, apply_provenance_core
        from hhemt.provenance import emit_provenance

        _core_json, _graph_json = emit_provenance(self._analysis)
        apply_provenance_core(tree, core_json_str=_core_json)

        # ADR-15 Phase 1: re-derive the scalar producing-stamp fast-path on the
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
        _stamp_coupled_resume_evidence(tree, self._analysis)

        write_datatree_zarr(tree, fname_out, compression_level=compression_level)

        from hhemt.metadata import write_rocrate_sidecar

        write_rocrate_sidecar(self._analysis.analysis_paths.analysis_dir, graph_json=_graph_json)

        self._analysis._refresh_log()
        if hasattr(self._analysis.log, "datatree_consolidation_complete"):
            self._analysis.log.datatree_consolidation_complete.set(True)
        if hasattr(self._analysis.log, "consolidation_version"):
            self._analysis.log.consolidation_version.set(self.CONSOLIDATION_VERSION)
        elapsed_s = time.time() - start_time
        self._analysis.log.add_sim_processing_entry(fname_out, get_file_size_MiB(fname_out), elapsed_s, True)

        # Write the analysis-level DU sentinel. Compare-and-write semantics in
        # du_sentinels.write_du_sentinel preserve mtime on idempotent re-runs,
        # so consumer rules that declare _du.json as input: are not cascade-rerun.
        from hhemt.du_sentinels import sum_child_sentinels

        sum_child_sentinels(
            self._analysis.analysis_paths.analysis_dir,
            scope="analysis",
            child_scope_dirs=["subanalyses", "sims"],
        )

        if verbose:
            print(f"Wrote DataTree zarr to {fname_out} in {elapsed_s:.1f}s")
        return fname_out

    def open_datatree(self) -> "xr.DataTree":
        """Open the consolidated hierarchical DataTree zarr lazily.

        Per Option B (render_bundle plan): the canonical signal that the
        DataTree is present and complete is
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
        # Mirrors the producer (_refresh_log before .set ~L167) and the
        # build_sensitivity_datatree per-sub precedent (sensitivity_analysis.py
        # ~L1195). The lost-update WRITE race is already closed (compute-on-read
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
        return xr.open_datatree(path, engine="zarr", chunks="auto", consolidated=False)

    def _retrieve_combined_timeseries(self, ts_mode: str) -> xr.Dataset:  # type: ignore
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
                raise ValueError(
                    f"Timeseries path is None for ts_mode '{ts_mode}' and event_iloc={event_iloc}."
                )
            if not ts_file.exists():
                raise FileNotFoundError(
                    f"Timeseries file not found: {ts_file}. Run timeseries processing before consolidating."
                )
            open_kwargs = {"chunks": "auto", "engine": self._open_engine(), "decode_timedelta": False}
            if open_kwargs["engine"] == "zarr":
                open_kwargs["consolidated"] = False
            lst_ds.append(xr.open_dataset(ts_file, **open_kwargs))
        ds_ts = xr.concat(lst_ds, dim="event_iloc", combine_attrs="drop_conflicts", join="outer")

        from hhemt.scenario import compute_event_id_slug

        event_ids = [
            compute_event_id_slug(self._analysis._retrieve_weather_indexer_using_integer_index(ei))
            for ei in self._analysis.df_sims.index
        ]
        return ds_ts.assign_coords(event_id=("event_iloc", event_ids))  # type: ignore

    def _retrieve_combined_output(self, mode: str) -> xr.Dataset:  # type: ignore
        """
        Load pre-created summary files for each scenario and concatenate them.

        Parameters
        ----------
        mode : str
            One of the keys in _MODE_CONFIG:
            "tritonswmm_triton", "tritonswmm_swmm_node", "tritonswmm_swmm_link",
            "triton_only", "swmm_only_node", "swmm_only_link"
        """
        if mode not in self._MODE_CONFIG:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {list(self._MODE_CONFIG.keys())}")

        scen_path_attr = self._MODE_CONFIG[mode][0]

        lst_ds = []
        for event_iloc in self._analysis.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self._analysis)

            summary_file = getattr(scen.scen_paths, scen_path_attr)

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
            # (R8) Defense-in-depth backstop. With the Phase-2 positive completion
            # marker gating the upstream generator emit (the d_process flag is
            # written only after all enabled-model summaries land), this raise is
            # unreachable on the happy path. Kept because the bare .exists() here is
            # acceptable AS A BACKSTOP once a positive DAG-enforced marker gates
            # upstream — it converts any residual out-of-band divergence into a clear
            # named error rather than a cryptic xr.open_dataset failure.

            open_kwargs = {
                "chunks": "auto",
                "engine": self._open_engine(),
                "decode_timedelta": False,
            }
            if open_kwargs["engine"] == "zarr":
                open_kwargs["consolidated"] = False
            ds = xr.open_dataset(summary_file, **open_kwargs)
            # ADR-15 Phase 1 (D6): normalize the producing-stamp coordinates before
            # concat so mixed stamped (v17+) / unstamped (legacy) summaries concat
            # cleanly. The "unknown" sentinel here exists ONLY in the in-memory
            # concatenated tree — it is never written back to the legacy flat
            # summary, so no historical provenance is fabricated.
            if "hhemt_producing_sha" not in ds.coords:
                ds = ds.assign_coords(hhemt_producing_sha=("event_iloc", ["unknown"]))
            if "hhemt_producing_version" not in ds.coords:
                ds = ds.assign_coords(hhemt_producing_version=("event_iloc", ["unknown"]))
            lst_ds.append(ds)

        ds_combined_outputs = xr.concat(lst_ds, dim="event_iloc", combine_attrs="drop_conflicts")

        from hhemt.scenario import compute_event_id_slug

        event_ids = [
            compute_event_id_slug(self._analysis._retrieve_weather_indexer_using_integer_index(ei))
            for ei in self._analysis.df_sims.index
        ]
        ds_combined_outputs = ds_combined_outputs.assign_coords(event_id=("event_iloc", event_ids))
        return ds_combined_outputs  # type: ignore

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

    def _open(self, f):
        if f.exists():
            open_kwargs = {
                "chunks": "auto",
                "engine": self._open_engine(),
                "decode_timedelta": False,
            }
            if open_kwargs["engine"] == "zarr":
                open_kwargs["consolidated"] = False
            return xr.open_dataset(f, **open_kwargs)  # type: ignore
        else:
            raise ValueError(
                f"could not open file because it does not exist: {f}. Run analysis.consolidate_[SWMM/TRITON]_outputs()."
            )

    # TRITON-SWMM coupled model accessors
    @property
    def tritonswmm_TRITON_summary(self):
        return self._open(self._analysis.analysis_paths.output_tritonswmm_triton_summary)

    @property
    def tritonswmm_SWMM_node_summary(self):
        return self._open(self._analysis.analysis_paths.output_tritonswmm_node_summary)

    @property
    def tritonswmm_SWMM_link_summary(self):
        return self._open(self._analysis.analysis_paths.output_tritonswmm_link_summary)

    @property
    def tritonswmm_performance_summary(self):
        return self._open(self._analysis.analysis_paths.output_tritonswmm_performance_summary)

    # TRITON-only model accessors
    @property
    def triton_only_summary(self):
        return self._open(self._analysis.analysis_paths.output_triton_only_summary)

    @property
    def triton_only_performance_summary(self):
        return self._open(self._analysis.analysis_paths.output_triton_only_performance_summary)

    # SWMM-only model accessors
    @property
    def swmm_only_node_summary(self):
        return self._open(self._analysis.analysis_paths.output_swmm_only_node_summary)

    @property
    def swmm_only_link_summary(self):
        return self._open(self._analysis.analysis_paths.output_swmm_only_link_summary)

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


def _stamp_triton_provenance(tree: "xr.DataTree", analysis) -> None:
    """Stamp the producing-TRITON provenance onto a consolidated-tree ROOT as plain attrs.

    Carries ``triton_producing_sha`` + ``triton_has_coupled_resume_fix`` — captured at
    COMPILE time onto the system log (a DIFFERENT process on HPC; see
    ``system.py::_capture_tritonswmm_provenance``) — onto the consolidated tree root. Kept
    as PLAIN root attrs (NOT ``metadata._EMBEDDED_PROV_KEYS``) so an additive provenance
    stamp does not churn the Gotcha-63 byte-identity golden fixtures and needs no
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
        _has_fix = _sys_log.triton_has_coupled_resume_fix.get()
    except Exception:
        return
    if _sha is not None:
        tree.attrs["triton_producing_sha"] = str(_sha)
    if _has_fix is not None:
        tree.attrs["triton_has_coupled_resume_fix"] = bool(_has_fix)


def _parse_replay_t(text: str, marker: str) -> "float | None":
    """Parse the numeric ``t=`` value from the LAST replay marker in a model log.

    The model log is ``"w"``-truncated per exec (Gotcha 71b), so this is the LAST
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


def _stamp_coupled_resume_evidence(tree: "xr.DataTree", analysis) -> None:
    """Stamp per-sub coupled-resume replay evidence onto the consolidated ROOT.

    Captured at CONSOLIDATION time (logs still live, pre-R7-purge) as a PLAIN root attr
    ``coupled_resume_replay_evidence`` = JSON ``{sub_id: {resumed, completed, replayed}}`` over
    each tritonswmm resume-candidate sim. Makes R9's acceptance evidence DURABLE (survives the
    ``"w"``-mode last-exec log being cleared/purged) and bundle-portable, so a downstream combine
    / reprex consumer can assert genuine replay without the raw logs. Mirrors
    ``_stamp_triton_provenance``: a plain root attr (NOT an ``_EMBEDDED_PROV_KEYS`` member) so no
    LAYOUT_VERSION bump / no golden churn. Best-effort; never raises.
    """
    import json

    from hhemt.analysis_validation import (
        _TRITON_CHECKPOINT_READ_MARKER,
        _TRITON_COMPLETION_MARKER,
        _TRITON_REPLAY_MARKER,
        _iter_subanalyses_or_self,
    )
    from hhemt.run_simulation import model_logfile_for

    try:
        import pandas as pd

        df = getattr(analysis, "df_status", None)
        if df is None or not {"model_type", "n_resumes", "event_iloc"}.issubset(getattr(df, "columns", [])):
            return
        n_res = pd.to_numeric(df["n_resumes"], errors="coerce").fillna(0)
        cands = df[(df["model_type"] == "tritonswmm") & (n_res >= 1)]
        if len(cands) == 0:
            return
        subs = {(str(k) if k is not None else None): v for k, v in _iter_subanalyses_or_self(analysis)}
        evidence: dict[str, dict[str, bool]] = {}
        for _, row in cands.iterrows():
            _sa = row.get("sa_id")
            sub = subs.get(str(_sa) if _sa is not None else None)
            if sub is None:
                continue
            try:
                text = model_logfile_for(sub, int(row["event_iloc"]), "tritonswmm").read_text()
            except Exception:  # noqa: BLE001 — log unreadable at consolidation: skip this sub, best-effort
                continue
            key = str(_sa) if _sa is not None else str(row.get("scenario_directory", ""))
            evidence[key] = {
                "resumed": _TRITON_CHECKPOINT_READ_MARKER in text,
                "completed": _TRITON_COMPLETION_MARKER in text,
                "replayed": _TRITON_REPLAY_MARKER in text,
                "replay_t": _parse_replay_t(text, _TRITON_REPLAY_MARKER),
            }
        if evidence:
            tree.attrs["coupled_resume_replay_evidence"] = json.dumps(evidence, sort_keys=True)
    except Exception:  # noqa: BLE001 — durable-evidence stamp is best-effort; never block consolidation
        return


def prev_power_of_two(n: int | float) -> int:
    n = int(n)
    if n < 1:
        return 1
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 << (n.bit_length() - 1)


def ds_memory_req_MiB(ds):
    return ds.nbytes / 1024**2


def make_sure_ds_are_compatible_for_concatenation(ds_ref, ds_comp, lst_common_dims=["x", "y"]):
    all_problems = ""
    problems = check_matching_dimensions(ds_ref, ds_comp)
    matching_dim_problems = check_for_matching_dim_values(ds_ref, ds_comp, lst_common_dims)
    all_problems += problems + matching_dim_problems
    # print(all_problems)
    return all_problems


def check_matching_dimensions(ds_ref, ds_comp):
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


def check_for_matching_dim_values(ds_ref, ds_comp, lst_common_dims=["x", "y"]):
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
    # Check for NaN values
    nan_mask = da.isnull()
    # Check if any NaN values are present
    any_nans = bool(nan_mask.any().values)
    return any_nans


def return_lst_dic_of_unique_storm_idxs(ds):
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
    # find unique indices for unique storm ids
    if "max_wlevel_m" in ds.data_vars:
        datavar = "max_wlevel_m"
        idx_loc = dict(x=1, y=1)
    elif "max_flow_cms" in ds.data_vars:
        datavar = "max_flow_cms"
        idx_loc = dict(link_id=1)
    elif "total_inflow_vol_10e6_ltr" in ds.data_vars:
        datavar = "total_inflow_vol_10e6_ltr"
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
