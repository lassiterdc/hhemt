Compute-config EDA: config-diff maps.

Cross-compute-configuration exploratory analysis for this sensitivity master. The figure reads the consolidated ``sensitivity_datatree.zarr`` (per-cell ``max_wlevel_m`` + per-conduit ``max_flow_cms`` + per-sub compute-config attrs) and renders a cross-config identity + absolute-diff table plus, per byte-identical config group, the signed-diff and percent-diff maps (DEM cells + SWMM conduits) versus the serial-CPU baseline. Compute-config labels derive from config attributes, not the ``sa_id`` name. Rendered as a config-selectable tab via the ``compute-sensitivity`` reporting set (``report_config.reporting_set``).

**Sources:**

{{ snakemake.params.source_paths_rst }}
