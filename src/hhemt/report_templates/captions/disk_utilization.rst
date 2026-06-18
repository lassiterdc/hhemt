Disk utilization breakdown for analysis **{{ snakemake.config.analysis_id }}**.

Total bytes-on-disk under the analysis directory plus a per-sub-path breakdown (one row per child of the analysis root). Values read from the ``_status/_du.json`` sentinel written by the analysis-scope consolidate step (Phase 1 DU sentinel infrastructure). When the sentinel is absent, the table is replaced by a re-run prompt.

**Sources:**

{{ snakemake.params.source_paths_rst }}
