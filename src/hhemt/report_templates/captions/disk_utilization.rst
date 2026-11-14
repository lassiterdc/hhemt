Disk utilization breakdown for analysis **{{ snakemake.config.analysis_id }}**.

One row per top-level entry of the analysis root that holds data, in mebibytes (MiB), leaving out status markers and the ephemeral test directory. The ``members/`` and ``sims/`` rows are the totals recorded in ``_status/_du.json``, rolled up from the member and scenario records; every other row is measured when this page is rendered.

**Sources:**

{{ snakemake.params.source_paths_rst }}
