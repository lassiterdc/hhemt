Full per-scenario status table for **{{ snakemake.config.analysis_id }}**.

Each row reports a single scenario × model_type combination's exit status (success / pending / failed), runtime, continuity error, and any per-row notes captured by ``export_scenario_status.py`` at workflow close.

**Sources:**

{{ snakemake.params.source_paths_rst }}
