# Report configurations

Each `*.yaml` here defines a `report_config` consumed by `analysis.run(report_config=...)`.

## Schema

See `src/TRITON_SWMM_toolkit/config/report.py` for the canonical Pydantic schema.

## Files

- `default_report_config.yaml` — built-in default; used when `analysis.run()` is called without `report_config=`.
- `synth_multisim_report_config.yaml` — used by `tests/test_synth_04_multisim_with_snakemake.py`.
- `synth_sensitivity_report_config.yaml` — used by `tests/test_synth_05_sensitivity_analysis_with_snakemake.py`. Demonstrates a benchmarking-mode sensitivity section.

## Validation

`report_config.yaml` is validated against the Pydantic schema at `analysis.run()` entry. Validation errors raise `ConfigurationError` (CLI exit code 2). The `sensitivity.independent_vars` field is cross-validated against the actual sensitivity CSV columns at the same entry, and each name must match the Snakemake-safe charset `^[A-Za-z0-9_.]+$`.
