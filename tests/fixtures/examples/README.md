# Example fixtures

Static example files referenced by documentation and external users. Not consumed
by the test suite directly — see `tests/fixtures/bundles/` and
`tests/fixtures/synthetic_model/` for runtime test inputs.

## `sensitivity_per_sa_system_configs_example.csv`

Demonstrates the optional `system_config_yaml` column in sensitivity-analysis
CSVs (added in the `per_sub_analysis_system_configs` feature). Two distinct
system YAMLs vary `target_dem_resolution` (10 m vs. 20 m); each is paired with
two analysis-level configurations (`run_mode` × `n_omp_threads`), producing four
sub-analyses on two compile targets (rows 0+2 collapse onto the 10 m target;
rows 1+3 onto the 20 m target).

Replace the placeholder paths with real system-config YAMLs before use. The
preflight validator enforces:

1. Each referenced YAML exists on disk and passes Pydantic validation
   (`load_system_config()`).
2. Every sub-analysis YAML's model-type toggles match the master system config.
3. YAMLs whose `(target_dem_resolution, gpu_hardware, gpu_compilation_backend)`
   tuples collide must agree on every other `cfg_system` field — divergence on
   any non-key field raises `ConfigurationError`.

See `library/prompts/workspaces/projects/hhemt/TRITON SWMM toolkit architecture.md`
§ "Per-Sub-Analysis System Configs" in the agentic-workspace for the full design.
