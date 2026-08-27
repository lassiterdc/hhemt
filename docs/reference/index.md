# Reference

Information-oriented lookup material: the API surface, the configuration
schema, and what the toolkit produces.

- [API Reference](api.md) — the `hhemt` package API.
- [Configuration schema](config-schema.md) — annotated system and analysis
  configs, and the toggle-dependency table.
- [CLI reference](cli.md) — every `hhemt` command and the structured exit codes.
- [Output data model](output-data-model.md) — what a completed analysis writes,
  the three output tiers, and the CF-1.13 variables.
- [FAIR scope table](fair-scope-table.md) — item-by-item F/A/I/R posture over
  the whole reproducibility dataset.
- [The interactive analysis report](example-report.md) — what a rendered report
  contains and how to open one.

## Example experiments

The repository ships the anonymized UVA and Frontier benchmarking experiment
definitions under `test_data/norfolk_coastal_flooding/`: the production suites
`full_benchmarking_experiment_uva.xlsx` and
`full_benchmarking_experiment_frontier.xlsx` (with their paired
`report_config_*.yaml`), and a lighter `benchmarking_uva_minimal.xlsx` used by
the [Norfolk tutorial](../tutorials/norfolk-end-to-end.md). Substitute
`{your-allocation}` in the example HPC profiles
(`hpc_system_config_{uva,frontier}.yaml`) to run them on your own allocation.
Note that the definitions describe the suites; reproducing the published results
also requires the input datasets, which are fetched separately by DOI (see
[Publishing and fetching](../how-to/publishing.md)).
