# Reference

Information-oriented lookup material: the API surface and example interactive
report bundles.

- [API Reference](api.md) — the `hhemt` package API.
- [FAIR scope table](fair-scope-table.md) — item-by-item F/A/I/R posture over
  the whole reproducibility dataset.
- [Example interactive report](example-report.md) — embedded interactive HTML
  report bundle.

## Example experiments

The repository ships the anonymized UVA and Frontier benchmarking experiment
definitions under `test_data/norfolk_coastal_flooding/`: the production suites
`full_benchmarking_experiment_uva.xlsx` and
`full_benchmarking_experiment_frontier.xlsx` (with their paired
`report_config_*.yaml`), and a lighter `benchmarking_uva_minimal.xlsx` used by
the [Norfolk tutorial](../tutorials/norfolk-end-to-end.md). Substitute
`{your-allocation}` in the example HPC profiles
(`hpc_system_config_{uva,frontier}.yaml`) to reproduce them.
