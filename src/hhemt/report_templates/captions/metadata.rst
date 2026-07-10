Experiment provenance and reproduction metadata for **{{ snakemake.config.analysis_id }}**:

- **Provenance summary** — the RO-Crate / PROV record (dataset identity and license,
  toolkit git-SHA, container environment and its SIF digest, input file digests, the
  per-run process graph, and the consolidated outputs with their CF variable
  dictionary), read from the co-located ``ro-crate-metadata.json`` provenance sidecar
  written at consolidation. The verifiability anchors — code git-SHA, SIF sha256, and
  input digests — are called out at the top. The producer's hostname and wall-clock
  are deliberately excluded so this page is safe to ship inside a render bundle.
- **Reproduction guide** — every configuration field grouped by what a reproducer must
  do with it: *supply* (user-specific, never bundled), *amend* (HPC-specific, bundled
  but machine-dependent), or *keep* (experiment-defining). Values shown are schema
  descriptions and placeholders only, never the producing user's configuration.
- **SLURM efficiency** — the resource-utilization report for SLURM executions. It is
  finalized at workflow teardown, *after* this report renders, so it is expected to be
  empty on the run that produces this page and populates on any later re-render.

**Sources:**

{{ snakemake.params.source_paths_rst }}
