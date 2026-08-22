Experiment provenance and reproduction metadata for **{{ snakemake.config.analysis_id }}**:

- **Provenance summary** — the RO-Crate / PROV record (dataset identity and license,
  toolkit git-SHA, container environment and its SIF digest, input file digests, the
  per-run process graph, and the consolidated outputs with their CF variable
  dictionary), read from the co-located ``ro-crate-metadata.json`` provenance sidecar
  written at consolidation. The verifiability anchors — code git-SHA, SIF sha256, and
  input digests — are called out at the top. The producer's hostname and wall-clock
  are deliberately excluded so this page is safe to ship inside a render bundle.
- **Data availability** — whether the post-processing reclaim was recorded and is
  consistent, projected from the ``Data availability`` check in the same
  ``validation_report.json`` the Errors-and-Warnings section reads. Reclaimed artifact
  classes were removed deliberately, after the toolkit verified the corresponding
  summary outputs were present and openable, so an absent timeseries or raw output here
  is a disclosed reclaim rather than a loss.
- **Reproduction guide** — every configuration field grouped by what a reproducer must
  do with it: *supply* (user-specific, never bundled), *amend* (HPC-specific, bundled
  but machine-dependent), or *keep* (experiment-defining). Values shown are schema
  descriptions and placeholders only, never the producing user's configuration.

**Sources:**

{{ snakemake.params.source_paths_rst }}
