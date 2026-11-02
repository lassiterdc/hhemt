Experiment provenance and reproduction metadata for **{{ snakemake.config.analysis_id }}**:

- **Provenance**: the provenance record of the analysis: dataset identity and license,
  toolkit git-SHA, container environment and its SIF digest, input file digests, the model
  runs of an analysis that is not a sweep, and the consolidated outputs with their CF
  variable dictionary. The verifiability anchors at the top give the git-SHA of the code
  that consolidated the data, the SIF sha256 and the input digests. The provenance chain
  names the builds that simulated, processed, consolidated and rendered the data. The
  producer's hostname and wall-clock are excluded. A part the analysis did not record is
  stated as not recorded.
- **Data Availability**: whether the post-processing reclaim was recorded and is consistent.
  An absent timeseries or raw output is a recorded reclaim, not a loss.
- **Reproduction Guide**: every configuration field grouped by what a reproducer must do
  with it: *supply* (user-specific, never bundled), *amend* (HPC-specific, bundled but
  machine-dependent), or *keep* (experiment-defining). Values shown are schema descriptions
  and placeholders only, never the producing user's configuration.

**Sources:**

{{ snakemake.params.source_paths_rst }}
