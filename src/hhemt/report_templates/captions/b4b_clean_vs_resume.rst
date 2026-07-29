Raw byte-for-byte identity: clean vs hotstart-resume.

Per-timestep binary identity of the raw TRITON output rasters comparing each hotstart-resumed sub-analysis against its clean config-counterpart, with vertical lines marking the requested resume-interruption boundaries (r1..rK). Green cells are byte-identical between the clean run and the SIGKILL-and-resumed run of the same compute config at that reporting timestep; crimson cells differ. This is the strongest raw resume-validity instrument: a difference localized to the resume boundaries would falsify the hotstart-replay determinism assumption. The comparison requires a single master carrying BOTH clean and resume subs; under the two-master campaign (clean and resume run as sibling masters) it is DEGRADED-BY-CONSTRUCTION and renders an honest-degradation panel — the clean-vs-resume result is delivered instead at the ``hhemt combine`` cross-experiment surface (``cross_experiment_intercomparison`` + spatial diff maps). Reads ``eda/b4b_clean_vs_resume.zarr``.

**Sources:**

{{ snakemake.params.source_paths_rst }}
