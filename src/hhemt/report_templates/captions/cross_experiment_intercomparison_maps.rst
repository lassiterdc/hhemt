Clean-vs-resume spatial difference maps (per-DEM-cell peak depth + per-conduit peak flow), resume minus clean, for each compute-config that DIFFERS across the combined experiments. RdBu diverging: blue = resume HIGHER than clean. Absolute magnitudes inherit the variable-dt SWMM final-period truncation (Nperiods=N-1). The truncation is one-sided rather than common-mode: the clean sweep drops the final period on all 28 sub-analyses and the resume sweep recovers it on 14, selected deterministically by the hotstart restart time. The difference shown here is therefore taken over the shared leading periods, which are timestamp-identical across both arms; TRITON-side fields are full-length on both arms and unaffected.

**Sources:**

{{ snakemake.params.source_paths_rst }}
