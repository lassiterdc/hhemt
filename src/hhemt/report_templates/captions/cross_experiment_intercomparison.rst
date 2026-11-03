Clean-versus-resume comparison of the combined experiments. Each row pairs one key-result variable (``max_wlevel_m``, peak flood depth, or ``max_flow_cms``, peak conduit flow) for one event and one compute configuration present in both the clean and the resume experiment. **clean_vs_resume** reports whether they are byte-for-byte identical or differ, and **max_abs_diff** is the largest absolute difference. The compute configuration is split into columns for run mode, MPI ranks, OpenMP threads, GPUs, nodes and partition. SWMM series are compared over the reporting periods both experiments share. When no result of a clean experiment can be compared with a result of a resume experiment, the page says so.

**Sources:**

{{ snakemake.params.source_paths_rst }}
