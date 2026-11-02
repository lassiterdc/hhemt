Spatial differences between the clean and the resume experiments (resume minus clean) for each compute configuration whose outputs differ: peak flood depth (``max_wlevel_m``, per cell) and, for TRITON-SWMM experiments, peak conduit flow (``max_flow_cms``, per SWMM link). Each model is compared against its own clean counterpart. The verdict table states, for each model, how many pairs were compared and how many differ. Each panel is one class of compute configurations with byte-identical outputs, and the table beside it lists each configuration in the class with its resume count in the resume experiment. Blue means the resume experiment is higher. When every compared pair is identical, no difference panel is drawn and the verdict table states the result. SWMM series are compared over the reporting periods both experiments share.

**Sources:**

{{ snakemake.params.source_paths_rst }}
