The table has one row per combined experiment: the experiment's role (clean or resume), its models (TRITON, TRITON-SWMM and SWMM, joined by "+" when its members ran more than one, or "model not recorded" when its consolidated tree is missing or holds no model), its number of members, and the builds that produced the experiment. The builds are the hhemt build that wrote the bundle, the hhemt build that produced the data and the solver build. A note under the table says that the hhemt build and the solver build are the same in every row, or that they differ. The combine stops before this page exists when the experiments are incompatible, so every experiment listed was accepted. When no experiment is recorded, the page says so.

**Sources:**

{{ snakemake.params.source_paths_rst }}
