**Purpose:** this table records WHAT WENT INTO this report. One row per combined experiment, giving its role (clean / resume), its model, how many sub-analyses it contributed, and the exact toolkit and solver builds that produced it. Read it as: *which experiments were combined here, and were they built from the same code?* Compatibility itself is enforced upstream rather than shown here: a BLOCKING divergence aborts the combine before this report is ever produced, so if you are reading this the experiments are combine-compatible. Projected from the ``combined_compatibility.json`` read-model written at combine time.

**Sources:**

{{ snakemake.params.source_paths_rst }}
