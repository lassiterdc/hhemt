**Purpose:** this table is the guard that catches two experiments that should NOT be compared side by side. Each row is a metadata field that DIFFERS between the combined experiments; the *Severity* column says whether that difference is merely INFORMATIONAL (expected — e.g. the same experiment run on different compute configs / UVA vs Frontier) or would have been BLOCKING. A blocking divergence aborts the combine before this report is ever produced, so if you are reading this the experiments are combine-compatible — an EMPTY table means every compared identity field agrees. Read it as: *what, if anything, differs about the two experiments' provenance, and does it matter?* Projected from the ``combined_compatibility.json`` read-model written at combine time.

**Sources:**

{{ snakemake.params.source_paths_rst }}
