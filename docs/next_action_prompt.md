# Phase 6: Extract SWMM Output Parsing

**Date:** January 27, 2026 | **Status:** Ready for Implementation | **Goal:** Extract SWMM parsing logic into dedicated module

---

## Objective

Extract SWMM output parsing functions from `process_simulation.py` (~1100 lines) into a new dedicated module `swmm_output_parser.py`. This creates clear separation between simulation processing orchestration and SWMM output parsing logic.

**Expected Impact:**
- Reduces `process_simulation.py` from ~1100 to ~600 lines
- Makes SWMM parsing reusable across different contexts
- Easier to test SWMM parsing logic in isolation
- Clear separation of concerns

**Risk Level:** Low - These are pure functions with no class dependencies

---

## Functions to Extract

Create new file: `src/TRITON_SWMM_toolkit/swmm_output_parser.py`

### Primary SWMM Parsing Functions (18 total)

**Main Entry Points:**
1. `retrieve_SWMM_outputs_as_datasets()` - Main entry point for SWMM output retrieval
2. `return_swmm_outputs()` - Returns SWMM outputs from .out or .rpt file
3. `return_swmm_system_outputs()` - Returns system-level SWMM outputs (continuity errors, flooding)

**RPT File Parsing:**
4. `return_lines_for_section_of_rpt()` - Extracts lines from RPT file sections
5. `return_data_from_rpt()` - Generic RPT data extraction with error handling
6. `format_rpt_section_into_dataframe()` - Formats RPT sections as DataFrames
7. `return_analysis_end_date()` - Parses analysis end date from RPT

**Timeseries Parsing:**
8. `return_node_time_series_results_from_rpt()` - Parses node timeseries from RPT
9. `return_node_time_series_results_from_outfile()` - Parses node timeseries from .out file
10. `create_tseries_ds()` - Creates timeseries xarray Dataset from parsed data
11. `convert_pyswmm_output_to_df()` - Converts pyswmm output to DataFrame

**Data Type Conversion:**
12. `convert_swmm_tdeltas_to_minutes()` - Converts SWMM time deltas to minutes
13. `convert_coords_to_dtype()` - Coerces xarray coordinates to proper dtypes
14. `convert_datavars_to_dtype()` - Coerces xarray data variables to proper dtypes

**String Parsing Helpers:**
15. `extract_substring()` - Extracts substring using index tuple
16. `find_substring_indices()` - Finds all occurrences of substring
17. `split_at_index()` - Splits string at given index
18. `isel_first_and_slice_longest()` - xarray selection helper for sampling

---

## Functions to Keep in process_simulation.py

**Keep these in `process_simulation.py`:**
- `TRITONSWMM_sim_post_processing` class (entire class stays)
- `parse_performance_file()` - TRITON performance parsing
- `return_filelist_by_tstep()` - TRITON file listing
- `return_fpath_wlevels()` - TRITON file path retrieval
- `load_triton_output_w_xarray()` - TRITON output loading
- `summarize_swmm_simulation_results()` - Summarization function
- `summarize_triton_simulation_results()` - Summarization function

---

## Required Imports for swmm_output_parser.py

```python
import sys
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr
import swmmio

from pyswmm import Output, NodeSeries, LinkSeries

from TRITON_SWMM_toolkit.constants import (
    LST_COL_HEADERS_NODE_FLOOD_SUMMARY,
    LST_COL_HEADERS_NODE_FLOW_SUMMARY,
    LST_COL_HEADERS_LINK_FLOW_SUMMARY,
)
```

---

## Implementation Steps

### Step 1: Create swmm_output_parser.py
1. Create new file: `src/TRITON_SWMM_toolkit/swmm_output_parser.py`
2. Add required imports (see above)
3. Copy all 18 functions listed above from `process_simulation.py`
4. Verify no class dependencies exist (all functions should be standalone)

### Step 2: Update process_simulation.py
1. Add import at top: `from TRITON_SWMM_toolkit.swmm_output_parser import retrieve_SWMM_outputs_as_datasets`
2. Remove all 18 functions that were moved
3. Verify `_export_SWMM_outputs()` method still works (it calls `retrieve_SWMM_outputs_as_datasets()`)

### Step 3: Validate
Run all smoke tests:
```bash
conda activate triton_swmm_toolkit
cd /home/***REMOVED***/dev/TRITON-SWMM_toolkit
python -m pytest tests/test_PC_01_singlesim.py tests/test_PC_02_multisim.py tests/test_PC_04_multisim_with_snakemake.py tests/test_PC_05_sensitivity_analysis_with_snakemake.py -v
```

**Success Criteria:** All 22 tests passing

---

## Key Constraints

✅ **DO:**
- Move all 18 functions as standalone functions
- Keep function signatures identical
- Update imports in `process_simulation.py`
- Verify no circular import issues

❌ **DON'T:**
- Modify function logic or signatures
- Move TRITON-specific functions
- Move summarization functions
- Touch the `TRITONSWMM_sim_post_processing` class

---

## Expected File Sizes After Refactoring

- `swmm_output_parser.py`: ~500 lines (new file)
- `process_simulation.py`: ~600 lines (reduced from ~1100)

---

## Validation Checklist

- [ ] Created `swmm_output_parser.py` with all 18 functions
- [ ] Updated imports in `process_simulation.py`
- [ ] Removed moved functions from `process_simulation.py`
- [ ] No circular import issues
- [ ] All 22 smoke tests passing (test_PC_01, test_PC_02, test_PC_04, test_PC_05)
- [ ] `_export_SWMM_outputs()` method still works correctly
- [ ] No changes to public API or log file structures

---

## Notes

- This is a pure code movement refactoring - no logic changes
- All functions are standalone with no class dependencies
- The `TRITONSWMM_sim_post_processing` class will call `retrieve_SWMM_outputs_as_datasets()` from the new module
- TRITON-specific parsing remains in `process_simulation.py` for now (may be extracted in future phase)

---

**Last Updated:** January 27, 2026 - Phase 6 Implementation Guide
