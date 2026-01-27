# Phase 8: Extract Scenario Preparation Logic

**Date:** January 27, 2026 | **Status:** Ready for Implementation | **Goal:** Decompose `TRITONSWMM_scenario` into focused components for input generation and model building

**Previous Phase:** Phase 7a (Subprocess Logging Consolidation) completed - 22/22 tests passing ✅

---

## Objective

Extract scenario preparation logic from `scenario.py` (~700 lines, 25+ methods) into two focused modules: `scenario_inputs.py` for weather/boundary condition file generation and `swmm_model_builder.py` for SWMM model creation and modification.

**Expected Impact:**
- Reduces `scenario.py` by ~300 lines
- Clear separation of concerns: input generation vs. model building
- Easier to test each component independently
- Makes scenario preparation logic more maintainable

**Risk Level:** Medium - These methods have interdependencies with scenario state

---

## New Modules to Create

### 1. `scenario_inputs.py` - Weather/Boundary Condition File Generation

**Purpose:** Handle all external input file generation for scenarios

**Methods to Extract:**
- `_write_swmm_rainfall_dat_files()` - Generates rainfall input files from weather data
- `_write_swmm_waterlevel_dat_files()` - Generates water level input files for boundary conditions
- `_create_external_boundary_condition_files()` - Creates boundary condition files for TRITON
- `_write_hydrograph_files()` - Generates hydrograph input files

**Design Pattern:**
```python
class ScenarioInputGenerator:
    """
    Generates external input files for TRITON-SWMM scenarios.
    
    Handles weather data, boundary conditions, and hydrograph file generation.
    """
    
    def __init__(self, scenario: "TRITONSWMM_scenario"):
        self.scenario = scenario
        self.cfg_analysis = scenario.cfg_analysis
        self.system = scenario._system
        
    def write_swmm_rainfall_dat_files(self) -> None:
        """Generate rainfall input files from weather data."""
        ...
        
    def write_swmm_waterlevel_dat_files(self) -> None:
        """Generate water level input files for boundary conditions."""
        ...
```

### 2. `swmm_model_builder.py` - SWMM Model Generation

**Purpose:** Handle SWMM model creation, modification, and execution

**Methods to Extract:**
- `_create_swmm_model_from_template()` - Creates SWMM model from template file
- `_update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell()` - Updates SWMM model structure to match DEM grid
- `_run_swmm_hydro_model()` - Executes SWMM hydraulic model

**Design Pattern:**
```python
class SWMMModelBuilder:
    """
    Builds and modifies SWMM models for TRITON-SWMM scenarios.
    
    Handles model template processing, structural updates, and execution.
    """
    
    def __init__(self, scenario: "TRITONSWMM_scenario"):
        self.scenario = scenario
        self.cfg_analysis = scenario.cfg_analysis
        self.system = scenario._system
        
    def create_swmm_model_from_template(self) -> None:
        """Create SWMM model from template file."""
        ...
        
    def update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(self) -> None:
        """Update SWMM model structure to match DEM grid."""
        ...
```

---

## Implementation Steps

### Step 1: Analyze Dependencies
1. Read `scenario.py` to understand method dependencies
2. Identify which methods depend on scenario state vs. pure functions
3. Map out data flow between methods
4. Identify any circular dependencies

### Step 2: Create `scenario_inputs.py`
1. Create new file `src/TRITON_SWMM_toolkit/scenario_inputs.py`
2. Define `ScenarioInputGenerator` class
3. Extract and adapt the 4 input generation methods:
   - `_write_swmm_rainfall_dat_files()`
   - `_write_swmm_waterlevel_dat_files()`
   - `_create_external_boundary_condition_files()`
   - `_write_hydrograph_files()`
4. Update method signatures to accept necessary parameters
5. Add comprehensive docstrings

### Step 3: Create `swmm_model_builder.py`
1. Create new file `src/TRITON_SWMM_toolkit/swmm_model_builder.py`
2. Define `SWMMModelBuilder` class
3. Extract and adapt the 3 model building methods:
   - `_create_swmm_model_from_template()`
   - `_update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell()`
   - `_run_swmm_hydro_model()`
4. Update method signatures to accept necessary parameters
5. Add comprehensive docstrings

### Step 4: Update `scenario.py`
1. Import new classes: `ScenarioInputGenerator` and `SWMMModelBuilder`
2. Initialize in `__init__`:
   ```python
   self._input_generator = ScenarioInputGenerator(self)
   self._model_builder = SWMMModelBuilder(self)
   ```
3. Replace method calls with delegation:
   - `self._write_swmm_rainfall_dat_files()` → `self._input_generator.write_swmm_rainfall_dat_files()`
   - `self._create_swmm_model_from_template()` → `self._model_builder.create_swmm_model_from_template()`
   - etc.
4. Remove the extracted method definitions

### Step 5: Validate
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
- Maintain scenario state access through `self.scenario` reference
- Keep method behavior identical (pure refactoring)
- Add type hints and docstrings to new classes
- Test each component independently if possible

❌ **DON'T:**
- Change any logic or behavior
- Modify public API methods
- Touch log file structures
- Break scenario preparation workflow

---

## Expected Results

**Before:**
```python
class TRITONSWMM_scenario:
    def prepare_scenario(self):
        self._write_swmm_rainfall_dat_files()  # 50 lines
        self._write_swmm_waterlevel_dat_files()  # 40 lines
        self._create_swmm_model_from_template()  # 80 lines
        self._update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell()  # 60 lines
        self._run_swmm_hydro_model()  # 70 lines
        # Total: ~300 lines in scenario.py
```

**After:**
```python
# scenario.py (~400 lines, down from ~700)
class TRITONSWMM_scenario:
    def __init__(self, ...):
        self._input_generator = ScenarioInputGenerator(self)
        self._model_builder = SWMMModelBuilder(self)
    
    def prepare_scenario(self):
        self._input_generator.write_swmm_rainfall_dat_files()
        self._input_generator.write_swmm_waterlevel_dat_files()
        self._model_builder.create_swmm_model_from_template()
        self._model_builder.update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell()
        self._model_builder.run_swmm_hydro_model()

# scenario_inputs.py (~150 lines)
class ScenarioInputGenerator:
    # Input generation methods

# swmm_model_builder.py (~150 lines)
class SWMMModelBuilder:
    # Model building methods
```

---

## Validation Checklist

- [ ] Analyzed method dependencies in scenario.py
- [ ] Created scenario_inputs.py with ScenarioInputGenerator class
- [ ] Extracted 4 input generation methods
- [ ] Created swmm_model_builder.py with SWMMModelBuilder class
- [ ] Extracted 3 model building methods
- [ ] Updated scenario.py to use new components
- [ ] Removed extracted method definitions from scenario.py
- [ ] Added type hints and docstrings to new classes
- [ ] All 22 smoke tests passing (test_PC_01, test_PC_02, test_PC_04, test_PC_05)
- [ ] No changes to public API
- [ ] No changes to log file structures
- [ ] Scenario preparation workflow unchanged

---

## Notes

- This phase has medium risk due to method interdependencies
- Careful attention needed to maintain scenario state access
- Consider keeping methods as instance methods (not static) to maintain access to scenario state
- The new classes act as "strategy" objects that encapsulate related behavior
- This follows the same pattern as Phases 1-3 (ResourceManager, ExecutionStrategy, SnakemakeWorkflowBuilder)

---

**Last Updated:** January 27, 2026 - Phase 8 Implementation Guide
