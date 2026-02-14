# Snakemake Reporting Integration Plan (REVISED)

## Executive Summary

This plan integrates Snakemake's built-in reporting capabilities into the TRITON-SWMM toolkit to provide:
1. **Interactive HTML/ZIP reports** with execution statistics, provenance, and workflow topology
2. **SLURM efficiency reports** for HPC resource utilization analysis
3. **Annotated outputs** with captions and categories for stakeholder communication
4. **Automated report generation** after workflow completion

**Revision History:**
- **2025-02-13**: Initial revision based on codebase review and Snakemake source code analysis
  - Fixed report directive placement (copy template to analysis_dir)
  - Fixed consolidation rule output handling (keep flag file separate)
  - Changed caption files to dynamic generation (no static templates)
  - Removed AnalysisConfig changes (use parameters only)
  - Verified rulegraph is automatically included in reports

## Key Capabilities from Snakemake

### 1. HTML/ZIP Reports (`--report`)
- **Self-contained reports** with runtime statistics, provenance info, and **workflow rulegraph (always included)**
- **HTML mode** (< 20 MB flagged content): Single-file, base64-embedded results
- **ZIP mode** (> 20 MB or directories): Archive with separate result files
- **Requires completed runs**: Metadata generated during execution (`.snakemake/metadata/`)

### 2. Report Annotations (`report()`)
- Annotate rule outputs with captions, categories, subcategories, and labels
- Support for images (SVG, PNG), HTML, PDFs, CSVs, and any file type
- In-browser viewers for images/HTML/PDF; downloads for data files

### 3. SLURM Efficiency Reports (`--slurm-efficiency-report`)
- Per-job efficiency data (CPU/memory/GPU utilization)
- Critical for validating resource requests on HPC clusters
- Logfile: `efficiency_report_<workflow_id>.log`

### 4. Global Workflow Description (`report: "path/to/workflow.rst"`)
- Top-level narrative describing the analysis
- Supports Jinja2 templating with access to workflow config

## Recommended Integration Strategy

### Phase 1: Basic Report Infrastructure (Immediate)
1. Add `--report-after-run` to workflow submission methods
2. Generate SLURM efficiency reports for HPC runs
3. Create and copy workflow description template dynamically

### Phase 2: Output Annotations (Short-term)
1. Annotate consolidation outputs (zarr summaries, CSVs)
2. Generate dynamic caption files per analysis
3. Add categories for different result types

### Phase 3: Enhanced Reporting (Medium-term)
1. Custom stylesheets for branding (Snakemake already supports `--report-stylesheet`)
2. Generate diagnostic plots and include in reports
3. Per-scenario summary visualizations

## Implementation Details

### 1. Workflow Description Template (Source Template)

Create source template in toolkit at `workflow/report/workflow_template.rst`:

```rst
TRITON-SWMM Analysis Report
===========================

Analysis Configuration
----------------------

- **Analysis Name**: {{ snakemake.config.get('analysis_name', 'N/A') }}
- **System**: {{ snakemake.config.get('system_name', 'N/A') }}
- **Number of Simulations**: {{ snakemake.config.get('n_sims', 'N/A') }}
- **Execution Mode**: {{ snakemake.config.get('multi_sim_run_method', 'local') }}

{% if snakemake.config.get('multi_sim_run_method') in ['batch_job', '1_job_many_srun_tasks'] %}
HPC Configuration
-----------------

- **Partition**: {{ snakemake.config.get('hpc_ensemble_partition', 'N/A') }}
- **CPUs per Simulation**: {{ snakemake.config.get('cpus_per_sim', 1) }}
- **GPUs per Simulation**: {{ snakemake.config.get('n_gpus', 0) }}
- **Memory per CPU**: {{ snakemake.config.get('mem_gb_per_cpu', 2) }} GB
{% endif %}

Model Types Enabled
-------------------

{% if snakemake.config.get('toggle_triton_model', False) %}
- **TRITON-only**: 2D hydrodynamic modeling
{% endif %}
{% if snakemake.config.get('toggle_tritonswmm_model', False) %}
- **TRITON-SWMM**: Coupled 2D surface + 1D drainage
{% endif %}
{% if snakemake.config.get('toggle_swmm_model', False) %}
- **SWMM-only**: Standalone stormwater network analysis
{% endif %}

Simulation Parameters
---------------------

- **Run Mode**: {{ snakemake.config.get('run_mode', 'N/A') }}
- **MPI Ranks**: {{ snakemake.config.get('n_mpi_procs', 1) }}
- **OMP Threads**: {{ snakemake.config.get('n_omp_threads', 1) }}

Results
-------

This report contains:

- Consolidated timeseries summaries (Results_)
- Per-scenario execution logs (Rules_)
- Workflow rulegraph showing rule dependencies
- Workflow provenance and execution timing
- Resource utilization statistics (if HPC run)

Navigate using the sidebar to explore results by category.
```

### 2. Modified `generate_snakefile_content()` Method

Add report directive and annotate consolidation outputs:

```python
def generate_snakefile_content(
    self,
    # ... existing parameters ...
    enable_reporting: bool = True,  # NEW PARAMETER
) -> str:
    """
    Generate Snakefile content with optional report annotations.

    Parameters
    ----------
    enable_reporting : bool
        If True, add report() annotations to outputs and include global report directive
    """
    # ... existing setup code ...

    # Add global report directive at top of Snakefile
    report_directive = ""
    if enable_reporting:
        # Copy workflow description template to analysis directory
        import shutil
        triton_toolkit_root = Path(__file__).parent.parent.parent
        workflow_template_source = triton_toolkit_root / "workflow" / "report" / "workflow_template.rst"
        workflow_template_dest = self.analysis_paths.analysis_dir / "workflow_description.rst"

        # Copy template if it exists
        if workflow_template_source.exists():
            shutil.copy(workflow_template_source, workflow_template_dest)
            report_directive = f'report: "workflow_description.rst"\n\n'
        else:
            # Fallback: generate inline if template not found
            logger.warning(f"Workflow template not found at {workflow_template_source}, skipping report directive")

    snakefile_content = f'''{report_directive}# Auto-generated by TRITONSWMM_analysis
# Analysis: {self.analysis.cfg_analysis.analysis_name}
# System: {self.system.cfg_system.system_name}

import os
import glob
import subprocess

# Read simulation IDs from config
SIM_IDS = {list(range(n_sims))}

# Snakemake config for workflow description templating
config = {{
    "analysis_name": "{self.analysis.cfg_analysis.analysis_name}",
    "system_name": "{self.system.cfg_system.system_name}",
    "n_sims": {n_sims},
    "multi_sim_run_method": "{self.cfg_analysis.multi_sim_run_method}",
    "hpc_ensemble_partition": "{self.cfg_analysis.hpc_ensemble_partition or 'N/A'}",
    "cpus_per_sim": {cpus_per_sim},
    "n_gpus": {n_gpus},
    "mem_gb_per_cpu": {self.cfg_analysis.mem_gb_per_cpu},
    "run_mode": "{self.cfg_analysis.run_mode}",
    "n_mpi_procs": {mpi_ranks},
    "n_omp_threads": {omp_threads},
    "toggle_triton_model": {self.system.cfg_system.toggle_triton_model},
    "toggle_tritonswmm_model": {self.system.cfg_system.toggle_tritonswmm_model},
    "toggle_swmm_model": {self.system.cfg_system.toggle_swmm_model},
}}

rule all:
    input: "_status/output_consolidation_complete.flag"

# ... existing rules ...
'''

    # ... rest of Snakefile generation ...

    return snakefile_content
```

### 3. Helper Method: Dynamic Caption Generation

Add to `SnakemakeWorkflowBuilder`:

```python
def _generate_caption_file(
    self,
    model_type: str,
    output_path: Path | str,
) -> Path:
    """
    Generate RST caption file for a specific model type's consolidated outputs.

    This creates analysis-specific captions dynamically, ensuring self-contained
    analysis directories and enabling AI-friendly single-source documentation.

    Parameters
    ----------
    model_type : str
        Model type: "triton", "tritonswmm", or "swmm"
    output_path : Path | str
        Path to the output file being captioned

    Returns
    -------
    Path
        Path to the generated caption file
    """
    caption_dir = self.analysis_paths.analysis_dir / "report_captions"
    caption_dir.mkdir(exist_ok=True, parents=True)

    caption_path = caption_dir / f"{model_type}_caption.rst"

    # Model-specific content
    model_name_map = {
        "triton": "TRITON-only",
        "tritonswmm": "TRITON-SWMM Coupled",
        "swmm": "SWMM-only",
    }

    model_desc_map = {
        "triton": "2D hydrodynamic surface flow modeling",
        "tritonswmm": "Coupled 2D surface + 1D stormwater network modeling",
        "swmm": "Standalone 1D stormwater network analysis",
    }

    model_name = model_name_map.get(model_type, model_type.upper())
    model_desc = model_desc_map.get(model_type, "Model simulation results")

    # Generate caption content with analysis-specific details
    caption_content = f"""{model_name} Consolidated Results
{'=' * (len(model_name) + 23)}

Analysis: {self.analysis.cfg_analysis.analysis_name}
System: {self.system.cfg_system.system_name}
Number of Scenarios: {len(self.analysis.df_sims)}
Model Type: {model_desc}

This zarr store contains consolidated timeseries summaries for all {model_name} simulations.

Contents
--------

"""

    # Add model-specific content sections
    if model_type in ["triton", "tritonswmm"]:
        caption_content += """- Peak flow depths at all mesh nodes
- Maximum flow velocities
- Inundation duration statistics
- Water surface elevation timeseries

"""

    if model_type in ["swmm", "tritonswmm"]:
        caption_content += """- Link flows and velocities
- Node depths and flooding
- Conduit surcharge statistics
- System flow balance

"""

    caption_content += f"""Data Access
-----------

The data is organized by scenario (event_iloc) and can be accessed using xarray:

.. code-block:: python

    import xarray as xr
    ds = xr.open_zarr("{output_path}")
    print(ds)

    # Example: Extract peak depths for scenario 0
    peak_depths = ds["peak_depth"].sel(event_iloc=0)

Provenance
----------

- **Generated**: During workflow consolidation phase
- **Format**: Zarr (chunked, compressed multi-dimensional arrays)
- **Compression**: Level {self.cfg_analysis.compression_level if hasattr(self.cfg_analysis, 'compression_level') else 5}
- **Analysis Directory**: {self.analysis_paths.analysis_dir}
"""

    caption_path.write_text(caption_content)
    return caption_path
```

### 4. Helper Method for Report-Annotated Outputs

```python
def _annotate_output_for_report(
    self,
    output_path: str | Path,
    category: str = "Results",
    subcategory: str | None = None,
    caption_file: str | Path | None = None,
) -> str:
    """
    Generate report() annotation string for a Snakemake output.

    Parameters
    ----------
    output_path : str | Path
        Path to the output file
    category : str
        Report category (e.g., "Results", "Diagnostics", "Logs")
    subcategory : str | None
        Optional subcategory for organization
    caption_file : str | Path | None
        Optional RST file with caption text

    Returns
    -------
    str
        Formatted report() annotation string for Snakemake rule
    """
    path_str = str(output_path)

    # Build report annotation
    parts = [f'"{path_str}"']
    parts.append(f'category="{category}"')

    if subcategory:
        parts.append(f'subcategory="{subcategory}"')

    if caption_file:
        parts.append(f'caption="{caption_file}"')

    # Generate label from filename
    filename = Path(output_path).stem
    parts.append(f'labels={{"output": "{filename}"}}')

    return f"report({', '.join(parts)})"
```

### 5. Modified Consolidation Rule with Reporting

Update the consolidation rule generation (in `generate_snakefile_content()`):

```python
# Build list of consolidation rule outputs
# ALWAYS include the flag file for workflow DAG integrity
consolidate_outputs = ['"_status/output_consolidation_complete.flag"']

if enable_reporting:
    # Generate caption files and annotate outputs for each enabled model
    for model_type in enabled_models:
        # Determine output path
        output_path = f"consolidated_outputs/{model_type}/summaries.zarr"

        # Generate dynamic caption file
        caption_path = self._generate_caption_file(model_type, output_path)

        # Create report annotation
        model_display_name = {
            "triton": "TRITON-only",
            "tritonswmm": "TRITON-SWMM",
            "swmm": "SWMM-only",
        }.get(model_type, model_type.upper())

        annotation = self._annotate_output_for_report(
            output_path,
            category="Consolidated Results",
            subcategory=model_display_name,
            caption_file=str(caption_path.relative_to(self.analysis_paths.analysis_dir)),
        )
        consolidate_outputs.append(annotation)

consolidate_output_str = ", ".join(consolidate_outputs)

snakefile_content += f'''
rule consolidate:
    input: {consolidate_input_str}
    output: {consolidate_output_str}
    log: "logs/consolidate.log"
    conda: "{conda_env_path}"
    resources:
{consolidate_resources}
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.consolidate_workflow \\
            {config_args} \\
            --compression-level {compression_level} \\
            {"--overwrite-outputs-if-already-created " if overwrite_outputs_if_already_created else ""}\\
            --which {which} \\
            > {{log}} 2>&1
        touch _status/output_consolidation_complete.flag
        """
'''
```

### 6. Modified `submit_workflow()` Method

Add report generation to workflow submission:

```python
def submit_workflow(
    self,
    # ... existing parameters ...
    generate_report: bool = True,  # NEW PARAMETER
    report_path: Path | None = None,  # NEW PARAMETER
) -> dict:
    """
    Submit workflow using Snakemake with optional report generation.

    Parameters
    ----------
    generate_report : bool
        If True, generate an HTML/ZIP report after workflow completion
    report_path : Path | None
        Custom path for report file. If None, auto-determines format:
        - analysis_dir/report.html (< 20 MB flagged outputs, no directories)
        - analysis_dir/report.zip (> 20 MB or contains directories)
    """
    # Determine report path if reporting enabled
    if generate_report:
        if report_path is None:
            # Auto-select format based on expected output size
            _, report_path = self._determine_report_format()

        # Warn if expected report size is very large
        self._check_report_size_warning(report_path)

        # Store report config for submission methods to access
        self._report_config = {
            "enabled": True,
            "path": report_path,
        }
    else:
        self._report_config = {"enabled": False}

    # ... rest of existing submission logic ...
```

### 7. Helper Method: Report Path Selection with Size Warning

```python
def _determine_report_format(self) -> tuple[str, Path]:
    """
    Determine optimal report format (HTML vs ZIP) based on expected output size.

    Returns
    -------
    tuple[str, Path]
        Format name ("html" or "zip") and recommended path

    Notes
    -----
    - HTML mode: < 20 MB flagged content, single-file output
    - ZIP mode: > 20 MB or directories, archive with separate files
    - Zarr directories always trigger ZIP mode
    """
    # Check for zarr directories (always use ZIP)
    has_directories = False
    total_size_mb = 0

    # Determine which model types will produce outputs
    enabled_models = []
    if self.system.cfg_system.toggle_triton_model:
        enabled_models.append("triton")
    if self.system.cfg_system.toggle_tritonswmm_model:
        enabled_models.append("tritonswmm")
    if self.system.cfg_system.toggle_swmm_model:
        enabled_models.append("swmm")

    # Check if zarr outputs exist and estimate size
    for model_type in enabled_models:
        zarr_path = self.analysis_paths.analysis_dir / "consolidated_outputs" / model_type / "summaries.zarr"
        if zarr_path.exists() and zarr_path.is_dir():
            has_directories = True
            # Estimate zarr directory size
            total_size_mb += sum(
                f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()
            ) / (1024 * 1024)

    # Decision logic
    if has_directories or total_size_mb > 20:
        format_name = "zip"
        report_path = self.analysis_paths.analysis_dir / "report.zip"
    else:
        format_name = "html"
        report_path = self.analysis_paths.analysis_dir / "report.html"

    return format_name, report_path

def _check_report_size_warning(self, report_path: Path) -> None:
    """
    Warn if expected report size exceeds recommended thresholds.

    Parameters
    ----------
    report_path : Path
        Intended report output path
    """
    # Calculate expected size from consolidated outputs
    total_size_mb = 0
    enabled_models = []
    if self.system.cfg_system.toggle_triton_model:
        enabled_models.append("triton")
    if self.system.cfg_system.toggle_tritonswmm_model:
        enabled_models.append("tritonswmm")
    if self.system.cfg_system.toggle_swmm_model:
        enabled_models.append("swmm")

    for model_type in enabled_models:
        zarr_path = self.analysis_paths.analysis_dir / "consolidated_outputs" / model_type / "summaries.zarr"
        if zarr_path.exists() and zarr_path.is_dir():
            total_size_mb += sum(
                f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()
            ) / (1024 * 1024)

    # Warn if exceeding recommended size (500 MB)
    if total_size_mb > 500:
        logger.warning(
            f"Expected report size ({total_size_mb:.1f} MB) exceeds recommended "
            f"threshold (500 MB). Consider excluding large zarr outputs from report."
        )
```

### 8. Modified `run_snakemake_local()` Method

```python
def run_snakemake_local(
    self,
    snakefile_path: Path,
    verbose: bool = True,
    dry_run: bool = False,
) -> dict:
    """Run Snakemake workflow on local machine with optional reporting."""
    # ... existing setup ...

    cmd_args = self._get_snakemake_base_cmd() + [
        "--profile",
        str(config_dir),
        "--snakefile",
        str(snakefile_path),
    ]

    # Add report generation if enabled (skip for dry runs)
    if not dry_run and hasattr(self, "_report_config") and self._report_config["enabled"]:
        report_path = self._report_config["path"]
        cmd_args.extend([
            "--report",
            str(report_path),
            "--report-after-run",
        ])
        if verbose:
            print(f"[Snakemake] Report will be generated: {report_path}", flush=True)

    # Add cores for multicore local runs
    # ... existing logic ...

    # Add dry-run flag last
    if dry_run:
        cmd_args.append("--dry-run")

    # ... existing execution and result handling ...

    # Validate report was created and add to result dict
    if hasattr(self, "_report_config") and self._report_config["enabled"]:
        report_path = self._report_config["path"]
        if report_path.exists():
            result_dict["report_path"] = report_path
            result_dict["report_created"] = True
        else:
            result_dict["report_path"] = None
            result_dict["report_created"] = False
            result_dict["report_warning"] = "Report generation failed or was skipped"

    return result_dict
```

### 9. Modified SLURM Batch Script Generation

For `batch_job` mode, add efficiency reporting:

```python
def _submit_batch_job_workflow(
    self,
    snakefile_path: Path,
    wait_for_completion: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Submit Snakemake workflow with SLURM efficiency reporting.
    """
    # ... existing setup ...

    # Build snakemake command with efficiency report
    snakemake_cmd_parts = [
        "${CONDA_PREFIX}/bin/python -m snakemake",
        f"--profile {config_dir}",
        f"--snakefile {snakefile_path}",
        "--executor slurm",
        "--printshellcmds",
        "--slurm-efficiency-report",  # Enable SLURM job efficiency tracking
    ]

    # Add report generation if enabled
    if hasattr(self, "_report_config") and self._report_config["enabled"]:
        report_path = self._report_config["path"]
        snakemake_cmd_parts.extend([
            f"--report {report_path}",
            "--report-after-run",
        ])

    snakemake_cmd = " \\\n    ".join(snakemake_cmd_parts)

    script_content = f"""#!/bin/bash
#SBATCH --job-name=triton_snakemake_orchestrator
# ... existing SBATCH directives ...

{module_load_cmd}

{conda_init_cmd}

# Run Snakemake with reporting
{snakemake_cmd}

# Post-execution: move efficiency report to analysis logs directory
if ls efficiency_report_*.log 1> /dev/null 2>&1; then
    mv efficiency_report_*.log {str(self.analysis_paths.analysis_dir / "logs")}/
    echo "[Reporting] SLURM efficiency report saved to logs/"
fi
"""

    # ... rest of script generation and submission ...
```

### 10. Single-Job Mode Batch Script

For `1_job_many_srun_tasks` mode:

```python
def _generate_single_job_submission_script(
    self, snakefile_path: Path, config_dir: Path
) -> Path:
    """
    Generate SLURM batch script with report generation.
    """
    # ... existing setup and resource calculations ...

    # Build snakemake command
    snakemake_cmd_parts = [
        "${CONDA_PREFIX}/bin/python -m snakemake",
        f"--profile {config_dir}",
        f"--snakefile {snakefile_path}",
        "--cores $TOTAL_CPUS",
    ]

    if gpu_cli_arg:
        snakemake_cmd_parts.append(f"--resources gpu=$TOTAL_GPUS")

    # Add report generation
    if hasattr(self, "_report_config") and self._report_config["enabled"]:
        report_path = self._report_config["path"]
        snakemake_cmd_parts.extend([
            f"--report {report_path}",
            "--report-after-run",
        ])

    snakemake_cmd = " \\\n    ".join(snakemake_cmd_parts)

    script_content = f"""#!/bin/bash
#SBATCH --job-name=triton_workflow
# ... existing SBATCH directives ...

{module_load_cmd}

{conda_init_cmd}

# Calculate resources
# ... existing resource calculation ...

# Run Snakemake with reporting
{snakemake_cmd}
"""

    # ... rest of script generation ...
```

## Usage Examples

### Example 1: Local Run with Report

```python
# In test or analysis script
analysis = system.analysis

# Submit workflow with automatic report generation
result = analysis.workflow_builder.submit_workflow(
    mode="local",
    generate_report=True,  # Enable reporting
    process_timeseries=True,
    verbose=True,
)

# After completion, report is at: analysis_dir/report.html or report.zip
if result["success"] and result.get("report_created"):
    report_path = result.get("report_path")
    print(f"Report available at: {report_path}")
```

### Example 2: HPC Batch Job with Efficiency Report

```python
# In HPC submission script
analysis.cfg_analysis.multi_sim_run_method = "batch_job"

result = analysis.workflow_builder.submit_workflow(
    mode="slurm",
    generate_report=True,
    wait_for_completion=False,  # Submit and return
    verbose=True,
)

# After job completes, check:
# - analysis_dir/report.html or report.zip (workflow report)
# - analysis_dir/logs/efficiency_report_*.log (SLURM efficiency data)
```

### Example 3: Disable Reporting for Quick Runs

```python
# For development/testing, skip report generation
result = analysis.workflow_builder.submit_workflow(
    mode="local",
    generate_report=False,  # No report overhead
    dry_run=True,
    verbose=True,
)
```

### Example 4: Custom Report Path

```python
# Save report to specific location
custom_report_path = Path("/shared/reports/") / f"{analysis.cfg_analysis.analysis_name}_report.zip"

result = analysis.workflow_builder.submit_workflow(
    mode="slurm",
    generate_report=True,
    report_path=custom_report_path,
    verbose=True,
)
```

## Testing Strategy

### Unit Tests (New File: `tests/test_reporting.py`)

```python
import pytest
from pathlib import Path
from TRITON_SWMM_toolkit import examples as tst

class TestReporting:
    """Test Snakemake reporting integration."""

    def test_report_directive_in_snakefile(self):
        """Verify global report directive is included when reporting enabled."""
        case = tst.retrieve_norfolk_multi_sim_test_case(start_from_scratch=True)
        analysis = case.system.analysis

        # Generate Snakefile with reporting
        snakefile_content = analysis.workflow_builder.generate_snakefile_content(
            enable_reporting=True,
        )

        # Check for report directive
        assert 'report: "workflow_description.rst"' in snakefile_content

    def test_caption_file_generation(self):
        """Test dynamic caption file generation."""
        case = tst.retrieve_norfolk_multi_sim_test_case(start_from_scratch=True)
        builder = case.system.analysis.workflow_builder

        caption_path = builder._generate_caption_file(
            model_type="triton",
            output_path="consolidated_outputs/triton/summaries.zarr",
        )

        assert caption_path.exists()
        content = caption_path.read_text()
        assert "TRITON-only Consolidated Results" in content
        assert builder.analysis.cfg_analysis.analysis_name in content

    def test_report_annotation_helper(self):
        """Test report annotation string generation."""
        case = tst.retrieve_norfolk_multi_sim_test_case(start_from_scratch=True)
        builder = case.system.analysis.workflow_builder

        annotation = builder._annotate_output_for_report(
            output_path="consolidated_outputs/triton/summaries.zarr",
            category="Results",
            subcategory="TRITON",
            caption_file="report_captions/triton_caption.rst",
        )

        expected = (
            'report("consolidated_outputs/triton/summaries.zarr", '
            'category="Results", '
            'subcategory="TRITON", '
            'caption="report_captions/triton_caption.rst", '
            'labels={"output": "summaries"})'
        )
        assert annotation == expected

    def test_report_format_selection_html(self):
        """Test HTML format selection for small outputs."""
        case = tst.retrieve_norfolk_multi_sim_test_case(start_from_scratch=True)
        builder = case.system.analysis.workflow_builder

        # Before consolidation runs, should default to HTML
        format_name, report_path = builder._determine_report_format()

        # HTML expected when no zarr directories exist
        assert report_path.suffix == ".html"

    def test_report_format_selection_zip(self):
        """Test ZIP format selection for zarr directory outputs."""
        case = tst.retrieve_norfolk_multi_sim_test_case(start_from_scratch=True)
        builder = case.system.analysis.workflow_builder

        # Create dummy zarr directory
        zarr_path = builder.analysis_paths.analysis_dir / "consolidated_outputs" / "triton" / "summaries.zarr"
        zarr_path.mkdir(parents=True, exist_ok=True)
        (zarr_path / ".zarray").write_text("{}")

        format_name, report_path = builder._determine_report_format()

        # ZIP expected when zarr directory exists
        assert report_path.suffix == ".zip"

    def test_consolidation_outputs_include_flag(self):
        """Verify consolidation rule always includes flag file."""
        case = tst.retrieve_norfolk_multi_sim_test_case(start_from_scratch=True)
        builder = case.system.analysis.workflow_builder

        # Generate Snakefile with reporting enabled
        snakefile_content = builder.generate_snakefile_content(
            enable_reporting=True,
        )

        # Extract consolidation rule outputs
        import re
        match = re.search(r'rule consolidate:.*?output:\s*([^\n]+)', snakefile_content, re.DOTALL)
        assert match, "Consolidation rule not found in Snakefile"

        outputs = match.group(1)

        # Flag file must always be present
        assert '"_status/output_consolidation_complete.flag"' in outputs

class TestReportGeneration:
    """Integration tests for full report generation."""

    @pytest.mark.slow
    def test_local_workflow_with_report(self):
        """Test full local workflow with report generation."""
        case = tst.retrieve_norfolk_multi_sim_test_case(start_from_scratch=True)
        analysis = case.system.analysis

        result = analysis.workflow_builder.submit_workflow(
            mode="local",
            generate_report=True,
            process_timeseries=True,
            verbose=True,
        )

        assert result["success"]

        # Check report was created
        assert result.get("report_created") is True
        report_path = result.get("report_path")
        assert report_path is not None
        assert Path(report_path).exists()

        # Verify report contains expected sections
        if report_path.suffix == ".html":
            from bs4 import BeautifulSoup
            with open(report_path) as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            # Check for workflow description
            assert soup.find(string=lambda t: t and "TRITON-SWMM Analysis" in t)

            # Check for results category
            assert soup.find(attrs={"class": lambda c: c and "Consolidated Results" in c})

    @pytest.mark.skipif(not tst_ut.on_UVA_HPC(), reason="UVA HPC only")
    @pytest.mark.slow
    def test_slurm_workflow_with_efficiency_report(self):
        """Test SLURM workflow with efficiency report generation."""
        case = tst.retrieve_norfolk_multi_sim_test_case(start_from_scratch=True)
        analysis = case.system.analysis
        analysis.cfg_analysis.multi_sim_run_method = "batch_job"

        result = analysis.workflow_builder.submit_workflow(
            mode="slurm",
            generate_report=True,
            wait_for_completion=True,
            verbose=True,
        )

        assert result["success"]

        # Check efficiency report exists
        efficiency_logs = list(
            (analysis.analysis_paths.analysis_dir / "logs").glob("efficiency_report_*.log")
        )
        assert len(efficiency_logs) > 0

        # Verify report has job efficiency data
        with open(efficiency_logs[0]) as f:
            content = f.read()
            assert "Job ID" in content or "CPU Utilized" in content
```

### Integration Tests

Add to existing HPC test files:

```python
# In test_UVA_02_multisim.py or test_frontier_02_multisim.py

def test_multisim_with_reporting(norfolk_multi_sim_analysis):
    """Test multi-simulation workflow with full reporting."""
    analysis = norfolk_multi_sim_analysis

    result = analysis.workflow_builder.submit_workflow(
        generate_report=True,
        process_timeseries=True,
        wait_for_completion=True,
        verbose=True,
    )

    assert result["success"]

    # Verify report exists
    assert result.get("report_created") is True
    report_path = result.get("report_path")
    assert report_path is not None
    assert Path(report_path).exists()

    # If ZIP format, verify structure
    if report_path.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(report_path) as zf:
            namelist = zf.namelist()
            # Check for report HTML
            assert any("report.html" in name or "index.html" in name for name in namelist)
```

## File Structure Changes

New files to create:

```
workflow/
└── report/
    ├── workflow_template.rst              # Regular analysis workflow description template
    └── workflow_sensitivity_template.rst  # Sensitivity analysis workflow description template
```

**Note**: Caption files are now generated dynamically in each analysis directory at:
```
<analysis_dir>/
└── report_captions/
    ├── triton_caption.rst       (generated dynamically)
    ├── tritonswmm_caption.rst   (generated dynamically)
    └── swmm_caption.rst         (generated dynamically)
```

Modified files:
- `src/TRITON_SWMM_toolkit/workflow.py`:
  - `SnakemakeWorkflowBuilder`: Add reporting methods and parameters
  - `SensitivityAnalysisWorkflowBuilder`: Add sensitivity-specific reporting

New test file:
- `tests/test_reporting.py`:
  - Regular analysis reporting tests
  - Sensitivity analysis reporting tests
  - Report content validation

## Sensitivity Analysis Considerations

Sensitivity analyses present unique reporting challenges due to their hierarchical structure (master → sub-analyses → simulations) and multi-dimensional parameter space exploration.

### Architecture Overview

```
Master Sensitivity Analysis
├── Sub-analysis 0 (param combo 1)
│   ├── Simulation 0 (event 0)
│   ├── Simulation 1 (event 1)
│   └── ...
├── Sub-analysis 1 (param combo 2)
│   └── ...
└── Consolidated multi-dimensional results (param dimensions + event dimension)
```

### Key Differences from Regular Analysis

1. **Single flattened Snakefile**: All sub-analysis simulations in one master workflow
2. **Multi-level consolidation**: Per-subanalysis consolidation → Master consolidation
3. **Multi-dimensional outputs**: Results indexed by parameter values AND event IDs
4. **Single model type**: No multi-model support (would explode parameter space)

### Reporting Challenges

| Challenge | Impact | Solution |
|-----------|--------|----------|
| **Massive output scale** | Hundreds of sub-analyses × simulations = thousands of outputs | Only flag master consolidated outputs for report |
| **Parameter space complexity** | Need to show which parameters were varied and their ranges | Extended workflow description template with sensitivity info |
| **Multi-dimensional data** | Standard report viewers can't navigate N-dimensional zarr stores | Generate summary plots/tables as report artifacts |
| **Provenance tracking** | Which sub-analysis used which parameter combination? | Include sensitivity setup table (CSV) in report |

### Recommended Reporting Strategy for Sensitivity Analysis

#### 1. Master Report Focus

**Include in report:**
- ✅ Master consolidated outputs (multi-dimensional zarr stores)
- ✅ Sensitivity setup table (`df_setup.csv` showing parameter combinations)
- ✅ Summary statistics across parameter space (generated plots/tables)
- ✅ Master workflow execution statistics

**Exclude from report:**
- ❌ Individual sub-analysis outputs (too numerous)
- ❌ Per-simulation outputs (redundant with master consolidation)
- ❌ Intermediate processing artifacts

#### 2. Sensitivity Workflow Description Template

Create `workflow/report/workflow_sensitivity_template.rst`:

```rst
TRITON-SWMM Sensitivity Analysis Report
========================================

Master Analysis Configuration
------------------------------

- **Analysis Name**: {{ snakemake.config.get('analysis_name', 'N/A') }}
- **System**: {{ snakemake.config.get('system_name', 'N/A') }}
- **Execution Mode**: {{ snakemake.config.get('multi_sim_run_method', 'local') }}

Sensitivity Analysis Configuration
-----------------------------------

This sensitivity analysis explores the parameter space defined by:

**Parameter Combinations:**
- Total sub-analyses: {{ snakemake.config.get('n_subanalyses', 'N/A') }}
- Simulations per sub-analysis: {{ snakemake.config.get('n_sims_per_subanalysis', 'N/A') }}
- **Total simulations**: {{ snakemake.config.get('total_simulations', 'N/A') }}

**Sensitivity Setup Table:**

The complete parameter combination matrix is available in the report
(see "Sensitivity Configuration" in the sidebar).

Model Configuration
-------------------

- **Model Type**: {{ snakemake.config.get('model_type', 'N/A') }}
  (sensitivity analysis supports single model type only)
- **Run Mode**: {{ snakemake.config.get('run_mode', 'N/A') }}

Results
-------

This report contains:

- **Master Consolidated Outputs**: Multi-dimensional datasets indexed by parameter values
  and event IDs (Results_)
- **Sensitivity Setup Table**: Complete parameter combination matrix (Configuration_)
- **Workflow Rulegraph**: Rule dependency visualization
- **Workflow Provenance**: Execution timing and resource utilization (Rules_)

Navigate using the sidebar to explore results by category.
```

#### 3. Modified Sensitivity Analysis Snakefile Generation

In `SensitivityAnalysisWorkflowBuilder.generate_master_snakefile_content()`:

```python
def generate_master_snakefile_content(
    self,
    # ... existing parameters ...
    enable_reporting: bool = True,  # NEW PARAMETER
) -> str:
    """Generate master Snakefile with report annotations for sensitivity analysis."""

    # Add report directive pointing to sensitivity-specific template
    report_directive = ""
    if enable_reporting:
        import shutil
        triton_toolkit_root = Path(__file__).parent.parent.parent
        workflow_template_source = triton_toolkit_root / "workflow" / "report" / "workflow_sensitivity_template.rst"
        workflow_template_dest = self.analysis_paths.analysis_dir / "workflow_description.rst"

        if workflow_template_source.exists():
            shutil.copy(workflow_template_source, workflow_template_dest)
            report_directive = f'report: "workflow_description.rst"\n\n'

    snakefile_content = f'''{report_directive}# Auto-generated flattened master Snakefile for sensitivity analysis
# Analysis: {self.sensitivity_analysis.cfg_analysis.analysis_name}

import os
import glob
import subprocess

# Snakemake config for workflow description
config = {{
    "analysis_name": "{self.sensitivity_analysis.cfg_analysis.analysis_name}",
    "system_name": "{self.sensitivity_analysis.system.cfg_system.system_name}",
    "n_subanalyses": {len(self.sensitivity_analysis.df_setup)},
    "n_sims_per_subanalysis": {len(self.sensitivity_analysis.df_sims)},
    "total_simulations": {len(self.sensitivity_analysis.df_setup) * len(self.sensitivity_analysis.df_sims)},
    "model_type": "{self.sensitivity_analysis.model_type}",
    "multi_sim_run_method": "{self.sensitivity_analysis.cfg_analysis.multi_sim_run_method}",
    "run_mode": "{self.sensitivity_analysis.cfg_analysis.run_mode}",
}}

# ... rest of Snakefile ...
'''

    # In master consolidation rule, annotate outputs
    if enable_reporting:
        # Export sensitivity setup table
        df_setup_path = self.analysis_paths.analysis_dir / "sensitivity_setup.csv"
        self.sensitivity_analysis.df_setup.to_csv(df_setup_path, index=True)

        # Generate caption for setup table
        setup_caption = self._generate_sensitivity_setup_caption()

        # Annotate setup table
        setup_annotation = self._base_builder._annotate_output_for_report(
            output_path="sensitivity_setup.csv",
            category="Sensitivity Configuration",
            subcategory="Parameter Matrix",
            caption_file=str(setup_caption.relative_to(self.analysis_paths.analysis_dir)),
        )

        # Generate caption for master outputs
        master_caption = self._generate_sensitivity_master_caption()

        # Annotate master consolidated outputs
        master_annotation = self._base_builder._annotate_output_for_report(
            output_path="consolidated_outputs/sensitivity_master/summaries.zarr",
            category="Sensitivity Results",
            subcategory="Master Consolidated",
            caption_file=str(master_caption.relative_to(self.analysis_paths.analysis_dir)),
        )

        # Consolidation rule outputs (keep flag file separate!)
        consolidate_outputs = [
            '"_status/master_consolidation_complete.flag"',
            setup_annotation,
            master_annotation,
        ]
    else:
        consolidate_outputs = ['"_status/master_consolidation_complete.flag"']

    consolidate_output_str = ", ".join(consolidate_outputs)

    # ... continue with consolidation rule generation ...
```

#### 4. Helper Methods for Sensitivity Captions

```python
def _generate_sensitivity_setup_caption(self) -> Path:
    """Generate caption file for sensitivity setup table."""
    caption_dir = self.analysis_paths.analysis_dir / "report_captions"
    caption_dir.mkdir(exist_ok=True, parents=True)

    caption_path = caption_dir / "sensitivity_setup_caption.rst"

    # Get parameter names
    param_names = list(self.sensitivity_analysis.df_setup.columns)

    caption_content = f"""Sensitivity Analysis Parameter Setup
=====================================

This CSV table defines all parameter combinations explored in the sensitivity analysis.

Parameter Space
---------------

**Independent Variables:**

{chr(10).join(f'- **{param}**' for param in param_names)}

**Total Combinations**: {len(self.sensitivity_analysis.df_setup)}

Table Structure
---------------

- **Index**: Sub-analysis identifier (0, 1, 2, ...)
- **Columns**: Independent variables (parameters being varied)
- **Values**: Specific parameter values for each sub-analysis

Usage
-----

Download this table to:

1. **Identify parameter combinations**: Map sub-analysis index to parameter values
2. **Reproduce specific runs**: Extract exact configuration for any sub-analysis
3. **Analyze parameter space**: Understand sampling strategy

Example
-------

.. code-block:: python

    import pandas as pd
    df_setup = pd.read_csv("sensitivity_setup.csv", index_col=0)

    # View all parameter combinations
    print(df_setup)

    # Find sub-analysis with specific parameters
    matching = df_setup[df_setup["{param_names[0]}"] == value]
    print(f"Sub-analysis index: {{matching.index[0]}}")

Provenance
----------

- **Analysis**: {self.sensitivity_analysis.cfg_analysis.analysis_name}
- **System**: {self.sensitivity_analysis.system.cfg_system.system_name}
- **Generated**: During sensitivity analysis setup
"""

    caption_path.write_text(caption_content)
    return caption_path

def _generate_sensitivity_master_caption(self) -> Path:
    """Generate caption file for sensitivity master consolidated outputs."""
    caption_dir = self.analysis_paths.analysis_dir / "report_captions"
    caption_dir.mkdir(exist_ok=True, parents=True)

    caption_path = caption_dir / "sensitivity_master_caption.rst"

    param_names = list(self.sensitivity_analysis.df_setup.columns)

    caption_content = f"""Master Consolidated Sensitivity Results
========================================

This zarr store contains multi-dimensional results indexed by:

1. **Parameter dimensions**: All independent variables ({', '.join(param_names)})
2. **Event dimension**: All weather events/scenarios
3. **Spatial dimensions**: Mesh nodes or network elements (model-dependent)
4. **Time dimension**: Simulation timesteps

Data Structure
--------------

The dataset is organized as:

.. code-block:: python

    import xarray as xr
    ds = xr.open_zarr("consolidated_outputs/sensitivity_master/summaries.zarr")

    # Example dimensions:
    # - {param_names[0]}: Parameter 1 values
    # - event_iloc: Weather event index
    # - node_id: Spatial identifiers
    # - time: Simulation timesteps (if applicable)

    # Extract results for specific parameter combination
    subset = ds.sel({param_names[0]}=value1, event_iloc=0)

Analysis Workflow
-----------------

1. **Download** the zarr store or access via xarray's lazy loading
2. **Slice** by parameter values to compare specific scenarios
3. **Aggregate** across parameter dimensions to identify sensitivities
4. **Visualize** multi-dimensional relationships

See the Sensitivity Configuration table for the complete parameter combination matrix.

Provenance
----------

- **Analysis**: {self.sensitivity_analysis.cfg_analysis.analysis_name}
- **System**: {self.sensitivity_analysis.system.cfg_system.system_name}
- **Model Type**: {self.sensitivity_analysis.model_type}
- **Total Simulations**: {len(self.sensitivity_analysis.df_setup) * len(self.sensitivity_analysis.df_sims)}
"""

    caption_path.write_text(caption_content)
    return caption_path
```

### Size Considerations for Sensitivity Analysis Reports

**Expected sizes:**
- Master consolidated zarr: **Large** (100+ MB for extensive parameter sweeps)
- Sensitivity setup CSV: **Small** (< 1 MB)
- Summary plots (if generated): **Small-Medium** (< 10 MB total)

**Format recommendation:** Always use **ZIP mode** for sensitivity analysis reports due to:
1. Multi-dimensional zarr stores are directories (HTML mode doesn't support directories)
2. Total size likely exceeds 20 MB threshold

**Auto-detection:** The `_determine_report_format()` method will automatically select ZIP format when it detects zarr directories.

## Migration Path

### Phase 1 (Week 1): Core Infrastructure
1. [ ] Create workflow description templates (`workflow/report/workflow_template.rst`, `workflow_sensitivity_template.rst`)
2. [ ] Add `enable_reporting` parameter to `generate_snakefile_content()`
3. [ ] Implement report directive with template copying logic
4. [ ] Implement `_generate_caption_file()` method for dynamic caption generation
5. [ ] Test locally with existing analysis (verify template copying and caption generation)
6. [ ] **Update this document**: Mark Phase 1 items complete, document any deviations from plan

### Phase 2 (Week 2): Report Annotations
1. [ ] Implement `_annotate_output_for_report()` method
2. [ ] Modify consolidation rule to include report annotations (ensure flag file always included)
3. [ ] Implement `_determine_report_format()` helper
4. [ ] Implement `_check_report_size_warning()` helper
5. [ ] Test with Norfolk case study (verify zarr directories trigger ZIP mode)
6. [ ] **Sensitivity analysis**: Implement `_generate_sensitivity_setup_caption()` and `_generate_sensitivity_master_caption()`
7. [ ] **Sensitivity analysis**: Modify `SensitivityAnalysisWorkflowBuilder.generate_master_snakefile_content()` for reporting
8. [ ] Test sensitivity analysis reporting with existing sensitivity test case
9. [ ] **Update this document**: Mark Phase 2 items complete, note any implementation changes

### Phase 3 (Week 3): Submission Integration
1. [ ] Add `generate_report` and `report_path` parameters to `submit_workflow()`
2. [ ] Modify `run_snakemake_local()` to add `--report-after-run` and validate report creation
3. [ ] Modify `_submit_batch_job_workflow()` to include `--slurm-efficiency-report` and report flags
4. [ ] Modify `_generate_single_job_submission_script()` to include report flags
5. [ ] Test on UVA HPC cluster (verify efficiency report generation)
6. [ ] **Update this document**: Mark Phase 3 items complete, document HPC-specific findings

### Phase 4 (Week 4): Testing & Documentation
1. [ ] Write unit tests in `tests/test_reporting.py` (regular analysis)
2. [ ] Write sensitivity analysis reporting tests
3. [ ] Add integration tests to existing HPC test files
4. [ ] Add report content validation tests (using BeautifulSoup)
5. [ ] Update user documentation with examples (both regular and sensitivity)
6. [ ] Test on Frontier cluster
7. [ ] Validate sensitivity analysis reports with actual parameter sweep
8. [ ] **Update this document**: Mark all phases complete, add "Implementation Complete" section with final notes
9. [ ] **Update CLAUDE.md**: Add reporting section summarizing:
   - Regular analysis reporting features
   - Sensitivity analysis reporting considerations
   - Report format selection (HTML vs ZIP)
   - HPC efficiency report integration
   - Dynamic caption generation approach

## Benefits

### For Users
1. **Self-contained provenance**: Complete workflow documentation in a single file
2. **Execution transparency**: Runtime statistics, rulegraph visualization, and resource utilization
3. **Stakeholder communication**: Interactive reports with analysis-specific captions
4. **Debugging support**: Failed job logs and efficiency data for optimization

### For Developers
1. **HPC resource validation**: Efficiency reports show CPU/GPU/memory utilization
2. **Workflow debugging**: Automatic rulegraph visualization helps identify bottlenecks
3. **Result verification**: Consolidated outputs easily accessible in one report
4. **Reproducibility**: Full provenance information for each run

## Risks & Mitigations

### Risk 1: Report Generation Overhead
**Concern**: Report generation may add significant time to workflow completion

**Mitigation**:
- Reports use existing metadata (minimal overhead)
- Make reporting optional via `generate_report` parameter (defaults to True)
- For quick dev runs, disable with `generate_report=False`
- Skip report generation during dry runs

### Risk 2: Large Output Files
**Concern**: Embedding large zarr directories may create huge reports

**Mitigation**:
- Auto-select ZIP format for outputs > 20 MB or directories
- Warn if expected report size exceeds 500 MB
- Only flag summary CSVs and plots in future enhancements, can exclude raw zarr stores
- Document recommended output sizes in dynamically-generated captions

### Risk 3: SLURM Efficiency Report Availability
**Concern**: `--slurm-efficiency-report` requires `seff` command on cluster

**Mitigation**:
- Snakemake handles missing `seff` gracefully (logs warning, continues without)
- Document cluster requirements in user guide
- Efficiency report is optional enhancement, not required for workflow execution

## Future Enhancements

### Short-term (Next Release)
- Add diagnostic plots to reports (convergence, mass balance)
- Include scenario status table (from `df_status`)
- Generate per-simulation summary cards
- Support custom stylesheets via `--report-stylesheet` (already supported by Snakemake)

### Medium-term (Future Releases)
- Interactive HTML tables with Datavzrd for CSVs
- Per-scenario detailed reports linked from main report
- Summary visualization generation for sensitivity analyses (tornado plots, heatmaps)

### Long-term (Research Features)
- Comparison reports for sensitivity analyses
- Time-lapse animations of inundation extent
- Interactive 3D visualizations (if TRITON supports WebGL export)

## Conclusion

This revised plan provides a **production-ready integration** of Snakemake's reporting capabilities with the following key improvements:

1. **Self-contained analyses**: Templates and captions copied/generated per analysis directory
2. **AI-friendly**: All documentation in code, minimal external template files
3. **Automatic rulegraph**: Workflow topology visualization always included in reports
4. **Robust**: Flag files always preserved in consolidation rules
5. **Simple configuration**: No new config fields, just submit_workflow() parameters

The phased approach allows incremental testing and validation, while the modular design preserves backward compatibility (reporting enabled by default but can be disabled per-run).

**Recommendation**: Proceed with Phase 1 implementation, validate with existing test cases, then progressively add annotations and HPC efficiency reporting.
