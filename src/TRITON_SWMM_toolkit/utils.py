import json
import os
import warnings
from pathlib import Path
import importlib.util
from platformdirs import user_data_dir
from string import Template
import re
import datetime
import yaml
import xarray as xr
import shutil
from typing import Any
import subprocess
from typing import Optional, Literal, Callable


class BatchJobSubmissionError(Exception):
    """
    Custom exception for batch job submission failures.

    Provides detailed information about why a batch job submission failed,
    including the script path, command, dependency information, and stderr output.
    """

    def __init__(
        self,
        script_path: Path,
        command: list,
        return_code: int,
        stderr: str,
        dependent_job_id: Optional[int | str | list] = None,
        dependency_type: str = "afterok",
    ):
        self.script_path = script_path
        self.command = command
        self.return_code = return_code
        self.stderr = stderr
        self.dependent_job_id = dependent_job_id
        self.dependency_type = dependency_type

        # Format the error message
        error_lines = [
            "Failed to submit batch job script",
            f"  Script: {script_path}",
        ]

        if dependent_job_id:
            error_lines.append(f"  Dependency: {dependency_type}:{dependent_job_id}")

        error_lines.extend(
            [
                f"  Command: {' '.join(str(c) for c in command)}",
                f"  Return code: {return_code}",
            ]
        )

        if stderr.strip():
            error_lines.append(f"  Error output:\n{self._indent_text(stderr)}")

        self.message = "\n".join(error_lines)
        super().__init__(self.message)

    @staticmethod
    def _indent_text(text: str, indent: str = "    ") -> str:
        """Indent each line of text for better readability."""
        return "\n".join(indent + line for line in text.split("\n"))


def fast_rmtree(
    path: str | Path,
    *,
    missing_ok: bool = True,
    onerror: Optional[Callable] = None,
) -> None:
    """Fast, cross-platform directory delete.

    Uses OS-native delete commands for speed; falls back to shutil.rmtree.

    Parameters
    ----------
    path : str | Path
        Directory path to delete.
    missing_ok : bool
        If True, silently return when path does not exist.
    onerror : callable, optional
        Error handler passed to shutil.rmtree (fallback only).
    """
    path = Path(path)

    if not path.exists():
        if missing_ok:
            return
        raise FileNotFoundError(path)

    if path.is_symlink() or path.is_file():
        path.unlink()
        return

    try:
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "rmdir", "/s", "/q", str(path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(
                ["rm", "-rf", str(path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return
    except Exception:
        shutil.rmtree(path, onerror=onerror)


def fix_line_endings(file_path, target_ending="\n"):
    """
    Convert line endings in a file to the target format.
    Only rewrites the file if line endings are incorrect.

    Args:
        file_path (str): Path to the file to fix
        target_ending (str): Target line ending ('\n' for LF, '\r\n' for CRLF)

    Returns:
        bool: True if file was modified, False if already correct
    """
    try:
        # Read file in binary mode
        with open(file_path, "rb") as f:
            original_content = f.read()

        # Normalize to LF first, then convert to target
        normalized = original_content.replace(b"\r\n", b"\n")  # CRLF -> LF
        normalized = normalized.replace(b"\r", b"\n")  # CR -> LF

        # Convert to target ending if needed
        if target_ending == "\r\n":
            normalized = normalized.replace(b"\n", b"\r\n")

        # Only write if content changed
        if normalized != original_content:
            with open(file_path, "wb") as f:
                f.write(normalized)
            print(f"✓ Fixed line endings in: {file_path}")
            return True
        else:
            # print(f"✓ Already correct: {file_path}")
            return False

    except Exception as e:
        print(f"✗ Error fixing {file_path}: {e}")
        return False


def run_bash_script(
    bash_script: Path,
    dependent_job_id: Optional[int | str | list] = None,
    dependency_type: Literal["afterok", "afterany"] = "afterok",
    verbose: bool = True,
):
    cmd = ["sbatch"]
    dpdndncy = ""
    if dependent_job_id:
        if isinstance(dependent_job_id, list):
            dependent_job_id = ",".join(dependent_job_id)
        cmd.append(
            f"--dependency={dependency_type}:{dependent_job_id}",
        )
        dpdndncy = (
            f"\n dependent on job {dependent_job_id} using dependency={dependency_type}"
        )
    cmd.append(str(bash_script))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise BatchJobSubmissionError(
            script_path=bash_script,
            command=cmd,
            return_code=e.returncode,
            stderr=e.stderr,
            dependent_job_id=dependent_job_id if dependent_job_id else None,
            dependency_type=dependency_type,
        ) from e

    job_id = proc.stdout.strip().split()[-1]
    if verbose:
        print(f"Submitted script {bash_script}{dpdndncy}\njob id: {job_id}", flush=True)
    return job_id


def archive_directory_contents(dir: Path):
    archive_dir = dir / "_archive"
    archive_dir.mkdir(exist_ok=True, parents=True)
    for item in dir.iterdir():
        if item.name == "_archive":
            continue
        shutil.move(str(item), archive_dir / item.name)


def create_mask_from_shapefile(
    da_to_mask, shapefile_path=None, series_single_row_of_gdf=None
):  # , COORD_EPSG):
    # da_to_mask, shapefile_path = da_sim_wlevel, f_mitigation_aois
    og_shape = da_to_mask.shape
    xs = da_to_mask.x.to_series()
    ys = da_to_mask.y.to_series()
    from shapely.geometry import mapping
    import geopandas as gpd
    import rasterio.features

    if shapefile_path is not None:
        gdf = gpd.read_file(shapefile_path)
        shapes = [
            mapping(geom) for geom in gdf.geometry
        ]  # Convert geometries to GeoJSON-like format
    if series_single_row_of_gdf is not None:
        shapes = [mapping(series_single_row_of_gdf.geometry)]
    mask = rasterio.features.geometry_mask(
        shapes,
        transform=da_to_mask.rio.transform(),
        invert=True,
        out_shape=(og_shape),
    )
    return mask


def read_yaml(f_yaml: Path | str):
    return yaml.safe_load(Path(f_yaml).read_text())


def write_yaml(data: dict, f_yaml: Path | str):
    with open(f_yaml, "w") as file:
        yaml.dump(data, file)
    return


def get_package_root(package_name: str) -> Path:
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"Package {package_name} not found")
    return Path(spec.origin).parent


def get_package_data_root(package_name) -> Path:
    return Path(user_data_dir(package_name))


def fill_template(f_template: Path, mapping: dict):
    with open(f_template, "r") as T:
        template = Template(T.read())
        filled = template.safe_substitute(mapping)
    return filled


def create_from_template(f_template: Path, mapping: dict, f_out: Path):
    filled = fill_template(f_template, mapping)
    f_out.parent.mkdir(parents=True, exist_ok=True)
    with open(f_out, "w+") as f1:
        f1.write(filled)
    return filled


def find_all_keys_in_template(f_template):
    with open(f_template, "r") as f:
        text = f.read()
    keys = re.findall(r"\{([^}]+)\}", text)
    unique_keys = list(dict.fromkeys(keys))
    return unique_keys


def load_json(file: Path):
    with open(file) as f:
        log = json.load(f)
    return log


def write_json(data: dict, file: Path):
    file.parent.mkdir(exist_ok=True, parents=True)
    pid = os.getpid()
    tmp_path = file.with_suffix(file.suffix + f".{pid}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(file)


def replace_substring_in_file(file_path, old_substring, new_substring, verbose=False):
    """
    Replace all occurrences of old_substring with new_substring in a text file.

    Parameters:
        file_path (str): Path to the text file.
        old_substring (str): The substring to be replaced.
        new_substring (str): The substring to replace with.
    """
    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace substring
    content = content.replace(old_substring, new_substring)

    # Write back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    if verbose:
        print(f"Replaced '{old_substring}' with '{new_substring}' in {file_path}")


def read_text_file_as_string(file):
    with open(file) as f:
        contents = f.read()
    return contents


def current_datetime():
    return (
        datetime.datetime.now()
        .astimezone()
        .isoformat(
            timespec="seconds",
        )
    )


def current_datetime_string(filepath_friendly: bool = False):
    """
    Docstring for current_datetime_string

    Generates a datetime string following following  ISO 8601 format conventions.

    :param filepath_friendly: If True, colons are replaced with nothing, e.g., 2026-01-07T10:03:37-05:00 becomes 2026-01-07T100337-0500
    :type filepath_friendly: bool
    """
    dts = current_datetime()
    if filepath_friendly:
        dts = dts.replace(":", "")

    return dts


def string_to_datetime(dt: str):
    return datetime.datetime.fromisoformat(dt)


def read_header(file, nlines):
    lst_lines = []
    with open(file, "r") as f:
        for i in range(nlines):
            line = f.readline()
            if not line:
                break  # Stop if file has fewer than 6 lines
            lst_lines.append(line)
    return lst_lines


def read_text_file_as_list_of_strings(file):
    with open(file) as f:
        contents = f.readlines()
    return contents


def parse_triton_log_file(log_file_path: Path) -> dict[str, Any]:
    """
    Parse TRITON log.out file to extract actual compute resource usage.

    Parameters
    ----------
    log_file_path : Path
        Path to the log.out file

    Returns
    -------
    dict
        Dictionary containing:
        - nTasks: int - Number of MPI tasks
        - omp_threads_per_task: int - OpenMP threads per task
        - gpus_per_task: int - GPUs per task
        - total_gpus: int - Total GPUs used
        - gpu_backend: str - GPU backend (HIP/CUDA/none)
        - build_type: str - Build type (e.g., "CPU+OMP", "GPU+HIP")
        - triton_git_version: str - TRITON git version
        - wall_time_s: float - Total wall time in seconds
        - machine: str - Machine name
        - cpu: str - CPU model

    Returns None for all fields if file doesn't exist or parsing fails.
    """
    if not log_file_path.exists():
        return {
            "nTasks": None,
            "omp_threads_per_task": None,
            "gpus_per_task": None,
            "total_gpus": None,
            "gpu_backend": None,
            "build_type": None,
            "triton_git_version": None,
            "wall_time_s": None,
            "machine": None,
            "cpu": None,
        }

    try:
        content = read_text_file_as_string(log_file_path)

        # Initialize result dictionary with None values
        result = {
            "nTasks": None,
            "omp_threads_per_task": None,
            "gpus_per_task": None,
            "total_gpus": None,
            "gpu_backend": None,
            "build_type": None,
            "triton_git_version": None,
            "wall_time_s": None,
            "machine": None,
            "cpu": None,
        }

        # Parse each field using regex
        # Machine name
        match = re.search(r"Machine\s*:\s*(.+)", content)
        if match:
            result["machine"] = match.group(1).strip()  # type: ignore

        # CPU model
        match = re.search(r"CPU\s*:\s*(.+)", content)
        if match:
            result["cpu"] = match.group(1).strip()  # type: ignore

        # nTasks
        match = re.search(r"nTasks\s*:\s*(\d+)", content)
        if match:
            result["nTasks"] = int(match.group(1))  # type: ignore

        # OMP threads per task
        match = re.search(r"OMP threads per task\s*:\s*(\d+)", content)
        if match:
            result["omp_threads_per_task"] = int(match.group(1))  # type: ignore

        # GPUs per task (handle "0 (CPU-only)" case)
        match = re.search(r"GPUs per task\s*:\s*(\d+)", content)
        if match:
            result["gpus_per_task"] = int(match.group(1))  # type: ignore

        # GPU backend
        match = re.search(r"GPU backend\s*:\s*(\S+)", content)
        if match:
            result["gpu_backend"] = match.group(1).strip()  # type: ignore

        # Total GPUs
        match = re.search(r"Total GPUs\s*:\s*(\d+)", content)
        if match:
            result["total_gpus"] = int(match.group(1))  # type: ignore

        # TRITON git version
        match = re.search(r"TRITON_GIT_VERSION\s*:\s*(.+)", content)
        if match:
            result["triton_git_version"] = match.group(1).strip()  # type: ignore

        # Build type
        match = re.search(r"Build type\s*:\s*(.+)", content)
        if match:
            result["build_type"] = match.group(1).strip()  # type: ignore

        # Wall time
        match = re.search(r"TRITON total wall time \[s\]\s*:\s*([\d.]+)", content)
        if match:
            result["wall_time_s"] = float(match.group(1))  # type: ignore

        return result

    except Exception as e:
        warnings.warn(
            f"Failed to parse TRITON log file {log_file_path}: {str(e)}",
            UserWarning,
        )
        return {
            "nTasks": None,
            "omp_threads_per_task": None,
            "gpus_per_task": None,
            "total_gpus": None,
            "gpu_backend": None,
            "build_type": None,
            "triton_git_version": None,
            "wall_time_s": None,
            "machine": None,
            "cpu": None,
        }


def return_dic_zarr_encodings(ds: xr.Dataset, clevel: int = 5) -> dict:
    """
    Create a dictionary of Zarr encodings for an xarray Dataset.

    Uses Blosc compression for numeric variables and preserves
    maximum string length for Unicode coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to encode.
    clevel : int, default=5
        Compression level for Blosc.

    Returns
    -------
    encoding : dict
        Dictionary suitable for xarray.to_zarr(..., encoding=encoding)
    """
    encoding = {}

    # Compressor for numeric data
    import zarr

    compressor = zarr.codecs.BloscCodec(  # type: ignore
        cname="zstd", clevel=clevel, shuffle=zarr.codecs.BloscShuffle.shuffle  # type: ignore
    )

    # Handle data variables
    for var in ds.data_vars:  # type: ignore
        dtype_kind = ds[var].dtype.kind
        if dtype_kind in {"i", "u", "f"}:  # int / unsigned int / float
            encoding[var] = {"compressors": compressor}
        # Optionally handle other types if needed

    # Handle coordinate encoding
    for coord in ds.coords:  # type: ignore
        dtype_kind = ds[coord].dtype.kind  # type: ignore
        if dtype_kind == "U":  # Unicode string coordinates
            max_len = ds[coord].str.len().max().item()
            encoding[coord] = {"dtype": f"<U{max_len}"}  # type: ignore

    return encoding


def return_dic_autochunk(ds):
    chunk_dict = {}
    for var in ds.dims:
        chunk_dict[var] = "auto"
    return chunk_dict


def estimate_timesteps_per_chunk(
    rds_dem: xr.DataArray,
    n_variables: int,
    memory_budget_MiB: float,
    dtype: Any = None,
) -> int:
    """
    Estimate how many timesteps can fit in memory budget.

    Uses simple memory arithmetic to calculate how many timesteps can be
    loaded simultaneously for all variables within the specified memory budget.
    This is used for chunked processing of TRITON binary outputs.

    Parameters
    ----------
    rds_dem : xr.DataArray
        DEM raster with x and y coordinates (used to get grid dimensions)
    n_variables : int
        Number of variables per timestep (e.g., 4 for H, QX, QY, MH)
    memory_budget_MiB : float
        Target memory budget in MiB
    dtype : np.dtype or None
        Data type (default: np.float64). If None, uses float64.

    Returns
    -------
    int
        Number of timesteps per chunk (minimum 1)

    Examples
    --------
    >>> # For a 513x526 grid with 4 variables and 200 MiB budget
    >>> chunk_size = estimate_timesteps_per_chunk(
    ...     rds_dem=dem,
    ...     n_variables=4,
    ...     memory_budget_MiB=200.0
    ... )
    >>> # Returns number of timesteps that fit in 200 MiB

    Notes
    -----
    Memory calculation:
        memory_per_timestep = n_variables × n_y × n_x × bytes_per_element
        timesteps_per_chunk = memory_budget / memory_per_timestep

    This simple approach is appropriate for timeseries processing where we need
    to determine how much data to load BEFORE creating the dataset. For chunking
    existing datasets for zarr writes, use compute_optimal_chunks() instead.
    """
    import numpy as np

    if dtype is None:
        dtype = np.float64

    n_y = len(rds_dem.y)
    n_x = len(rds_dem.x)
    bytes_per_element = np.dtype(dtype).itemsize

    # Memory for ONE timestep across ALL variables
    memory_per_timestep_bytes = n_variables * n_y * n_x * bytes_per_element
    memory_per_timestep_MiB = memory_per_timestep_bytes / (1024**2)

    # How many timesteps fit in budget?
    timesteps_per_chunk = int(memory_budget_MiB / memory_per_timestep_MiB)

    # Ensure at least 1 timestep per chunk
    return max(1, timesteps_per_chunk)


def prev_power_of_two(n: int | float) -> int:
    """
    Return the largest power of 2 less than or equal to n.

    Parameters
    ----------
    n : int or float
        Input number (must be positive)

    Returns
    -------
    int
        Largest power of 2 <= n

    Examples
    --------
    >>> prev_power_of_two(100)
    64
    >>> prev_power_of_two(256)
    256
    """
    n = int(n)
    if n < 1:
        return 1
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 << (n.bit_length() - 1)


def ds_memory_req_MiB(ds: xr.Dataset) -> float:
    """
    Calculate memory requirement of xarray Dataset in MiB.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to measure

    Returns
    -------
    float
        Memory requirement in MiB
    """
    return ds.nbytes / 1024**2


def compute_optimal_chunks(
    ds: xr.Dataset,
    spatial_coords: list[str] | str | None,
    max_mem_usage_MiB: float,
    spatial_coord_size: int = 65536,  # 256x256 for x,y coords
    verbose: bool = True,
) -> dict | str:
    """
    Compute optimal chunk sizes for writing xarray datasets to disk.

    This function determines chunk sizes that:
    1. Keep memory usage under max_mem_usage_MiB
    2. Use efficient spatial chunks (~256x256 for x,y)
    3. Handle sparse multi-dimensional coordinates (sensitivity analysis)

    Extracted from processing_analysis.py to make it reusable for both
    per-simulation processing and analysis-level consolidation.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to compute chunks for
    spatial_coords : List[str] | str | None
        Spatial coordinate names (e.g., ['x', 'y'] or 'node_id').
        If None, returns 'auto'.
    max_mem_usage_MiB : float
        Maximum memory per chunk in MiB
    spatial_coord_size : int
        Target total cells per spatial chunk (default 65536 = 256^2)
    verbose : bool
        Print chunk information if True

    Returns
    -------
    dict or "auto"
        Chunk specification for each dimension

    Examples
    --------
    >>> # For TRITON spatial outputs
    >>> chunks = compute_optimal_chunks(
    ...     ds=ds_triton,
    ...     spatial_coords=["x", "y"],
    ...     max_mem_usage_MiB=200.0
    ... )

    >>> # For SWMM node outputs
    >>> chunks = compute_optimal_chunks(
    ...     ds=ds_swmm_nodes,
    ...     spatial_coords="node_id",
    ...     max_mem_usage_MiB=200.0
    ... )

    Notes
    -----
    This function is used for chunking EXISTING datasets for zarr writes.
    For determining how many timesteps to load during processing, use
    estimate_timesteps_per_chunk() instead.
    """
    from typing import List

    # Handle non-spatial data (e.g., performance summaries)
    if spatial_coords is None:
        if verbose:
            print("spatial_coords are None. Returning chunks = 'auto'", flush=True)
        return "auto"

    if isinstance(spatial_coords, str):
        spatial_coords = [spatial_coords]

    # Validation: Check that all spatial coords exist in dataset
    missing_coords = [c for c in spatial_coords if c not in ds.coords]
    if missing_coords:
        error_msg = (
            f"Spatial coordinates {missing_coords} not found in dataset. "
            f"Available coordinates: {list(ds.coords.keys())}"
        )
        raise ValueError(error_msg)

    size_per_spatial_coord = spatial_coord_size ** (1 / len(spatial_coords))

    if len(spatial_coords) not in [1, 2]:
        raise ValueError("Spatial dimension can only be 1 or 2 dimensional")

    lst_non_spatial_coords = []
    for coord in ds.coords:
        if coord not in spatial_coords:
            lst_non_spatial_coords.append(coord)

    # Categorize variables by whether they have spatial dimensions
    spatial_vars = []
    nonspatial_vars = []  # system-wide vars
    for var in ds.data_vars:
        var_dims = set(ds[var].dims)
        if any(coord in var_dims for coord in spatial_coords):
            spatial_vars.append(var)
        else:
            nonspatial_vars.append(var)

    # Get average bytes per element (for float64/float32 estimation)
    # Use first spatial variable if available, otherwise use a default
    if spatial_vars:
        sample_var = ds[spatial_vars[0]]
        bytes_per_element = sample_var.dtype.itemsize
    else:
        bytes_per_element = 8  # default to float64

    # Calculate spatial chunk size first (fixed target)
    chunks: dict = {}
    spatial_chunk_points = 1
    for coord in spatial_coords:
        coord_len = len(ds[coord])
        chunk_size = int(min(size_per_spatial_coord, coord_len))
        chunks[coord] = chunk_size
        spatial_chunk_points *= chunk_size

    # Calculate non-spatial budget accounting for heterogeneous variable shapes
    # Chunk memory = (n_spatial_vars * spatial_points * nonspatial_points +
    #                 n_nonspatial_vars * nonspatial_points) * bytes_per_element
    # Solving for nonspatial_points:
    # nonspatial_points = max_mem_bytes /
    #                     (bytes_per_element * (n_spatial_vars * spatial_points + n_nonspatial_vars))

    bytes_available = max_mem_usage_MiB * 1024**2

    # Calculate the "weight" of one nonspatial point in the chunk
    # Each nonspatial point contributes:
    # - spatial_chunk_points elements for each spatial variable
    # - 1 element for each non-spatial variable
    elements_per_nonspatial_point = len(spatial_vars) * spatial_chunk_points + len(
        nonspatial_vars
    )

    if elements_per_nonspatial_point > 0:
        target_nonspatial_points = bytes_available / (
            bytes_per_element * elements_per_nonspatial_point
        )
        target_nonspatial_points = max(1, int(target_nonspatial_points))
    else:
        # Edge case: no variables (shouldn't happen in practice)
        target_nonspatial_points = 1

    # Use power-of-2 for better compression
    target_nonspatial_chunk = prev_power_of_two(target_nonspatial_points)

    # Sort non-spatial coords by size (largest first) for better chunking
    sorted_nonspatial = sorted(
        lst_non_spatial_coords,
        key=lambda c: len(ds[c]),
        reverse=True,
    )

    nonspatial_chunk_product = 1
    for coord in sorted_nonspatial:
        coord_len = len(ds[coord])

        # Determine chunk size for this dimension
        if nonspatial_chunk_product >= target_nonspatial_chunk:
            # Already reached target, chunk remaining dims minimally
            chunk_size = 1
        elif coord_len == 1:
            # Singleton dimension
            chunk_size = 1
        else:
            # Calculate how much "budget" remains for chunking
            remaining_budget = target_nonspatial_chunk // nonspatial_chunk_product
            chunk_size = min(coord_len, prev_power_of_two(remaining_budget))
            # Ensure at least some chunking for large dimensions
            if chunk_size < 1:
                chunk_size = 1

        chunks[coord] = chunk_size
        nonspatial_chunk_product *= chunk_size

    # Build test slice to verify memory usage
    test_slice = {}
    for coord, chunk_size in chunks.items():
        test_slice[coord] = slice(0, min(chunk_size, len(ds[coord])))

    # Estimate test chunk memory without forcing rechunking (avoid dask overhead)
    test_ds = ds.isel(test_slice)
    test_size_MiB = ds_memory_req_MiB(test_ds)

    # Validation: Check chunk efficiency
    if test_size_MiB < 1:
        msg = (
            f"Warning: chunks are less than 1 MiB ({test_size_MiB:.3f} MiB), "
            "which could lead to inefficient reading and writing. "
            f"Consider increasing max_mem_usage_MiB or spatial_coord_size."
        )
        print(msg, flush=True)

    if test_size_MiB > max_mem_usage_MiB * 1.2:
        msg = (
            f"Chunk size ({test_size_MiB:.1f} MiB) exceeds "
            f"max_mem_usage_MiB ({max_mem_usage_MiB} MiB). "
            f"Chunks: {chunks}"
        )
        raise ValueError(msg)

    if verbose:
        print(
            f"Memory per chunk: {test_size_MiB:.3f} MiB\nChunks: {chunks}",
            flush=True,
        )

    return chunks


def get_file_size_MiB(f: Path):
    if f.name.split(".")[-1] == "zarr":
        size_bytes = zarr_size_bytes(f)
    else:
        size_bytes = f.stat().st_size
    return size_bytes / 1024**2


def zarr_size_bytes(zarr_path: Path) -> int:
    return sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file())


def write_zarr(ds, fname_out, compression_level, chunks: str | dict = "auto"):
    encoding = return_dic_zarr_encodings(ds, compression_level)
    if chunks == "auto":
        chunks = return_dic_autochunk(ds)
    ds = ds.chunk(chunks)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*does not have a Zarr V3 specification.*",
            category=Warning,
        )
        ds.to_zarr(fname_out, mode="w", encoding=encoding, consolidated=False)


def write_zarr_then_netcdf(
    ds, fname_out, compression_level: int = 5, chunks: str | dict = "auto"
):
    # encoding = return_dic_zarr_encodings(ds, compression_level)
    if chunks == "auto":
        chunks = return_dic_autochunk(ds)
    ds = ds.chunk(chunks)
    # first write to zarr, then write to netcdf
    write_zarr(ds, f"{fname_out}.zarr", compression_level, chunks)
    # open and write
    ds = xr.open_dataset(
        f"{fname_out}.zarr", engine="zarr", chunks="auto", consolidated=False
    )
    write_netcdf(ds, fname_out, compression_level, chunks)
    # delete zarr
    try:
        fast_rmtree(f"{fname_out}.zarr")
    except Exception as e:
        print(f"Could not remove zarr folder {fname_out}.zarr due to error {e}")
    return


def return_dic_netcdf_encodings(ds: xr.Dataset, clevel: int = 5) -> dict:
    encoding = {}
    for var in ds.data_vars:
        if ds[var].dtype.kind in {"i", "u", "f"}:
            encoding[var] = {"zlib": True, "complevel": clevel, "shuffle": True}
    # Coordinates usually don’t need compression
    return encoding


def write_netcdf(
    ds, fname_out, compression_level: int = 5, chunks: str | dict = "auto"
):
    encoding = return_dic_netcdf_encodings(ds, compression_level)
    if chunks == "auto":
        chunk_dict = return_dic_autochunk(ds)
    else:
        chunk_dict = chunks
    try:
        ds = ds.chunk(chunk_dict)
    except NotImplementedError:
        ds = ds.copy(deep=False)
    ds.to_netcdf(fname_out, encoding=encoding, engine="h5netcdf")
    return


def paths_to_strings(obj: Any) -> Any:
    """
    Recursively convert all pathlib.Path objects to strings
    in arbitrarily nested dictionaries / containers.
    """
    if isinstance(obj, Path):
        return str(obj)

    elif isinstance(obj, dict):
        return {k: paths_to_strings(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [paths_to_strings(v) for v in obj]

    elif isinstance(obj, tuple):
        return tuple(paths_to_strings(v) for v in obj)

    elif isinstance(obj, set):
        return {paths_to_strings(v) for v in obj}

    return obj


def convert_datetime_to_str(obj: Any) -> Any:
    """
    Recursively convert all datetime objects to ISO format strings
    in arbitrarily nested dictionaries / containers.

    This ensures that datetime objects can be serialized to JSON
    when writing xarray datasets to zarr format.
    """
    import pandas as pd

    # Handle datetime objects
    if isinstance(obj, (datetime.datetime, pd.Timestamp)):
        return obj.isoformat()

    elif isinstance(obj, dict):
        return {k: convert_datetime_to_str(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [convert_datetime_to_str(v) for v in obj]

    elif isinstance(obj, tuple):
        return tuple(convert_datetime_to_str(v) for v in obj)

    elif isinstance(obj, set):
        return {convert_datetime_to_str(v) for v in obj}

    return obj
