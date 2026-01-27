import json
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
from typing import Optional, Literal


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
    with open(file, "w") as f:
        json.dump(data, f, indent=2, default=str)


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
    ds.to_zarr(fname_out, mode="w", encoding=encoding, consolidated=False)


def write_zarr_then_netcdf(
    ds, fname_out, compression_level: int = 5, chunks: str | dict = "auto"
):
    encoding = return_dic_zarr_encodings(ds, compression_level)
    if chunks == "auto":
        chunk_dict = return_dic_autochunk(ds)
    ds = ds.chunk(chunk_dict)
    # first write to zarr, then write to netcdf
    write_zarr(ds, fname_out, compression_level, chunks)
    # open and write
    ds = xr.open_dataset(
        f"{fname_out}.zarr", engine="zarr", chunks="auto", consolidated=False
    )
    write_netcdf(ds, fname_out, compression_level, chunks)
    # delete zarr
    try:
        shutil.rmtree(f"{fname_out}.zarr")
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
    ds = ds.chunk(chunk_dict)
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
