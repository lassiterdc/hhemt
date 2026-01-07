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
    from numcodecs import Blosc

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
    for var in ds.data_vars:
        dtype_kind = ds[var].dtype.kind
        if dtype_kind in {"i", "u", "f"}:  # int / unsigned int / float
            encoding[var] = {"compressors": compressor}
        # Optionally handle other types if needed

    # Handle coordinate encoding
    for coord in ds.coords:
        dtype_kind = ds[coord].dtype.kind
        if dtype_kind == "U":  # Unicode string coordinates
            max_len = ds[coord].str.len().max().item()
            encoding[coord] = {"dtype": f"<U{max_len}"}

    return encoding


def return_dic_autochunk(ds):
    chunk_dict = {}
    for var in ds.dims:
        chunk_dict[var] = "auto"
    return chunk_dict


def write_zarr(ds, fname_out, compression_level):
    encoding = return_dic_zarr_encodings(ds)
    chunks = return_dic_autochunk(ds)
    ds.chunk(chunks).to_zarr(fname_out, mode="w", encoding=encoding, consolidated=False)


def write_zarr_then_netcdf(ds, fname_out, compression_level):
    encoding = return_dic_zarr_encodings(ds)
    chunk_dict = return_dic_autochunk(ds)
    ds = ds.chunk(chunk_dict)
    # first write to zarr, then write to netcdf
    write_zarr(ds, fname_out, compression_level)
    # open and write
    ds = xr.open_dataset(
        f"{fname_out}.zarr", engine="zarr", chunks="auto", consolidated=False
    )
    ds.to_netcdf(fname_out, encoding=encoding, engine="h5netcdf")
    # delete zarr
    try:
        shutil.rmtree(f"{fname_out}.zarr")
    except Exception as e:
        print(f"Could not remove zarr folder {fname_out}.zarr due to error {e}")
    return
