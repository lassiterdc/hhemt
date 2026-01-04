import json
from pathlib import Path
import os
import importlib.util
from platformdirs import user_data_dir
from string import Template
import re
import sys
from datetime import datetime


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


def create_logfile(inital_data: dict, file: Path):
    if file.exists():
        log = load_json(file)
        print(log)
        sys.exit("handle when logfile already exists")
    log = inital_data.copy()
    log["logfile"] = str(file)
    write_json(log, file)
    return log


def update_logfile(log):
    write_json(log, Path(log["logfile"]))
    return load_json(Path(log["logfile"]))


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


def current_datetime_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
