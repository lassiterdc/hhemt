"""`hhemt reprex config --bundle DIR` writes the bundle's runnable system config and prints the path it wrote.

The command is a thin front for `hhemt.bundle._emit.reconstitute_runnable_config`: `--output`
maps to the helper's `target_path`, `--software-dir` maps to its `software_dir`, and the
printed line is the path the helper returns. The tests run the command through Typer's
`CliRunner` on a minimal bundle directory (a `cfg_system.yaml` with bundle-relative paths)
and compare the written file with what the helper writes for the same bundle. The bundle
needs no solver, so the tests run on the laptop.
"""

from __future__ import annotations

import yaml
from typer.testing import CliRunner

from hhemt.bundle._emit import reconstitute_runnable_config
from hhemt.cli import app


def _bundle(root):
    """A reprex bundle directory holding a scrubbed `cfg_system.yaml` with a relative input path."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "cfg_system.yaml").write_text(
        yaml.safe_dump(
            {
                "system_directory": ".",
                "target_dem_resolution": 2.0,
                "watershed_gis_polygon": "external/ws.shp",
                "SWMM_software_directory": None,
                "TRITONSWMM_software_directory": None,
            }
        )
    )
    return root.resolve()


def _invoke(*args):
    return CliRunner().invoke(app, ["reprex", "config", *args])


def _flat(result):
    """The command's combined output with line wraps collapsed, for substring matching."""
    return " ".join(result.output.split())


def test_default_writes_the_helpers_file_inside_the_bundle_and_prints_its_path(tmp_path):
    """Property: with neither optional flag, the command writes `system_config.yaml` in the bundle
    exactly as the helper does, leaves both software directories null, and prints that path.

    Class: a bundle directory with a relative input path. Class NEW CAPABILITY; the plausible
    wrong implementations are a command that prints another path, passes a `software_dir` by
    default, or writes elsewhere; each is red here (printed path, null directories, file location).
    Why required: this is the page's replacement for calling the private helper, so its file must
    equal the helper's output byte for byte.
    What kills it: the printed value changed, a default `software_dir`, a default `target_path`.
    A second correct implementation: print `Path(written).as_posix()` or the resolved path.
    """
    bundle = _bundle(tmp_path / "bundle")
    expected = reconstitute_runnable_config(bundle, target_path=tmp_path / "expected.yaml")
    result = _invoke("--bundle", str(bundle))
    assert result.exit_code == 0, result.output
    written = bundle / "system_config.yaml"
    assert result.output.strip() == str(written)
    assert written.read_bytes() == expected.read_bytes()
    loaded = yaml.safe_load(written.read_text())
    assert loaded["SWMM_software_directory"] is None
    assert loaded["TRITONSWMM_software_directory"] is None
    assert loaded["watershed_gis_polygon"] == str(bundle / "external" / "ws.shp")


def test_output_maps_to_the_helpers_target_path(tmp_path):
    """Property: `--output FILE` writes FILE and prints it, and the bundle's default file stays absent.

    Class: an `--output` path outside the bundle. Class NEW CAPABILITY; the plausible wrong
    implementation drops the keyword, so the file lands at the default location; red here.
    Why required: the flag exists to keep the bundle directory unchanged.
    What kills it: `target_path` not passed, or passed the wrong value.
    A second correct implementation: write at the default location and move the file.
    """
    bundle = _bundle(tmp_path / "bundle")
    target = tmp_path / "elsewhere" / "mine.yaml"
    target.parent.mkdir()
    result = _invoke("--bundle", str(bundle), "--output", str(target))
    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(target)
    assert target.read_bytes() == reconstitute_runnable_config(bundle, target_path=tmp_path / "e.yaml").read_bytes()
    assert not (bundle / "system_config.yaml").exists()


def test_software_dir_maps_to_the_helpers_software_dir(tmp_path):
    """Property: `--software-dir DIR` points both build directories at DIR's `swmm` and `tritonswmm`
    subdirectories, as the helper does for the same argument.

    Class: a `--software-dir` path that does not exist yet (the build directories are created
    later). Class NEW CAPABILITY; the plausible wrong implementation drops the keyword, so both
    stay null; red here.
    Why required: a run needs real build paths where a render needs null.
    What kills it: `software_dir` not passed, or passed as the bundle directory.
    A second correct implementation: patch the two keys in the YAML after the helper returns.
    """
    bundle = _bundle(tmp_path / "bundle")
    software = tmp_path / "builds"
    result = _invoke("--bundle", str(bundle), "--software-dir", str(software))
    assert result.exit_code == 0, result.output
    written = bundle / "system_config.yaml"
    expected = reconstitute_runnable_config(bundle, target_path=tmp_path / "e.yaml", software_dir=software)
    assert written.read_bytes() == expected.read_bytes()
    loaded = yaml.safe_load(written.read_text())
    assert loaded["SWMM_software_directory"] == str(software.resolve() / "swmm")
    assert loaded["TRITONSWMM_software_directory"] == str(software.resolve() / "tritonswmm")


def test_a_directory_without_cfg_system_yaml_exits_2_with_the_reason(tmp_path):
    """Property: a directory that holds no `cfg_system.yaml` exits 2 and says it is not a reprex bundle.

    Class: an existing empty directory. Class NEW CAPABILITY; the plausible wrong implementation
    has no check, so the helper's bare `FileNotFoundError` surfaces under another exit code and
    another message; red here. Why required: the user gets the argument and a fix hint.
    What kills it: the `cfg_system.yaml` check removed. A second correct implementation: catch
    the helper's `FileNotFoundError` and re-raise it as `CLIValidationError`.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _invoke("--bundle", str(empty))
    assert result.exit_code == 2, result.output
    assert "not a reprex bundle directory" in _flat(result)
    assert not (empty / "system_config.yaml").exists()


def test_a_file_or_a_missing_path_is_refused_by_the_option_check(tmp_path):
    """Property: `--bundle` must be an existing directory; a file and a missing path exit non-zero with Typer's reason.

    Class: a file path and a nonexistent path. Class NEW CAPABILITY; the plausible wrong
    implementation lets either through to the `cfg_system.yaml` check, which refuses with a
    different message; red here. What kills it: `file_okay=True` or `exists=False` on the option.
    A second correct implementation: test `bundle.is_dir()` in the body and raise the same text.
    """
    a_file = tmp_path / "bundle.zip"
    a_file.write_text("x")
    file_result = _invoke("--bundle", str(a_file))
    assert file_result.exit_code == 2
    assert "is a file" in _flat(file_result)
    missing_result = _invoke("--bundle", str(tmp_path / "nope"))
    assert missing_result.exit_code == 2
    assert "does not exist" in _flat(missing_result)


def _flat_unboxed(result):
    """The output with the box characters Typer draws around an option error removed and wraps collapsed."""
    return " ".join(result.output.replace("│", " ").split())


def test_output_naming_an_existing_directory_is_refused(tmp_path):
    """Property: `--output` rejects an existing directory with exit 2 and Typer's reason, and writes nothing.

    Class: an `--output` value that is an existing directory. Class NEW CAPABILITY; the plausible
    wrong implementation declares the option with `dir_okay=True`, so the helper is asked to
    write a file at a directory path; red here.
    Why required: a directory passed where a file is asked is an ordinary mistake, and the
    refusal names it before anything is written.
    What kills it: `dir_okay=True` on `--output`. A second correct implementation: test
    `output.is_dir()` in the body and raise the same refusal.
    """
    bundle = _bundle(tmp_path / "bundle")
    result = _invoke("--bundle", str(bundle), "--output", str(tmp_path))
    assert result.exit_code == 2, result.output
    assert "is a directory" in _flat_unboxed(result)
    assert not (bundle / "system_config.yaml").exists()


def test_software_dir_naming_an_existing_file_is_refused(tmp_path):
    """Property: `--software-dir` rejects an existing file with exit 2 and Typer's reason, and writes nothing.

    Class: a `--software-dir` value that is an existing regular file. Class NEW CAPABILITY; the
    plausible wrong implementation declares the option with `file_okay=True`, so the command
    writes a config whose build directories sit under a regular file and exits 0; red here.
    Why required: the build directories are created under this path at run time.
    What kills it: `file_okay=True` on `--software-dir`. A second correct implementation: test
    `software_dir.is_file()` in the body and raise the same refusal.
    """
    bundle = _bundle(tmp_path / "bundle")
    a_file = tmp_path / "builds.txt"
    a_file.write_text("x")
    result = _invoke("--bundle", str(bundle), "--software-dir", str(a_file))
    assert result.exit_code == 2, result.output
    assert "is a file" in _flat_unboxed(result)
    assert not (bundle / "system_config.yaml").exists()
