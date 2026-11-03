"""Reviewer tests for chunk 1a.3: the combine's zipped output and its Python entry point.

Each test builds small files under ``tmp_path`` with the standard library and the module under
test, so none compiles or runs a solver and all run on a laptop as
``PYTHONPATH=${PWD}/src pytest tests/test_combine_zip_output_reviewer.py``. The behaviours are new
in this chunk, so no test has a pre-change tree to be red on; each names the wrong implementation
it reds under and a second correct implementation it passes under.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import hhemt.bundle as bundle_pkg
from hhemt.bundle import _combine as C
from hhemt.version_migration.constants import BUNDLE_MANIFEST_FILENAME


def test_combine_command_called_from_python_passes_real_values_for_every_option(tmp_path: Path, monkeypatch) -> None:
    """Property: ``combine_command`` called as a function hands ``combine_bundle`` no typer placeholder.

    Input class: Python callers of ``combine_command`` that omit some options, so each omitted
    parameter holds its ``typer.Option(...)`` default object. Approved-shape row: section 5.1 row
    1a.3 ("``--unzipped`` and ``declare_solver_split`` passed on as ``value is True``", item N2)
    and section 5.7 1a test (12) ("``combine_command(...)`` called from Python without the new
    keywords writes a zip").

    Why required: an omitted parameter is a truthy ``OptionInfo`` object, not its default. The two
    boolean options are guarded with ``is True``; ``workdir`` is not, so ``combine_bundle`` receives
    an ``OptionInfo`` as a directory and fails inside ``Path`` or ``mkdtemp`` (measured: the call
    forwards ``workdir`` as ``OptionInfo``). The estate driver calls ``combine_command`` this way.

    Class: NEW CAPABILITY. Plausible wrong implementation: the shipped call, which forwards
    ``workdir=workdir`` unguarded; red under it. Reachability: the estate driver imports
    ``combine_command`` and calls it with a subset of keywords. Both-states anchor: the check is on
    the types of the forwarded values. Invocation not isolatable from this tree - discrimination
    shown instead: forwarding ``workdir if isinstance(workdir, Path) else None`` greens it. Second
    correct implementation: validating the options in a wrapper that Typer and Python both call.
    """
    seen: dict = {}

    def fake_combine(bundles, output_path=None, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(root=tmp_path / "out.zip")

    monkeypatch.setattr(bundle_pkg, "combine_bundle", fake_combine)
    from hhemt.cli import combine_command

    combine_command([tmp_path / "a", tmp_path / "b"], output=tmp_path / "out")
    assert seen["declare_solver_split"] is False and seen["unzipped"] is False
    assert seen["workdir"] is None or isinstance(seen["workdir"], Path)


def _bundle_dir(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / BUNDLE_MANIFEST_FILENAME).write_text("{}")
    (root / "cfg_analysis.yaml").write_text("analysis_id: exp\n")
    (root / "sims" / "0" / "out_triton" / "bin").mkdir(parents=True)
    (root / "sims" / "0" / "out_triton" / "bin" / "H.bin").write_bytes(b"raw")
    (root / "sims" / "0" / "out_triton" / "run.cfg").write_text("cfg\n")
    return root


def test_a_directory_member_zip_has_bundle_relative_entry_names(tmp_path: Path) -> None:
    """Property: a directory input's member zip names its entries relative to the bundle root.

    Input class: directory inputs of a zipped combine, whatever the open question on what else the
    member zip holds. Approved-shape row: section 5.1 row 1a.3 (``child_crates/{eid}.zip`` per
    member) and section 4A item 2 (1a.4's materializer reads each nested member as a bundle with
    its files at the zip root).

    Why required: 1a.4 extracts a nested member and opens it with ``Bundle.from_directory``, which
    needs ``bundle_manifest.json`` at the top of what was extracted; a member zip whose entries
    carry a leading folder name would extract one level too deep. The test deliberately does not say
    whether raw ``out_*/bin`` files are present, because the row does not decide that for a
    directory input (open item for the developer).

    Class: NEW CAPABILITY, held green. Plausible wrong implementation: ``write_zip`` fed
    ``(path.name + "/" + name, ...)`` pairs; red under it. Reachability: any directory input of a
    zipped combine. Both-states anchor: the test reads the zip's entry list and compares with
    literals. Invocation not isolatable from this tree - discrimination shown instead: prefixing the
    names reds it. Second correct implementation: zipping the render set only.
    """
    bundle = _bundle_dir(tmp_path / "in" / "exp")
    (tmp_path / "work").mkdir()
    (zipped,) = C._member_zips([bundle], ["exp"], tmp_path / "work")
    with zipfile.ZipFile(zipped) as zf:
        names = set(zf.namelist())
    assert {BUNDLE_MANIFEST_FILENAME, "cfg_analysis.yaml", "sims/0/out_triton/run.cfg"} <= names
    assert not any(n.startswith("exp/") for n in names)


def test_the_schema_version_is_one_above_the_previous_generation() -> None:
    """Property: ``BUNDLE_SCHEMA_VERSION`` is 8, one above the 7 the combined layout started from.

    Input class: the single constant. Approved-shape row: section 5.1 row 1a.3
    ("``BUNDLE_SCHEMA_VERSION`` bumped by one with the bump's reason in its comment") and section 5.1
    gate facts ("bumped once, in 1a.3").

    Why required: the consume-side guard is an exact match, so a bump of two, or none, either strands
    every bundle emitted at 7 for no reason or lets a v7 reader open a v8 layout.

    Class: NEW CAPABILITY, held green. Plausible wrong implementation: 9, or 7 left in place; red
    under either. Reachability: the constant is read at import. Both-states anchor: the check
    compares with a literal and with the comment's own `v8` label. Invocation not isolatable from
    this tree - discrimination shown instead: setting the constant to 9 reds it. Second correct
    implementation: none; the value is fixed by the row.
    """
    import hhemt.version_migration.constants as K

    assert K.BUNDLE_SCHEMA_VERSION == 8
    assert "v8" in Path(K.__file__).read_text()
