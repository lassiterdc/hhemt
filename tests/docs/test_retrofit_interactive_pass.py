"""Tests for the interactive Python tutorial notebook and the committed example result that the notebook reads.

The tutorial `docs/tutorials/interactive-python.ipynb` is a notebook page. The
plugin `mkdocs-jupyter` renders the notebook and never executes it, so these tests are the only reader of the
notebook before a reader of the site.

Properties, each with its own control that turns it red on a planted defect:

- The committed store `docs/tutorials/assets/irene-example-result.zarr` holds the variables and dimensions the
  notebook reads, stays under 5,000,000 bytes (unit: bytes; matcher: sum of the file sizes under the store
  directory), carries a provenance attribute free of an absolute path and a host name, and is tracked by git
  (the repository ignores `*.zarr`, so an untracked store would be missing from a clone and the build would pass).
- The notebook is valid nbformat, opens with the exact title, and carries either no saved output or output free of
  a cluster path and a host name, so the test passes before and after the notebook is executed.
- The markdown cells face the matchers the pages face: the plugin renders them as page text, and a notebook is not
  a tracked `.md` page, so `tests/docs/test_docs_style.py` never reads them.
- `mkdocs.yml` sets `execute: false` for the plugin, and the plugin sits in the docs extra only.
- The trim script writes a store that passes the same checks from a synthetic source.
"""

from __future__ import annotations

import importlib.util
import json
import re
import socket
import subprocess
import tomllib
from pathlib import Path

import nbformat
import numpy as np
import pytest
import xarray as xr
import yaml

from tests.docs.test_docs_style import check_style

pytestmark = pytest.mark.docs

_REPO = Path(__file__).resolve().parents[2]
STORE_REL = "docs/tutorials/assets/irene-example-result.zarr"
STORE = _REPO / STORE_REL
NOTEBOOK = _REPO / "docs/tutorials/interactive-python.ipynb"
NOTEBOOK_PAGE = "docs/tutorials/interactive-python.ipynb"
TITLE = "# Running a simulation interactively in Python"
CAP_BYTES = 5_000_000
IRENE = "year.9_event_type.compound_event_id.1"

# Group -> (variable, dimensions) the notebook reads.
READ_SET = {
    "tritonswmm/swmm_link": ("max_flow", ("event_iloc", "link_id")),
    "swmm_only/swmm_link": ("max_flow", ("event_iloc", "link_id")),
    "tritonswmm/triton": ("max_flood_depth", ("event_iloc", "y", "x")),
    "triton_only/triton": ("max_flood_depth", ("event_iloc", "y", "x")),
}
_READ_CALL = re.compile(r'tree\["([^"]+)"\]\.to_dataset\(\)\["([^"]+)"\]')
_ABSOLUTE_PATH = re.compile(r"(?<![\w.:/-])/(?!/)[\w.-]+/[\w./-]+|[A-Za-z]:\\")
_HOST_NAME = re.compile(r"\b[\w-]+\.(?:edu|gov|local|internal|lan)\b", re.IGNORECASE)
_PRIVATE_LOCATION = re.compile(r"/scratch|/home|/Users|/tmp|/project|/sfs|/standard|/gpfs", re.IGNORECASE)
# Planted defects are assembled at run time, so no literal absolute path of a person sits in this file (the
# anonymization guard reads every tracked file, tests included).
_PLANTED_HOME = "/" + "home" + "/someone/analysis"
_PLANTED_SCRATCH = "/" + "scratch" + "/someone/run/store"


# --- the store ---------------------------------------------------------------------------------------------


def store_bytes(store: Path) -> int:
    """Sum of the sizes in bytes of the files under `store`."""
    return sum(p.stat().st_size for p in store.rglob("*") if p.is_file())


def host_leaks(text: str) -> list[str]:
    """Host names in `text`: the name of this machine and any name under a network domain."""
    found = _HOST_NAME.findall(text)
    host = socket.gethostname()
    if host and host.lower() in text.lower():
        found.append(host)
    return found


def leaks(text: str) -> list[str]:
    """Absolute paths and host names in `text`."""
    found = _ABSOLUTE_PATH.findall(text) + _HOST_NAME.findall(text)
    host = socket.gethostname()
    if host and host.lower() in text.lower():
        found.append(host)
    return found


def store_problems(store: Path, cap: int = CAP_BYTES) -> list[str]:
    """What is wrong with the store: size, groups, variables, dimensions, event, provenance."""
    if not store.is_dir():
        return [f"the store {store.name} does not exist"]
    problems: list[str] = []
    size = store_bytes(store)
    if size >= cap:
        problems.append(f"the store is {size} bytes, not under {cap}")
    tree = xr.open_datatree(store, engine="zarr", consolidated=False)
    for group, (variable, dims) in READ_SET.items():
        if f"/{group}" not in tree.groups:
            problems.append(f"group {group} is missing")
            continue
        ds = tree[group].to_dataset()
        if variable not in ds.data_vars:
            problems.append(f"{group} lacks the variable {variable}")
            continue
        if tuple(ds[variable].dims) != dims:
            problems.append(f"{group}/{variable} has dimensions {ds[variable].dims}, the notebook reads {dims}")
        if ds.sizes.get("event_iloc") != 1:
            problems.append(f"{group} holds {ds.sizes.get('event_iloc')} events, expected one")
        elif str(ds["event_id"].values[0]) != IRENE:
            problems.append(f"{group} holds the event {ds['event_id'].values[0]}, expected {IRENE}")
        for dim in dims[1:]:
            if dim not in ds.coords:
                problems.append(f"{group} has no coordinate for {dim}")
    provenance = tree.attrs.get("provenance")
    if not isinstance(provenance, str):
        problems.append("the root attribute provenance is missing")
        return problems
    record = json.loads(provenance)
    for key in ("source_analysis_id", "event_id", "variables", "hhemt_producing_sha"):
        if key not in record:
            problems.append(f"provenance lacks {key}")
    problems += [f"provenance holds {leak}" for leak in leaks(provenance)]
    return problems


def synthetic_group(variable: str, dims: tuple[str, ...], events: int, with_sha: bool) -> xr.Dataset:
    """A small stand-in for one group of the analysis store, shaped like the real groups."""
    sizes = {"event_iloc": events, "link_id": 5, "y": 4, "x": 3}
    shape = tuple(sizes[d] for d in dims)
    ids = np.array([f"other_{i}" for i in range(events)], dtype=object)
    ids[min(2, events - 1)] = IRENE
    coords: dict = {"event_iloc": ("event_iloc", np.arange(events)), "event_id": ("event_iloc", ids)}
    if with_sha:
        coords["hhemt_producing_sha"] = ("event_iloc", np.array(["abc123def456"] * events, dtype=object))
        coords["hhemt_producing_version"] = ("event_iloc", np.array(["9.9.9+x"] * events, dtype=object))
    for dim in dims[1:]:
        coords[dim] = (
            dim,
            np.arange(sizes[dim]) if dim != "link_id" else np.array([f"n{i}" for i in range(5)], dtype=object),
        )
    data = np.arange(int(np.prod(shape)), dtype="float64").reshape(shape)
    return xr.Dataset(
        {variable: (dims, data), "other_variable": (dims, data * 2)},
        coords=coords,
    )


def write_synthetic_source(path: Path, events: int = 4) -> Path:
    """A source store with the four groups the script reads, an extra variable per group and no root metadata."""
    for group, (variable, dims) in READ_SET.items():
        ds = synthetic_group(variable, dims, events, with_sha=group.startswith("swmm_only"))
        ds.to_zarr(path, group=group, mode="a", consolidated=False)
    return path


def write_synthetic_store(path: Path, provenance: str | None = None) -> Path:
    """A store that satisfies `store_problems`, with `provenance` as the root attribute."""
    record = {
        "source_analysis_id": "norfolk_doc_observed_events",
        "event_id": IRENE,
        "variables": {g: [v] for g, (v, _) in READ_SET.items()},
        "hhemt_producing_sha": "abc123def456",
    }
    groups = {}
    for group, (variable, dims) in READ_SET.items():
        ds = synthetic_group(variable, dims, 1, with_sha=False)
        ds = ds[[variable]].drop_vars([c for c in ds.coords if c not in {*dims, "event_id"}])
        ds["event_id"] = ("event_iloc", np.array([IRENE], dtype=object))
        groups[f"/{group}"] = ds
    tree = xr.DataTree.from_dict(groups)
    tree.attrs["provenance"] = provenance if provenance is not None else json.dumps(record, sort_keys=True)
    tree.to_zarr(path, mode="w", consolidated=False)
    return path


def test_committed_store_holds_what_the_notebook_reads():
    assert store_problems(STORE) == []


def test_committed_store_is_tracked_by_git():
    """The repository ignores `*.zarr`, so the store needs a forced add or a negation line to reach a clone."""
    listed = subprocess.run(
        ["git", "-C", str(_REPO), "ls-files", "--", f"{STORE_REL}/zarr.json"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert listed == [f"{STORE_REL}/zarr.json"], "the root metadata of the store is not tracked"


def test_notebook_reads_exactly_the_variables_the_store_must_hold():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert set(_READ_CALL.findall(code)) == {(group, variable) for group, (variable, _) in READ_SET.items()}


def test_control_a_good_store_passes_and_each_planted_defect_is_found(tmp_path):
    good = write_synthetic_store(tmp_path / "good.zarr")
    assert store_problems(good) == []
    oversize = write_synthetic_store(tmp_path / "oversize.zarr")
    (oversize / "filler.bin").write_bytes(b"0" * CAP_BYTES)
    assert any("not under" in p for p in store_problems(oversize))
    pathy = write_synthetic_store(tmp_path / "pathy.zarr", provenance=json.dumps({"source": _PLANTED_HOME}))
    assert any("provenance holds" in p for p in store_problems(pathy))
    hosty = write_synthetic_store(tmp_path / "hosty.zarr", provenance=json.dumps({"host": "login1.cluster.edu"}))
    assert any("provenance holds" in p for p in store_problems(hosty))
    assert any("does not exist" in p for p in store_problems(tmp_path / "absent.zarr"))


def test_control_a_store_without_a_read_variable_or_with_two_events_is_found(tmp_path):
    source = write_synthetic_source(tmp_path / "source.zarr")
    problems = store_problems(source)
    assert any("holds 4 events" in p for p in problems)
    assert any("provenance is missing" in p for p in problems)
    no_variable = write_synthetic_store(tmp_path / "no_variable.zarr")
    ds = (
        xr.open_zarr(no_variable, group="triton_only/triton", consolidated=False)
        .load()
        .rename({"max_flood_depth": "other_name"})
    )
    ds.to_zarr(no_variable, group="triton_only/triton", mode="w", consolidated=False)
    assert any("lacks the variable max_flood_depth" in p for p in store_problems(no_variable))


# --- the trim script ---------------------------------------------------------------------------------------


def _load_trim_script():
    spec = importlib.util.spec_from_file_location(
        "trim_irene_example_result", _REPO / "scripts/trim_irene_example_result.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trim_script_writes_a_store_that_passes_the_same_checks(tmp_path):
    script = _load_trim_script()
    source = write_synthetic_source(tmp_path / "source.zarr")
    destination = tmp_path / "trimmed.zarr"
    assert script.main([str(source), str(destination)]) == 0
    assert store_problems(destination) == []
    tree = xr.open_datatree(destination, engine="zarr", consolidated=False)
    record = json.loads(tree.attrs["provenance"])
    assert record["hhemt_producing_sha"] == "abc123def456"
    assert record["source_analysis_id"] == "norfolk_doc_observed_events"
    for group, (variable, _) in READ_SET.items():
        assert list(tree[group].to_dataset().data_vars) == [variable]
        assert "hhemt_producing_version" not in tree[group].to_dataset().variables


def test_control_trim_script_refuses_a_source_without_the_event(tmp_path):
    script = _load_trim_script()
    source = tmp_path / "source.zarr"
    for group, (variable, dims) in READ_SET.items():
        ds = synthetic_group(variable, dims, 2, with_sha=True)
        ds["event_id"] = ("event_iloc", np.array(["other_a", "other_b"], dtype=object))
        ds.to_zarr(source, group=group, mode="a", consolidated=False)
    with pytest.raises(SystemExit, match="holds no event"):
        script.main([str(source), str(tmp_path / "trimmed.zarr")])


# --- the notebook ------------------------------------------------------------------------------------------


def output_text(cell) -> str:
    """The text a saved output of a cell carries: streams, text representations and error traces."""
    parts: list[str] = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append("".join(out["text"]))
        elif out.get("output_type") == "error":
            parts += [out.get("ename", ""), out.get("evalue", ""), *out.get("traceback", [])]
        else:
            for mime, value in out.get("data", {}).items():
                if mime.startswith("image/"):
                    continue
                parts.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(parts)


def notebook_problems(nb) -> list[str]:
    """What is wrong with the notebook: validity, title, code that names a private location, saved output."""
    problems: list[str] = []
    try:
        nbformat.validate(nb)
    except nbformat.ValidationError as error:
        problems.append(f"not valid nbformat: {error.message}")
    if nb.nbformat != 4:
        problems.append(f"nbformat {nb.nbformat}, expected 4")
    first = nb.cells[0] if nb.cells else None
    if first is None or first.cell_type != "markdown" or first.source.splitlines()[0] != TITLE:
        problems.append("the first cell is not a markdown cell that opens with the exact title")
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            problems += [f"code cell {i} names {m}" for m in _PRIVATE_LOCATION.findall(cell.source)]
            problems += [f"code cell {i} holds {leak}" for leak in leaks(cell.source)]
        text = output_text(cell)
        problems += [f"saved output of cell {i} holds {m}" for m in _PRIVATE_LOCATION.findall(text)]
        # A saved output may print group paths such as `/swmm_only/swmm_node`, so the path check of an output reads
        # the private locations only; host names are read as everywhere else.
        problems += [f"saved output of cell {i} holds {leak}" for leak in host_leaks(text)]
    return problems


def markdown_page(nb) -> dict[str, str]:
    """The markdown cells as one page of text, keyed as the style test keys a page."""
    return {NOTEBOOK_PAGE: "\n\n".join(c.source for c in nb.cells if c.cell_type == "markdown")}


def style_problems(nb) -> list[str]:
    """Hits of every rule of the style test over the markdown cells, with no allowlist entry."""
    return check_style(markdown_page(nb), [])


def test_notebook_is_valid_and_titled_and_free_of_private_locations():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    assert notebook_problems(nb) == []


def test_the_title_line_occurs_once_under_the_tutorials():
    lines = [
        f"{p.relative_to(_REPO)}:{n}"
        for p in sorted((_REPO / "docs/tutorials").glob("*"))
        if p.is_file() and p.suffix in {".md", ".ipynb"}
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1)
        if TITLE in line
    ]
    assert len(lines) == 1, lines


def test_markdown_cells_face_the_page_style_rules():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    assert style_problems(nb) == []


def test_the_notebook_runs_no_simulation():
    """A code cell that calls a solver entry point would run a simulation at the one execution on a cluster."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert not re.search(r"\.run\(|compile_TRITON|\.test\(|NorfolkIreneExperiment|Toolkit", code)


def fresh_notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_control_a_changed_title_is_found():
    nb = fresh_notebook()
    nb.cells[0].source = nb.cells[0].source.replace(TITLE, "# Running a simulation in Python", 1)
    assert any("exact title" in p for p in notebook_problems(nb))
    nb = fresh_notebook()
    nb.cells.insert(0, nbformat.v4.new_code_cell("print(1)"))
    assert any("exact title" in p for p in notebook_problems(nb))


def test_control_a_saved_output_with_a_cluster_path_or_host_is_found_and_a_clean_output_is_not():
    nb = fresh_notebook()
    code_cell = next(c for c in nb.cells if c.cell_type == "code")
    code_cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="loaded 12 groups\n")]
    assert notebook_problems(nb) == []
    code_cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=f"opened {_PLANTED_SCRATCH}\n")]
    assert any("saved output" in p and "/scratch" in p for p in notebook_problems(nb))
    code_cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="host login1.cluster.edu\n")]
    assert any("saved output" in p for p in notebook_problems(nb))
    code_cell.outputs = [
        nbformat.v4.new_output("error", ename="OSError", evalue="no store", traceback=[f"File {_PLANTED_HOME}/x.py"])
    ]
    assert any("saved output" in p for p in notebook_problems(nb))


def test_control_an_absolute_path_in_a_code_cell_is_found():
    nb = fresh_notebook()
    next(c for c in nb.cells if c.cell_type == "code").source = f'store = "{_PLANTED_HOME}.zarr"'
    assert any("code cell" in p for p in notebook_problems(nb))


def test_control_a_colon_list_sentence_in_a_markdown_cell_is_found():
    nb = fresh_notebook()
    assert style_problems(nb) == []
    nb.cells[1].source += "\n\nThe options are: serial and hybrid."
    problems = style_problems(nb)
    assert any("rule colon" in p for p in problems), problems


def test_control_a_which_clause_and_a_number_word_in_a_markdown_cell_are_found():
    nb = fresh_notebook()
    nb.cells[0].source += " The result has two plots, which the cells draw."
    rules = {m.group(1) for p in style_problems(nb) if (m := re.search(r"rule (\w+)", p))}
    assert {"which", "number_word"} <= rules


# --- the plugin configuration ------------------------------------------------------------------------------


class _Loader(yaml.SafeLoader):
    """`mkdocs.yml` carries `!!python/name:` tags; the test reads the plugin block and needs no object."""


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def plugin_config(text: str) -> dict | None:
    """The mapping configured for `mkdocs-jupyter` in the plugin list, or `None` when the plugin is absent."""
    for entry in yaml.load(text, Loader=_Loader)["plugins"]:
        if isinstance(entry, dict) and "mkdocs-jupyter" in entry:
            return entry["mkdocs-jupyter"] or {}
    return None


def mkdocs_problems(text: str) -> list[str]:
    problems: list[str] = []
    config = plugin_config(text)
    if config is None:
        problems.append("the mkdocs-jupyter plugin is not configured")
    elif config.get("execute") is not False:
        problems.append(f"the plugin sets execute: {config.get('execute')!r}, expected false")
    nav = yaml.load(text, Loader=_Loader)["nav"]
    tutorials = next(item["Tutorials"] for item in nav if isinstance(item, dict) and "Tutorials" in item)
    targets = [v for item in tutorials if isinstance(item, dict) for v in item.values()]
    if "tutorials/interactive-python.ipynb" not in targets:
        problems.append("the Tutorials nav does not name the notebook")
    if any("norfolk-end-to-end" in str(t) for t in targets):
        problems.append("the Tutorials nav still names the deleted page")
    return problems


def test_mkdocs_executes_no_cell_at_build_and_navigates_to_the_notebook():
    assert mkdocs_problems((_REPO / "mkdocs.yml").read_text(encoding="utf-8")) == []


def test_plugin_is_in_the_docs_extra_only():
    project = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    assert any(re.match(r"mkdocs-jupyter\b", d) for d in extras["docs"])
    elsewhere = [d for d in project["dependencies"] if re.match(r"mkdocs-jupyter\b", d)]
    elsewhere += [
        d for name, deps in extras.items() if name != "docs" for d in deps if re.match(r"mkdocs-jupyter\b", d)
    ]
    assert elsewhere == []


def test_control_execute_true_a_missing_key_or_a_missing_plugin_is_found():
    text = (_REPO / "mkdocs.yml").read_text(encoding="utf-8")
    assert mkdocs_problems(text) == []
    assert any(
        "execute: True" in p
        for p in mkdocs_problems(text.replace("      execute: false\n", "      execute: true\n", 1))
    )
    assert any("execute: None" in p for p in mkdocs_problems(text.replace("      execute: false\n", "", 1)))
    assert any(
        "not configured" in p for p in mkdocs_problems(text.replace("- mkdocs-jupyter:", "- mkdocs-jupyter-absent:", 1))
    )
    assert any(
        "deleted page" in p
        for p in mkdocs_problems(
            text.replace("tutorials/interactive-python.ipynb", "tutorials/norfolk-end-to-end.md", 1)
        )
    )
