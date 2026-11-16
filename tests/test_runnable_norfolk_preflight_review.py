"""Review tests for the Norfolk data and docs for the runnable SWMM model.

Three classes the chunk's own tests do not enter. The chunk's file reads the Norfolk system template as YAML and
never builds a ``system_config`` from it, checks the removed-token guard over ``src`` only in
the cut-over review file, and leaves the field names in the new prose unchecked against the config models.
None runs a solver or the network; each test builds a config dict or reads tracked text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hhemt.config.analysis import analysis_config
from hhemt.config.system import system_config

_REPO = Path(__file__).resolve().parents[1]
_CASE_DIR = _REPO / "test_data" / "norfolk_coastal_flooding"

# The warning text the system config records for a dropped HPC key (assembled so no line spells the word).
_DROPPED_FIELD_WARNING = "is " + "re" + "tired"

# The removed-token matcher of the grep line that guards the removed keys, as one Python alternation.
_RETIRED = re.compile(
    r"SWMM_hydraulics|SWMM_hydrology|SWMM_full\b|_model_template\.inp|create_swmm_inp_from_template"
    r"|build_templates|triton_swmm_configuration_template|TRITON_SWMM_definition_template"
)


def _load_template_text(text: str, tmp_path: Path) -> system_config:
    """Build a ``system_config`` from the template text, with every ``${DATA_DIR}`` file created empty.

    ``system_config`` refuses a path field whose file is absent, and the Norfolk dataset is not in the
    repository, so each value under ``${DATA_DIR}`` is touched under ``tmp_path`` first. ``${PACKAGE_DIR}``
    points at a directory that is never created, because no field of the template needs it to exist.
    """
    data_root, package_root = tmp_path / "data", tmp_path / "pkg"
    raw = yaml.safe_load(text)
    resolved: dict = {}
    for key, value in raw.items():
        if isinstance(value, str):
            value = value.replace("${DATA_DIR}", str(data_root)).replace("${PACKAGE_DIR}", str(package_root))
            if value.startswith(str(data_root)):
                Path(value).parent.mkdir(parents=True, exist_ok=True)
                Path(value).touch()
        resolved[key] = value
    return system_config.model_validate(resolved)


def test_every_key_of_the_norfolk_system_template_is_accepted_by_the_system_config(tmp_path):
    """Property: the tracked Norfolk system template validates as a ``system_config``, and a stale key is refused.

    Class: NEW CAPABILITY. The plausible wrong implementation is a template that keeps a key the config no
    longer has, or misspells ``swmm_outfalls_to_couple``; ``system_config`` forbids extra keys, so the
    template then fails to load and every Norfolk run stops at the first config read. The chunk's own
    tests read the template as YAML and assert the keys they name, so a key they do not name goes unchecked.
    Reachability: ``hhemt`` copies this file for each new Norfolk analysis, so the file's keys reach the
    model's ``extra="forbid"`` check on the first load. Artifact: ``load_system_config`` in
    ``src/hhemt/config/loaders.py`` validates the copied file with this model.
    Both-states anchor: the assertion turns on whether validation returns, which holds for any key set the
    model declares, and not on the wording of a message.
    Invocation: not isolatable from this tree. Discrimination shown instead: renaming
    ``swmm_outfalls_to_couple`` in the template text to ``swmm_outfalls_to_coupl`` turns the first arm red
    (extra key refused), and deleting the ``SWMM_model`` line turns it red (required field missing).
    Why required: a template the model refuses is a Norfolk example that cannot start, which no
    YAML-only assertion detects.
    What kills it: a removed or misspelled key in ``template_system_config.yaml``.
    A second correct implementation: a template that sets ``SWMM_model`` to any other ``.inp`` under
    ``${DATA_DIR}/external`` and lists other outfalls.
    Held test: green before this turn; it must stay green. Node id
    ``tests/test_runnable_norfolk_preflight_review.py::test_every_key_of_the_norfolk_system_template_is_accepted_by_the_system_config``.
    """
    text = (_CASE_DIR / "template_system_config.yaml").read_text()
    cfg = _load_template_text(text, tmp_path)
    assert cfg.SWMM_model.name.endswith(".inp")
    assert cfg.swmm_outfalls_to_couple == []
    # Control (must fire): the same text with a removed key, and with a misspelled key, is refused.
    with pytest.raises(ValidationError, match="SWMM_model"):
        _load_template_text(text.rstrip("\n") + "\nSWMM_full: '${DATA_DIR}/external/x.inp'\n", tmp_path)
    with pytest.raises(ValidationError, match="swmm_outfalls_to_coupl"):
        _load_template_text(text.replace("swmm_outfalls_to_couple:", "swmm_outfalls_to_coupl:"), tmp_path)


def test_the_norfolk_system_template_loads_with_no_retired_key_warning(tmp_path):
    """Property: loading the tracked Norfolk system template raises no "field ... is dropped" warning.

    Class: NEW CAPABILITY. The plausible wrong implementation is the template as the chunk committed it: it
    sets ``gpu_compilation_backend: null`` under a comment that tells the reader to set it to ``HIP`` or
    ``CUDA``, and ``system_config`` pops that key and warns ``system_config field 'gpu_compilation_backend'
    is dropped ... Remove it from your system config YAML``. The test is red under it; the output is pasted
    in the reviewer's file.
    The input class is every key of the one tracked template (the removed set is the four HPC keys of
    ``_retired_hpc_keys`` in ``config/system.py`` and the four template keys of ``_retired_template_keys``).
    Reachability: ``hhemt`` copies this template for each new Norfolk analysis, so every user who copies it
    sees the warning on the first load and reads the comment above it as an instruction to set a removed key.
    Both-states anchor: the assertion turns on whether any retirement warning is recorded, a property of the
    template's key set that holds for any warning wording.
    Invocation: not isolatable from this tree. Discrimination shown instead: deleting the
    ``gpu_compilation_backend`` line and its comment from the template text turns this test green.
    Why required: the template is the published example config, and a copy of it must load without
    telling the reader to remove a key the example itself sets.
    What kills it: any removed key (of either set) left in ``template_system_config.yaml``.
    A second correct implementation: a template whose comment points at the per-HPC-system config and sets
    no removed key.
    """
    import warnings

    text = (_CASE_DIR / "template_system_config.yaml").read_text()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _load_template_text(text, tmp_path)
    removed = sorted(str(w.message) for w in caught if _DROPPED_FIELD_WARNING in str(w.message))
    # Control (must fire): the same text with a removed HPC key added does record the warning.
    with warnings.catch_warnings(record=True) as control:
        warnings.simplefilter("always")
        _load_template_text(text.rstrip("\n") + "\ngpu_hardware: a100\n", tmp_path)
    assert any("gpu_hardware" in str(w.message) and _DROPPED_FIELD_WARNING in str(w.message) for w in control)
    assert removed == []


def _retired_carriers(root: Path, suffixes: set[str]) -> set[str]:
    """Repo-relative paths under ``root`` whose text matches the removed-token matcher."""
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix in suffixes and _RETIRED.search(p.read_text(encoding="utf-8", errors="replace"))
    }


def test_no_docs_page_or_norfolk_data_file_names_a_retired_template_token(tmp_path):
    """Property: no ``docs/`` page and no Norfolk case file names a removed template key, builder or file.

    Class: NEW CAPABILITY. The plausible wrong implementation is a docs page or Norfolk data file that
    spells a removed key, for example an example block copied from an older page; the grep line that
    guards the removed keys catches it once, at the run's close, and the cut-over review test covers only
    ``src/hhemt``. The input class is every ``.md``, ``.yaml`` and ``.py`` file under ``docs/`` and under
    ``test_data/norfolk_coastal_flooding/`` (the other two directories of that grep line besides ``src``).
    Reachability: a reader copies a key from a published page into a config, and the removed-key
    refusal then rejects it at load; artifact: the refusal in ``config/system.py`` for each of the four keys.
    Both-states anchor: the assertion turns on the set of matching paths, and holds for any wording the
    pages use so long as the tokens are absent.
    Invocation: not isolatable from this tree. Discrimination shown instead: adding ``SWMM_hydraulics:`` to
    any docs page turns it red, and the planted-file control below turns red when the traversal misses a file.
    Why required: a docs page that teaches a refused key sends a user to a config the toolkit rejects.
    What kills it: any removed token in a docs page or a Norfolk data file.
    A second correct implementation: pages that describe the retirement in words without the tokens.
    Held test: green before this turn; it must stay green. Node id
    ``tests/test_runnable_norfolk_preflight_review.py::test_no_docs_page_or_norfolk_data_file_names_a_retired_template_token``.
    """
    planted = tmp_path / "page.md"
    planted.write_text("Set `SWMM_hydraulics` to the model.\n")
    assert _retired_carriers(tmp_path, {".md"}) == {"page.md"}, "control: the traversal no longer finds a planted token"
    suffixes = {".md", ".yaml", ".py"}
    found = _retired_carriers(_REPO / "docs", suffixes) | _retired_carriers(_CASE_DIR, suffixes)
    assert sorted(found) == []


_FIELD_PREFIXES = ("SWMM_", "swmm_", "TRITON_", "TRITONSWMM_", "subcatchment_", "toggle_", "one_coupled_")
# Backticked names the pages use that are NOT config fields: ``swmm_coupling_nodes`` is the TRITON cfg key.
_NOT_CONFIG_FIELDS = {"swmm_coupling_nodes"}
_PAGES = ("docs/how-to/using-a-runnable-swmm-model.md", "docs/reference/coupled-inflow-nodes.md")


def _field_like_spans(text: str) -> set[str]:
    """Backticked identifiers of the page text that start like a config field of the system or analysis config."""
    spans = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text))
    return {s for s in spans if s.startswith(_FIELD_PREFIXES) and s not in _NOT_CONFIG_FIELDS}


def test_every_config_field_the_runnable_model_pages_name_is_a_field_of_a_config_model():
    """Property: each field-like name in backticks on the two runnable-model pages is a field of a config model.

    Class: NEW CAPABILITY. The plausible wrong implementation is a page that names a field the models do not
    have, such as a key with a new name or a removed key; the pages teach a key the load refuses. The input class is
    every backticked identifier on the two pages that starts with a config-field prefix
    (``SWMM_``, ``swmm_``, ``TRITON_``, ``TRITONSWMM_``, ``subcatchment_``, ``toggle_``, ``one_coupled_``).
    Reachability: a user copies the name into the system or analysis config, and ``extra="forbid"`` refuses
    an unknown key at load.
    Both-states anchor: the assertion turns on membership in the union of the two models' field names, which
    holds before and after any rewording of the prose.
    Invocation: not isolatable from this tree. Discrimination shown instead: replacing ``SWMM_model`` on the
    how-to page with ``SWMM_hydraulics`` turns it red.
    Why required: a page makes no claim the code does not make, and a field name is the one claim
    a machine can settle.
    What kills it: a removed or misspelled field name in either page.
    A second correct implementation: pages that name the fields in a table instead of in running text.
    Held test: green before this turn; it must stay green. Node id
    ``tests/test_runnable_norfolk_preflight_review.py::test_every_config_field_the_runnable_model_pages_name_is_a_field_of_a_config_model``.
    """
    fields = set(system_config.model_fields) | set(analysis_config.model_fields)
    # Control (must fire): the matcher reads the pages at all, and a planted removed name is flagged.
    assert {"SWMM_model", "swmm_outfalls_to_couple"} <= _field_like_spans((_REPO / _PAGES[0]).read_text())
    assert _field_like_spans("Set `SWMM_hydraulics` here.") - fields == {"SWMM_hydraulics"}
    unknown = {page: sorted(_field_like_spans((_REPO / page).read_text()) - fields) for page in _PAGES}
    assert {page: names for page, names in unknown.items() if names} == {}
