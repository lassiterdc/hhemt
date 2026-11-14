"""The analysis front page prints only counts it has a source for and names only what exists.

The front page is a template filled from a ``config`` block that each route writes: the
workflow, a reprocess, and a bundle regenerate. A count with no source on a route must
be left off the page, never printed as zero, and the template must read only keys that a
route writes.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import jinja2
import pytest

from hhemt.bundle.snakefile_generator import generate_regeneration_snakefile
from hhemt.report_renderers._reporting_sets import _STANDARD_CATEGORY_ORDER

TESTS = Path(__file__).resolve().parent
TEMPLATE = TESTS.parent / "src" / "hhemt" / "report_templates" / "workflow_description.rst.j2"
BUNDLES = TESTS / "fixtures" / "bundles"

# Keys the analysis-level workflow writes besides the ones a bundle block writes.
_BASE = {"analysis_id": "a", "toolkit_version": "0.1.0+1.gabc", "report": {"generated_at": "t"}}


def _render(config: dict, template: str | None = None) -> str:
    text = template if template is not None else TEMPLATE.read_text(encoding="utf-8")
    return jinja2.Environment().from_string(text).render(snakemake=SimpleNamespace(config=config))


def _template_keys(text: str | None = None) -> set[str]:
    text = text if text is not None else TEMPLATE.read_text(encoding="utf-8")
    pattern = r"snakemake\.config\[['\"](\w+)['\"]\]|snakemake\.config\.get\(['\"](\w+)['\"]"
    return {a or b for a, b in re.findall(pattern, text)}


def _bundle_config(tmp_path: Path, fixture: str, csv: str | None) -> dict:
    """The ``config`` dict the regenerated Snakefile of a bundle builds, by running its assignments."""
    bundle = tmp_path / fixture
    shutil.copytree(BUNDLES / fixture, bundle)
    if csv is not None:
        (bundle / "scenario_status.csv").write_text(csv)
    text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    lines = [ln for ln in text.splitlines() if ln.startswith("config[")]
    config: dict = {}
    scope = {"config": config, "_toolkit_version": "0.1.0+1.gabc", "_dt": __import__("datetime").datetime}
    exec("\n".join(lines), scope)  # noqa: S102 -- assignments emitted by the generator under test
    return config


_PLAIN_CSV = "event_iloc,model_type,run_completed\n" + "".join(
    f"{e},tritonswmm,True\n{e},swmm,True\n" for e in range(4)
)
_SWEEP_CSV = "member_id,event_iloc,model_type,run_completed\n" + "".join(
    f"{m},{e},tritonswmm,True\n" for m in (1, 2, 3) for e in range(2)
)


def test_a_bundle_counts_scenarios_from_its_scenario_status_csv(tmp_path: Path) -> None:
    """Four events run by two models are 4 scenarios, not 8 rows and not the configuration's absent field."""
    config = _bundle_config(tmp_path, "multi_sim", _PLAIN_CSV)
    assert config["n_sims"] == 4
    assert "The analysis covers 4 scenarios." in _render(config)


def test_a_sweep_bundle_counts_members_and_scenarios_from_its_csv(tmp_path: Path) -> None:
    config = _bundle_config(tmp_path, "sensitivity_master", _SWEEP_CSV)
    assert (config["n_sims"], config["n_members"]) == (6, 3)


@pytest.mark.parametrize("fixture", ["multi_sim", "sensitivity_master"])
def test_a_bundle_without_the_csv_prints_no_scenario_count(tmp_path: Path, fixture: str) -> None:
    """Must-fire: the base wrote ``config["n_sims"] = 0`` and the page read "comprising 0 scenarios"."""
    config = _bundle_config(tmp_path, fixture, None)
    page = _render(config)
    assert config["n_sims"] is None
    assert not re.search(r"\b0 scenarios?\b", page)
    assert "scenario" not in page


def test_one_scenario_is_singular() -> None:
    assert "The analysis covers 1 scenario." in _render({**_BASE, "is_sensitivity": False, "n_sims": 1})


def test_the_sweep_sentence_needs_every_value_it_prints() -> None:
    sweep = {**_BASE, "is_sensitivity": True, "n_sims": 6, "n_members": 3, "independent_vars": ["run_mode"]}
    assert "across 3 members" in _render(sweep)
    for missing in ({"n_members": None}, {"independent_vars": []}):
        assert "sweep" not in _render({**sweep, **missing})


def test_the_template_reads_only_keys_a_bundle_writes(tmp_path: Path) -> None:
    """A key the template reads and no route writes is a block that never renders."""
    written = set(_bundle_config(tmp_path, "sensitivity_master", _SWEEP_CSV)) | {"group_by_var"}
    assert sorted(_template_keys() - written) == []


def test_a_template_block_reading_an_unwritten_key_is_caught() -> None:
    mutated = (
        TEMPLATE.read_text(encoding="utf-8") + "{% if snakemake.config.get('rule_completion_summary') %}x{% endif %}"
    )
    assert "rule_completion_summary" in _template_keys(mutated)


def test_bold_names_in_the_template_are_labels_or_real_sidebar_categories() -> None:
    """The page may point a reader only to a sidebar category the report has."""
    text = TEMPLATE.read_text(encoding="utf-8")
    bold = set(re.findall(r"\*\*([^*\n]+?)\*\*", text))
    assert sorted(bold - set(_STANDARD_CATEGORY_ORDER) - {"Generated:", "hhemt build:"}) == []


def test_a_sweep_bundle_reads_its_varied_columns_from_the_report_configuration(tmp_path: Path) -> None:
    """The sweep sentence names the columns the sweep varies, so a bundle must carry them to the page.

    The bundled analysis configuration holds them under ``report.sensitivity``; a block that
    reads another key prints no sweep sentence on a regenerated sweep page.
    """
    import yaml

    bundle = tmp_path / "sensitivity_master"
    shutil.copytree(BUNDLES / "sensitivity_master", bundle)
    cfg_path = bundle / "cfg_analysis.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.setdefault("report", {})["sensitivity"] = {"independent_vars": ["run_mode"], "group_by_var": "run_mode"}
    cfg_path.write_text(yaml.safe_dump(cfg))
    (bundle / "scenario_status.csv").write_text(_SWEEP_CSV)
    text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    config: dict = {}
    scope = {"config": config, "_toolkit_version": "v", "_dt": __import__("datetime").datetime}
    exec("\n".join(ln for ln in text.splitlines() if ln.startswith("config[")), scope)  # noqa: S102
    assert config["independent_vars"] == ["run_mode"]
    assert "The sweep varies run_mode" in _render(config)


# ---- The two sweep master emitters: no count when counting fails ----------------------------


class _Stop(Exception):
    """Raised by a stub just after the scenario count is decided, to read what was decided."""


class _Members(dict):
    pass


def _master_builder(cls, counting_fails: bool, n_events_per_member: int = 2):
    """A sweep master builder built without its constructor, stubbed to the point the count is decided."""
    from hhemt import workflow as wf

    class _Member:
        def __init__(self, n):
            self._n = n

        @property
        def df_sims(self):
            if counting_fails:
                raise RuntimeError("cannot enumerate scenarios")
            return list(range(self._n))

    builder = object.__new__(cls)
    members = _Members({str(i): _Member(n_events_per_member) for i in range(3)})
    cfg_analysis = SimpleNamespace(
        report=SimpleNamespace(sensitivity=None),
        process_in_sim_rule=False,
        analysis_id="a",
    )
    builder.sensitivity_analysis = SimpleNamespace(members=members, _materialize_target_yamls=lambda: None)
    builder.experiment = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=Path("."), analysis_log_directory=Path(".")),
        analysis_config_yaml=Path("a.yaml"),
        cfg_analysis=cfg_analysis,
    )
    builder.system = SimpleNamespace(
        cfg_system=SimpleNamespace(toggle_triton_model=False, toggle_tritonswmm_model=True, toggle_swmm_model=False)
    )
    builder._base_builder = SimpleNamespace(
        _get_conda_env_path=lambda: Path("env.yaml"), _get_config_args=lambda **k: ""
    )

    def _stop(*args, **kwargs):
        raise _Stop

    builder._enumerate_member_event_pairs = _stop
    return builder, wf


def _decided_counts(cls, method: str, counting_fails: bool, monkeypatch) -> dict:
    """The emitter's scenario-count locals, read from its frame when the stub stops it."""
    builder, wf = _master_builder(cls, counting_fails)
    monkeypatch.setattr(wf, "_emit_report_artifacts", lambda *a, **k: None)
    with pytest.raises(_Stop) as caught:
        getattr(builder, method)()
    tb = caught.value.__traceback__
    while tb.tb_next is not None:
        tb = tb.tb_next
    frame = tb.tb_frame.f_back
    while frame is not None and "total_n_sims" not in frame.f_locals:
        frame = frame.f_back
    assert frame is not None, "the emitter never decided a scenario count"
    return dict(frame.f_locals)


_MASTER_SITES = [
    ("SensitivityAnalysisWorkflowBuilder", "generate_master_snakefile_content"),
    ("SensitivityAnalysisWorkflowBuilder", "generate_reprocess_master_snakefile_content"),
]


@pytest.mark.parametrize(("cls_name", "method"), _MASTER_SITES)
def test_a_master_writes_the_scenario_count_when_counting_works(cls_name, method, monkeypatch) -> None:
    from hhemt import workflow as wf

    decided = _decided_counts(getattr(wf, cls_name), method, False, monkeypatch)
    assert decided["total_n_sims"] == 6
    assert decided.get("_n_sims_config") == 'config["n_sims"] = 6\n'


@pytest.mark.parametrize(("cls_name", "method"), _MASTER_SITES)
def test_a_master_writes_no_scenario_count_when_counting_fails(cls_name, method, monkeypatch) -> None:
    """Must-fire: the old fallback wrote the member count (3) as the scenario count."""
    from hhemt import workflow as wf

    decided = _decided_counts(getattr(wf, cls_name), method, True, monkeypatch)
    assert decided["total_n_sims"] is None
    assert decided.get("_n_sims_config") == ""
