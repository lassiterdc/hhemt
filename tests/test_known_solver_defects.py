"""The every-coupled-run solver-defect check: which builds it flags, and how it scopes US units.

The problem this file pins. Some solver defects affect every coupled run a build produces, not
only resumed ones, so they cannot be reported by the resume-defect check, which returns before
it reads any trigger when no sim was resumed. ``check_known_solver_defects`` resolves the
registry's ``coupled`` entries against the analysis's producing TRITON sha, and scopes the
US-unit rim entry per scenario from that scenario's prepared ``hydraulics.inp``, so a CMS model is
never told it carries a US-unit defect. Every test here builds a fake analysis and writes its
``.inp`` fixtures as text; none compiles or runs a solver, so the file runs on a laptop.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

_PRE_FIX = "c7b2fd1da3c2ed48d0d8cc0bf0e4b703b1f86e5e"
_PIN = "320b3313c99c3e929e77a258ff6e4b041bae2927"


def _analysis(tmp_path, units: dict, *, coupled=True):
    """A fake analysis with one scenario per entry of ``units``: a FLOW_UNITS word, or None for no file."""
    from hhemt.scenario import compute_event_id_slug

    sim_dir = tmp_path / "sims"
    for i, unit in units.items():
        if unit is None:
            continue
        inp = sim_dir / compute_event_id_slug({"event_id": i}) / "swmm" / "hydraulics.inp"
        inp.parent.mkdir(parents=True)
        inp.write_text(f"[OPTIONS]\nFLOW_UNITS {unit}\n")
    return SimpleNamespace(
        _system=SimpleNamespace(cfg_system=SimpleNamespace(toggle_tritonswmm_model=coupled)),
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False),
        analysis_paths=SimpleNamespace(simulation_directory=sim_dir),
        df_sims=pd.DataFrame(index=list(units)),
        _retrieve_weather_indexer_using_integer_index=lambda i: {"event_id": i},
    )


def _run(monkeypatch, analysis, sha):
    """Run the check with ``sha`` as the analysis's recorded producing TRITON sha."""
    import hhemt.analysis_validation as av

    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: sha)
    return av.check_known_solver_defects(analysis)


def test_a_pre_fix_build_reports_its_coupled_defects_and_scopes_us_units_per_scenario(tmp_path, monkeypatch):
    """Property 80, first case: at ``c7b2fd1d`` (a) and (c) are present, (b) only for the US scenario.

    Class: non-resumed coupled analyses at a build before T7a, with one US, one CMS and one
    reclaimed hydraulics model. Why: these defects affect every coupled run, resumed or not, and a
    US-unit defect must not be reported for a CMS model or invented for a missing file. What kills
    it: a resume gate, reading the unit for every model regardless of the build, or treating a
    missing file as US. Second correct implementation: counting per scenario first and resolving the
    sha afterwards; the assertions read the verdict, the defect ids and the per-unit counts.
    """
    res = _run(monkeypatch, _analysis(tmp_path, {0: "CFS", 1: "CMS", 2: None}), _PRE_FIX)
    assert res.passed is False
    for did in ("TRITON-COUPLED-RIM-BELOW-CROWN", "TRITON-COUPLED-LIST-MISALIGNED", "TRITON-COUPLED-US-UNIT-RIM"):
        assert did in res.summary
    assert "1 US, 1 not applicable, 1 not determinable" in res.summary
    us_rows = [r for r in res.details if r["detail"].startswith("TRITON-COUPLED-US-UNIT-RIM")]
    assert [r["scenario"] for r in us_rows] == ["event_id.0"]


def test_a_pre_fix_cms_only_analysis_does_not_carry_the_us_unit_defect(tmp_path, monkeypatch):
    """Property 80, control: at ``c7b2fd1d`` a CMS-only analysis carries (a) and (c) but not (b).

    Class: coupled analyses whose every hydraulics model is SI. What kills it: reporting (b) from the
    build verdict alone. Second correct implementation: filtering the scenarios before resolving.
    """
    res = _run(monkeypatch, _analysis(tmp_path, {0: "CMS"}), _PRE_FIX)
    assert res.passed is False
    assert "TRITON-COUPLED-US-UNIT-RIM" not in [r["detail"].split(":")[0] for r in res.details]
    assert "TRITON-COUPLED-US-UNIT-RIM:" in res.summary  # its counts are still disclosed
    assert "0 US, 1 not applicable, 0 not determinable" in res.summary


def test_the_package_3a_pin_passes_and_reads_no_hydraulics_file(tmp_path, monkeypatch):
    """Property 80: at ``320b3313`` the check passes, and (b) is absent even with the file missing.

    Class: coupled analyses at the pin. Why: the pin carries all three fixes, and a build without
    (b) must never report it not determinable. What kills it: reading the unit when the sha already
    resolves (b) absent. Second correct implementation: resolving every entry first and skipping
    the per-scenario pass when none is present.
    """
    res = _run(monkeypatch, _analysis(tmp_path, {0: None, 1: "CFS"}), _PIN)
    assert res.passed is True
    assert "not determinable" not in res.summary


_UNREGISTERED = "deadbeefcafef00ddeadbeefcafef00ddeadbeef"
_COUPLED_IDS = ("TRITON-COUPLED-RIM-BELOW-CROWN", "TRITON-COUPLED-US-UNIT-RIM", "TRITON-COUPLED-LIST-MISALIGNED")


def test_an_unregistered_pin_pass_discloses_that_nothing_was_assessed(tmp_path, monkeypatch):
    """Property 80: at a sha in no known set the pass names all three INDETERMINATE; at ``320b3313`` all three ABSENT.

    Class: passing coupled analyses whose producing sha the registry either cleared or has never
    heard of. Why: both pass, and a reader acts on the pass, so the summary must say which basis
    produced it; with the sha masked the two summaries must differ. What kills it: a summary that
    states only the present entries, which reads byte-identically at the two pins once the sha is
    masked. Second correct implementation: a per-entry verdict list in the summary; the assertions
    read the two counts, the named ids and the masked inequality.
    """
    unreg = _run(monkeypatch, _analysis(tmp_path / "u", {0: "CFS"}), _UNREGISTERED)
    pin = _run(monkeypatch, _analysis(tmp_path / "p", {0: "CFS"}), _PIN)
    assert unreg.passed is True and pin.passed is True
    assert "3 INDETERMINATE and therefore NOT assessed at this pin" in unreg.summary
    for did in _COUPLED_IDS:
        assert did in unreg.summary
    assert "assessed ABSENT" not in unreg.summary
    assert "3 assessed ABSENT" in pin.summary
    assert "INDETERMINATE" not in pin.summary
    assert unreg.summary.replace(_UNREGISTERED[:12], "{PIN}") != pin.summary.replace(_PIN[:12], "{PIN}")


def test_a_no_us_analysis_counts_the_us_unit_entry_not_determinable_unless_every_file_was_read(tmp_path, monkeypatch):
    """Property 80: at ``c7b2fd1d``, no US scenario: (b) not determinable if a file is absent, else not applicable.

    Class: coupled analyses at a pre-fix build whose read scenarios are all SI. Why: (a) and (c)
    are present at that sha, so the check fails either way; for (b), an absent file could have
    been a US model, so "not applicable to any scenario" is claimable only when every file was
    read. What kills it: counting (b) not applicable whenever no scenario read as US, or omitting
    the entry-level count so the counts no longer sum to three. Second correct implementation:
    deciding (b)'s entry count from the not-determinable scenario count before the US count; the
    assertions read the verdict, the entry-level phrase and the summed counts.
    """
    gap = _run(monkeypatch, _analysis(tmp_path / "g", {0: "CMS", 1: None}), _PRE_FIX)
    full = _run(monkeypatch, _analysis(tmp_path / "f", {0: "CMS", 1: "CMS"}), _PRE_FIX)
    for res in (gap, full):
        assert res.passed is False
        assert "TRITON-COUPLED-RIM-BELOW-CROWN" in res.summary
        assert "TRITON-COUPLED-LIST-MISALIGNED" in res.summary
        assert "2 PRESENT" in res.summary
    assert "1 not determinable (TRITON-COUPLED-US-UNIT-RIM: 1 scenario(s)" in gap.summary
    assert "not applicable to any scenario" not in gap.summary
    assert "1 not applicable to any scenario (TRITON-COUPLED-US-UNIT-RIM)" in full.summary
    assert "1 not determinable (" not in full.summary


@pytest.mark.parametrize(("coupled", "sha"), [(False, _PRE_FIX), (True, None)])
def test_not_applicable_without_the_coupled_model_or_without_a_sha(tmp_path, monkeypatch, coupled, sha):
    """Property 80: a TRITON-only or SWMM-only analysis, or one with no readable sha, is not applicable."""
    res = _run(monkeypatch, _analysis(tmp_path, {0: "CFS"}, coupled=coupled), sha)
    assert res.applicable is False
    assert res.passed is True


def test_the_resume_check_reports_t4a_and_does_not_count_the_coupled_entries(monkeypatch):
    """Property 80, last case: a resumed coupled analysis at ``c7b2fd1d`` gets T4a from the resume check, not (a)-(c).

    Class: resumed coupled analyses at a build that carries both the clean-start resume defect
    (T4a) and the three every-coupled-run defects. Why: (a)-(c) are not resume defects and have
    their own check, so the resume check must neither name them nor count them in its
    denominator. What kills it: the resume check's registry filter without its ``coupled``
    exclusion, which names all three and raises the denominator by three. Second correct
    implementation: a whitelist of the resume triggers; the assertions read only the summary.
    """
    import hhemt.analysis_validation as av
    from hhemt.model_defects import REGISTRY

    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _PRE_FIX)
    analysis = SimpleNamespace(_system=SimpleNamespace(cfg_system=SimpleNamespace(toggle_tritonswmm_model=True)))
    df = pd.DataFrame({"model_type": ["tritonswmm"], "n_resumes": [1], "event_iloc": [0]})
    res = av.check_known_resume_defects(analysis, df_status=df)
    assert res.passed is False
    assert "TRITON-SNAPSHOT-STALE-AT-CLEAN-START" in res.summary
    coupled_ids = [d.defect_id for d in REGISTRY if d.trigger == "coupled"]
    assert len(coupled_ids) == 3
    for did in coupled_ids:
        assert did not in res.summary
    n_resume = sum(1 for d in REGISTRY if d.trigger != "coupled")
    assert f"{n_resume} applicable registry defect(s)" in res.summary
