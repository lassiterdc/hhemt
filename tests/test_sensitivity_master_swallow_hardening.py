"""Silent-zero hardening for the sensitivity master generators (R13/D10, Phase 6).

The ``(sa, event)`` plot-pair enumeration that feeds ``SA_EVENT_PAIRS_SA`` /
``SA_EVENT_PAIRS_EVT`` is a best-effort step: on failure it must leave the paired
lists EMPTY (so ``rule all`` does not demand the per-sim panels and the master
report skips them). Phase 6 extracts that enumeration + its best-effort
``except`` into ONE shared builder helper (``_enumerate_sa_event_pairs``), called
by both ``generate_master_snakefile_content`` and
``generate_reprocess_master_snakefile_content``, and adds a WARNING so the
silent-zero becomes a DETECTED-zero (R13 is plural — both call sites share this
one hardened ``except``).

Testing the helper directly is deliberate: it proves the warning + empty-list
behavior without driving the full reprocess generator, whose sub-inclusion
invariants require on-disk completion state the unrun synth fixture does not
provide (and which a single-gate monkeypatch would violate).
"""

from __future__ import annotations

import logging


def _builder(synth_sensitivity_builder):
    return synth_sensitivity_builder.sensitivity._workflow_builder


def _break_indexer(monkeypatch, builder) -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("injected indexer failure")

    for sub in builder.sensitivity_analysis.sub_analyses.values():
        monkeypatch.setattr(sub, "_retrieve_weather_indexer_using_integer_index", _boom)


def test_enumerate_sa_event_pairs_happy_path(synth_sensitivity_builder) -> None:
    """No failure, no filter: every sub-analysis/event pair is enumerated (the
    byte-identical behavior the two generators relied on before extraction)."""
    builder = _builder(synth_sensitivity_builder)
    n_events = sum(len(sub.df_sims.index) for sub in builder.sensitivity_analysis.sub_analyses.values())
    sa, evt = builder._enumerate_sa_event_pairs(context_label="sensitivity master")
    assert n_events > 0  # fixture sanity
    assert len(sa) == len(evt) == n_events


def test_enumerate_sa_event_pairs_warns_and_empties_on_failure(synth_sensitivity_builder, monkeypatch, caplog) -> None:
    """R13/D10 (production caller label): an enumeration failure logs a WARNING
    naming the exception type AND returns two EMPTY lists."""
    builder = _builder(synth_sensitivity_builder)
    _break_indexer(monkeypatch, builder)

    with caplog.at_level(logging.WARNING, logger="hhemt.workflow"):
        sa, evt = builder._enumerate_sa_event_pairs(context_label="sensitivity master")

    assert sa == [] and evt == []
    assert any(
        r.levelno == logging.WARNING
        and "Per-sim panel enumeration failed for the sensitivity master" in r.getMessage()
        and "RuntimeError" in r.getMessage()
        for r in caplog.records
    ), "expected a WARNING naming the exception type for the sensitivity-master label"


def test_enumerate_sa_event_pairs_reprocess_label(synth_sensitivity_builder, monkeypatch, caplog) -> None:
    """The reprocess caller passes ``context_label='reprocess master'``; the same
    hardened ``except`` names it (proving the second call site's message)."""
    builder = _builder(synth_sensitivity_builder)
    _break_indexer(monkeypatch, builder)

    with caplog.at_level(logging.WARNING, logger="hhemt.workflow"):
        sa, evt = builder._enumerate_sa_event_pairs(context_label="reprocess master")

    assert sa == [] and evt == []
    assert any(
        r.levelno == logging.WARNING
        and "Per-sim panel enumeration failed for the reprocess master" in r.getMessage()
        and "RuntimeError" in r.getMessage()
        for r in caplog.records
    ), "expected a WARNING naming the exception type for the reprocess-master label"


def test_enumerate_sa_event_pairs_include_sub_filter(synth_sensitivity_builder) -> None:
    """``include_sub`` gates enumeration (the reprocess generator's completion
    filter); excluding every sub yields empty lists and no warning."""
    builder = _builder(synth_sensitivity_builder)
    sa, evt = builder._enumerate_sa_event_pairs(
        context_label="reprocess master",
        include_sub=lambda _sa_id, _sub: False,
    )
    assert sa == [] and evt == []
