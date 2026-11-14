"""A model log's NAME is a function of the event's identity, never of its row position.

Class of defect this pins: the name was keyed on the events CSV's ROW ORDINAL, so inserting,
deleting or reordering a row renumbered every ordinal at or after the edit and three durable
record families -- the model log, its ``_walltime`` ledger and its ``_resume_events`` ledger,
the last two DERIVED from the log's stem -- changed hands between events with no identity
marker in any of the three names. Why required: a consumer that reads a foreign event's
completion marker SKIPS a simulation that never ran, which is silent work loss; the
``_perf`` conjunct in ``_coupled_swmm_report_finalized`` defended exactly one of the three
model types against it.

The discriminator is a PAIR of fixtures over one shared event set:

* the same event occupying DIFFERENT row positions must resolve the SAME path (a name that
  followed the row would give two), and
* two DIFFERENT events occupying the SAME row position must resolve DIFFERENT paths (a name
  that followed the row gives one, which is the pre-fix collision).

Neither arm alone is sufficient: the first is satisfied by any constant name and the second
by any name that hashes the weather indexers without the event set being involved.

Every analysis stub here carries the REAL ``_retrieve_weather_indexer_using_integer_index``
bound off ``TRITONSWMM_analysis`` rather than a re-implementation, so the projection semantics
the resolver depends on -- including an absent ``event_iloc`` raising out of ``.loc`` -- are
the production ones. Re-implementing it would let this file and the producer disagree about
that failure mode while both stayed green.

Laptop-safe: no solver is compiled or executed and every path is under ``tmp_path``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

#: The one weather-indexer column the fixtures key events on. The production config names its
#: own columns; the slug's SHAPE is what matters here, and `compute_event_id_slug` renders it
#: as ``{column}.{value}``.
_INDEX_COLUMN = "event"


def _analysis(df_sims, simlog_directory, *, is_experiment_member=False, analysis_id=None, analysis_dir=None):
    """A duck-typed analysis carrying the projection the resolver requires.

    ``_retrieve_weather_indexer_using_integer_index`` is bound off ``TRITONSWMM_analysis``, so
    this stub projects ``df_sims.loc[event_iloc, weather_event_indices]`` exactly as the
    production method does. An ``event_iloc`` outside ``df_sims.index`` therefore raises the
    same ``KeyError`` it raises in production, which is what makes the resolver's
    ``ConfigurationError`` a statement about real behaviour rather than about a stub.
    """
    from hhemt.analysis import TRITONSWMM_analysis

    cfg = SimpleNamespace(
        weather_event_indices=[_INDEX_COLUMN],
        is_experiment_member=is_experiment_member,
    )
    if analysis_id is not None:
        cfg.analysis_id = analysis_id
    paths = SimpleNamespace(simlog_directory=simlog_directory)
    if analysis_dir is not None:
        paths.analysis_dir = analysis_dir
    stub = SimpleNamespace(df_sims=df_sims, cfg_analysis=cfg, analysis_paths=paths)
    stub._retrieve_weather_indexer_using_integer_index = (
        TRITONSWMM_analysis._retrieve_weather_indexer_using_integer_index.__get__(stub)
    )
    return stub


def _events(*values):
    """A minimal events table with the production index shape (RangeIndex, one indexer col)."""
    return pd.DataFrame({_INDEX_COLUMN: list(values)})


def test_same_event_at_different_row_positions_resolves_one_path(tmp_path):
    """THE IDENTITY ARM. Row ORDER differs; the event set does not.

    A name keyed on the row gives ``evt0`` for ``A`` in the first table and ``evt1`` for the
    SAME ``A`` in the second, so the assertion fails the moment the ordinal comes back into
    the expression.
    """
    from hhemt.run_simulation import model_logfile_for

    forward = _analysis(_events("A", "B"), tmp_path)
    reversed_ = _analysis(_events("B", "A"), tmp_path)

    assert model_logfile_for(forward, 1, "tritonswmm") == model_logfile_for(reversed_, 0, "tritonswmm")
    assert model_logfile_for(forward, 0, "tritonswmm") == model_logfile_for(reversed_, 1, "tritonswmm")


def test_two_events_at_the_same_row_position_resolve_different_paths(tmp_path):
    """THE COLLISION ARM, and it is RED pre-fix.

    Two tables whose row 0 holds DIFFERENT events. Under an ordinal key both resolve to
    ``model_{m}_evt0.log`` -- one name, two events, which is the defect: whichever event ran
    last owns the file and the other event's consumer reads it as its own evidence.
    """
    from hhemt.run_simulation import model_logfile_for

    first = _analysis(_events("A"), tmp_path)
    second = _analysis(_events("B"), tmp_path)

    assert model_logfile_for(first, 0, "triton") != model_logfile_for(second, 0, "triton")


def test_name_carries_the_stable_slug_in_the_flag_grammar_form(tmp_path):
    """The rendered name, asserted as a NAME, against the ``evt-{event_id}`` form.

    ``constants.py``'s flag-name GRAMMAR block mints ``c_run_{model}_evt-{event_id}_complete``
    for the ``_status/`` flags, and the scenario directory is keyed on the same slug. The model
    log joining that family is the whole of the change, so the form is pinned rather than left
    to follow from the pair above.
    """
    from hhemt.run_simulation import model_logfile_for

    plain = _analysis(_events("A"), tmp_path / "logs" / "sims")
    assert model_logfile_for(plain, 0, "tritonswmm").name == "model_tritonswmm_evt-event.A.log"

    # The sensitivity-member arm keeps the full analysis_id segment (not `member{N}`) and
    # anchors under the MASTER's logs/sims via `.parent.parent`.
    master_dir = tmp_path / "master"
    member = _analysis(
        _events("A"),
        tmp_path / "unused",
        is_experiment_member=True,
        analysis_id="member_gpu_0_r1",
        analysis_dir=master_dir / "members" / "member_gpu_0_r1",
    )
    path = model_logfile_for(member, 0, "triton")
    assert path == master_dir / "logs" / "sims" / "model_triton_member_gpu_0_r1_evt-event.A.log"


def test_both_ledgers_derive_from_the_rekeyed_stem(tmp_path):
    """The two append-only ledgers re-key BY DERIVATION, which is why neither file was edited.

    ``_walltime`` and ``_resume_events`` are sibling directories under the log's own parent and
    both filenames are the log's STEM. Pinning that here is what makes "re-keying the log
    re-keys both ledgers with zero edits at their sites" a checked claim rather than a
    description of the implementation.
    """
    from hhemt.resume_events import LEDGER_DIRNAME, ledger_path_for
    from hhemt.run_simulation import model_logfile_for

    log = model_logfile_for(_analysis(_events("A"), tmp_path / "logs" / "sims"), 0, "tritonswmm")

    walltime = log.parent / "_walltime" / f"{log.stem}.jsonl"
    resume_ledger = ledger_path_for(log)
    assert walltime.name == resume_ledger.name == f"{log.stem}.jsonl"
    assert resume_ledger.parent.name == LEDGER_DIRNAME
    assert resume_ledger.parent.parent == log.parent
    # The stem itself carries the identity, so neither ledger can drift onto another event.
    assert log.stem == "model_tritonswmm_evt-event.A"


def test_an_ordinal_named_walltime_ledger_reads_none_not_zero(tmp_path):
    """THE DISCLOSED CONSEQUENCE of the re-key, asserted rather than left to be discovered.

    A tree written before this change keeps its per-attempt wall ledger at an ORDINAL stem.
    The re-keyed resolver asks for the identity stem, so ``read_walltime_ledger_total_s``
    finds nothing and returns None -- and None, NOT 0.0, is what makes ``df_status``'s
    ``wall_clock_ledger_s`` fall back to the perf-summary total. A zero would report a
    simulation that took zero seconds, which is the silent-zero failure this arm exists to
    exclude.

    The must-fire control is in the same test: the reader DOES return the summed total when
    the ledger sits at the identity stem. Without it, a reader that returned None for
    everything would pass the first assertion, and the pair would not discriminate.
    """
    from hhemt.resume_events import LEDGER_DIRNAME  # noqa: F401  (documents the sibling set)
    from hhemt.run_simulation import model_logfile_for, read_walltime_ledger_total_s

    simlogs = tmp_path / "logs" / "sims"
    (simlogs / "_walltime").mkdir(parents=True)
    (simlogs / "_walltime" / "model_tritonswmm_evt0.jsonl").write_text('{"wall_s": 61.5}\n', encoding="utf-8")
    analysis = _analysis(_events("A"), simlogs)

    # The ordinal-named ledger is not this event's ledger under the re-keyed name, so the
    # answer is None (a fallback signal), never 0.0 (a measured zero).
    assert read_walltime_ledger_total_s(model_logfile_for(analysis, 0, "tritonswmm")) is None

    # CONTROL: same reader, same file content, the identity stem -- it sums for real.
    (simlogs / "_walltime" / "model_tritonswmm_evt-event.A.jsonl").write_text('{"wall_s": 61.5}\n', encoding="utf-8")
    assert read_walltime_ledger_total_s(model_logfile_for(analysis, 0, "tritonswmm")) == 61.5


def test_absent_event_iloc_raises_configuration_error_naming_the_iloc(tmp_path):
    """LEGIBILITY on a LATENT partiality, not a new guard.

    Resolving the slug inside the path resolver is what makes the function able to fail: an
    ``event_iloc`` that is not a row of ``df_sims`` raises ``KeyError`` out of
    ``_retrieve_weather_indexer_using_integer_index``. No caller in the current population
    passes an out-of-range iloc, so this cannot fire today -- but if it ever does, a bare
    ``KeyError`` from inside a path resolver names neither the row nor the caller. The
    resolver therefore raises ``ConfigurationError`` whose message names the iloc.

    RED pre-fix in the other direction: the function reads two attributes off ``analysis`` and
    cannot raise at all, so ``pytest.raises`` fails on the exception type.
    """
    from hhemt.exceptions import ConfigurationError
    from hhemt.run_simulation import model_logfile_for

    analysis = _analysis(_events("A"), tmp_path / "logs" / "sims")

    with pytest.raises(ConfigurationError) as excinfo:
        model_logfile_for(analysis, 5, "triton")
    assert "5" in str(excinfo.value)
    assert "event_iloc" in str(excinfo.value)
