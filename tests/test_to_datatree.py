"""Regression tests for the removal of the analysis-level flat-summary readers.

Before the removal, several names reached analysis-level flat summary files that nothing
writes. The eight summary properties on the post-processing object, and the ``_open``
helper behind them, could only raise ``ValueError``; its ``to_datatree`` overlay skipped
the missing files and returned a tree holding only its root. ``TRITONSWMM_analysis``
forwarded the eight properties through ``_``-prefixed forwarders, and
``TRITONSWMM_sensitivity_analysis`` carried three summary properties of its own. All of
these were deleted, together with the ``AnalysisPaths`` fields that named the files, the
analysis plotting class and the ``hhemt.plot_analysis`` module.

These tests check that each deleted name stays gone, that the consolidation surface
through which results are now read survives, and that the two mode tables still agree.
"""

import dataclasses
import importlib.util

import pytest

import hhemt.analysis as analysis_module
from hhemt.analysis import TRITONSWMM_analysis
from hhemt.paths import AnalysisPaths, ScenarioPaths
from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing
from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis


def test_mode_to_tree_path_keys_match_mode_config(synth_all_models_analysis):
    """`_MODE_TO_TREE_PATH` must cover every `_MODE_CONFIG` key, so the two mode tables cannot drift apart."""
    proc = synth_all_models_analysis.process
    assert set(proc._MODE_TO_TREE_PATH.keys()) == set(proc._MODE_CONFIG.keys())


#: The eight analysis-level flat summary readers on the post-processing object. Each opened
#: a file in the analysis directory that no code under ``src/hhemt`` writes, so each could
#: only raise ``ValueError``; results are read from the consolidated trees instead.
_DELETED_PROCESS_PROPERTIES = (
    "tritonswmm_TRITON_summary",
    "tritonswmm_SWMM_node_summary",
    "tritonswmm_SWMM_link_summary",
    "tritonswmm_performance_summary",
    "triton_only_summary",
    "triton_only_performance_summary",
    "swmm_only_node_summary",
    "swmm_only_link_summary",
)

#: The eight ``AnalysisPaths`` fields that named those never-written files. The per-scenario
#: fields of the SAME names on ``ScenarioPaths`` are written by ``process_simulation.py``
#: and must survive; the control test below holds them.
_DELETED_ANALYSIS_PATH_FIELDS = (
    "output_tritonswmm_triton_summary",
    "output_tritonswmm_node_summary",
    "output_tritonswmm_link_summary",
    "output_tritonswmm_performance_summary",
    "output_triton_only_summary",
    "output_triton_only_performance_summary",
    "output_swmm_only_node_summary",
    "output_swmm_only_link_summary",
)

#: The deleted plotting class, spelled in two pieces so that a repository-wide search for
#: its name finds only code that READS it; this test asserts it is gone and reads nothing.
_DELETED_PLOTTING_CLASS = "TRITONSWMM_analysis_" + "plotting"

_DELETED_SENSITIVITY_PROPERTIES = (
    "tritonswmm_SWMM_node_summary",
    "tritonswmm_SWMM_link_summary",
    "tritonswmm_TRITON_summary",
)


def _field_names(dataclass_type) -> set[str]:
    """The declared field names of a dataclass, read without constructing an instance."""
    return {f.name for f in dataclasses.fields(dataclass_type)}


def _absence_cases():
    """One (case id, predicate) pair per name that must be gone; the predicate is True when it is."""
    cases = []
    for name in (*_DELETED_PROCESS_PROPERTIES, "to_datatree", "_open"):
        cases.append((f"post_processing.{name}", lambda n=name: not hasattr(TRITONSWMM_analysis_post_processing, n)))
    for name in _DELETED_PROCESS_PROPERTIES:
        forwarder = f"_{name}"
        cases.append((f"analysis.{forwarder}", lambda n=forwarder: not hasattr(TRITONSWMM_analysis, n)))
    cases.append(
        (
            f"analysis_module.{_DELETED_PLOTTING_CLASS}",
            lambda: not hasattr(analysis_module, _DELETED_PLOTTING_CLASS),
        )
    )
    for name in _DELETED_SENSITIVITY_PROPERTIES:
        cases.append((f"sensitivity.{name}", lambda n=name: not hasattr(TRITONSWMM_sensitivity_analysis, n)))
    for name in _DELETED_ANALYSIS_PATH_FIELDS:
        cases.append((f"AnalysisPaths.{name}", lambda n=name: n not in _field_names(AnalysisPaths)))
    cases.append(("module.hhemt.plot_analysis", lambda: importlib.util.find_spec("hhemt.plot_analysis") is None))
    for mode in TRITONSWMM_analysis_post_processing._MODE_CONFIG:
        cases.append(
            (
                f"_MODE_CONFIG[{mode!r}] has two elements",
                lambda m=mode: len(TRITONSWMM_analysis_post_processing._MODE_CONFIG[m]) == 2,
            )
        )
    return cases


_ABSENCE_CASES = _absence_cases()


@pytest.mark.parametrize("predicate", [p for _, p in _ABSENCE_CASES], ids=[i for i, _ in _ABSENCE_CASES])
def test_never_written_flat_summary_readers_are_gone(predicate):
    """No public or private reader of the analysis-level flat summary files survives.

    Property: the post-processing class carries none of the eight summary properties,
    ``to_datatree`` or ``_open``; the analysis class carries none of the eight ``_..._summary``
    forwarders and its module no plotting class; the sensitivity analysis carries none of its
    three summary properties; ``AnalysisPaths`` names none of the eight file fields; the
    ``hhemt.plot_analysis`` module does not exist; and every ``_MODE_CONFIG`` value is
    ``(scenario_path_attr, spatial_coords)``.

    Class: every name a caller could have used to reach a file in the analysis directory
    that nothing writes, one parametrized case per name, so a single surviving name reds
    exactly one case and names it.

    Why it is required: each of these readers can only raise ``ValueError`` (the file is
    never written), and ``to_datatree`` can only return a root-only tree, so a surviving one
    is a public call that fails or silently returns nothing for every user who tries it.

    What kills it: restoring any one of the deleted names, re-adding a field to
    ``AnalysisPaths``, recreating ``plot_analysis.py``, or giving a ``_MODE_CONFIG`` value back
    its analysis-path element reds the matching case.

    A second correct implementation under which it still passes: one that stores each
    ``_MODE_CONFIG`` value as a two-field named tuple or a two-element list instead of a
    tuple; the cases assert absence and arity, never the container type or the order of
    deletion.

    Not established: the analysis's ``self.plot`` instance attribute is not reached, because
    reading it needs a constructed analysis; the module-absence and plotting-class cases are
    what hold it, since the attribute had no value to bind once its class is gone.

    Fixture-free by construction: it imports modules and reads class attributes, and it
    constructs no analysis, so it needs no compile, no solver and no clean working tree.
    """
    assert predicate()


def test_surviving_consolidation_surface_is_present():
    """Control: the same instruments report the surface that must survive as present.

    Property: the post-processing class keeps ``consolidate_to_datatree``,
    ``_retrieve_combined_output`` and ``_open_engine``; ``ScenarioPaths`` keeps all eight
    same-named per-scenario summary fields; ``_MODE_TO_TREE_PATH``'s keys equal
    ``_MODE_CONFIG``'s; every ``_MODE_CONFIG`` value's first element names a ``ScenarioPaths``
    field; and ``find_spec`` resolves a module that does exist.

    Class: the must-be-nonzero control for the absence test above. Each assertion uses the
    instrument an absence case uses (``hasattr``, ``dataclasses.fields``, ``find_spec``), so a
    broken instrument that reports every name absent reds here instead of passing there.

    Why it is required: an absence test passes against an instrument that sees nothing, and
    deleting too much — the consolidation writer, the per-scenario fields, the mode table —
    would also make every absence case pass while breaking consolidation.

    What kills it: deleting ``_open_engine`` (two live readers call it), removing a
    ``ScenarioPaths`` summary field, or deleting a mode from only one of the two tables.

    A second correct implementation under which it still passes: one that reorders
    ``_MODE_CONFIG``'s entries, or stores its values as lists rather than tuples, keeps every
    assertion here green.
    """
    for name in ("consolidate_to_datatree", "_retrieve_combined_output", "_open_engine"):
        assert hasattr(TRITONSWMM_analysis_post_processing, name), name
    scenario_fields = _field_names(ScenarioPaths)
    assert set(_DELETED_ANALYSIS_PATH_FIELDS) <= scenario_fields
    proc_cls = TRITONSWMM_analysis_post_processing
    assert set(proc_cls._MODE_TO_TREE_PATH) == set(proc_cls._MODE_CONFIG)
    assert len(proc_cls._MODE_CONFIG) == 8
    for mode, value in proc_cls._MODE_CONFIG.items():
        assert value[0] in scenario_fields, mode
    assert importlib.util.find_spec("hhemt.processing_analysis") is not None
    assert hasattr(analysis_module, "TRITONSWMM_analysis")
    assert hasattr(TRITONSWMM_sensitivity_analysis, "open_sensitivity_datatree")
