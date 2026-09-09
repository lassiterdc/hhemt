"""Regression tests for hooks/config_reference.py -- the `### D99` closure population.

Phase 1 measured that NO test referenced this hook. The assertions below pin the
population BY NAME rather than by a count: a count cannot say WHICH models it
reached, so it would pass on any change that happened to reach twenty-five of
something.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_HOOK = _REPO / "hooks" / "config_reference.py"


def _load():
    spec = importlib.util.spec_from_file_location("config_reference", _HOOK)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["config_reference"] = mod
    spec.loader.exec_module(mod)
    mod._bind_local_src()
    return mod


cfgref = _load()

#: The twenty-one models the closure reaches that the old four-model hand-list did
#: not. Pinned BY NAME. Sixteen are report style sub-models -- that is the cost
#: `### D99` accepted, and it is written here so a later reader meets it as a
#: recorded decision rather than as page bloat of unknown origin.
CLOSURE_GAINED = frozenset(
    {
        "CRSConfig",
        "ContainerSpec",
        "ElevationPanelStyle",
        "ErrorsAndWarningsConfig",
        "FigureDefaults",
        "ForceRerunSpec",
        "HydraulicsPanelStyle",
        "HydrologyMapPanelStyle",
        "HydrologyPanelConfig",
        "InteractiveBackendConfig",
        "PerAnalysisSummaryConfig",
        "PerSimConfig",
        "PerSimFigureSpec",
        "PerSimMapConfig",
        "PerSimMapInteractiveConfig",
        "ScenarioStatusAppendixConfig",
        "SensitivityReportConfig",
        "SystemMapConfig",
        "TableInteractiveConfig",
        "eda_config",
        "report_config",
    }
)

#: The four the old hand-list tabled. Present in the closure too, so the widening
#: is purely additive -- the property that decided `### D99` over its alternatives.
PREVIOUSLY_TABLED = frozenset(
    {
        "analysis_config",
        "system_config",
        "hpc_system_config",
        "PartitionSpec",
    }
)


def _forward_refs_in(annotation) -> bool:
    """True when a ForwardRef survives anywhere inside `annotation`."""
    import typing

    stack = [annotation]
    while stack:
        node = stack.pop()
        if isinstance(node, typing.ForwardRef):
            return True
        stack.extend(typing.get_args(node) or ())
    return False


def test_closure_reaches_every_model_the_hand_list_missed():
    """THE NON-VACUITY ARM. Red against the four-model hand-list, green after.

    The old generator tabled four models and this asserts twenty-one further ones
    by name, so it cannot pass against the pre-repair page. A count assertion
    could: 25 is reachable by many wrong routes.
    """
    names = {m.__name__ for m in cfgref._closure_models()}
    missing = sorted(CLOSURE_GAINED - names)
    assert not missing, f"{len(missing)} model(s) the D99 closure must reach are absent: {missing[:5]}"


def test_the_widening_removes_nothing():
    """Purely additive, which is what `### D99` was selected for."""
    names = {m.__name__ for m in cfgref._closure_models()}
    assert PREVIOUSLY_TABLED <= names, sorted(PREVIOUSLY_TABLED - names)


def test_the_walk_leaves_no_unresolved_forward_reference():
    """Pins the PROPERTY, not one model's presence.

    An earlier version asserted ``"ContainerSpec" in names``. That pins a MODEL'S
    PRESENCE: it passes if ContainerSpec is reached by any other edge and it never
    exercises resolution at all, so the property a regression would destroy was
    pinned by nothing. Measured: 1 unresolved ForwardRef without the rebuild, 0
    with it, so this assertion discriminates where the old one did not.

    Removing the rebuild makes `_closure_models()` raise instead of returning, so
    this goes red by exception rather than by assertion -- red either way, which is
    what the test is for.
    """
    unresolved = sorted(
        f"{m.__name__}.{fname}"
        for m in cfgref._closure_models()
        for fname, info in m.model_fields.items()
        if _forward_refs_in(info.annotation)
    )
    assert not unresolved, (
        f"{len(unresolved)} field(s) still carry an unresolved ForwardRef after the "
        f"walk: {unresolved}. model_rebuild() must run before the annotation walk."
    )


def test_every_closure_model_is_tabled_on_the_page():
    """The completeness sentence must be TRUE of the rendered page.

    Asserted on each model's own rendered TABLE, not on its name appearing
    somewhere: the three entry configs are tabled under prose headings ("System
    config"), so a name-presence heuristic reports them missing while they are
    plainly there. `_table(model)` is the exact string the page must contain.
    """
    page = cfgref._render()
    missing = sorted(m.__name__ for m in cfgref._closure_models() if cfgref._table(m) not in page)
    assert not missing, f"closure members absent from the rendered page: {missing}"
