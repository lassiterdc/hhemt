"""The `(renderer_kind, model_type) -> consolidated-DataTree group` matrix.

Single source of truth for WHICH model arms each per-sim renderer can plot, and
for the order they appear in on the composite per-scenario page.

Why a table rather than the `if/elif` chains this replaces. The prior form --
`if "tritonswmm" in enabled: ... elif "triton" in enabled: ...` in each
renderer's `render()` -- reads as a per-model matrix and is not one: the `elif`
is unreachable whenever the coupled arm is enabled, so a three-model analysis
rendered the coupled arm and silently dropped the other two. A fallback chain
can drop an arm without saying so; an enumerable table cannot.

The group strings mirror `processing_analysis._MODE_TO_TREE_PATH`, which is the
write side. The asymmetry is physical, not incidental: standalone SWMM produces
no 2D depth field and standalone TRITON produces no conduits, so `peak_flood_depth`
has no `swmm` arm and `conduit_flow` has no `triton` arm.
"""

from __future__ import annotations

#: renderer_kind -> {model_type: leading-slash DataTree group path}.
ARM_GROUPS: dict[str, dict[str, str]] = {
    "peak_flood_depth": {
        "tritonswmm": "/tritonswmm/triton",
        "triton": "/triton_only/triton",
    },
    "conduit_flow": {
        "tritonswmm": "/tritonswmm/swmm_link",
        "swmm": "/swmm_only/swmm_link",
    },
}

#: Section order on the composite page, per the user's stated order
#: (TRITON-SWMM, then SWMM, then TRITON). ALSO the tie-break order for the
#: legacy single-figure path, where it reproduces the historical coupled-wins
#: choice exactly: "tritonswmm" is first, so a legacy caller that passes no
#: model_type still gets the coupled arm it got before.
MODEL_ORDER: tuple[str, ...] = ("tritonswmm", "swmm", "triton")

#: Within one model section, the renderer order (depth before conduits).
RENDERER_ORDER: tuple[str, ...] = ("peak_flood_depth", "conduit_flow")

#: Display names for section headers. Kept here rather than in
#: report_plot_ids._RENDERER_KIND_LABELS because those key on renderer_kind;
#: these key on model_type.
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "tritonswmm": "TRITON-SWMM",
    "swmm": "SWMM",
    "triton": "TRITON",
}


def arms_for(renderer_kind: str, enabled_model_types) -> list[str]:
    """Every model arm this renderer can plot for this analysis, in MODEL_ORDER.

    Returns [] when the renderer is not applicable to the analysis at all --
    which is the condition the callers' skip placeholder answers.
    """
    groups = ARM_GROUPS.get(renderer_kind, {})
    enabled = set(enabled_model_types)
    return [m for m in MODEL_ORDER if m in groups and m in enabled]


def resolve_arm_group(renderer_kind: str, enabled_model_types, *, model_type: str | None = None) -> str | None:
    """The DataTree group for one arm, or None when there is no applicable arm.

    `model_type=None` selects the FIRST applicable arm in MODEL_ORDER, which
    reproduces the retired coupled-wins behaviour byte-for-byte for the legacy
    single-figure rules. An explicit `model_type` selects that arm and returns
    None if it is not applicable, so a caller iterating `arms_for` never has to
    re-check applicability.
    """
    arms = arms_for(renderer_kind, enabled_model_types)
    if not arms:
        return None
    if model_type is None:
        return ARM_GROUPS[renderer_kind][arms[0]]
    if model_type not in arms:
        return None
    return ARM_GROUPS[renderer_kind][model_type]


def groups_for(renderer_kind: str, enabled_model_types) -> list[str]:
    """Every applicable group for this renderer -- the union a shared colour
    scale must span so one arm's range is never borrowed by another."""
    groups = ARM_GROUPS.get(renderer_kind, {})
    return [groups[m] for m in arms_for(renderer_kind, enabled_model_types)]


def page_sections(enabled_model_types) -> list[tuple[str, str, str]]:
    """Ordered `(model_type, renderer_kind, group)` sections for one scenario page.

    Grouped by MODEL first (so the page reads as model headers) and by
    RENDERER_ORDER within a model. For a three-model analysis this yields FOUR
    sections: TRITON-SWMM depth, TRITON-SWMM conduits, SWMM conduits, TRITON
    depth. The count is derived, never hardcoded -- a coupled-only analysis
    yields two and a tritonswmm+swmm analysis yields three.
    """
    out: list[tuple[str, str, str]] = []
    enabled = set(enabled_model_types)
    for model in MODEL_ORDER:
        if model not in enabled:
            continue
        for kind in RENDERER_ORDER:
            group = ARM_GROUPS.get(kind, {}).get(model)
            if group is not None:
                out.append((model, kind, group))
    return out
