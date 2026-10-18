"""Every report-config read in a renderer names a field the config actually declares.

Renderers read the report configuration through attribute chains such as
`report_cfg.system_map.elevation_panel.wall_threshold_buffer_m`. Pydantic models
raise `AttributeError` on an undeclared name, and a renderer that wraps the read in
`except AttributeError` turns a misspelled field into a silent fallback: the user's
setting is ignored and nothing reports it. These tests resolve every such chain
statically against the config models, and check the one place where two renderers
share a config-derived threshold.

These tests read source files and config models only. They compile no solver and
run no simulation.
"""

import ast
import typing
from pathlib import Path

from pydantic import BaseModel

import hhemt.report_renderers as _renderers_pkg
from hhemt.config.report import ElevationPanelStyle, report_config

_RENDERERS_DIR = Path(_renderers_pkg.__file__).parent


def _attribute_chain(node):
    """Return the dotted names of an attribute expression, outermost name first."""
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    return list(reversed(names))


def _config_suffix(chain):
    """Return the part of a chain read off a `report_config`, or None.

    Two roots name a report config in the renderers: the `report_cfg` argument
    every renderer receives, and `cfg_analysis.report` on the analysis object.
    """
    if "report_cfg" in chain:
        return chain[chain.index("report_cfg") + 1 :]
    for i in range(1, len(chain) - 1):
        if chain[i - 1] == "cfg_analysis" and chain[i] == "report":
            return chain[i + 1 :]
    return None


def _nested_model(annotation):
    """Return the Pydantic model a field holds, unwrapping Optional, else None."""
    for candidate in (annotation, *typing.get_args(annotation)):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def _first_unresolved_name(suffix):
    """Walk a chain through the config models; return the first undeclared name, or None.

    A name is resolved if it is a declared field (descend into it when it holds a
    model) or any other attribute of the model class, such as a method; the walk
    stops at the first name that is not itself a model-valued field.
    """
    model = report_config
    for name in suffix:
        if name in model.model_fields:
            model = _nested_model(model.model_fields[name].annotation)
            if model is None:
                return None
        elif hasattr(model, name):
            return None
        else:
            return name
    return None


def _config_chains():
    """Yield (file, line, chain) for every outermost attribute chain read off a report config."""
    for path in sorted(_RENDERERS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        inner = {child for node in ast.walk(tree) if isinstance(node, ast.Attribute) for child in [node.value]}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node not in inner:
                chain = _attribute_chain(node)
                suffix = _config_suffix(chain)
                if suffix:
                    yield path.name, node.lineno, chain, suffix


def test_every_report_config_read_in_a_renderer_names_a_declared_field():
    """Property: every attribute chain a renderer reads off a report config resolves,
    name by name, to fields the config models declare.

    Class: all outermost attribute chains in `src/hhemt/report_renderers/*.py`
    rooted at `report_cfg` or at `cfg_analysis.report`. This is a fitness function:
    it constrains how renderers name config fields, not what they compute.

    Why required: a chain naming an undeclared field raises `AttributeError` when it
    runs, and where the read sits inside `except AttributeError` the renderer
    silently substitutes a hardcoded value for the user's setting.

    What kills it: renaming a config field without updating a renderer that reads
    it, or reading a field from the wrong sub-model, at any site in any renderer.

    A second correct implementation that still passes: renderers that bind the
    sub-model to a local name first (`ep = report_cfg.system_map.elevation_panel`)
    and read the field off the local; this test does not see those reads, and it
    passes them.
    """
    chains = list(_config_chains())
    assert chains, "found no report-config reads at all, so the scan is not reading the renderers"
    unresolved = [
        f"{name}:{line} {'.'.join(chain)} (undeclared: {_first_unresolved_name(suffix)})"
        for name, line, chain, suffix in chains
        if _first_unresolved_name(suffix) is not None
    ]
    assert unresolved == []


class _SystemConfig:
    """The two DEM sentinel heights the wall threshold reads, and nothing else."""

    def __init__(self, building, outside):
        self.dem_building_height = building
        self.dem_outside_watershed_height = outside


def test_wall_threshold_honours_a_user_set_buffer():
    """Property: the per-sim peak-flood-depth wall threshold subtracts the buffer the
    user configured on `system_map.elevation_panel`, not a hardcoded one.

    Class: any buffer value, tested at the default and at a non-default, with both
    sentinel heights set. The one-sentinel and no-sentinel cases are the next test's.

    Why required: the overview map and the per-sim map both draw a building mask
    from this threshold, and a user who changes the buffer must see both maps change.

    What kills it: substituting a constant 40.0 for the configured buffer, which the
    non-default case reds.

    A second correct implementation that still passes: computing the threshold
    inline at each call site from the same two inputs.
    """
    from hhemt.report_renderers.per_sim_peak_flood_depth import _wall_threshold_m

    sys_cfg = _SystemConfig(80.0, 100.0)
    assert _wall_threshold_m(sys_cfg, ElevationPanelStyle()) == 80.0 - 40.0
    assert _wall_threshold_m(sys_cfg, ElevationPanelStyle(wall_threshold_buffer_m=10.0)) == 70.0


def test_wall_threshold_uses_whichever_sentinel_is_set():
    """Property: with only one DEM sentinel height set, the wall threshold is that
    height less the buffer; with neither set there is no threshold.

    Class: the three sentinel states other than both-set: building height only,
    outside-watershed height only, and neither. Each one-sentinel case uses a
    different height, so a threshold read from the wrong field is caught.

    Why required: the system-overview map paints walls from whichever sentinel is
    set, and the per-sim map must mask the same cells, or the two maps of one
    analysis disagree about which cells are buildings.

    What kills it: returning None whenever either sentinel is unset, which reds both
    one-sentinel cases; or subtracting the buffer from the unset field.

    A second correct implementation that still passes: filtering the two heights to
    the ones that are set and taking `min()` of that list, as the overview map does.
    """
    from hhemt.report_renderers.per_sim_peak_flood_depth import _wall_threshold_m

    panel = ElevationPanelStyle(wall_threshold_buffer_m=10.0)
    assert _wall_threshold_m(_SystemConfig(80.0, None), panel) == 70.0
    assert _wall_threshold_m(_SystemConfig(None, 100.0), panel) == 90.0
    assert _wall_threshold_m(_SystemConfig(None, None), panel) is None
