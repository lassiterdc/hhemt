"""The crate's `variableMeasured` descriptors are STORE-SOURCED through the production path.

WHY THIS MODULE EXISTS AND WHY IT IS NOT COVERED BY `tests/test_metadata.py`. The
store-sourcing capability (`build_analysis_crate(emitted_attrs=...)`) landed one commit before
the callers that supply it, and for that commit the crate was override-blind in production while
the whole suite was green. `test_metadata.py::test_omitting_emitted_attrs_is_byte_identical_to_
before` passes PRECISELY BECAUSE no caller threads the mapping, and it passes equally once they
do -- so green there is not evidence about production, in either state. The absence of a
post-condition over the PRODUCTION path is the entire mechanism by which the gap stayed
invisible, and this module is that post-condition.

THE TWO ARMS ARE BLIND TO DIFFERENT THINGS AND NEITHER SUBSTITUTES FOR THE OTHER.

- `test_the_crate_publishes_the_store_long_name_...` is the BEHAVIOUR arm. It drives the real
  helper -> `emit_provenance` -> `build_analysis_crate` chain and asserts the published
  `description` equals the value the store carries. It would stay green if a future edit removed
  `emitted_attrs=` from a CALL SITE, because it supplies the mapping itself.
- `test_both_production_call_sites_thread_emitted_attrs` is the CALLER arm. It reads the two
  production modules' source with `ast` and asserts each `emit_provenance(...)` call passes
  `emitted_attrs`. It would stay green if the descriptor projection were broken, because it never
  evaluates a descriptor.

The assertion in the behaviour arm is anchored on a property that EXISTS IN BOTH the pre-fix and
post-fix worlds -- the string value of `description` -- rather than on anything introduced by the
fix. Measured against the pre-threading code it fails with the base-map label
`'Maximum flood velocity'` where the store stamps `'Maximum conduit velocity'`; that divergence
is the only one in the live tables, which is why `max_velocity_mps` is the probe.
"""

from __future__ import annotations

import ast
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import xarray as xr

from hhemt import metadata
from hhemt.cf_conventions import (
    _CF_VARIABLE_MAP,
    _CF_VARIABLE_OVERRIDES_BY_MODE,
    apply_cf_attributes,
)
from hhemt.processing_analysis import emitted_vars_and_attrs

#: The one variable whose link-mode override `long_name` diverges from its base-map entry. It is
#: the probe BECAUSE it diverges: a name whose override agrees with its base entry cannot
#: discriminate a store-sourced descriptor from a static one.
_PROBE = "max_velocity_mps"
_LINK_MODE = "tritonswmm_swmm_link"


def _link_node() -> xr.Dataset:
    """A link-mode node stamped by the real `apply_cf_attributes`, not by a hand-written attr."""
    ds = xr.Dataset({_PROBE: (("conduit_id",), np.array([1.25]))}, coords={"conduit_id": ["C1"]})
    return apply_cf_attributes(ds, _LINK_MODE)


def _fake_case():
    return SimpleNamespace(case_name="probe", description="", manifest={})


def _crate_descriptions(emitted_vars, emitted_attrs) -> dict[str, str | None]:
    crate = metadata.build_analysis_crate(
        analysis_id="a1",
        system_id=None,
        layout_version=24,
        toolkit_git_sha="deadbeef",
        code_repository="https://example/repo",
        cfg_case=_fake_case(),
        sif_spec=None,
        consolidated_zarr_relpath="analysis_datatree.zarr",
        input_parts=[],
        emitted_vars=emitted_vars,
        emitted_attrs=emitted_attrs,
    )
    doc = json.loads(metadata.canonical_jsonld(crate))
    return {e["name"]: e.get("description") for e in doc["@graph"] if e.get("@type") == "PropertyValue" and "name" in e}


def test_the_probe_variable_actually_diverges_between_store_and_base_map():
    """The discriminator precondition. Without it the behaviour arm is vacuously green."""
    base = _CF_VARIABLE_MAP[_PROBE]["long_name"]
    override = _CF_VARIABLE_OVERRIDES_BY_MODE[_LINK_MODE][_PROBE]["long_name"]
    assert base != override, (
        f"{_PROBE} no longer diverges between the base map ({base!r}) and the {_LINK_MODE} "
        "override: this module's behaviour arm can no longer discriminate a store-sourced "
        "descriptor from a static one. Re-point the probe at a variable that does diverge, or "
        "state in this module why no such variable remains."
    )
    assert _link_node()[_PROBE].attrs["long_name"] == override


def test_the_crate_publishes_the_store_long_name_not_the_base_map_label():
    """THE POST-CONDITION: `description == store long_name`, through the production path."""
    ds = _link_node()
    stored = ds[_PROBE].attrs["long_name"]
    emitted_vars, emitted_attrs = emitted_vars_and_attrs([ds])
    descriptions = _crate_descriptions(emitted_vars, emitted_attrs)

    assert _PROBE in descriptions, f"{_PROBE} was not advertised at all: {sorted(descriptions)}"
    assert descriptions[_PROBE] == stored, (
        f"crate description {descriptions[_PROBE]!r} != store long_name {stored!r}. The crate is "
        "override-blind: either a production caller stopped threading emitted_attrs, or the "
        "helper stopped reading the node attrs."
    )
    assert descriptions[_PROBE] != _CF_VARIABLE_MAP[_PROBE]["long_name"], (
        "the crate published the STATIC base-map label, which is the pre-fix state"
    )


def test_omitting_the_mapping_reproduces_the_pre_fix_divergence():
    """The violating arm: with no mapping the crate publishes the base-map label.

    This is what the two production call sites were doing. It is asserted rather than merely
    described so the post-condition above is known to discriminate rather than assumed to.
    """
    ds = _link_node()
    emitted_vars, _ = emitted_vars_and_attrs([ds])
    descriptions = _crate_descriptions(emitted_vars, None)
    assert descriptions[_PROBE] == _CF_VARIABLE_MAP[_PROBE]["long_name"]
    assert descriptions[_PROBE] != ds[_PROBE].attrs["long_name"]


def test_two_nodes_describing_one_name_differently_fall_back_to_the_static_entry():
    """The DECIDED conflict policy, asserted so it is a contract rather than an accident.

    A flat mapping cannot express one name described two ways, so the helper omits a disagreeing
    name and `_descriptor` falls back to its static entry. Inert on the current corpus (the two
    override modes carrying `max_velocity_mps` are byte-identical); reachable by a future
    heterogeneous tree.
    """
    agreed = _link_node()
    divergent = _link_node()
    divergent[_PROBE].attrs["long_name"] = "Something else entirely"

    emitted_vars, emitted_attrs = emitted_vars_and_attrs([agreed, divergent])
    assert _PROBE in emitted_vars
    assert _PROBE not in emitted_attrs, "a disagreeing name must be omitted, not last-wins"
    assert _crate_descriptions(emitted_vars, emitted_attrs)[_PROBE] == _CF_VARIABLE_MAP[_PROBE]["long_name"]


def test_an_unstamped_node_contributes_no_descriptor():
    """`_descriptor`'s predicate is a non-empty `long_name`; an unstamped node must not shadow."""
    bare = xr.Dataset({_PROBE: (("conduit_id",), np.array([1.25]))}, coords={"conduit_id": ["C1"]})
    emitted_vars, emitted_attrs = emitted_vars_and_attrs([bare])
    assert emitted_vars == {_PROBE}
    assert emitted_attrs == {}


#: The two production `emit_provenance` call sites, each of which MUST pass `emitted_attrs`.
_PRODUCTION_CALLERS = ("src/hhemt/processing_analysis.py", "src/hhemt/sensitivity_analysis.py")


def test_both_production_call_sites_thread_emitted_attrs():
    """THE CALLER ARM: a source-level assertion the behaviour arm structurally cannot make.

    The behaviour arm supplies the mapping itself, so deleting `emitted_attrs=` from either call
    site leaves it green. This reads the call expression instead. It is the cheapest instrument
    that fails on the exact edit that produced the original gap.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    found: dict[str, list[bool]] = {}
    for rel in _PRODUCTION_CALLERS:
        tree = ast.parse((repo_root / rel).read_text())
        threaded = [
            any(kw.arg == "emitted_attrs" for kw in node.keywords)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "emit_provenance"
        ]
        found[rel] = threaded

    for rel, threaded in found.items():
        assert threaded, f"{rel} has no emit_provenance(...) call at all; re-point this guard"
        assert all(threaded), (
            f"{rel} calls emit_provenance without emitted_attrs at {threaded.count(False)} of "
            f"{len(threaded)} site(s). The crate is then override-blind in production even though "
            "every descriptor test in the suite stays green."
        )
