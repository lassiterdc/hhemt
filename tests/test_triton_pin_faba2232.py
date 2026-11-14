"""The ORNL `triton-swmm` tip `faba2232` is admitted to the model-defect registry, and a native
clone at it is judged as reading the coupling-node list.

Neither test compiles or runs a solver: the first reads the registry's explicit sets, the second
reads one file written into a temporary directory.
"""

from __future__ import annotations

from hhemt import model_defects as _md
from hhemt.model_defects import (
    REGISTRY,
    REGISTRY_BY_ID,
    resolve,
)
from hhemt.scenario import SWMM_COUPLING_NODES_READ, triton_coupling_read

_FABA2232 = "faba223212408f04b7a3dbf6580adedf4a4e6b13"

#: The registry's defect ids at the time faba2232 was admitted (a floor, never a size).
_ADMITTED_DEFECT_IDS = frozenset(
    {
        "TRITON-COUPLED-RESUME-REPLAY",
        "TRITON-RESUME-DEPTH-SCATTER",
        "TRITON-RESUME-EXTBC-GHOST-RING",
        "TRITON-SNAPSHOT-TRAVERSAL-GUARD",
        "TRITON-SNAPSHOT-STALE-AT-CLEAN-START",
        "TRITON-COUPLED-RIM-BELOW-CROWN",
        "TRITON-COUPLED-US-UNIT-RIM",
        "TRITON-COUPLED-LIST-MISALIGNED",
    }
)

#: A well-formed 40-hex sha that is in no cached set and is not a registered fix, so resolving it
#: without an ancestry function MUST fall through to `ancestry_unresolvable`. It is the control that
#: makes "no defect is indeterminate at faba2232" capable of failing.
_UNREGISTERED_SHA = "0123456789abcdef0123456789abcdef01234567"


def test_no_registered_defect_is_indeterminate_at_faba2232():
    """Every registered defect resolves absent at `faba2232` from the explicit sets alone.

    Class: every REGISTRY entry, read without a clone (the read path's only rule is the cached
    sets). Why: `analysis_validation` resolves with no ancestry function, so a sha missing from an
    entry reads `indeterminate` for that defect and the coupling-flux experiment's report cannot
    say whether the build carries it. What kills it: dropping `faba2232` from any one entry's
    `known_absent_in`, or moving it to `also_present_in`. Second correct implementation: deriving
    the membership from the pin constant's comment; the assertions read only status and rule.
    """
    # Read by attribute so a tree without the registration fails here, on an assertion, not at import.
    pin = getattr(_md, "SHA_ORNL_COUPLING_DECLARATION_PIN", None)
    assert pin is not None, "faba2232 is not registered: model_defects has no SHA_ORNL_COUPLING_DECLARATION_PIN"
    assert pin == _FABA2232
    # The eight defects the registry held when faba2232 was admitted must still be present; the
    # registry may GROW, so no size is pinned and a ninth correctly registered defect stays green.
    assert _ADMITTED_DEFECT_IDS <= {d.defect_id for d in REGISTRY}
    for d in REGISTRY:
        v = resolve(d, _FABA2232)
        assert v.status != "indeterminate", (d.defect_id, v)
        assert (v.status, v.rule) == ("absent", "known_absent_set"), (d.defect_id, v)
        assert _FABA2232 not in d.also_present_in, d.defect_id


def test_the_unregistered_sha_control_does_read_indeterminate():
    """The no-indeterminate assertion can fail: a sha in no set reads `indeterminate` on every entry.

    Class: every REGISTRY entry that carries a `fixed_in` (all of them), resolved at a sha no set
    holds. Why: an assertion over a population that is `indeterminate` for no reason would pass
    vacuously; this control varies the property under test (membership in the sets), not a token.
    What kills it: a registry that resolves an unknown sha to `present` or `absent` without
    ancestry. Second correct implementation: resolving through a failing `is_ancestor` stub.
    """
    for did, d in REGISTRY_BY_ID.items():
        v = resolve(d, _UNREGISTERED_SHA)
        assert v.status == "indeterminate", (did, v)
        assert v.rule == "ancestry_unresolvable", (did, v)


def test_a_native_clone_carrying_the_faba2232_key_line_reads_the_coupling_list(tmp_path):
    """`triton_coupling_read` returns `"list"` for a native directory whose `config_utils.h` has the key.

    Class: the native branch of `triton_coupling_read`, the judgement this experiment relies on
    because it runs no container. Why: a native clone is judged by its own source, so no
    container-sha registration is needed for `faba2232`. The line is the one ORNL carries at
    `faba2232` (`src/config_utils.h:536`). What kills it: a native branch that consults the
    container sha sets, or a key constant that no longer matches the solver's line. Second correct
    implementation: asserting `SWMM_COUPLING_NODES_READ in line` and the return value separately.
    """
    src = tmp_path / "src"
    src.mkdir()
    key_line = '\t\targlist.swmm_coupling_nodes = argsd("swmm_coupling_nodes", argmap, "");\n'
    assert SWMM_COUPLING_NODES_READ in key_line
    (src / "config_utils.h").write_text("// stub of the solver's argument reader\n" + key_line)
    assert (
        triton_coupling_read(
            execution_environment="native",
            software_directory=tmp_path,
            branch_key=_FABA2232,
            recorded_sha=None,
        )
        == "list"
    )
