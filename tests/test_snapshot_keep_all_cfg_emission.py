"""The coupled TRITON cfg carries `swmm_snapshot_keep_all=1` iff the arm asked for it.

Sibling module to `test_snapshot_disable_cfg_emission.py`, deliberately FIXTURE-FREE and
importing no `*_compiled` fixture, so it is not compile-bearing and runs in any venue. The
property under test is a text transform over the REAL shipped cfg template, and every
assertion reads a WRITTEN FILE back off disk rather than inspecting the template — because
`utils.fill_template` routes to `string.Template.safe_substitute`, which sits between the
template and the emitted bytes and can leave an unmapped placeholder in place SILENTLY.

Four properties are pinned here, and the FIRST is the one that makes this change non-breaking
for every analysis that already exists:

1. DEFAULT OFF, BYTE-IDENTICAL. With the field unset the emitted cfg must be byte-identical to
   what the pre-change code emitted. Asserted three ways: on OBJECT IDENTITY through the
   single helper, on object identity through the COMPOSED pair of appends (which is the real
   emission path, so composing is what the call site actually does), and positively by reading
   a written file back and finding no trace of the key.
2. ON WRITES THE KEY EXACTLY ONCE, at column 0, with the value `1` and not `0` — the solver
   reads it through `atoi(argsd(..., "0"))`, so `0` and ABSENT are the same thing to it and
   only `1` arms the arm. Re-generation does not accumulate a duplicate line.
3. THE FIELD IS DECLARED LIKE ITS SIBLING: a `bool` defaulting to False, not required, and
   carrying a description. `field_meta()` is deliberately absent on BOTH fields and that is
   correct rather than an omission — it builds `applies_when`/`required_when`/`options`, all
   three of which are empty for an unconditional `bool` with no closed-set glossary, so
   `field_meta()` would return `{}` and add nothing. This test asserts the sibling PARITY
   rather than restating a declaration form, so the two fields cannot drift apart.
4. THE FLAG SURVIVES THE PER-SCENARIO REGENERATION PATH the same way the sibling does: one
   call site, in the COUPLED generator only, reading the CONFIG FIELD rather than a literal.

A fifth property is asserted because the two flags are NOT inverses and a reader will assume
they are: they are independently settable, and the toolkit arbitrates nothing between them.
"""

import ast
import re
from pathlib import Path

import pytest

from hhemt import utils
from hhemt.scenario import (
    SWMM_SNAPSHOT_DISABLE_KEY,
    SWMM_SNAPSHOT_KEEP_ALL_KEY,
    append_swmm_snapshot_disable,
    append_swmm_snapshot_keep_all,
)
from hhemt.synthetic_model import triton_cfg

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SCENARIO_SRC = Path(utils.__file__).parent / "scenario.py"


def _filled_cfg(tmp_path: Path) -> str:
    """Fill the real template with a value per placeholder, via the real filler.

    The mapping is DERIVED from the template's own placeholder set rather than restated, so a
    future placeholder cannot make this helper silently emit a literal `${...}` and leave the
    tests below asserting about a corrupted baseline.
    """
    template = tmp_path / "tritonswmm.cfg"
    template.write_text(triton_cfg._TEMPLATE, encoding="utf-8")
    mapping = {name: f"VALUE_{name}" for name in set(_PLACEHOLDER.findall(triton_cfg._TEMPLATE))}
    return utils.fill_template(template, mapping)


# ---------------------------------------------------------------- property 1: default off


@pytest.mark.parametrize("trailing_newline", [True, False])
def test_unset_field_returns_the_input_object_unchanged(trailing_newline):
    """Default preservation, asserted on OBJECT IDENTITY so no byte can differ."""
    text = "a=1\nb=2\n" if trailing_newline else "a=1\nb=2"
    assert append_swmm_snapshot_keep_all(text, False) is text


def test_both_flags_off_returns_the_input_object_through_the_composed_pair(tmp_path):
    """The REAL emission path is the two appends COMPOSED, so compose them here.

    Asserting each helper's no-op in isolation leaves the composition untested, and the
    composition is what `_generate_TRITON_SWMM_cfg` runs. Object identity through both is the
    strongest available form of "byte-identical to the pre-change code": not merely equal
    bytes, but the SAME str object the pre-change post-processing block held.
    """
    base = _filled_cfg(tmp_path)
    out = append_swmm_snapshot_keep_all(append_swmm_snapshot_disable(base, False), False)
    assert out is base


def test_unset_field_emits_a_cfg_with_no_keep_all_key(tmp_path):
    """Positively: the written FILE carries no trace of the key."""
    out = tmp_path / "config_unset.cfg"
    out.write_text(append_swmm_snapshot_keep_all(_filled_cfg(tmp_path), False), encoding="utf-8")
    assert SWMM_SNAPSHOT_KEEP_ALL_KEY not in out.read_text(encoding="utf-8")


# ------------------------------------------------- property 2: on writes the key exactly once


def test_set_field_emits_the_key_at_line_start_with_value_one(tmp_path):
    """The written FILE carries the literal arm-membership evidence, exactly once."""
    out = tmp_path / "config_set.cfg"
    out.write_text(append_swmm_snapshot_keep_all(_filled_cfg(tmp_path), True), encoding="utf-8")
    lines = out.read_text(encoding="utf-8").splitlines()
    hits = [ln for ln in lines if ln.startswith(f"{SWMM_SNAPSHOT_KEEP_ALL_KEY}=")]
    assert hits == [f"{SWMM_SNAPSHOT_KEEP_ALL_KEY}=1"], (
        f"expected exactly one line `{SWMM_SNAPSHOT_KEEP_ALL_KEY}=1` at column 0; got {hits}. "
        f"Arm membership is established by this literal, so a commented, indented or "
        f"differently-valued line unassigns the member from its arm. `=0` in particular is "
        f'indistinguishable from ABSENT to the solver\'s atoi(argsd(..., "0")) read.'
    )


def test_the_two_arms_differ_in_exactly_one_line(tmp_path):
    """Read from two WRITTEN FILES rather than from the template.

    A template diff cannot establish this — `safe_substitute` sits between the template and
    the emitted bytes. These are the two files two arms would actually hand the solver.
    """
    base = _filled_cfg(tmp_path)
    off, on = tmp_path / "off.cfg", tmp_path / "on.cfg"
    off.write_text(append_swmm_snapshot_keep_all(base, False), encoding="utf-8")
    on.write_text(append_swmm_snapshot_keep_all(base, True), encoding="utf-8")
    left = off.read_text(encoding="utf-8").splitlines()
    right = on.read_text(encoding="utf-8").splitlines()
    assert right[: len(left)] == left, "the armed cfg must be a pure suffix-extension of the unarmed one"
    assert right[len(left) :] == [f"{SWMM_SNAPSHOT_KEEP_ALL_KEY}=1"]


def test_append_is_idempotent(tmp_path):
    """A re-prepared scenario must not accumulate duplicate key lines."""
    once = append_swmm_snapshot_keep_all(_filled_cfg(tmp_path), True)
    assert append_swmm_snapshot_keep_all(once, True) == once


def test_idempotence_is_not_satisfied_by_a_substring_of_another_key(tmp_path):
    """The idempotence probe must anchor at LINE START, not match anywhere.

    A probe written as `KEY in cfg_content` would be satisfied by a longer key that merely
    CONTAINS this one, and would then silently refuse to emit the line for a member that asked
    for it. This pins the anchored form by feeding exactly that adversary.
    """
    decoy = f"a=1\nx_{SWMM_SNAPSHOT_KEEP_ALL_KEY}=1\n"
    out = append_swmm_snapshot_keep_all(decoy, True)
    assert out != decoy, (
        "a key whose name merely CONTAINS this one must not suppress the emission; the "
        "membership probe has to anchor at line start."
    )
    assert out.splitlines()[-1] == f"{SWMM_SNAPSHOT_KEEP_ALL_KEY}=1"


# ----------------------------------------------- property 3: field declared like its sibling


def test_the_field_defaults_to_false_on_analysis_config():
    """Default preservation at the CONFIG layer, so an omitted field cannot arm an arm."""
    from hhemt.config.analysis import analysis_config

    field = analysis_config.model_fields["swmm_snapshot_keep_all"]
    assert field.default is False
    assert not field.is_required()


def test_the_field_declaration_matches_its_siblings_shape():
    """Declaration PARITY with the sibling, so the two cannot drift apart.

    Asserted as parity rather than as a restated form: `field_meta()` is absent on BOTH
    fields, which is correct for an unconditional `bool` with no closed-set glossary
    (`field_meta()` would return `{}`), and parity is what keeps a future change to one
    field's declaration from leaving the other behind.
    """
    from hhemt.config.analysis import analysis_config

    new = analysis_config.model_fields["swmm_snapshot_keep_all"]
    sibling = analysis_config.model_fields["swmm_snapshot_disable"]
    assert new.annotation is sibling.annotation is bool
    assert new.default == sibling.default
    assert new.is_required() == sibling.is_required()
    assert (new.json_schema_extra is None) == (sibling.json_schema_extra is None)
    assert new.description, "the field must carry a description — it is the rendered-config-reference text"


def test_the_field_is_bucketed_experiment_for_the_reprex_guide():
    """A reproducer must KEEP this value, never Supply or Amend it.

    Amending it changes which resume mechanism the arm exercises — the one thing the arm
    exists to hold fixed — so the bucket is part of the arm's definition rather than
    bookkeeping. Sibling-consistent with `swmm_snapshot_disable`, also "experiment".
    """
    from hhemt.config.reprex_taxonomy import all_field_bucket

    assert all_field_bucket("swmm_snapshot_keep_all") == "experiment"


# ------------------------------------- property 4: survives the per-scenario regeneration path


def _call_sites(func_name: str) -> set[str]:
    tree = ast.parse(_SCENARIO_SRC.read_text(encoding="utf-8"))
    callers = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func_name:
                callers.add(fn.name)
    return callers


def test_only_the_coupled_generator_appends_the_key():
    """The TRITON-only cfg can never carry the key, asserted over live source.

    The key gates the SWMM coupling surface, so it is meaningless in a TRITON-only cfg. This
    is the half a comment cannot hold: it names the single call site and fails if a second
    appears. It is ALSO the guard a generalized flag-table helper would silently weaken — one
    shared helper carries N flags through ONE call site, after which you can no longer tell
    which flags that site emits.
    """
    assert _call_sites("append_swmm_snapshot_keep_all") == {"_generate_TRITON_SWMM_cfg"}, (
        f"append_swmm_snapshot_keep_all is called from "
        f"{sorted(_call_sites('append_swmm_snapshot_keep_all'))}; it must be called ONLY from "
        f"the COUPLED generator. A call from _generate_TRITON_cfg would put a SWMM-coupling "
        f"key into a cfg that runs no SWMM."
    )


def test_the_call_site_reads_the_config_field_not_a_literal():
    """The emission must be CONFIG-DRIVEN, not a hardcoded arm.

    A call site passing a literal `True` would emit the key for every analysis and would pass
    every other test in this module — the byte-identity property would be violated with no
    assertion observing it, because the helper itself is still correct. The defect lives at
    the call site, so it is asserted at the call site.
    """
    tree = ast.parse(_SCENARIO_SRC.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "append_swmm_snapshot_keep_all":
                found.append(node)
    assert len(found) == 1, f"expected exactly one call; found {len(found)}"
    (call,) = found
    assert len(call.args) == 2, f"expected two positional args; got {ast.dump(call)}"
    second = call.args[1]
    assert isinstance(second, ast.Attribute) and second.attr == "swmm_snapshot_keep_all", (
        f"the second argument must be the config field `...cfg_analysis.swmm_snapshot_keep_all`, "
        f"not {ast.dump(second)}. A literal there arms every analysis unconditionally."
    )


# ------------------------------------------------- fifth property: the flags are independent


@pytest.mark.parametrize(
    ("disable", "keep_all", "expect_disable_line", "expect_keep_all_line"),
    [
        (False, False, False, False),
        (True, False, True, False),
        (False, True, False, True),
        (True, True, True, True),
    ],
)
def test_the_two_flags_are_independently_emitted(
    tmp_path, disable, keep_all, expect_disable_line, expect_keep_all_line
):
    """All four combinations are expressible and the toolkit arbitrates NONE of them.

    The two flags are not inverses: one removes the snapshot path, the other changes its
    retention depth. Both-True is contradictory in EFFECT and is resolved BY THE SOLVER
    (disable wins — there is no stem to retain into). The toolkit's job is to emit what the
    configuration asked for; a toolkit-side guard that suppressed one key would make the
    emitted cfg stop being evidence of what the configuration said.
    """
    base = _filled_cfg(tmp_path)
    out = tmp_path / "combo.cfg"
    text = append_swmm_snapshot_keep_all(append_swmm_snapshot_disable(base, disable), keep_all)
    out.write_text(text, encoding="utf-8")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert (f"{SWMM_SNAPSHOT_DISABLE_KEY}=1" in lines) is expect_disable_line
    assert (f"{SWMM_SNAPSHOT_KEEP_ALL_KEY}=1" in lines) is expect_keep_all_line


def test_the_two_keys_are_distinct_spellings():
    """A copy-paste that left the sibling's key name in the new helper would pass most of this
    module — the emission, the idempotence and the one-line-diff tests would all still hold,
    against the WRONG key. This is the assertion that separates them."""
    assert SWMM_SNAPSHOT_KEEP_ALL_KEY != SWMM_SNAPSHOT_DISABLE_KEY
    assert SWMM_SNAPSHOT_KEEP_ALL_KEY == "swmm_snapshot_keep_all"
