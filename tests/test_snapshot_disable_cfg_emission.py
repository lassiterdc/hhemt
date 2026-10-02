"""The coupled TRITON cfg carries `swmm_snapshot_disable=1` iff the arm asked for it.

This module is deliberately FIXTURE-FREE and imports no `*_compiled` fixture, so it is not
compile-bearing and runs in any venue. The property under test is a text transform over the
REAL shipped cfg template, and every assertion here reads a WRITTEN FILE back off disk rather
than inspecting the template — because `utils.fill_template` routes to
`string.Template.safe_substitute`, which sits between the template and the emitted bytes and
can leave an unmapped placeholder in place SILENTLY. A template-level assertion would not
observe that; a read-back of the written file does.

Two invariants are load-bearing beyond this arm:

1. DEFAULT PRESERVATION. With the field unset, the emitted cfg must be byte-identical to what
   the pre-change code emitted. That is asserted positively (no key anywhere) and structurally
   (the transform returns its input object unchanged).
2. THE TRITON-ONLY CFG NEVER CARRIES THE KEY. One template feeds two mapping dicts, so this is
   the cross-arm corruption that a one-sided edit would cause. It is asserted over the live
   source of both generators rather than over a comment promising it.
"""

import ast
import re
from pathlib import Path

import pytest

from hhemt import utils
from hhemt.scenario import SWMM_SNAPSHOT_DISABLE_KEY, append_swmm_snapshot_disable
from hhemt.synthetic_model import triton_cfg

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SCENARIO_SRC = Path(utils.__file__).parent / "scenario.py"


def _template_placeholders() -> set[str]:
    return set(_PLACEHOLDER.findall(triton_cfg._TEMPLATE))


def _filled_cfg(tmp_path: Path) -> str:
    """Fill the real template with a value per placeholder, via the real filler.

    The mapping is DERIVED from the template's own placeholder set rather than restated, so a
    future placeholder cannot make this helper silently emit a literal `${...}` and leave the
    tests below asserting about a corrupted baseline.
    """
    template = tmp_path / "tritonswmm.cfg"
    template.write_text(triton_cfg._TEMPLATE, encoding="utf-8")
    mapping = {name: f"VALUE_{name}" for name in _template_placeholders()}
    return utils.fill_template(template, mapping)


def test_template_has_no_unmapped_placeholder_today(tmp_path):
    """The baseline every other test here rests on: no literal `${...}` survives the filler.

    If this fails, `safe_substitute` has left a placeholder in place and the "byte-identical"
    claim below would be measured against an already-corrupt baseline.
    """
    filled = _filled_cfg(tmp_path)
    assert not _PLACEHOLDER.findall(filled), (
        f"unmapped placeholder(s) {_PLACEHOLDER.findall(filled)} survived fill_template; "
        f"safe_substitute leaves these as LITERAL text in the emitted cfg with no error."
    )


def test_both_mapping_dicts_cover_every_template_placeholder():
    """R3's hazard, guarded for any FUTURE placeholder rather than only for today's set.

    One template is consumed by BOTH `_generate_TRITON_SWMM_cfg` and `_generate_TRITON_cfg`.
    A placeholder added for one mapping is emitted into the OTHER cfg as the literal `${...}`,
    silently, because `safe_substitute` does not raise on an unmapped key. This is the static
    half of that guard: every placeholder is a key of BOTH dicts.
    """
    tree = ast.parse(_SCENARIO_SRC.read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "TRITONSWMM_scenario")
    wanted = {"_generate_TRITON_SWMM_cfg", "_generate_TRITON_cfg"}
    found: dict[str, set[str]] = {}
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name not in wanted:
            continue
        for node in ast.walk(fn):
            # the `mapping = dict(KEY=..., ...)` literal in each generator
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "dict"
                and node.keywords
            ):
                found[fn.name] = {kw.arg for kw in node.keywords if kw.arg}
    assert found.keys() == wanted, f"could not locate both mapping dicts; found {sorted(found)}"
    placeholders = _template_placeholders()
    for name, keys in found.items():
        missing = sorted(placeholders - keys)
        assert not missing, (
            f"{name}'s mapping omits template placeholder(s) {missing}. safe_substitute emits "
            f"these into that generator's cfg as LITERAL '${{...}}' text with no error."
        )


@pytest.mark.parametrize("trailing_newline", [True, False])
def test_unset_field_returns_the_input_object_unchanged(trailing_newline):
    """Default preservation, asserted on OBJECT IDENTITY so no byte can differ."""
    text = "a=1\nb=2\n" if trailing_newline else "a=1\nb=2"
    assert append_swmm_snapshot_disable(text, False) is text


def test_unset_field_emits_a_cfg_with_no_snapshot_disable_key(tmp_path):
    """An A1/A2-shaped member: the written FILE carries no trace of the key."""
    out = tmp_path / "config_unset.cfg"
    out.write_text(append_swmm_snapshot_disable(_filled_cfg(tmp_path), False), encoding="utf-8")
    assert SWMM_SNAPSHOT_DISABLE_KEY not in out.read_text(encoding="utf-8")


def test_set_field_emits_the_key_at_line_start_with_value_one(tmp_path):
    """An A3-shaped member: the written FILE carries the literal arm-membership evidence.

    `=1` and not `=0` is the whole point — the solver reads the value through
    `atoi(argsd(..., "0"))`, so `0` and ABSENT are the same thing to it and only `1` arms the arm.
    """
    out = tmp_path / "config_set.cfg"
    out.write_text(append_swmm_snapshot_disable(_filled_cfg(tmp_path), True), encoding="utf-8")
    lines = out.read_text(encoding="utf-8").splitlines()
    hits = [ln for ln in lines if ln.startswith(f"{SWMM_SNAPSHOT_DISABLE_KEY}=")]
    assert hits == [f"{SWMM_SNAPSHOT_DISABLE_KEY}=1"], (
        f"expected exactly one line `{SWMM_SNAPSHOT_DISABLE_KEY}=1` at column 0; got {hits}. "
        f"Arm membership for the b4b re-run is established by this literal, so a commented, "
        f"indented or differently-valued line unassigns the member from its arm."
    )


def test_the_two_arms_differ_in_exactly_one_line(tmp_path):
    """Attack target 1, read from two WRITTEN FILES rather than from the template.

    A template diff cannot establish this — `safe_substitute` sits between the template and the
    emitted bytes. These are the two files two arms would actually hand the solver.
    """
    base = _filled_cfg(tmp_path)
    a2 = tmp_path / "a2.cfg"
    a3 = tmp_path / "a3.cfg"
    a2.write_text(append_swmm_snapshot_disable(base, False), encoding="utf-8")
    a3.write_text(append_swmm_snapshot_disable(base, True), encoding="utf-8")
    left = a2.read_text(encoding="utf-8").splitlines()
    right = a3.read_text(encoding="utf-8").splitlines()
    assert right[: len(left)] == left, "the armed cfg must be a pure suffix-extension of the unarmed one"
    assert right[len(left) :] == [f"{SWMM_SNAPSHOT_DISABLE_KEY}=1"]


def test_append_is_idempotent(tmp_path):
    """A re-prepared scenario must not accumulate duplicate key lines."""
    once = append_swmm_snapshot_disable(_filled_cfg(tmp_path), True)
    assert append_swmm_snapshot_disable(once, True) == once


def test_only_the_coupled_generator_appends_the_key():
    """The TRITON-only cfg can never carry the key, asserted over live source.

    The key gates the SWMM coupling surface, so it is meaningless in a TRITON-only cfg. This is
    the half a comment cannot hold: it names the single call site and fails if a second appears.
    """
    tree = ast.parse(_SCENARIO_SRC.read_text(encoding="utf-8"))
    callers = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "append_swmm_snapshot_disable"
            ):
                callers.add(fn.name)
    assert callers == {"_generate_TRITON_SWMM_cfg"}, (
        f"append_swmm_snapshot_disable is called from {sorted(callers)}; it must be called ONLY "
        f"from the COUPLED generator. A call from _generate_TRITON_cfg would put a "
        f"SWMM-coupling key into a cfg that runs no SWMM."
    )


def test_the_field_defaults_to_false_on_analysis_config():
    """Default preservation at the CONFIG layer, so an omitted field cannot arm an arm."""
    from hhemt.config.analysis import analysis_config

    field = analysis_config.model_fields["swmm_snapshot_disable"]
    assert field.default is False
    assert not field.is_required()


def test_the_field_is_bucketed_experiment_for_the_reprex_guide():
    """A reproducer must KEEP this value, never Supply or Amend it.

    Amending it converts a replay arm into a snapshot arm — the one comparison the arm exists
    to make — so the bucket is part of the arm's definition rather than bookkeeping.
    """
    from hhemt.config.reprex_taxonomy import all_field_bucket

    assert all_field_bucket("swmm_snapshot_disable") == "experiment"
