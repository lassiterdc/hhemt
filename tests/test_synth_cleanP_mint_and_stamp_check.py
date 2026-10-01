"""The A1 (same-pin clean) mint name and the mint-time stamp-check instrument.

VENUE: these tests are HPC-FREE and compile-FREE by construction. They exercise two things only —
the analysis-NAME composition of ``clean_case`` (asserted without constructing a case) and
``check_arm_pin_stamp`` against zarr-v3 store fixtures this module writes itself. No fixture in the
``*_compiled`` family is reached, no solver is built, and no simulation runs.

WHY THE NAME TEST DOES NOT CONSTRUCT A CASE. ``clean_case`` materializes a synthetic model and a
``TRITONSWMM_system``, which compiles. The property under test is a pure string composition, so the
test asserts it against the function's own source rather than by running it — see
``test_variant_composes_the_mandated_a1_name`` for why that is the honest instrument here and what
it deliberately does not cover.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiments.synth_compute_config import (
    _MEMBER_STORE_GLOBS,
    _read_member_stamp,
    _shas_agree,
    check_arm_pin_stamp,
)

#: Pin P of the b4b re-run: WP-1C + its output_end C-linkage repair + the linkage-guard test.
_PIN_P = "e53c2fa01a64583fb57bc58082245fd687882b8f"
#: The DIFFERENT pin the already-existing clean tree declares. The two must never compare equal.
_PIN_OTHER = "a38338b09e62e57c936f51516bbdbe495d89a546"


def _write_v3_store(member_dir: Path, *, stamp: str | None, extra_attrs: dict | None = None) -> Path:
    """Write a minimal zarr-V3 per-member store. ``stamp=None`` writes NO stamp attribute.

    Deliberately writes ``zarr.json`` and NO ``.zattrs``, because that asymmetry IS the thing the
    instrument has to get right and a fixture carrying both would hide it.
    """
    store = member_dir / "analysis_datatree.zarr"
    store.mkdir(parents=True, exist_ok=True)
    attributes: dict[str, object] = {"Conventions": "CF-1.13", "analysis_id": member_dir.name}
    # The NEIGHBOURING attribute is populated on exactly the trees whose triton stamp is absent.
    # Every fixture carries it so a loose read cannot pass these tests by accident.
    attributes["hhemt_producing_sha"] = "0123456789abcdef0123456789abcdef01234567"
    if stamp is not None:
        attributes["triton_producing_sha"] = stamp
    if extra_attrs:
        attributes.update(extra_attrs)
    (store / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attributes}))
    return store


def _arm(tmp_path: Path, name: str, *, layout: str = "members", n: int = 3, stamp: str | None = _PIN_P) -> Path:
    arm_root = tmp_path / name
    for i in range(n):
        child = "member_" if layout == "members" else "sa_"
        _write_v3_store(arm_root / layout / f"{child}{i}", stamp=stamp)
    return arm_root


# --------------------------------------------------------------------------------------
# The A1 mint name
# --------------------------------------------------------------------------------------


def test_variant_composes_the_mandated_a1_name() -> None:
    """``variant='P'`` must compose exactly ``synth_cc_cleanP_tritonswmm``.

    The assertion is on the composition EXPRESSION in source, not on a constructed case, because
    constructing one compiles a solver. What this therefore does NOT cover: that ``_build_case``
    forwards the name unchanged. That is covered by reading the single call site, and it is stated
    as a limitation rather than implied to be tested.
    """
    source = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "synth_compute_config.py"
    text = source.read_text()
    assert 'analysis_name=f"synth_cc_clean{variant}_{model_arm}"' in text, (
        "clean_case must compose its analysis name from BOTH the variant infix and model_arm; "
        "a hardcoded name cannot mint A1 and a full override lets the arm silently disagree."
    )
    # The composition, evaluated the way the source evaluates it.
    for variant, model_arm, expected in (
        ("P", "tritonswmm", "synth_cc_cleanP_tritonswmm"),
        ("", "tritonswmm", "synth_cc_clean_tritonswmm"),
        ("", "triton", "synth_cc_clean_triton"),
    ):
        assert f"synth_cc_clean{variant}_{model_arm}" == expected


def test_default_variant_is_byte_identical_to_the_historical_name() -> None:
    """The default must not move any existing arm's path. ``variant=''`` is the historical name."""
    import inspect

    from scripts.experiments.synth_compute_config import clean_case

    assert inspect.signature(clean_case).parameters["variant"].default == ""


# --------------------------------------------------------------------------------------
# The stamp-check instrument — the PASS arm and the two STOP arms
# --------------------------------------------------------------------------------------


def test_pass_requires_every_store_stamped_and_equal(tmp_path: Path) -> None:
    arm = _arm(tmp_path, "synth_cc_cleanP_tritonswmm", n=4, stamp=_PIN_P)
    v = check_arm_pin_stamp(arm, _PIN_P)
    assert v["verdict"] == "PASS"
    assert v["examined"] == 4 and v["stamped"] == 4
    assert v["missing"] == [] and v["mismatched"] == {}


def test_one_absent_stamp_is_a_stop_not_a_pass(tmp_path: Path) -> None:
    """The measured base rate is intermittent (~32%), so a SINGLE unstamped member must STOP."""
    arm = _arm(tmp_path, "arm", n=3, stamp=_PIN_P)
    _write_v3_store(arm / "members" / "member_9", stamp=None)
    v = check_arm_pin_stamp(arm, _PIN_P)
    assert v["verdict"] == "STOP"
    assert v["examined"] == 4 and v["missing"] == ["member_9"]


def test_a_differing_pin_is_a_stop(tmp_path: Path) -> None:
    """A fully-stamped arm at the WRONG pin is the cross-pin confound presented as same-pin."""
    arm = _arm(tmp_path, "arm", n=3, stamp=_PIN_OTHER)
    v = check_arm_pin_stamp(arm, _PIN_P)
    assert v["verdict"] == "STOP"
    assert v["stamped"] == 3 and len(v["mismatched"]) == 3


# --------------------------------------------------------------------------------------
# ATTACK TARGET 3 — the instrument's failure modes, EXHIBITED
# --------------------------------------------------------------------------------------


def test_zero_stores_is_not_evaluated_and_never_a_pass(tmp_path: Path) -> None:
    """The disclosed-denominator guard.

    A verdict derived as "no bad store was found" is identical on "examined 30, all good" and
    "examined 0", and zero is the likelier of the two on a wrong root because every glob returns
    empty there. This is the single assertion standing between the instrument and one that cannot
    fail — the same class as the b4b criterion's own G1 non-empty guard.
    """
    empty = tmp_path / "typo_root"
    empty.mkdir()
    v = check_arm_pin_stamp(empty, _PIN_P)
    assert v["verdict"] == "NOT-EVALUATED"
    assert v["examined"] == 0
    assert v["verdict"] != "PASS"


def test_zattrs_is_absent_on_a_v3_store_so_a_v2_read_raises(tmp_path: Path) -> None:
    """EXHIBIT the false-absent read rather than asserting the instrument avoids it.

    On a zarr-v3 store there is no ``.zattrs``. A ``.zattrs``-based instrument raises
    ``FileNotFoundError`` on EVERY store including a correctly stamped one, so mapping that
    exception to "no stamp" yields a uniform definite absence that can never observe a present
    stamp. Both halves are asserted here: that the v2 read raises, and that the v3 read on the SAME
    store returns the stamp.
    """
    store = _write_v3_store(tmp_path / "members" / "member_0", stamp=_PIN_P)
    assert not (store / ".zattrs").exists()
    with pytest.raises(FileNotFoundError):
        (store / ".zattrs").read_text()
    assert _read_member_stamp(store) == _PIN_P


def test_whole_tree_grep_finds_the_literal_where_no_store_carries_a_value(tmp_path: Path) -> None:
    """EXHIBIT the grep false positive the instrument is specified against.

    Reproduces the measured shape of the different-pin clean tree: the literal
    ``triton_producing_sha`` appears in sibling report artifacts while ZERO per-member stores carry a
    value. A grep-based check reports that arm as stamped; the specified read reports it absent.
    """
    arm = _arm(tmp_path, "arm", n=3, stamp=None)
    # Sibling artifacts that mention the literal without any store carrying a value — the measured
    # population on the real tree was three render_bundle zips plus validation_report.json.
    (arm / "validation_report.json").write_text(json.dumps({"checks": [{"name": "triton_producing_sha"}]}))
    bundles = arm / "render_bundle"
    bundles.mkdir()
    for i in range(3):
        (bundles / f"bundle_{i}.txt").write_text("triton_producing_sha")

    grep_hits = [p for p in arm.rglob("*") if p.is_file() and "triton_producing_sha" in p.read_text()]
    assert len(grep_hits) >= 4, "fixture must reproduce the literal-present/value-absent shape"

    v = check_arm_pin_stamp(arm, _PIN_P)
    assert v["verdict"] == "STOP"
    assert v["stamped"] == 0, "the specified read must find ZERO stamped stores where grep finds the literal"


def test_an_empty_string_stamp_is_not_a_stamp(tmp_path: Path) -> None:
    """A falsy attribute value present in the namespace is a non-stamp, not a stamp."""
    store = _write_v3_store(tmp_path / "members" / "member_0", stamp="   ")
    assert _read_member_stamp(store) is None


def test_the_neighbouring_hhemt_stamp_is_never_accepted(tmp_path: Path) -> None:
    """``hhemt_producing_sha`` is populated on exactly the trees whose triton stamp is absent."""
    store = _write_v3_store(tmp_path / "members" / "member_0", stamp=None)
    attrs = json.loads((store / "zarr.json").read_text())["attributes"]
    assert attrs["hhemt_producing_sha"], "fixture must carry the populated neighbour"
    assert _read_member_stamp(store) is None


# --------------------------------------------------------------------------------------
# Layout and abbreviation coverage
# --------------------------------------------------------------------------------------


def test_both_member_layouts_are_read(tmp_path: Path) -> None:
    """``subanalyses/sa_*`` is the CORRECT name on pre-rename trees and must not be missed."""
    assert _MEMBER_STORE_GLOBS == ("members/member_*", "subanalyses/sa_*")
    legacy = _arm(tmp_path, "legacy", layout="subanalyses", n=3, stamp=_PIN_P)
    v = check_arm_pin_stamp(legacy, _PIN_P)
    assert v["verdict"] == "PASS" and v["examined"] == 3


def test_a_tree_carrying_both_layouts_is_not_double_counted(tmp_path: Path) -> None:
    """The denominator must be wrong in NEITHER direction; one real tree carries both namings."""
    arm = tmp_path / "both"
    for i in range(2):
        _write_v3_store(arm / "members" / f"member_{i}", stamp=_PIN_P)
        _write_v3_store(arm / "subanalyses" / f"sa_{i}", stamp=_PIN_P)
    v = check_arm_pin_stamp(arm, _PIN_P)
    assert v["examined"] == 4, "two layouts x two members = four distinct stores"


@pytest.mark.parametrize(
    ("stamped", "expected", "agree"),
    [
        (_PIN_P, _PIN_P, True),
        (_PIN_P, "e53c2fa", True),  # abbreviated pin must agree with the full stamp
        (_PIN_P, _PIN_OTHER, False),
        (_PIN_P, "e53c2f", False),  # below the 7-char floor: refused, not accepted loosely
        (_PIN_P, _PIN_P.upper(), True),  # case-insensitive
    ],
)
def test_sha_agreement_under_abbreviation(stamped: str, expected: str, agree: bool) -> None:
    assert _shas_agree(stamped, expected) is agree
