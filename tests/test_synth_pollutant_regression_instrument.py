"""Gate for the synthetic model's defaulted-off pollutant knob.

The problem this file solves. The coupled solver refuses to write a snapshot when
it walks routed pollutant mass at an outfall that has no `RouteTo` — the guard is
`Nobjects[POLLUT] > 0` and the refusal needs `routeTo < 0`. The synthetic model
satisfies the second conjunct already (its single outfall declares no `RouteTo`)
and not the first (it declares no pollutant), so the defect cannot occur on it and
a green bit-for-bit verdict from this model says nothing about that defect class.
`SyntheticModelParams.pollutant_name` completes the trigger on demand.

Adding that knob puts a new field in the one dataclass every synthetic family's
inputs are generated from, so the thing most worth guarding is that the knob's
OFF state changed nothing: every existing family was run against the no-pollutant
model, and a perturbed default would silently turn each cross-family comparison
into a comparison across two different models.

So this file carries two assertions and, for each, a companion that proves the
assertion can fail. The companions are the point: a byte-identity assertion whose
golden was taken from the code it is asserting on is green by construction, and
without a named-and-run change that reds it there is no evidence it constrains
anything at all.

Venue note. These tests generate inputs only — no solver is compiled and none is
run, so they are not simulation-bearing and carry no HPC-venue requirement. What
they establish is a property of the emitted `.inp` text. That `Nobjects[POLLUT]`
is then > 0 inside the C engine follows from SWMM's own parse of a `[POLLUTANTS]`
section and is NOT re-measured here, because measuring it needs a solver run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from hhemt.synthetic_model import swmm_template
from hhemt.synthetic_model._build import SyntheticModelParams, build_synthetic_case

# sha256 of each default-parameter variant, taken from the generator as it stood
# BEFORE the knob existed (base 5e83c4c6, two independent builds agreeing). These
# are the operand of the byte-identity gate: the knob is a no-op at its default iff
# the post-change generator still reproduces them.
_PRE_CHANGE_INP_SHA256 = {
    "swmm_hydraulics.inp": "9c583983cb5f743356be6869243947f61063000a66599d42533b6175e39d8e0a",
    "swmm_hydrology.inp": "40a51e39d2dd15110007588acf8f2e61c21b530c7800959bc36d827089408a04",
    "swmm_full.inp": "9fabedff6f478cc0080bddcd3df5a5206f91ec38fb2fce7c3d5487321dfdc9a4",
}

# Scoped to the `.inp` variants deliberately. The knob reaches no other artifact,
# and the raster/NetCDF artifacts embed library-version metadata that would make a
# byte golden over them flaky for reasons unrelated to this change.
_VARIANTS = tuple(_PRE_CHANGE_INP_SHA256)


def _inp_digests(params: SyntheticModelParams, dest: Path) -> dict[str, str]:
    build_synthetic_case(params, dest)
    return {v: hashlib.sha256((dest / v).read_bytes()).hexdigest() for v in _VARIANTS}


def _section(text: str, header: str) -> str:
    """Return one `[SECTION]` block of an `.inp`, header line excluded.

    Reads the emitted TEXT rather than a parsed frame because the property under
    test is what the file DECLARES: a reader that normalizes a section to its full
    declared column set would report a `RouteTo` column for a file that names none,
    which is the one thing these assertions must be able to tell apart.
    """
    start = text.index(header) + len(header)
    rest = text[start:]
    end = rest.find("\n[")
    return rest if end < 0 else rest[:end]


def _always_emit_pollutant(params):  # noqa: ARG001 — signature parity with the real seam
    """The rejected implementation: emit the pollutant whatever the knob says.

    This is the alternative the design considered and refused (an unconditional
    pollutant, no knob). It is kept here as an executable mutation so the
    byte-identity gate has a named change that reds it.
    """
    return pd.DataFrame(
        {
            "MassUnits": ["MG/L"],
            "RainConcen": [0.0],
            "GWConcen": [0.0],
            "I&IConcen": [0.0],
            "DecayCoeff": [0.0],
            "SnowOnly": ["NO"],
            "CoPollutName": ["*"],
            "CoPollutFraction": [0.0],
            "DWFConcen": [0.0],
            "InitConcen": [0.0],
        },
        index=pd.Index(["TSS"], name="Name"),
    )


def test_default_inp_is_byte_identical_to_the_pre_knob_generator(tmp_path):
    """Property: at the default parameter every emitted `.inp` is byte-unchanged.

    Class: NEW CAPABILITY. A pre-change run of this assertion is green by
    construction — the golden was taken from the pre-change generator — so its
    pre-change state carries no information and the regression bar does not apply.
    The plausible wrong implementation is an unconditional pollutant, and
    `test_the_byte_identity_gate_reds_under_an_unconditional_pollutant` runs it and
    shows this assertion red under it.

    Why required: every synthetic family in the benchmarking record was generated
    from the no-pollutant model. If the default emission moves, those families stop
    being comparable to anything generated afterwards, and nothing else in the
    suite would report it — the other synth tests assert on model BEHAVIOUR, which
    a formatting-level perturbation of the `.inp` can leave intact.

    What kills it: emitting the `[POLLUTANTS]` section without consulting
    `params.pollutant_name`; or any other edit to the generator package that
    changes default-parameter `.inp` bytes.

    A second correct implementation under which it still passes: gate the knob the
    same way but emit the section by appending text post-save (the route
    `_inject_double_comment_separators` already takes) instead of via
    `m.inp.pollutants`. The default path emits nothing under either, so the goldens
    hold.
    """
    got = _inp_digests(SyntheticModelParams(), tmp_path)
    assert got == _PRE_CHANGE_INP_SHA256


def test_the_byte_identity_gate_reds_under_an_unconditional_pollutant(tmp_path, monkeypatch):
    """Property: the byte-identity gate above detects a perturbed default.

    This is the discrimination run forward, and it is what makes the gate evidence
    rather than ceremony. It replaces the one seam that owns the on/off decision
    with the rejected unconditional implementation and asserts the default-parameter
    digests then differ from the goldens.

    Class: NEW CAPABILITY — there is no pre-change seam to patch. The named change
    is to what the generator DOES (which frame it writes), never to how it is
    invoked, so it discharges the invocation-isolation requirement by discrimination
    rather than by re-running under a second tree.

    What kills it: restoring the knob's gate inside `_write_variant` instead of
    inside `_pollutants_df`, which would leave nothing to patch and make this test
    pass vacuously — it would stop being able to red.

    A second correct implementation under which it still passes: compare whole file
    bytes instead of sha256 digests; the inequality it asserts is unchanged.
    """
    monkeypatch.setattr(swmm_template, "_pollutants_df", _always_emit_pollutant)
    got = _inp_digests(SyntheticModelParams(), tmp_path)
    assert got != _PRE_CHANGE_INP_SHA256
    for variant in _VARIANTS:
        assert "[POLLUTANTS]" in (tmp_path / variant).read_text(encoding="utf-8")


@pytest.mark.parametrize("variant", _VARIANTS)
def test_non_default_completes_the_defect_trigger(tmp_path, variant):
    """Property: with the knob set, the `.inp` declares a pollutant AND no RouteTo.

    Both conjuncts of the solver's refusal condition, asserted together, because
    either alone leaves the instrument unable to exhibit the defect: a pollutant
    with a routed outfall never reaches the refusal, and a RouteTo-less outfall
    with no pollutant never enters the guarded traversal.

    Class: NEW CAPABILITY. Pre-change the knob does not exist, so the parameter
    cannot be constructed and the run is a TypeError rather than an assertion
    failure — a construction error, which carries no information. The plausible
    wrong implementation is declaring the pollutant and then "completing" the
    outfall row with a `RouteTo`, which satisfies the visible half and silently
    destroys the trigger; `test_the_trigger_gate_reds_when_the_outfall_is_routed`
    runs that and shows this assertion red under it.

    Why required: this is the only property that makes the synthetic tier able to
    exhibit the defect class at all. Without it the tier's green verdicts are
    silent about the defect rather than evidence against it.

    The RouteTo clause is anchored on a property that exists in BOTH the pre-change
    and post-change generator — the outfall frame has always been RouteTo-less — so
    it asserts a behaviour rather than wording this change introduced.

    What kills it: adding `RouteTo` (or the `TideGate`+`RouteTo` pair swmmio
    declares for OUTFALLS) to `_outfalls_df`; or gating the pollutant emission on
    something other than the knob so a set knob emits nothing.

    A second correct implementation under which it still passes: emit two
    pollutants, or a different name and mass unit. The assertion is `>= 1` row plus
    the requested name, because the solver guard is a count test — pinning exactly
    one row would constrain this implementation rather than the behaviour.
    """
    import swmmio

    params = SyntheticModelParams(pollutant_name="TSS")
    build_synthetic_case(params, tmp_path)
    path = tmp_path / variant
    text = path.read_text(encoding="utf-8")

    assert "[POLLUTANTS]" in text
    pollutants = swmmio.Model(str(path)).inp.pollutants
    assert len(pollutants) >= 1
    assert params.pollutant_name in set(pollutants.index)

    outfalls = _section(text, "[OUTFALLS]")
    data_rows = [ln for ln in outfalls.splitlines() if ln.strip() and not ln.lstrip().startswith(";")]
    assert data_rows, "the model must still declare an outfall for the trigger to be reachable"
    assert "RouteTo" not in outfalls


def test_the_trigger_gate_reds_when_the_outfall_is_routed(tmp_path, monkeypatch):
    """Property: the trigger assertion above detects a routed outfall.

    The second discrimination, and it guards the half that is easy to break while
    looking more complete: giving the outfall a `RouteTo` makes the `[OUTFALLS]`
    section match SWMM's full declared column set and leaves the pollutant in
    place, so every visible sign of the instrument survives while the refusal it
    exists to provoke becomes unreachable.

    Class: NEW CAPABILITY — the patched seam is reached only with the knob set.
    The named change is to what the generator emits, not to how it is invoked.

    What kills it: asserting the RouteTo clause against a parsed frame instead of
    the emitted text, since a reader that normalizes OUTFALLS to its declared
    column set reports `RouteTo` for a file that names none and the assertion could
    no longer tell the two files apart.

    A second correct implementation under which it still passes: route the outfall
    by adding only `RouteTo`, or by adding the `TideGate`+`RouteTo` pair; both emit
    the literal this asserts on.
    """
    real_outfalls_df = swmm_template._outfalls_df

    def routed_outfalls_df(params):
        """Wrap the real outfall frame and route it, adding the pair swmmio declares.

        `TideGate` comes along because SWMM's `[OUTFALLS]` places it before
        `RouteTo`; emitting `RouteTo` alone would put the value in the gate's
        column and route nothing, which would make this mutation pass for the
        wrong reason.
        """
        frame = real_outfalls_df(params).copy()
        frame["TideGate"] = "NO"
        frame["RouteTo"] = "S1"
        return frame

    monkeypatch.setattr(swmm_template, "_outfalls_df", routed_outfalls_df)
    build_synthetic_case(SyntheticModelParams(pollutant_name="TSS"), tmp_path)
    text = (tmp_path / "swmm_full.inp").read_text(encoding="utf-8")

    assert "[POLLUTANTS]" in text, "the pollutant half must survive, or this proves nothing"
    assert "RouteTo" in _section(text, "[OUTFALLS]")
