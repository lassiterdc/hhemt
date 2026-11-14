"""Unit coverage for the SWMM ``THREADS`` rewrite.

Asserts that ``analysis_config.n_omp_threads`` reaches the ``THREADS`` key of a SWMM
``.inp`` OPTIONS section, across several values. This runs in the fast tier: it calls
``update_swmm_threads_in_inp_file`` directly against a tmp ``.inp`` and a two-attribute
stub, so it needs no solver build, no scenario preparation, no SWMM execution and no
example data. The INTEGRATION property -- that the rewrite is wired into scenario
preparation for both ``hydro.inp`` and ``full.inp`` -- is asserted separately by
``test_synth_04_multisim_with_snakemake.py::test_snakemake_workflow_end_to_end``.

NOT COVERED HERE, deliberately. ``update_swmm_threads_in_inp_file`` returns having
changed nothing, and reports nothing, when the OPTIONS section carries no ``THREADS``
key -- its own docstring calls that "backward compatible".
``test_threads_absent_leaves_the_file_byte_unchanged`` PINS that contract so a future
change to it shows up as a test edit; it does not endorse it. What the pin does NOT
close is a template that DROPS the token: production would silently stop threading
while every test here stayed green, because each test supplies its own input. Closing
that needs a token-presence guard over the shipped templates, which is not this
module's job.

Assertions compare the PARSED value of the THREADS key, never the rendered line. Two
column widths are in play and neither is safe to assert against: the SEED's, which each
test below sets deliberately and which
``test_rewrite_is_independent_of_the_seed_files_padding`` varies on purpose, and the
REWRITTEN line's, which is a hardcoded 14-space literal in
``update_swmm_threads_in_inp_file`` (``scenario_inputs.py:261``) and which moves if that
literal changes. A substring assertion would couple to whichever of the two it happened
to be written against, and would then pass or fail on rendering rather than on value.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.scenario_inputs import ScenarioInputGenerator

# Differs from every value asserted below, so no case can pass on the seed alone.
_SEED_THREADS = 2

_INP_TEMPLATE = """[TITLE]
;;Project Title/Notes

[OPTIONS]
;;Option             Value
FLOW_UNITS           CMS
{threads_line}
IGNORE_QUALITY       YES

[REPORT]
INPUT                NO
"""


def _write_inp(tmp_path: Path, threads_line: str) -> Path:
    target = tmp_path / "full.inp"
    target.write_text(_INP_TEMPLATE.format(threads_line=threads_line), encoding="utf-8")
    return target


def _generator_for(n_omp_threads: int) -> ScenarioInputGenerator:
    """The two attributes ScenarioInputGenerator.__init__ binds, and nothing else."""
    stub_scenario = SimpleNamespace(
        _analysis=SimpleNamespace(cfg_analysis=SimpleNamespace(n_omp_threads=n_omp_threads)),
        _system=None,
    )
    return ScenarioInputGenerator(stub_scenario)


def _threads_value(inp_text: str) -> str | None:
    """Parsed THREADS value from the OPTIONS section, or None when the key is absent."""
    in_options = False
    for line in inp_text.splitlines():
        if "[OPTIONS]" in line:
            in_options = True
            continue
        if in_options and line.startswith("["):
            break
        if in_options and line.strip().startswith("THREADS"):
            parts = line.split()
            return parts[1] if len(parts) > 1 else None
    return None


@pytest.mark.parametrize("n_omp_threads", [1, 4, 8])
def test_threads_value_from_config_reaches_the_inp(tmp_path, n_omp_threads):
    inp = _write_inp(tmp_path, f"THREADS              {_SEED_THREADS}")
    _generator_for(n_omp_threads).update_swmm_threads_in_inp_file(inp)
    assert _threads_value(inp.read_text()) == str(n_omp_threads)


def test_rewrite_is_independent_of_the_seed_files_padding(tmp_path):
    inp = _write_inp(tmp_path, f"THREADS   {_SEED_THREADS}")
    _generator_for(4).update_swmm_threads_in_inp_file(inp)
    assert _threads_value(inp.read_text()) == "4"


def test_threads_absent_leaves_the_file_byte_unchanged(tmp_path):
    inp = _write_inp(tmp_path, "MIN_SLOPE            0")
    before = inp.read_bytes()
    _generator_for(8).update_swmm_threads_in_inp_file(inp)
    assert inp.read_bytes() == before
