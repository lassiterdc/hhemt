"""``--debug`` makes a coupled clean start keep the previous run's snapshot files.

THE PROBLEM. When a coupled TRITON-SWMM simulation starts from the beginning, the solver
disposes of the previous run's SWMM state snapshot files before writing new ones; by default
it removes them, because a stale snapshot sitting at a live name could be loaded by a later
resume. A developer debugging a run wants them kept. The toolkit carries that wish to the
solver as one line in the coupled base cfg, ``swmm_snapshot_keep_previous=1``, written by the
launcher at every clean-start launch and present exactly when the run was invoked with
``debug``. These tests pin the helper that writes the line, the launcher's use of it, and the
flag's path from the run entry points to every run-simulation rule shell.

No test here compiles or runs a solver. The launcher arms stop at a sentinel raised just past
the hotstart branch; the generator arms build a Snakefile's text and run nothing.
"""

from __future__ import annotations

import inspect
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.run_simulation import TRITONSWMM_run
from hhemt.scenario import SWMM_SNAPSHOT_KEEP_PREVIOUS_KEY, set_swmm_snapshot_keep_previous

_LINE = f"{SWMM_SNAPSHOT_KEEP_PREVIOUS_KEY}=1"
_BASE = 'output_folder="out_tritonswmm"\ndem_filename="d"\nswmm_snapshot_keep_all=1\n'


def _count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip().startswith(f"{SWMM_SNAPSHOT_KEEP_PREVIOUS_KEY}="))


# --------------------------------------------------------------------------------------
# (i) the helper
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "before",
    [
        _BASE,
        _BASE.rstrip("\n"),  # no trailing newline
        _BASE + f"{SWMM_SNAPSHOT_KEEP_PREVIOUS_KEY}=0\n",  # a different value
        _BASE + f"{_LINE}\n{_LINE}\n",  # duplicates
    ],
)
def test_keep_true_leaves_exactly_one_line_and_keep_false_leaves_none(tmp_path, before):
    """Property: the helper leaves one ``=1`` line when asked to keep and none when not.

    Class: base-cfg texts with no line, no trailing newline, a different value, and duplicates.

    Why required, checkable against the code: the solver reads the key additively and takes
    the first occurrence, so a stale ``=0`` or a duplicate would make the cfg say something
    other than this invocation's flag.

    What kills it: an append-only helper (the sibling keys' shape), which cannot remove the
    line after a debug run, or one that appends without collapsing an existing line.

    A second correct implementation under which it still passes: one that rewrites the line
    in place when present rather than removing and re-appending it.
    """
    cfg = tmp_path / "TRITONSWMM.cfg"
    cfg.write_text(before)
    set_swmm_snapshot_keep_previous(cfg, True)
    after_keep = cfg.read_text()
    assert _count(after_keep) == 1 and f"\n{_LINE}\n" in f"\n{after_keep}"
    set_swmm_snapshot_keep_previous(cfg, False)
    assert _count(cfg.read_text()) == 0
    assert "swmm_snapshot_keep_all=1" in cfg.read_text()


@pytest.mark.parametrize("keep", [True, False])
def test_a_matching_file_is_not_rewritten(tmp_path, keep):
    """Property: when the file already matches the flag, its bytes and mtime are untouched.

    Class: a cfg already carrying the line (for keep) or not carrying it (for not keep).

    Why required, checkable against the code: the base cfg is written at every clean-start
    launch; rewriting an unchanged file would change its mtime for nothing.

    What kills it: an unconditional ``write_text``.

    A second correct implementation under which it still passes: one that hashes the text
    instead of comparing strings.
    """
    cfg = tmp_path / "TRITONSWMM.cfg"
    cfg.write_text(_BASE + (f"{_LINE}\n" if keep else ""))
    os.utime(cfg, (1_000_000_000, 1_000_000_000))
    before = (cfg.read_bytes(), cfg.stat().st_mtime_ns)
    assert set_swmm_snapshot_keep_previous(cfg, keep) is False
    assert (cfg.read_bytes(), cfg.stat().st_mtime_ns) == before


# --------------------------------------------------------------------------------------
# (ii)-(iv) the launcher
# --------------------------------------------------------------------------------------


class _Field:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Stop(Exception):
    pass


class _System:
    @property
    def cfg_system(self):
        raise _Stop


def _run(tmp_path: Path, *, with_hotstart_cfgs: bool, failed_replay: bool = False):
    sim = tmp_path / "sim"
    out = sim / "out_tritonswmm"
    (out / "cfg").mkdir(parents=True)
    (out / "swmm").mkdir()
    (sim / "out_triton" / "cfg").mkdir(parents=True)
    base = sim / "TRITONSWMM.cfg"
    base.write_text(_BASE)
    triton_base = sim / "TRITON.cfg"
    triton_base.write_text('output_folder="out_triton"\n')
    if with_hotstart_cfgs:
        for k in (4, 5):
            (out / "cfg" / f"config_{k}.cfg").write_text(_BASE)
            (out / "swmm" / f"hydraulics_state_cp{k}.snapshot").write_bytes(b"x")
            (sim / "out_triton" / "cfg" / f"config_{k}.cfg").write_text(_BASE)
    log = SimpleNamespace(force_rerun_pending=_Field(False), n_resumes=_Field(0), resume_reporting_tsteps=_Field(None))
    analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(
            multi_sim_run_method="local",
            run_mode="serial",
            n_mpi_procs=1,
            n_omp_threads=1,
            n_gpus=0,
            n_nodes=1,
            execution_environment="native",
            is_experiment_member=False,
            swmm_snapshot_failed_replay=failed_replay,
        ),
        cfg_hpc_system=None,
        analysis_paths=SimpleNamespace(simlog_directory=tmp_path / "logs" / "sims"),
        _system=_System(),
        # The model log's NAME is derived from the event's stable identity, so a stub that
        # reaches `run_simulation.model_logfile_for` (through `prepare_simulation_command`)
        # must carry the weather-index projection the production method provides. Only the
        # INDEXERS are supplied here -- the filename grammar stays in `run_simulation`, so a
        # stub can never become a second expression of the convention.
        _retrieve_weather_indexer_using_integer_index=lambda event_iloc: {"event": int(event_iloc)},
    )
    scen = SimpleNamespace(
        event_iloc=0,
        weather_event_indexers={},
        _analysis=analysis,
        scen_paths=SimpleNamespace(
            out_tritonswmm=out,
            out_triton=sim / "out_triton",
            swmm_hydraulics_inp=sim / "swmm" / "hydraulics.inp",
            triton_swmm_cfg=base,
            sim_tritonswmm_executable=sim / "build" / "triton.exe",
            triton_cfg=triton_base,
            sim_triton_executable=sim / "build" / "triton_only.exe",
        ),
        get_log=lambda model_type: log,
        model_run_completed=lambda model_type: False,
    )
    return TRITONSWMM_run(scen), base, triton_base


def _prepare(run, *, model_type: str, pickup: bool, debug: bool) -> None:
    with pytest.raises(_Stop):
        run.prepare_simulation_command(pickup_where_leftoff=pickup, verbose=False, model_type=model_type, debug=debug)


@pytest.mark.parametrize("pickup", [False, True])
def test_a_coupled_clean_start_writes_the_line_exactly_when_debug(tmp_path, pickup):
    """Property: a coupled clean start leaves the base cfg with the line under debug and without it otherwise.

    Class: a coupled launch that hands the solver the base cfg -- pickup off, and pickup on
    with no hotstart cfg to resume from -- run with debug True then False on one base cfg.

    Why required, checkable against the code: the base cfg is reused across invocations, so a
    plain run after a debug run must remove the line or it inherits the debug behaviour.

    What kills it: writing the line only when debug is True, or only on the pickup-off path.

    A second correct implementation under which it still passes: one that writes a fresh copy
    of the base cfg with the line rather than editing it in place, if it is the file the
    solver is handed.
    """
    run, base, _ = _run(tmp_path, with_hotstart_cfgs=False)
    _prepare(run, model_type="tritonswmm", pickup=pickup, debug=True)
    assert _count(base.read_text()) == 1
    _prepare(run, model_type="tritonswmm", pickup=pickup, debug=False)
    assert _count(base.read_text()) == 0


def test_a_hotstart_launch_leaves_the_base_cfg_byte_unchanged(tmp_path):
    """Property: a resume from a ``config_NNNN.cfg`` does not touch the base cfg, whatever the flag.

    Class: a coupled launch with published hotstart cfgs, under debug True.

    Why required, checkable against the code: the key governs the CLEAN-START disposal only;
    a resume is handed a hotstart cfg, so editing the base cfg there would change a file the
    launch does not use.

    What kills it: writing the line on every coupled launch.

    A second correct implementation under which it still passes: one that decides "clean
    start" from ``hotstart_cfg is None`` rather than by comparing the chosen cfg to the base
    cfg path.
    """
    run, base, _ = _run(tmp_path, with_hotstart_cfgs=True)
    before = base.read_bytes()
    _prepare(run, model_type="tritonswmm", pickup=True, debug=True)
    assert base.read_bytes() == before


@pytest.mark.parametrize("pickup", [False, True])
def test_a_triton_only_base_cfg_never_gains_the_line(tmp_path, pickup):
    """Property: the TRITON-only base cfg never carries the coupled snapshot key.

    Class: ``model_type="triton"`` clean start and resume, under debug True.

    What kills it: writing the key on every model type's base cfg.

    A second correct implementation under which it still passes: one that gates on the cfg
    path rather than on the model type.
    """
    run, base, triton_base = _run(tmp_path, with_hotstart_cfgs=pickup)
    _prepare(run, model_type="triton", pickup=pickup, debug=True)
    assert _count(triton_base.read_text()) == 0
    assert _count(base.read_text()) == 0


# --------------------------------------------------------------------------------------
# (v) the flag's path to every run-simulation rule shell
# --------------------------------------------------------------------------------------


def test_debug_is_a_parameter_at_every_layer_the_flag_crosses():
    """Property: ``debug`` is accepted at every layer between the run entry points and the runner.

    Class: a FITNESS FUNCTION over the facade chain's signatures -- ``Toolkit.run``,
    ``run_experiment``, ``TRITONSWMM_analysis.run`` and ``.submit_workflow``, the sensitivity
    ``submit_workflow``, both builders' ``submit_workflow`` and both generators, and
    ``prepare_simulation_command``.

    Why required, checkable against the code: a parameter accepted at one layer and absent at
    the next is dropped silently -- the caller's ``debug=True`` never reaches the shell.

    What kills it: omitting ``debug`` from any listed signature.

    A second correct implementation under which it still passes: ``debug`` declared
    keyword-only at every layer rather than positional-or-keyword -- the assertion reads names,
    not kinds.
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.experiment_definition import run_experiment
    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
    from hhemt.toolkit import Toolkit
    from hhemt.workflow import SensitivityAnalysisWorkflowBuilder, SnakemakeWorkflowBuilder

    layers = [
        Toolkit.run,
        run_experiment,
        TRITONSWMM_analysis.run,
        TRITONSWMM_analysis.submit_workflow,
        TRITONSWMM_sensitivity_analysis.submit_workflow,
        SnakemakeWorkflowBuilder.submit_workflow,
        SnakemakeWorkflowBuilder.generate_snakefile_content,
        SensitivityAnalysisWorkflowBuilder.submit_workflow,
        SensitivityAnalysisWorkflowBuilder.generate_master_snakefile_content,
        TRITONSWMM_run.prepare_simulation_command,
    ]
    missing = [f.__qualname__ for f in layers if "debug" not in inspect.signature(f).parameters]
    assert missing == []


def _runner_blocks(text: str) -> list[str]:
    return re.findall(r"-m hhemt\.run_simulation_runner \\\n(?:.*\\\n)*.*", text)


_HPC = Path(__file__).parent / "fixtures" / "hpc_system_config_test.yaml"


def _multisim_builder():
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.test_case_catalog import Local_TestCases

    from hhemt.workflow import SnakemakeWorkflowBuilder

    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=False, hpc_system_config_yaml=_HPC)
    return SnakemakeWorkflowBuilder(tc.analysis)


def _sensitivity_builder():
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.test_case_catalog import Local_TestCases

    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
    from hhemt.workflow import SensitivityAnalysisWorkflowBuilder

    tc = Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(
        start_from_scratch=False, hpc_system_config_yaml=_HPC
    )
    return SensitivityAnalysisWorkflowBuilder(TRITONSWMM_sensitivity_analysis(tc.analysis))


@pytest.mark.parametrize("which", ["multisim", "sensitivity"])
def test_both_builders_emit_debug_on_every_run_shell_exactly_when_debug(which):
    """Property: every run-simulation rule shell carries ``--debug`` under debug and none does otherwise.

    Class: both Snakefile generators, each rendered with debug True and False.

    Why required, checkable against the code: the runner is the only process that writes the
    base cfg line, and it learns the flag only from its own command line.

    What kills it: threading the flag into one generator only, or dropping it from the shell.

    A second correct implementation under which it still passes: one that places ``--debug``
    elsewhere in the shell's argument list.
    """
    builder = _multisim_builder() if which == "multisim" else _sensitivity_builder()
    gen = builder.generate_snakefile_content if which == "multisim" else builder.generate_master_snakefile_content
    on = _runner_blocks(gen(debug=True))
    off = _runner_blocks(gen(debug=False))
    assert on and len(on) == len(off)
    assert all("--debug" in block for block in on)
    assert not any("--debug" in block for block in off)


_FAILED_REPLAY_LINE = "swmm_snapshot_failed_replay=1"


def _failed_replay_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip().startswith("swmm_snapshot_failed_replay="))


@pytest.mark.parametrize(
    ("field", "seeded", "expected_lines"),
    [
        (True, False, 1),
        (False, True, 0),
    ],
)
def test_a_coupled_clean_start_base_cfg_carries_the_failed_replay_line_iff_the_field(
    tmp_path, field, seeded, expected_lines
):
    """Property: a coupled clean start leaves the base cfg with exactly one replay line iff the field is True.

    Class: a coupled launch that hands the solver the base cfg (pickup off, no hotstart cfg),
    in the two states where the launch write must CHANGE the file -- the field on with no key
    in the base cfg, and the field off with a key already in it.

    Why required, checkable against the code: the solver copies the cfg it is launched with
    into every checkpoint cfg it writes, so a run started fresh with the field on must carry
    the key from its first launch, and a base cfg that kept a key from an earlier opted-in run
    must lose it. The other tests of this file leave the field off on a keyless base cfg, where
    the write changes nothing, so they cannot see a write that is skipped on a clean start.

    What kills it: a guard that confines the launch write to resumes, for example requiring the
    launched cfg to differ from the scenario's base cfg path.

    A second correct implementation under which it still passes: one that writes the key into
    the base cfg at preparation AND again at launch, since the cfg read back is the same.
    """
    run, base, _ = _run(tmp_path, with_hotstart_cfgs=False, failed_replay=field)
    if seeded:
        base.write_text(_BASE + _FAILED_REPLAY_LINE + "\n")
    _prepare(run, model_type="tritonswmm", pickup=False, debug=False)
    text = base.read_text()
    assert _failed_replay_lines(text) == expected_lines
    assert text.startswith(_BASE)
