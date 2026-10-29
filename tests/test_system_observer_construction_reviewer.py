"""An observer construction must persist nothing through the ABSTENTION-RESTORE branch.

The work chunk adds ``TRITONSWMM_system(skip_log_update=True)`` so a runner that only
READS compile state never writes the SHARED ``{system_dir}/system_log.json``. That
document has TWO write sites inside ``_sync_compilation_log_field``, and both are writes:
``LogField.set(success)`` on a measured verdict, and ``LogField.clear()`` in the
ABSTENTION-RESTORE branch -- taken when the build artifact is ABSENT and the persisted
value is ``False``, which is the ordinary state of a tree whose build dirs were cleaned
up after a simulation.

The chunk's own test file positions both of its observer arms on the ``set`` branch (a
``compilation.log`` that EXISTS and derives ``True`` against a persisted ``False``). An
implementation that gates only ``set`` while leaving ``clear()`` ungated passes all four
of those tests and still writes here, so this file adds the differently-positioned arm.

VENUE: laptop-safe. Construction only -- no solver is compiled or executed, and no
synthetic-analysis fixture is used (which is also why the system config is reused from
the sibling file rather than rebuilt: one hand-maintained copy of a schema-valid minimal
config, not two that can drift apart).
"""

import json
from pathlib import Path

from hhemt.system import TRITONSWMM_system
from tests.test_system_observer_construction import _system_yaml


def test_observer_construction_skips_the_abstention_restore_branch(tmp_path):
    """An observer leaves ``system_log.json`` byte- and mtime-identical on the restore input.

    Property: with the persisted compilation verdict ``False`` and the build artifact
    ABSENT, a ``skip_log_update=True`` construction changes neither the bytes nor the
    mtime of the shared system log.

    Class: NEW CAPABILITY -- the plausible wrong implementation is a guard placed on the
    ``set`` call alone, leaving ``LogField.clear()`` reachable. Red under it: the
    three-implementation differential and its pasted output are recorded in this chunk's
    review round record.

    Why it is required: ``_sync_compilation_log_field`` records in its own docstring that
    "the restore is itself a write", and ``_sync_compilation_status_on_init`` reads all
    five compilation properties at EVERY construction -- so with the build trees cleaned
    up (the normal post-simulation state) every runner lands on this branch. A guard that
    misses ``clear()`` reopens the concurrent-write race the chunk exists to close, on the
    one input the chunk's four tests do not construct.

    What kills it: moving ``if self._skip_log_update: return`` below the
    ``if not derived_from_resolvable_input:`` block in ``_sync_compilation_log_field``, or
    wrapping only the ``log_field.set(success)`` call.

    Second correct implementation: gate persistence inside ``TRITONSWMM_log.write()`` via
    a read-only flag consulted there, rather than in the helper. ``set`` and ``clear``
    then still run and the document is still unchanged, so this assertion still passes --
    the assertion turns on the file's state, not on which call site declines to write.
    """
    cfg_yaml = _system_yaml(tmp_path)
    probe = TRITONSWMM_system(cfg_yaml)  # default mode: creates and writes the document
    log_path = Path(probe.log.logfile)

    # The TRITON-only CPU build tree is absent on a fresh tmp root, so
    # compilation_triton_only_cpu_successful takes the ABSTENTION branch and passes
    # derived_from_resolvable_input=False into the helper.
    assert not (probe.sys_paths.TRITON_build_dir_cpu / "compilation.log").exists()

    # Force the persisted verdict to False so the restore branch has a value to clear.
    payload = json.loads(log_path.read_text())
    payload["compilation_triton_cpu_successful"] = False
    log_path.write_text(json.dumps(payload, indent=2))
    before = (log_path.stat().st_mtime_ns, log_path.read_bytes())

    observer = TRITONSWMM_system(cfg_yaml, skip_log_update=True)
    derived = observer.compilation_triton_only_cpu_successful
    after = (log_path.stat().st_mtime_ns, log_path.read_bytes())

    assert derived is False, "the abstention still returns False for the caller's boolean contract"
    assert after == before, "an observer must not persist through the abstention-restore branch"
