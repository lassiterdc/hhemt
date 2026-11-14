"""REVIEWER-AUTHORED failing tests for Item 3 (compile-tier messaging).

Written by the build-review half of the builder pair, deliberately in a module
DISTINCT from the coder's ``test_setup_workflow_compile_tier_message.py`` so the
two instruments stay separable: the coder's module pins what the repair did, this
one pins two properties the repair claims but does not yet hold.

Both tests below FAIL against the landed working tree. Each carries, in its
docstring, the measurement that produced it and the DECLINE arm -- the
measurement that closes the objection WITHOUT adopting the remedy this test
happens to encode.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib

import pytest

from hhemt import setup_workflow as sw

_MODEL_KEYS = ("tritonswmm", "triton_only", "swmm")

_CODER_MODULE = pathlib.Path(__file__).with_name("test_setup_workflow_compile_tier_message.py")


# ---------------------------------------------------------------------------
# Finding 1 -- the CPU refusal remedy drops `--hpc-system-config`, and with it
# the `module load` lines the rule's own compile emitted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_every_refusal_remedy_names_the_hpc_system_config_flag(model_key: str) -> None:
    """`--hpc-system-config` carries TWO payloads and only one of them is GPU-gated.

    ``resolve_gpu_target(cfg_hpc, partition)`` is the GPU payload, and gating it on
    ``gpu_target`` is correct. ``resolve_additional_modules(cfg_hpc)`` is the OTHER
    payload, it is not GPU-conditioned, and it returns ``None`` when no config is
    supplied -- which makes ``system.additional_modules`` falsy and causes all three
    compile paths to skip their ``module load`` emission outright
    (``system.py`` ``_compile_backend_locked``:1357, ``_compile_triton_only_backend_locked``:1778,
    ``_compile_SWMM_locked``:2139, each guarded ``if self.additional_modules:``).

    MEASURED, on a config this repository ships rather than a hypothetical:
    ``test_data/norfolk_coastal_flooding/hpc_system_config_uva.yaml`` declares
    ``additional_modules: ['gcc/12.4.0']`` and a ``standard`` partition whose
    ``gpu_compilation_backend`` is ``None``. A CPU analysis on that partition takes the
    ``gpu_target=False`` branch, so the refusal prints a command WITHOUT
    ``--hpc-system-config``; the generated setup rule DID pass it
    (``workflow.py::_get_config_args`` emits it whenever the analysis carries one).
    Following the printed remedy therefore compiles with no ``module load gcc/12.4.0``
    on a cluster whose libstdc++/MPI resolution depends on exactly that module.

    The message reads "Build it as the caller, before resubmitting:" and then names a
    build that is not the build the rule needed. That is the same defect class Item 3
    exists to remove -- a remedy the reader cannot act on -- relocated rather than
    fixed.

    DECLINE ARM (closes this objection without adopting this test): show that no
    compile path reachable from a refusal consults ``self.additional_modules`` -- e.g.
    a measurement that the three ``if self.additional_modules:`` guards above are
    unreachable from ``compile_TRITON_SWMM`` / ``compile_TRITON_only`` / ``compile_SWMM``,
    or that every deployment supplying ``additional_modules`` also resolves a GPU
    backend on its ensemble partition. Either makes the omission harmless and this
    test wrong.

    ASSERTION REPLACED (coder round 2, threaded form). The original assertion here was
    ``"--hpc-system-config" in sw._compile_tier_message(model_key, refused=True, gpu_target=False)``
    -- the simpler bar this spec explicitly offered, satisfied by naming the flag
    unconditionally. The threaded remedy was chosen instead (a third parameter
    ``hpc_config_present``, passed as ``args.hpc_system_config is not None`` at all three refusal
    sites), because ``setup_workflow`` rejects a non-existent ``--hpc-system-config`` path outright
    and so an unconditional placeholder would itself be a command the reader cannot run. Per this
    spec's own terms -- "the requirement is the property, never my assertion" -- the assertion
    below pins the SAME property in the threaded form's vocabulary: a CPU refusal printed by an
    invocation that DID receive the config names it, and ``--target-partition`` stays absent on
    that branch. The complementary negative (no config supplied -> flag not printed) is pinned in
    the coder module's ``test_refusal_names_target_partition_only_for_a_gpu_target``.
    """
    cpu_refusal = sw._compile_tier_message(model_key, refused=True, gpu_target=False, hpc_config_present=True)
    assert "--hpc-system-config" in cpu_refusal, (
        "the CPU refusal remedy omits --hpc-system-config, so a reader following it "
        "compiles without the hpc_system_config's additional_modules `module load` "
        f"lines that the generated setup rule supplied:\n{cpu_refusal}"
    )
    assert "--target-partition" not in cpu_refusal, (
        "--target-partition is genuinely GPU-only and must stay absent on the CPU branch"
    )


# ---------------------------------------------------------------------------
# Finding 2 -- the class post-condition's collector sees only bare literals.
# ---------------------------------------------------------------------------


def _load_coder_module():
    spec = importlib.util.spec_from_file_location("_item3_coder_tests", _CODER_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_class_post_condition_collector_is_not_blind_to_an_f_string() -> None:
    """A known-answer input the collector must catch, and today does not.

    ``test_no_logger_message_names_a_compile_flag`` asserts that no ``logger.*`` call
    passes a string constant containing ``--compile-``. Its collector accepts an argument
    only when it ``isinstance(arg, ast.Constant)``. MEASURED over the module it grades:
    16 of the 47 logger message arguments in ``setup_workflow.py`` are bare constants;
    20 are f-strings and 11 are other forms, so the walk is blind to 31 of 47 -- and the
    f-string is the file's DOMINANT logger idiom, not an exotic one.

    The class-boundary control (``--process-system-inputs`` held at exactly 1) does real
    work and this test does not dispute it: that message is itself a bare constant at
    ``setup_workflow.py:331``, so the control does falsify a walk that has gone to zero
    matches. What it cannot falsify is UNDER-match on the forms it never exercises. So
    the post-condition pins today's six literals; it does not protect the class, and a
    re-introduction written as ``logger.error(f"... --compile-triton-swmm ...")`` passes
    it silently.

    This test fails in TWO stages and both are the finding: first because the collector
    takes no tree argument and so cannot be fed a known-answer input at all, and then
    because the input it must be fed is not caught. Making the collector accept a tree
    is the smaller half of the remedy; widening it past ``ast.Constant`` to the
    statically-derivable segments of an ``ast.JoinedStr`` (and of a constant-only
    ``ast.BinOp``) is the substantive half.

    DECLINE ARM (closes this objection without adopting this test): show that no
    ``--compile-`` naming can re-enter in a non-``ast.Constant`` form -- e.g. a lint rule
    or an independent gate that forbids f-string logger arguments in this module, or a
    widened post-condition measured against a known-answer corpus of its own. The
    requirement is a collector with a demonstrated true-positive on a non-literal form,
    by whatever route; it is not this particular signature.
    """
    coder = _load_coder_module()
    reintroduction = ast.parse('logger.error(f"{label} not compiled and --compile-triton-swmm not specified")')
    found = coder._logger_message_constants(reintroduction)
    assert any("--compile-" in value for *_rest, value in found), (
        "the class post-condition's collector does not see a --compile-* re-introduction "
        "written as an f-string, which is the dominant logger idiom in the module it grades"
    )
