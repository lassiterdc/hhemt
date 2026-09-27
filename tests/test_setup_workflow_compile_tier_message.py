"""Tests for the compile-tier messaging in ``hhemt.setup_workflow`` (Item 3).

WHAT THIS PINS, and why the shape is what it is.

Since ``d98802556`` (D84 phases 1+2) no generated Snakefile emits ``--compile-triton-swmm``,
``--compile-triton-only`` or ``--compile-swmm``: the setup rule ASSERTS the compile tier
rather than building it. Six log messages in ``setup_workflow.py`` nevertheless told their
reader that a ``--compile-*`` flag was "not specified", naming a remedy no caller on the
generated path can supply. Three of the six are ``logger.info`` lines that fire on EVERY
native run including successful ones, so a healthy run's log taught the wrong thing before
the reader ever saw an error.

Two independent test families here, and they answer different questions:

1. CLASS POST-CONDITION -- TWO instruments, because neither spans the class alone.

   (a) ``test_no_logger_message_names_a_compile_flag``, AST-measured over the module as
   imported: no ``logger.*`` argument carries a STATICALLY-DERIVABLE string containing
   ``--compile-``. Measured 6 before the repair and 0 after, with the
   ``--process-system-inputs`` message held at 1 throughout as the class boundary control --
   that flag IS still emitted by two live sites in ``workflow.py``, so its message is honest
   and must NOT be swept in. This family is the one that fails against pre-repair code.

   The control can refute a VACUOUS walk and can never establish its EXTENT: it is drawn in
   the defect's own syntactic form (a bare ``ast.Constant``), so it exercises the one form the
   collector always handled. MEASURED over the graded file, only 16 of ~47 logger message
   arguments are bare constants and 20 are f-strings -- the module's dominant idiom -- so the
   original ``ast.Constant``-only collector was blind to the form a re-introduction would most
   likely take. ``_static_str_segments`` therefore also derives f-string literal segments,
   constant-only ``+`` concatenations, ``%`` templates and ``str.format`` receivers, and
   ``_logger_message_constants`` takes a ``tree`` parameter so it can be fed a KNOWN-ANSWER
   input at all -- ``test_the_collector_catches_a_non_literal_compile_flag_reintroduction``
   demonstrates a true-positive per widened form, and
   ``test_the_collector_residual_is_pinned_not_merely_stated`` pins what it still cannot see.

   (b) ``test_no_rendered_message_names_an_unactionable_or_foreign_compile_flag``, asserting on
   the formatter's OUTPUT per role. This exists because no static widening of a ``logger.*``
   argument walk can reach a string that arrives from a called helper -- and the repair itself
   is exactly that shape, since the six sites pass ``_compile_tier_message(...)`` whose return
   legitimately names the arm's own flag. (a) reads source, (b) reads rendered text; together
   they span the class.

2. BEHAVIOUR INVARIANTS (the control-flow and reachability tests). These are true in BOTH the
   pre- and post-repair trees by construction, and they exist so that a future edit which
   unifies the messages by accidentally dropping a refusal, or by making the class
   unreachable, trips a test instead of silently voiding the repair.

REACHABILITY, stated because a test asserting on any of the six can otherwise assert on a
branch no run enters. The six fire iff BOTH conjuncts hold: native mode
(``_native_compile = not _exec_env_container``) AND at least one model toggle enabled (the
early return keyed on ``_tier_must_be_asserted``). ``test_the_message_class_stays_reachable``
pins both conjuncts structurally.

WHY STRUCTURAL RATHER THAN A LIVE ``main()`` RUN -- stated as the residual it is, not as
closure. Exercising the arms for real requires constructing ``TRITONSWMM_system`` from
on-disk configs, which drags in the synthetic-model fixture tier; and the compile-bearing
tier does not run locally in this project (venue rule: compiles and solver runs go to
Rivanna scratch). So these tests assert the message CONTENT directly against the formatter
and the message ROUTING and REACHABILITY structurally over the module's AST. What they do
NOT prove is that a real workflow-driven invocation emits these strings -- only that the
sites which would emit them are the six, are routed, and remain reachable.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from hhemt import setup_workflow as sw

_MODEL_KEYS = ("tritonswmm", "triton_only", "swmm")


def _module_tree() -> ast.Module:
    """Parse the ``setup_workflow`` module that was actually IMPORTED.

    Keyed on ``sw.__file__`` rather than a repo-relative path on purpose: in a per-branch
    worktree a bare ``python``/``pytest`` invocation can import the MAIN checkout's copy via
    the editable-install ``.pth`` (hhemt gotcha 68), so a hardcoded path would let this test
    grade a different file than the one under test.
    """
    return ast.parse(pathlib.Path(sw.__file__).read_text(encoding="utf-8"))


def _static_str_segments(node: ast.expr) -> str | None:
    """The statically-derivable string content of ``node``, or ``None`` if there is none.

    Deliberately wider than ``isinstance(node, ast.Constant)``. MEASURED over
    ``setup_workflow.py``: of its ~47 ``logger.*`` message arguments only 16 are bare string
    constants; 20 are f-strings, which is the module's DOMINANT logger idiom. A collector
    restricted to ``ast.Constant`` therefore pins the literals it happens to see and is silent
    about the class -- a re-introduction written ``logger.error(f"... --compile-swmm ...")``
    passes it without a word.

    Forms derived here, each a known-answer case in
    ``test_the_collector_catches_a_non_literal_compile_flag_reintroduction``:

    * ``ast.Constant`` -- the bare literal, and implicit adjacent-literal concatenation, which
      CPython folds into one ``Constant`` before this walk ever sees it.
    * ``ast.JoinedStr`` -- the f-string, reduced to its literal segments. Interpolations
      (``FormattedValue``) contribute nothing, which is correct: their content is not static.
    * ``ast.BinOp`` / ``Add`` -- constant-only ``+`` concatenation, recursively.
    * ``ast.BinOp`` / ``Mod`` -- the ``%``-format TEMPLATE (the left operand). The flag name
      lives in the template, not the operands.
    * ``str.format()`` -- the template RECEIVER, for the same reason.

    Forms NOT derived, stated as the residual rather than left to be discovered: a module
    constant referenced by name, a runtime concatenation of non-constants, and a string
    produced by a called helper. The last is the one that matters here, because the repair
    itself uses it -- the six sites pass ``_compile_tier_message(...)``, an ``ast.Call``, whose
    return value legitimately contains a flag name. NO static widening of a ``logger.*``-argument
    walk can reach that class; it is covered instead by
    ``test_no_rendered_message_names_an_unactionable_or_foreign_compile_flag``, which asserts on
    the formatter's OUTPUT per role. The negative half of the known-answer test pins this
    boundary so the scope is falsifiable rather than asserted.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        segments = [v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        return "".join(segments) if segments else None
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            left = _static_str_segments(node.left)
            right = _static_str_segments(node.right)
            if left is not None and right is not None:
                return left + right
            return left if right is None else right
        if isinstance(node.op, ast.Mod):
            return _static_str_segments(node.left)
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        return _static_str_segments(node.func.value)
    return None


def _logger_message_constants(tree: ast.Module | None = None) -> list[tuple[int, str, str]]:
    """Every ``(lineno, level, value)`` where a ``logger.*`` call carries a derivable string.

    ``tree`` defaults to the imported ``setup_workflow`` module, which is how the class
    post-condition and the boundary control call it. Passing a tree explicitly is what makes
    the collector testable AT ALL: a walk with no input parameter can be exercised only against
    the one file it hardcodes, so its coverage can be asserted but never demonstrated.

    The name is retained rather than corrected to ``_logger_message_strings`` because the
    review module ``test_setup_workflow_compile_tier_message_review.py`` calls it by this name;
    renaming it here would break a sibling instrument to fix a label.
    """
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree if tree is not None else _module_tree()):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
        ):
            for arg in node.args:
                value = _static_str_segments(arg)
                if value is not None:
                    found.append((arg.lineno, node.func.attr, value))
    return found


def _formatter_calls() -> list[ast.Call]:
    return [
        node
        for node in ast.walk(_module_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_compile_tier_message"
    ]


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


# ---------------------------------------------------------------------------
# 1. Class post-condition -- the family that fails against pre-repair code.
# ---------------------------------------------------------------------------


def test_no_logger_message_names_a_compile_flag() -> None:
    """No STATICALLY-DERIVABLE string in a ``logger.*`` call may name a ``--compile-*`` flag.

    Single-sided over the defect shape. Measured 6 offenders pre-repair, 0 after.

    SCOPE, narrowed to what the instrument measures. This reads the literal content a
    ``logger.*`` argument carries in source -- constants, f-string segments, constant-only
    ``+`` concatenations, ``%`` templates and ``str.format`` receivers (see
    ``_static_str_segments``). It does NOT read a string that arrives from a called helper, and
    the repair itself is that shape: the six sites pass ``_compile_tier_message(...)``, whose
    return legitimately contains the arm's own flag. The earlier wording -- "No log message may
    name a ``--compile-*`` flag" -- claimed that larger class while measuring this smaller one,
    which is exactly the gap the class post-condition existed to close. The helper-routed class
    is covered by ``test_no_rendered_message_names_an_unactionable_or_foreign_compile_flag``,
    which asserts on the formatter's OUTPUT. The two together span the class; neither does
    alone, and saying so is the point.
    """
    offenders = [(ln, lvl, val) for ln, lvl, val in _logger_message_constants() if "--compile-" in val]
    assert offenders == [], (
        "log message(s) still name a --compile-* flag that no generated workflow emits; "
        f"route them through _compile_tier_message: {offenders}"
    )


def test_the_process_system_inputs_message_is_left_alone() -> None:
    """Class-boundary control: ``--process-system-inputs`` IS emitted, so its message is honest.

    This is also the scan's own sanity check -- without it, a broken AST walk would satisfy
    the post-condition above by finding nothing at all.
    """
    survivors = [val for _, _, val in _logger_message_constants() if "--process-system-inputs" in val]
    assert len(survivors) == 1, f"expected exactly one --process-system-inputs message, found {len(survivors)}"


# The control above can refute a VACUOUS walk and can never establish its EXTENT, because it is
# drawn in the defect's own syntactic form (``setup_workflow.py``'s --process-system-inputs
# message is itself a bare ``ast.Constant``), so it exercises the one form the collector always
# handled. Extent needs a known-answer input in a form the control does not use. That is what
# the next two tests supply: a demonstrated true-positive per widened form, and the OUTPUT-side
# post-condition for the one class no static widening can reach.

_CAUGHT_BY_DESIGN = (
    ("bare literal", 'logger.error("TRITON-SWMM needs --compile-triton-swmm")'),
    (
        "implicit adjacent-literal concatenation (CPython folds it to one Constant)",
        'logger.info("TRITON-SWMM needs " "--compile-triton-swmm")',
    ),
    ("f-string -- the module's dominant logger idiom", 'logger.error(f"{label} needs --compile-triton-swmm")'),
    ("constant-only + concatenation", 'logger.error("needs " + "--compile-triton-swmm" + " first")'),
    ("nested constant-only + concatenation", 'logger.error(("needs " + "--compile-swmm") + " first")'),
    (
        "%-format template (BinOp/Mod -- the flag lives in the template, not the operand)",
        'logger.error("%s needs --compile-triton-only" % label)',
    ),
    ("str.format template receiver", 'logger.error("{0} needs --compile-swmm".format(label))'),
    ("lazy-logging stdlib idiom (message is still arg 0)", 'logger.info("needs --compile-swmm: %s", path)'),
)

_OUT_OF_REACH_BY_DESIGN = (
    ("module constant referenced by name", "logger.error(_MSG_NAMING_COMPILE_TRITON_SWMM)"),
    ("runtime concatenation of non-constants", "logger.error(prefix + suffix)"),
    (
        "routed through a called helper -- the form the repair itself uses",
        'logger.error(_compile_tier_message("tritonswmm", refused=True))',
    ),
)


@pytest.mark.parametrize(("label", "source"), _CAUGHT_BY_DESIGN, ids=[lbl for lbl, _ in _CAUGHT_BY_DESIGN])
def test_the_collector_catches_a_non_literal_compile_flag_reintroduction(label: str, source: str) -> None:
    """A DEMONSTRATED true-positive per form, not an asserted one.

    Each source above is a known-answer re-introduction of a ``--compile-*`` flag that a walk
    protecting the class must catch. Six of the eight are written in forms the class-boundary
    control never exercises, so they measure the collector's EXTENT rather than its liveness.
    """
    found = _logger_message_constants(ast.parse(source))
    assert any("--compile-" in value for *_rest, value in found), (
        f"the collector is blind to a --compile-* re-introduction written as a {label}: {source}"
    )


@pytest.mark.parametrize(("label", "source"), _OUT_OF_REACH_BY_DESIGN, ids=[lbl for lbl, _ in _OUT_OF_REACH_BY_DESIGN])
def test_the_collector_residual_is_pinned_not_merely_stated(label: str, source: str) -> None:
    """The NEGATIVE half: what the collector cannot see, asserted so the scope is falsifiable.

    A residual named only in prose drifts; pinned here, a future widening that closes one of
    these fails this test and forces the docstring and the post-condition's stated scope to be
    updated in the same edit. The third case is the load-bearing one -- it is the shape the six
    live sites use, and it is why the OUTPUT-side post-condition below exists rather than being
    optional.
    """
    found = _logger_message_constants(ast.parse(source))
    assert not any("--compile-" in value for *_rest, value in found), (
        f"the collector now reaches a {label}; widen the post-condition's stated scope to match: {source}"
    )


def test_no_rendered_message_names_an_unactionable_or_foreign_compile_flag() -> None:
    """OUTPUT-side post-condition, closing the helper-routed class the AST walk cannot see.

    For every role the formatter has -- three model arms x skip/refusal x both flag branches --
    the RENDERED string may name at most ONE ``--compile-*`` flag, it must be that arm's own
    flag, it must appear only as a command-block line the reader can copy, and it must never sit
    in prose of the "not specified" shape that sends the reader to a flag no generated workflow
    emits. The skip notice, which fires on every native run including successful ones, may name
    no flag at all.
    """
    all_flags = {flag for _label, flag in sw._COMPILE_TARGETS.values()}
    for model_key in _MODEL_KEYS:
        _label, own_flag = sw._COMPILE_TARGETS[model_key]

        skip = sw._compile_tier_message(model_key, refused=False)
        assert "--compile-" not in skip, f"{model_key}'s skip notice names a compile flag: {skip}"

        for gpu_target in (False, True):
            for hpc_config_present in (False, True):
                message = sw._compile_tier_message(
                    model_key,
                    refused=True,
                    gpu_target=gpu_target,
                    hpc_config_present=hpc_config_present,
                )
                for flag in all_flags - {own_flag}:
                    assert flag not in message, f"{model_key}'s refusal names the foreign flag {flag}"
                naming_lines = [line for line in message.splitlines() if "--compile-" in line]
                assert len(naming_lines) == 1, (
                    f"{model_key} refusal (gpu={gpu_target}, hpc={hpc_config_present}) names a compile flag on "
                    f"{len(naming_lines)} lines, expected exactly 1: {naming_lines}"
                )
                assert naming_lines[0].strip() == own_flag, (
                    "the flag must appear only as a copyable command-block line, never inside prose: "
                    f"{naming_lines[0]!r}"
                )
                assert "not specified" not in message


# ---------------------------------------------------------------------------
# 2. Routing -- all six sites express the convention once.
# ---------------------------------------------------------------------------


def test_all_six_message_sites_route_through_the_one_formatter() -> None:
    """Six calls: one skip notice and one refusal per model arm."""
    calls = _formatter_calls()
    assert len(calls) == 6, f"expected 6 _compile_tier_message call sites, found {len(calls)}"

    seen: dict[tuple[str, bool], int] = {}
    for call in calls:
        assert call.args and isinstance(call.args[0], ast.Constant), "model key must be a literal at the call site"
        model_key = call.args[0].value
        refused_node = _keyword(call, "refused")
        assert isinstance(refused_node, ast.Constant), "`refused` must be passed as a literal at the call site"
        key = (model_key, bool(refused_node.value))
        seen[key] = seen.get(key, 0) + 1

    expected = {(model, refused): 1 for model in _MODEL_KEYS for refused in (False, True)}
    assert seen == expected, f"call-site role coverage is wrong: {seen}"


def test_every_refusal_site_passes_the_resolved_gpu_target() -> None:
    """A refusal must not hardcode ``gpu_target`` -- including the SWMM arm.

    The flag answers "does this INVOCATION need the GPU-resolving flags", not "does this model
    have a GPU build". Under ``n_gpus > 0`` the preflight guard rejects ANY direct
    ``setup_workflow`` invocation missing them, a ``--compile-swmm`` one included.
    """
    refusals = [
        call
        for call in _formatter_calls()
        if isinstance(_keyword(call, "refused"), ast.Constant) and _keyword(call, "refused").value is True
    ]
    # Assert the DENOMINATOR before iterating. Without this the loop body is skipped when no
    # refusal site exists and the test passes having examined nothing -- measured: it was one
    # of only 4 tests in this module that passed against pre-repair code, and it passed for
    # that reason rather than because the property held.
    assert len(refusals) == 3, f"expected one refusal site per model arm, found {len(refusals)}"

    for call in refusals:
        gpu_node = _keyword(call, "gpu_target")
        assert gpu_node is not None, f"refusal at line {call.lineno} passes no gpu_target"
        assert not isinstance(gpu_node, ast.Constant), (
            f"refusal at line {call.lineno} hardcodes gpu_target; it must resolve it from the system"
        )
        assert "gpu_compilation_backend" in ast.unparse(gpu_node), (
            f"refusal at line {call.lineno} does not resolve gpu_target from gpu_compilation_backend: "
            f"{ast.unparse(gpu_node)}"
        )


def test_every_refusal_site_passes_the_resolved_hpc_config_presence() -> None:
    """``hpc_config_present`` defaults to ``False``, so a site that forgets it reverts the fix.

    ``--hpc-system-config`` carries the cluster's ``module load`` set as well as the GPU target,
    and ``additional_modules`` is a CLUSTER-level field, so the flag's presence in the printed
    remedy must track whether the INVOCATION received a config -- not whether a GPU backend
    resolved. That is a separate parameter from ``gpu_target``, and a separate parameter is a
    thing a later call site can omit: the default is ``False``, so omitting it silently prints
    the pre-fix command again with no error anywhere. This pins the parameter at every refusal
    site and rejects a hardcoded value, mirroring the sibling pin on ``gpu_target``.
    """
    refusals = [
        call
        for call in _formatter_calls()
        if isinstance(_keyword(call, "refused"), ast.Constant) and _keyword(call, "refused").value is True
    ]
    assert len(refusals) == 3, f"expected one refusal site per model arm, found {len(refusals)}"

    for call in refusals:
        node = _keyword(call, "hpc_config_present")
        assert node is not None, f"refusal at line {call.lineno} passes no hpc_config_present"
        assert not isinstance(node, ast.Constant), (
            f"refusal at line {call.lineno} hardcodes hpc_config_present; it must resolve it from args"
        )
        assert "hpc_system_config" in ast.unparse(node), (
            f"refusal at line {call.lineno} does not resolve hpc_config_present from args.hpc_system_config: "
            f"{ast.unparse(node)}"
        )


# ---------------------------------------------------------------------------
# 3. Message content.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_skip_notice_is_one_line_and_names_no_flag(model_key: str) -> None:
    """The skip notice fires on every native run, successful ones included.

    So it must stay one line (no command block, three times per healthy run) and must name no
    flag at all -- naming one is what taught the reader the wrong thing.
    """
    message = sw._compile_tier_message(model_key, refused=False)
    assert "\n" not in message, "the skip notice fires on every native run and must stay one line"
    assert "--compile-" not in message
    assert "not specified" not in message, "'not specified' implies a caller who could specify it"
    label, _flag = sw._COMPILE_TARGETS[model_key]
    assert label in message
    assert "caller" in message


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_refusal_names_the_direct_invocation_as_the_remedy(model_key: str) -> None:
    label, flag = sw._COMPILE_TARGETS[model_key]
    message = sw._compile_tier_message(model_key, refused=True, gpu_target=False)

    assert label in message
    assert "python -m hhemt.setup_workflow" in message, "the refusal must name the direct invocation"
    assert "--system-config" in message
    assert "--analysis-config" in message
    assert flag in message, "the direct invocation's own compile flag is valid and must be named"
    assert "not specified" not in message
    # Item 1b has not hardened this route; naming it would re-commit the GPU-path defect.
    assert "compile_and_preprocess_all_targets" not in message


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_refusal_names_target_partition_only_for_a_gpu_target(model_key: str) -> None:
    """``--target-partition`` is GPU-only; ``--hpc-system-config`` is NOT, and the split is the point.

    Both are ``required=False`` and ``resolve_gpu_target`` returns ``(None, None)`` SILENTLY when
    either is absent, which is why a GPU refusal must name both. But ``--hpc-system-config`` also
    carries ``resolve_additional_modules`` -- the cluster's ``module load`` set, a CLUSTER-level
    field with no GPU conditioning -- so gating it on ``gpu_target`` dropped it from every CPU
    refusal even when the generated rule had supplied it. The four cases below are the whole
    contract: partition tracks ``gpu_target``, config tracks ``gpu_target or hpc_config_present``.
    """
    gpu = sw._compile_tier_message(model_key, refused=True, gpu_target=True, hpc_config_present=True)
    assert "--hpc-system-config" in gpu
    assert "--target-partition" in gpu

    cpu_with_config = sw._compile_tier_message(model_key, refused=True, gpu_target=False, hpc_config_present=True)
    assert "--hpc-system-config" in cpu_with_config, (
        "a CPU refusal printed by an invocation that DID receive --hpc-system-config must name it: "
        "omitting it tells the reader to compile without the additional_modules `module load` lines "
        "the rule's own compile emitted"
    )
    assert "--target-partition" not in cpu_with_config, "--target-partition is genuinely GPU-only"

    cpu_without_config = sw._compile_tier_message(model_key, refused=True, gpu_target=False, hpc_config_present=False)
    assert "--hpc-system-config" not in cpu_without_config, (
        "an invocation that received no hpc-system-config must not print a flag the reader has no "
        "file for -- setup_workflow rejects a non-existent --hpc-system-config path outright, so an "
        "unconditional placeholder would itself be a remedy the reader cannot run"
    )
    assert "--target-partition" not in cpu_without_config

    # gpu_target=True implies a config was supplied (resolve_gpu_target cannot resolve a backend
    # without one), so the disjunction is satisfied on every reachable path even when a caller
    # passes only the GPU signal. Pinned so the disjunction is not "simplified" away.
    gpu_only_signal = sw._compile_tier_message(model_key, refused=True, gpu_target=True)
    assert "--hpc-system-config" in gpu_only_signal


def test_the_three_model_arms_do_not_share_a_compile_flag() -> None:
    """One mapping pairs each label with its flag, so a copy-paste slip is a test failure."""
    flags = [flag for _label, flag in sw._COMPILE_TARGETS.values()]
    assert sorted(flags) == sorted({"--compile-triton-swmm", "--compile-triton-only", "--compile-swmm"})
    for model_key in _MODEL_KEYS:
        _label, own_flag = sw._COMPILE_TARGETS[model_key]
        message = sw._compile_tier_message(model_key, refused=True, gpu_target=True)
        for other_flag in flags:
            if other_flag != own_flag:
                assert other_flag not in message, f"{model_key}'s refusal names {other_flag}"


# ---------------------------------------------------------------------------
# 4. Behaviour invariants -- true in BOTH the pre- and post-repair trees.
# ---------------------------------------------------------------------------


def _native_skip_arms(tree: ast.Module) -> list[ast.If]:
    """Every ``elif _native_compile:`` arm -- i.e. an ``If`` whose test is the bare name."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "_native_compile"
    ]


def test_each_native_skip_arm_still_refuses_with_return_1() -> None:
    """A unifying refactor must not drop a refusal; the refusal is the fail-closed half.

    Each ``elif _native_compile:`` arm holds a nested ``if`` whose body ends in ``return 1``.
    """
    arms = _native_skip_arms(_module_tree())
    assert len(arms) == 3, f"expected 3 `elif _native_compile:` arms (one per model), found {len(arms)}"
    for arm in arms:
        guards = [stmt for stmt in arm.body if isinstance(stmt, ast.If)]
        assert guards, f"arm at line {arm.lineno} carries no enabled-but-not-compiled guard"
        assert any(
            isinstance(guard.body[-1], ast.Return)
            and isinstance(guard.body[-1].value, ast.Constant)
            and guard.body[-1].value.value == 1
            for guard in guards
        ), f"arm at line {arm.lineno} no longer refuses with `return 1`"


def test_the_message_class_stays_reachable() -> None:
    """Pin BOTH reachability conjuncts, so a later edit cannot void the repair silently.

    Conjunct 1: ``_native_compile = not _exec_env_container`` -- container mode fires none of
    the six (the SIF carries the binary). Conjunct 2: the early return is gated on
    ``_tier_must_be_asserted``, which must still consult all three model toggles; if it
    stopped doing so, a workflow-driven invocation would return before reaching the arms.
    """
    assignments: dict[str, str] = {}
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = ast.unparse(node.value)

    assert assignments.get("_native_compile") == "not _exec_env_container"

    tier = assignments.get("_tier_must_be_asserted")
    assert tier is not None, "_tier_must_be_asserted is gone; the early return's gate has moved"
    assert "_exec_env_container" in tier
    for toggle in ("toggle_tritonswmm_model", "toggle_triton_model", "toggle_swmm_model"):
        assert toggle in tier, f"{toggle} no longer reaches _tier_must_be_asserted: {tier}"
