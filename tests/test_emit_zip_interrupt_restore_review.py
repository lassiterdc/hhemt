"""Reviewer test for chunk 2a.13: the interrupt test's restore of SIGALRM when the install returns ``None``.

``test_an_interrupt_mid_emit_leaves_the_old_destination_and_no_partial`` installs a SIGALRM
handler and restores what ``signal.signal`` returned. Where a C-level handler was installed before
Python started (a site-wide ``LD_PRELOAD`` library does this on the Rivanna cluster), the install
returns ``None``, and ``signal.signal(SIGALRM, None)`` raises ``TypeError``. On a laptop the
previous disposition is ``SIG_DFL``, so that branch never runs and the test is green there while
red on the cluster. This file reaches the branch on any machine by running the REAL test with
``signal.signal`` wrapped so that its first SIGALRM call still installs the handler and returns
``None``.

The file runs the driven test twice (``first`` and ``last``) and does not copy the install and
restore pair, because a copy would test the copy. It writes only under ``tmp_path`` and runs no
solver, so it runs on a laptop.
"""

from __future__ import annotations

import os
import signal

import pytest

import tests.test_emit_zip_policy_reviewer as _policy_reviewer

# Reached through the module so that pytest does not collect it a second time under this file.
_DRIVEN_TEST = _policy_reviewer.test_an_interrupt_mid_emit_leaves_the_old_destination_and_no_partial


@pytest.mark.skipif(
    not (hasattr(os, "mkfifo") and hasattr(signal, "setitimer")),
    reason="needs a FIFO and an interval timer to interrupt a blocked read",
)
@pytest.mark.parametrize("position", ["first", "last"])
def test_the_driven_interrupt_test_restores_sig_dfl_when_the_install_returned_none(tmp_path, monkeypatch, position):
    """Property: when the install returns ``None``, the driven test's last SIGALRM call is ``SIG_DFL``.

    It also raises nothing.

    Input class: the driven test run with ``signal.signal`` wrapped so that the first SIGALRM
    call installs the handler it is given and then returns ``None``, which is what the install
    returns where a C-level handler was already present. Calls after the first are recorded and
    passed to the real ``signal.signal``. The install is NOT stubbed: with no real handler the
    driven test's 0.5 s timer would deliver SIGALRM to the default disposition and end the pytest
    process.

    Why required: the cleanup is the only code in the driven test that can mask its own result.
    With ``None`` handed to the restore, the ``TypeError`` replaces whatever the body raised or
    failed to raise, and the handler that the test installed stays installed for every later
    test in the process, turning a later stray SIGALRM into a ``KeyboardInterrupt``. The final
    disposition is asserted as well as the call, because a restore that is called and does not
    take effect leaves the same leak.

    What kills it: the unguarded restore ``signal.signal(signal.SIGALRM, previous)`` (it raises
    ``TypeError`` on the ``None`` it is given); a restore of ``signal.SIG_IGN`` in place of
    ``SIG_DFL``; and a restore that is skipped when ``previous`` is ``None`` (it leaves the test's
    handler installed).

    A second correct implementation, which differs in what the test asserts on: a restore written
    ``signal.signal(signal.SIGALRM, previous or signal.SIG_DFL)`` still passes, because it also
    hands ``SIG_DFL`` to the last call and leaves ``SIG_DFL`` installed. So does one that makes
    two restore calls and ends on ``SIG_DFL``, since the assertion reads the last call, not the
    call count.

    Not established: this holds the ``None`` branch only. The branch where the install returns a
    Python handler is the driven test's other arm and is not constrained here.
    """
    real_signal = signal.signal
    real_getsignal = signal.getsignal
    disposition_before = real_getsignal(signal.SIGALRM)
    sigalrm_calls: list[object] = []
    installed_handler_was_visible: list[bool] = []

    def _install_returns_none_then_record(signum, handler):
        if signum != signal.SIGALRM:
            return real_signal(signum, handler)
        if not sigalrm_calls:
            real_signal(signum, handler)
            sigalrm_calls.append(handler)
            installed_handler_was_visible.append(real_getsignal(signum) is handler)
            return None
        sigalrm_calls.append(handler)
        return real_signal(signum, handler)

    monkeypatch.setattr(signal, "signal", _install_returns_none_then_record)
    raised: BaseException | None = None
    disposition_after = None
    try:
        try:
            _DRIVEN_TEST(tmp_path, position)
        except BaseException as exc:  # noqa: BLE001 -- the driven test's own failure is the result under test
            raised = exc
        disposition_after = real_getsignal(signal.SIGALRM)
    finally:
        # Through the original function and not the wrapper. A ``None`` read is restored as
        # ``SIG_DFL``, because ``signal.signal(SIGALRM, None)`` raises ``TypeError``, the defect
        # under test.
        signal.setitimer(signal.ITIMER_REAL, 0)
        real_signal(signal.SIGALRM, disposition_before if disposition_before is not None else signal.SIG_DFL)

    assert len(sigalrm_calls) >= 1 and callable(sigalrm_calls[0]), (
        f"the driven test did not install a callable SIGALRM handler; calls={sigalrm_calls!r}"
    )
    assert installed_handler_was_visible == [True], (
        "the install did not take effect before it returned None, so the 0.5 s timer would not "
        "have interrupted the driven test"
    )
    assert raised is None, f"the driven test raised {type(raised).__name__}: {raised}"
    assert len(sigalrm_calls) >= 2, "the driven test never restored SIGALRM after installing its handler"
    assert sigalrm_calls[-1] == signal.SIG_DFL, (
        f"the last SIGALRM call handed {sigalrm_calls[-1]!r} to signal.signal; expected SIG_DFL"
    )
    assert disposition_after == signal.SIG_DFL, (
        f"SIGALRM was left as {disposition_after!r} after the driven test; expected SIG_DFL"
    )
