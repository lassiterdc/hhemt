"""Reviewer tests for chunk 1j.3: the memory-bound tests must pass on the correct verification and fail on each
wrong one.

Each test runs the coder's own test function on a patched ``_stored_slice_equals`` (the correct source plus one
retention line), so the bounds under review are the bounds actually in the repo and not a copy of them.

Measured by the reviewer on the single-threaded scheduler (see the review scratch for the sample tables):
- linear-work test: a verification holding the previous span grows 4.51 times the extra span bytes against 2.5.
- member-count test: holding every span across members gives eight-minus-one about 403 MB and eight-member about
  463 MB (red on both bounds); holding a whole slice during one call gives an eight-member peak of about 47 to 62 MB,
  red on the 0.5 * slice bound only.
"""

from __future__ import annotations

import inspect
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import test_consolidation_linear_work_review as linear  # noqa: E402
import test_experiment_consolidation_memory_review as members  # noqa: E402

from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as Post  # noqa: E402

_WANT = "want_piece = want_piece.copy(data=want_piece.data.persist())"
_GOT = "got_piece = got_piece.copy(data=got_piece.data.persist())"


def _verification_with(kind: str):
    """The correct ``_stored_slice_equals`` source with one retention line added.

    ``previous_span``: the last two spans' persisted sides stay alive. ``every_span_across_calls``: every persisted
    side is kept in a module-level list for the life of the process. ``whole_slice_per_call``: every persisted side is
    kept until the call returns, so a call peaks near its whole slice.
    """
    import hhemt.processing_analysis as pa

    src = textwrap.dedent(inspect.getsource(Post.__dict__["_stored_slice_equals"].__func__))
    assert _WANT in src and _GOT in src, "the persist lines this reviewer test patches have moved"
    if kind == "previous_span":
        src = src.replace(_WANT, _WANT + "; _HELD.append(want_piece); _HELD[:] = _HELD[-4:]")
        src = src.replace(_GOT, _GOT + "; _HELD.append(got_piece); _HELD[:] = _HELD[-4:]")
    elif kind == "last_span":
        src = src.replace(_WANT, _WANT + "; _HELD[:] = [want_piece]")
        src = src.replace(_GOT, _GOT + "; _HELD.append(got_piece); _HELD[:] = _HELD[-2:]")
    elif kind == "every_span_across_calls":
        src = src.replace(_WANT, _WANT + "; _HELD.append(want_piece)").replace(_GOT, _GOT + "; _HELD.append(got_piece)")
    elif kind == "whole_slice_per_call":
        marker = "    cls = TRITONSWMM_analysis_post_processing"
        src = src.replace(marker, "    _LOCAL = []\n" + marker)
        src = src.replace(_WANT, _WANT + "; _LOCAL.append(want_piece)")
        src = src.replace(_GOT, _GOT + "; _LOCAL.append(got_piece)")
    else:
        raise AssertionError(kind)
    ns = dict(vars(pa))
    ns["_HELD"] = []
    exec(src, ns)
    return staticmethod(ns["_stored_slice_equals"])


def test_the_member_count_test_passes_on_the_correct_verification(tmp_path, monkeypatch):
    """Control: the coder's member-count test is green on the unmodified verification, so the mutant tests below are
    red because of the mutant and not because of the host."""
    members.test_the_peak_of_copying_n_members_one_after_another_does_not_grow_with_n(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("kind", "bound"),
    [
        ("every_span_across_calls", "both"),
        ("whole_slice_per_call", "absolute"),
    ],
)
def test_the_member_count_test_fails_on_a_verification_that_holds_more_than_one_span(
    tmp_path, monkeypatch, kind, bound
):
    """Property: the member-count test is red on a verification that keeps every span across members, on both bounds,
    and on one that holds a whole slice during a call, on the absolute 0.5 * slice bound.

    Class: the two wrong implementations the test's docstring names. Both-states anchor: tracemalloc peak. Must-fire
    control: the correct-code test above is green. Killed by: widening either bound past what the mutant reaches.
    """
    monkeypatch.setattr(Post, "_stored_slice_equals", _verification_with(kind))
    with pytest.raises(AssertionError) as raised:
        members.test_the_peak_of_copying_n_members_one_after_another_does_not_grow_with_n(tmp_path, monkeypatch)
    message = str(raised.value)
    if bound == "both":
        assert "from 1 to 8 members" in message, message
    else:
        assert "against a" in message and "one-event slice" in message, message


def test_the_span_growth_bound_of_the_linear_work_test_fails_on_a_verification_holding_two_spans(tmp_path, monkeypatch):
    """Property: the linear-work memory test is red on a verification that keeps the previous span's persisted sides
    alive (measured growth 4.51 times the extra span bytes against the 2.5 bound) and green on the correct one.

    Class: the two-spans-at-once wrong implementation named in that test's docstring.
    """
    (tmp_path / "correct").mkdir()
    linear.test_resident_memory_of_the_verification_grows_with_the_span_and_not_with_the_slice(
        tmp_path / "correct", monkeypatch
    )
    monkeypatch.setattr(Post, "_stored_slice_equals", _verification_with("previous_span"))
    (tmp_path / "mutant").mkdir()
    with pytest.raises(AssertionError) as raised:
        linear.test_resident_memory_of_the_verification_grows_with_the_span_and_not_with_the_slice(
            tmp_path / "mutant", monkeypatch
        )
    assert "peak grew by more than the extra span bytes" in str(raised.value), str(raised.value)


def _residual_after_a_warm_member(tmp_path, kind: str) -> int:
    """Traced bytes still live, after a collection, once one member's group has been copied and verified by ``kind``
    (``correct`` or a mutant of ``_verification_with``): what the verification leaves behind, which a peak sample
    cannot separate from its transient allocations. One member is first copied by the correct verification, so the
    imports and codec caches a first copy builds (5.7 MB under pytest) are not counted."""
    import gc
    import tracemalloc

    import dask

    from hhemt import sensitivity_analysis as sa
    from hhemt.utils import deferred_consolidation

    Post._TIMESERIES_BLOCK_BYTES = 1 << 18
    Post._TIMESERIES_VERIFY_SPAN_BYTES = 1 << 20
    src = tmp_path / "src.zarr"
    group = members._gridded_member(src, 384)
    master = tmp_path / "master.zarr"
    with dask.config.set(scheduler="single-threaded"):
        tracemalloc.start()
        try:
            with deferred_consolidation(master):
                sa._copy_series_group(src, group, master, f"member_warm/{group}", 5, member_id="warm")
                if kind != "correct":
                    Post._stored_slice_equals = _verification_with(kind)
                gc.collect()
                before = tracemalloc.get_traced_memory()[0]
                sa._copy_series_group(src, group, master, f"member_0/{group}", 5, member_id="0")
                gc.collect()
                return tracemalloc.get_traced_memory()[0] - before
        finally:
            tracemalloc.stop()


@pytest.fixture
def restore_span_constants():
    saved = (Post._TIMESERIES_BLOCK_BYTES, Post._TIMESERIES_VERIFY_SPAN_BYTES, Post.__dict__["_stored_slice_equals"])
    yield
    Post._TIMESERIES_BLOCK_BYTES, Post._TIMESERIES_VERIFY_SPAN_BYTES = saved[0], saved[1]
    Post._stored_slice_equals = saved[2]


@pytest.mark.parametrize("kind", ["correct", "last_span", "previous_span", "every_span_across_calls"])
def test_what_the_verification_leaves_live_after_a_member_is_below_one_megabyte(tmp_path, restore_span_constants, kind):
    """Property: after a member is copied, the bytes the verification leaves live are under 1 MB on correct code
    (measured 0.14 to 0.23 MB after a warm-up copy) and over it on a verification that keeps a span of the previous
    member alive (last span both sides about 2 MB, last two spans about 4 MB, every span 58 MB).

    Class: retention past the call, the two-at-once analogue of the spans case, which the peak bounds of the
    member-count test cannot see (a one-span hold raises the eight-minus-one difference by 2.8 to 3.9 MB against a
    correct-code range of 1.1 to 2.8 MB). Both-states anchor: traced bytes after a collection. Must-fire control: the
    correct code row is green and the mutant rows are red.
    """
    residual = _residual_after_a_warm_member(tmp_path, kind)
    if kind == "correct":
        assert residual < 1_000_000, f"correct code left {residual} bytes live"
    else:
        assert residual >= 1_000_000, f"{kind} left only {residual} bytes live"
