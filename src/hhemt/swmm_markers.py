"""The two EPA SWMM log literals this toolkit decides a standalone SWMM run by, kept once,
and the check that the third-party program printing them still prints them.

``TRITONSWMM_run.model_run_completed`` decides a ``model_type == "swmm"`` run succeeded by
looking for two literal strings in the model's log: ``SWMM_COMPLETION_MARKER``, printed
unconditionally by EPA SWMM's ``src/run/main.c`` after ``swmm_run`` returns, and
``SWMM_ERROR_MARKER``, appended to that same line when ``swmm_getError() > 0``. Both are
literals in third-party C source cloned at a tag the operator selects (``SWMM_tag_key``),
so no test in this repository compared them against the program that was actually built.
This module is that comparison: one home for the two literals, and one helper that reads a
clone's ``src/run/main.c`` and reports the markers it does NOT contain.

TWO PRODUCERS, ONE HELPER. The native path clones at compile time into
``{build_dir}/swmm_source``, and ``TRITONSWMM_system._compile_SWMM_locked`` emits a
non-failing ``grep -qF`` per literal into the generated ``compile_swmm.sh`` AFTER the clone
block and OUTSIDE its ``if`` -- so a clone that already exists at a different tag is checked
too. The container path is deliberately NOT covered by editing a recipe: a recipe edit
changes ``recipe_sha256``, which ``hhemt.sif.identity.SifIdentity.key`` hashes into the
image identity, so a one-line edit re-keys every built image and forces a rebuild. The four
recipes clone the standalone source to ``/opt/swmm-src`` and the built image retains it, so
:func:`producer_check` reads that tree instead, at the point a standalone-SWMM completion
verdict is evaluated.

ABSENCE IS NOT A FINDING. A source tree that is not there yields ``None``: a native run, a
binary-only install, or a container built from a recipe revision that prunes its source
tree (§ 9 of the shape states that last residual rather than closing it). An UNREADABLE
source tree yields the same ``None``, for the same reason the check exists at all: it is a
warning about third-party source, and a warning that can raise aborts the completion
verdict it was meant to annotate. Only a ``main.c`` that EXISTS, is READABLE, and lacks a
literal produces a result, because only then is there a producer to be wrong about.
"""

import os
from pathlib import Path

#: EPA SWMM's end-of-run line prefix, printed by the engine's ``main()`` after the run
#: returns, whether or not the analysis succeeded. Read by ``model_run_completed``'s
#: raw-marker fallback. ONE definition: ``run_simulation`` re-exports this name so a caller
#: that has always imported it from there keeps resolving, and the generated compile script
#: interpolates it into bash rather than retyping it.
SWMM_COMPLETION_MARKER = "EPA SWMM completed"

#: The clause EPA SWMM appends to that same line when the run had errors. Its presence turns
#: an apparently completed SWMM run into a failed one, so it is a NEGATIVE conjunct of the
#: verdict and its disappearance is the more dangerous of the two.
SWMM_ERROR_MARKER = "There are errors."

#: Both literals, in verdict order, for the checker and for the generated compile script.
SWMM_MARKERS: tuple[str, ...] = (SWMM_COMPLETION_MARKER, SWMM_ERROR_MARKER)

#: The file EPA SWMM prints both literals from, relative to a clone's root. Scoped to this
#: one file on purpose: ``src/solver/text.h`` also carries ``" There are errors."`` (as
#: ``#define FMT03``), so a match anywhere under ``src/`` would pass even if ``main.c``
#: stopped printing the clause.
MAIN_C_RELATIVE_PATH = "src/run/main.c"

#: Where the four container recipes clone the standalone source to, and where the built
#: image RETAINS it: ``openmpi-cpu.def`` clones into this tree and copies the built binary
#: out, and nothing removes it.
CONTAINER_SWMM_SOURCE_ROOT = Path("/opt/swmm-src")

#: absolute ``main.c`` path -> ``(st_mtime_ns, st_size, missing-at-that-stat)``. Module-level
#: because the lifetime being memoised is the process's and the chokepoint is one function.
_PRODUCER_CHECK_CACHE: dict[str, tuple[int, int, tuple[str, ...]]] = {}


def missing_markers(main_c: Path | str) -> tuple[str, ...]:
    """The literals ABSENT from ``main_c``, in :data:`SWMM_MARKERS` order.

    ``()`` means both literals were found, which is the healthy case. Reads with
    ``errors="replace"``: the file is third-party C source and this check must never fail a
    build over a decoding accident.

    THIS FUNCTION PROPAGATES ``OSError`` ON PURPOSE, and that is why it is the wrong place for
    the degradation the two callers below implement. Its return type is a tuple of missing
    literals with NO way to say "no verdict": ``()`` would claim both literals are present and
    the full tuple would claim both are missing, so an unreadable file would produce a FALSE
    VERDICT either way. Only a signature admitting ``None`` can express "this producer could
    not be checked", so the guard lives at :func:`missing_markers_in_source_tree` and
    :func:`producer_check` -- never here.
    """
    text = Path(main_c).read_text(errors="replace")
    return tuple(marker for marker in SWMM_MARKERS if marker not in text)


def missing_markers_in_source_tree(source_root: Path | str) -> tuple[str, ...] | None:
    """Missing literals for ``{source_root}/src/run/main.c``, or ``None`` when it cannot be checked.

    ``None`` covers BOTH ways a producer can be unavailable, and they are deliberately the same
    answer: there is no ``main.c`` (``is_file()`` false -- which is also what a DIRECTORY at that
    path reports), or the file exists and the READ fails (``PermissionError``, or an
    ``IsADirectoryError`` on a path that races into a directory between the two syscalls). The
    ``None``/``()`` distinction is load-bearing and is why this does not simply return the tuple
    from :func:`missing_markers`: reporting "both missing" for an unavailable producer would warn
    on every non-container run, and an exception would abort the caller.
    """
    main_c = Path(source_root) / MAIN_C_RELATIVE_PATH
    try:
        if not main_c.is_file():
            return None
        return missing_markers(main_c)
    except OSError:
        return None


def producer_check(source_root: Path | str = CONTAINER_SWMM_SOURCE_ROOT) -> tuple[str, ...] | None:
    """Memoised :func:`missing_markers_in_source_tree` -- one ``stat`` per call after the first.

    WHY THE MEMO EXISTS. This runs at the point a standalone-SWMM completion verdict is
    evaluated, which is once per model log read, and a file read there would put an
    uncached disk read on a hot path to answer a question whose answer cannot change while
    the process lives. ``stat`` is the cheapest invalidation available: the cache is keyed on
    the absolute ``main.c`` path and validated against its ``(st_mtime_ns, st_size)``, so a
    source tree swapped underneath a long-lived process is re-read rather than trusted.

    Returns ``None`` when no source tree was retained OR the retained tree cannot be read
    (see :func:`missing_markers_in_source_tree`) and the missing literals otherwise.

    WHY THE READ IS GUARDED AND NOT ONLY THE STAT. This function is called from a completion
    verdict, which must not fail: the check is a WARNING about third-party source, so a
    ``PermissionError`` or ``IsADirectoryError`` from the read has to degrade to the same
    "nothing to check" answer the absent-tree branch gives. A guard on the ``stat`` alone still
    lets an unreadable ``main.c`` propagate out of the verdict. A read FAILURE is not cached --
    the entry is dropped and ``None`` returned -- because a transient permission problem must
    not become a permanent silence for the rest of the process's life.
    """
    main_c = Path(source_root) / MAIN_C_RELATIVE_PATH
    key = os.path.abspath(os.fspath(main_c))
    try:
        stat = main_c.stat()
    except OSError:
        # No source tree here. Drop any entry a previous stat cached for this path so a
        # vanished tree cannot keep answering from stale content.
        _PRODUCER_CHECK_CACHE.pop(key, None)
        return None
    cached = _PRODUCER_CHECK_CACHE.get(key)
    if cached is not None and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]
    try:
        missing = missing_markers(main_c)
    except OSError:
        # Present but unreadable. Same answer as absent, and NOT memoised: a transient
        # permission problem must not be remembered as a standing verdict.
        _PRODUCER_CHECK_CACHE.pop(key, None)
        return None
    _PRODUCER_CHECK_CACHE[key] = (stat.st_mtime_ns, stat.st_size, missing)
    return missing


def clear_producer_check_cache() -> None:
    """Drop the process-lifetime memo. For tests, and for a caller that re-points a root."""
    _PRODUCER_CHECK_CACHE.clear()
