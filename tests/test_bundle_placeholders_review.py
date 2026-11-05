"""Reviewer tests for chunk 2a.1 (the placeholder table applied at archive-write time).

SOLVER-FREE: every arm writes small files under ``tmp_path`` or inspects classes; no fixture here
compiles or runs a solver, and none requests a ``synth_*`` fixture.

Producer-looking values are BUILT FROM FRAGMENTS, never written as literals, so this file is not
itself a hit for the repository's anonymization guard.

Each of the four tests pins a behaviour the approved shape states and the coder's own file passes
without pinning: each is GREEN on the coder's tree and goes RED under a named mutation of it (the
mutations are listed in each docstring and were run against a copy of the tree).
"""

from __future__ import annotations

import inspect
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

from hhemt.bundle import _placeholders as P
from hhemt.bundle._emit import _emit_bundle_zip
from hhemt.bundle._reprex_gate import _SNIFF_BYTES

_ROOT = "/scr" + "atch/" + "x" * 50  # 59 characters, an absolute path, producer-shaped


def _put(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


def test_a_rewritten_text_members_zip_entry_is_sized_from_its_rewritten_bytes(tmp_path, monkeypatch):
    """Property: the entry header of a rewritten text member is built from the REWRITTEN length, so a member
        that shrinks below the zip64 threshold is written without zip64 fields.

    Class: NEW CAPABILITY - the plausible wrong implementation is ``info.file_size`` left at the ``stat``
        value (the coder's mutant M8). The coder's own file says it cannot pin that, because ``zipfile``
        rewrites the header from the bytes written; it can, through the one place the OPENING size is read
        and never rewritten: ``ZipFile._open_to_write`` decides zip64 from ``zinfo.file_size * 1.05 >
        ZIP64_LIMIT`` at open, and ``_ZipWriteFile.close`` then writes a local header carrying the zip64
        extra whenever it was chosen. Red under the mutant: the local header's extra length is 20.
    Reachability: a harvested text member larger than the threshold whose producer paths shrink it below it;
        artifact: ``_emit_bundle_zip`` sets each rewritten text member's ``file_size`` from the rewritten
        length before its zip entry is opened. The threshold is lowered here so the 100 KiB-scale input stands in for a
        2 GiB one.
    Both-states anchor: the assertion is on a header field both the shipped and the mutant emitter write, not
        on any wording the change introduces.
    Invocation isolated: ``_emit_bundle_zip`` reads only its arguments; ``ZIP64_LIMIT`` is restored by
        ``monkeypatch``.
    What kills it: leaving ``file_size`` at ``src.stat().st_size`` for a rewritten text member.
    Second correct implementation: opening the entry with ``force_zip64=False`` and the length computed from
        a first pass, or buffering the rewritten bytes in a ``BytesIO`` and calling ``writestr``.
    """
    text = ("run " + _ROOT + "/data\n") * 100  # stat size well above the lowered limit
    staging = tmp_path / "staging"
    _put(staging / "a_first.txt", text)
    table = P.PlaceholderTable.from_values({P.TOKEN_ANALYSIS_DIR: [_ROOT]})
    rewritten = table.apply(text.encode())
    limit = 5000
    assert len(text) * 1.05 > limit > len(rewritten) * 1.05, "the input must straddle the lowered threshold"

    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", limit)
    out = tmp_path / "b.zip"
    _emit_bundle_zip(staging, out, streamed=None, placeholders=table, declared_arcnames=frozenset())

    raw = out.read_bytes()
    assert raw[:4] == b"PK\x03\x04"
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    assert name_len == len(b"a_first.txt")
    assert extra_len == 0, "a zip64 extra in the local header means the entry was opened with the stat size"
    with zipfile.ZipFile(out) as zf:
        assert zf.read("a_first.txt") == rewritten


def test_a_value_straddling_the_sniff_prefix_boundary_is_rewritten(tmp_path):
    """Property: a producer value is rewritten wherever it sits relative to the end of the sniffed prefix,
        including when it begins before byte 8,192 and ends after it.

    Class: NEW CAPABILITY - the plausible wrong implementation rewrites the prefix and the remainder
        separately (``apply(prefix) + apply(rest)``), which cannot see a value across the seam; red under it
        for every offset in the straddling class below. A second wrong implementation reads only the prefix.
    Reachability: ``Snakefile.source`` and the baseline html page are tens of kilobytes of text, so a value
        lands across the 8,192nd byte for some member in some emit; artifact: the 8,192-byte sniff prefix
        is read from the same handle as the rest of the member, which is then read whole and rewritten as one.
    Both-states anchor: the assertion is on the member's bytes after the emit (value absent, token present),
        true of any correct emitter.
    Invocation isolated: ``_emit_bundle_zip`` reads only its arguments.
    What kills it: applying the table to the two reads independently.
    Second correct implementation: reading the whole member in one call and sniffing its first 8,192 bytes.
    """
    value = _ROOT
    table = P.PlaceholderTable.from_values({P.TOKEN_ANALYSIS_DIR: [value]})
    straddling_offsets = range(_SNIFF_BYTES - len(value) + 1, _SNIFF_BYTES)  # begins before, ends after
    assert len(straddling_offsets) == len(value) - 1
    for offset in (straddling_offsets[0], straddling_offsets[len(straddling_offsets) // 2], straddling_offsets[-1]):
        staging = tmp_path / f"s{offset}"
        body = b" " * offset + value.encode() + b" tail\n"
        _put(staging / "m.txt", body)
        out = tmp_path / f"o{offset}.zip"
        _emit_bundle_zip(staging, out, placeholders=table, declared_arcnames=frozenset())
        with zipfile.ZipFile(out) as zf:
            got = zf.read("m.txt")
        assert value.encode() not in got, f"value left in place at offset {offset}"
        assert got == b" " * offset + P.TOKEN_ANALYSIS_DIR.encode() + b" tail\n"


def test_a_symlinked_directory_is_rewritten_in_both_the_given_and_the_resolved_spelling(tmp_path):
    """Property: when the analysis directory is reached through a symlink, text naming it by EITHER its given
        spelling or its resolved target is rewritten to the analysis token.

    Class: NEW CAPABILITY - the plausible wrong implementations keep one spelling (the coder's deviation D1
        is the only thing that keeps both, and no test of the coder's pins it): ``resolve()`` only, which
        leaves a Snakefile that names the symlink spelling in place; or the given spelling only, which
        leaves a log or a SLURM-written path that names the target. Red under each.
    Reachability: a cluster scratch directory that is a symlink to a parallel-filesystem path; artifact:
        the toolkit's per-user path patterns list both the short scratch root and the ``sfs``
        parallel-filesystem scratch root, which is the symlink pair, and the table is built from the unresolved
        ``analysis_paths.analysis_dir`` while a tool that resolves symlinks writes the target.
    Both-states anchor: the assertion is on both spellings leaving the output, a property of the bytes.
    Invocation isolated: only ``tmp_path`` is read; the symlink is made inside it.
    What kills it: a ``_spellings`` that returns one spelling.
    Second correct implementation: the table taking ``os.path.realpath`` and ``os.path.abspath`` separately.
    """
    real = tmp_path / "parallel_fs" / "run_a1"
    real.mkdir(parents=True)
    link = tmp_path / "scratch_link"
    link.symlink_to(real.parent, target_is_directory=True)
    given = link / "run_a1"
    assert str(given) != str(given.resolve()), "the fixture must make the two spellings differ"

    table = P.build_placeholder_table(SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=given)))
    for spelling in (str(given), str(given.resolve())):
        got = table.apply(f"cd {spelling}/work".encode()).decode()
        assert got == f"cd {P.TOKEN_ANALYSIS_DIR}/work", f"spelling not rewritten: {spelling}"


def test_every_link_build_placeholder_table_follows_exists_on_the_real_classes():
    """Property: each attribute the table builder reads from an analysis is one the REAL classes define, so a
        row is never silently empty because a link is misnamed.

    Class: NEW CAPABILITY, a fitness function - the plausible wrong implementation misnames one link (say
        ``_system_cfg``). ``_chain`` returns None at the first missing link BY DESIGN (a partial analysis
        yields a partial table), so a misnamed link yields an EMPTY row and nothing raises. The coder's
        stand-in analysis is built with the same names the builder reads, so it cannot catch this; only the
        real classes can. Red under the mutant: the ``_system`` link is not found.
    Reachability: any emit of a real analysis; the system directory would then reach the archive
        unrewritten and the only backstop is the emit-time scan of config values, which this tree does not yet have.
    Both-states anchor: the assertion reads the real classes' definitions, which exist in both trees.
    Invocation isolated: class and source inspection only; nothing is constructed.
    What kills it: a renamed link in ``build_placeholder_table``.
    Second correct implementation: reading each value through a typed accessor on the real class.
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.config.hpc_system import hpc_system_config
    from hhemt.config.system import system_config
    from hhemt.paths import AnalysisPaths
    from hhemt.system import TRITONSWMM_system

    builder_src = inspect.getsource(P.build_placeholder_table)
    analysis_init = inspect.getsource(TRITONSWMM_analysis.__init__)
    system_init = inspect.getsource(TRITONSWMM_system.__init__)

    assert '"analysis_paths", "analysis_dir"' in builder_src
    assert "self.analysis_paths = AnalysisPaths(" in analysis_init
    assert "analysis_dir" in {f for f in AnalysisPaths.__dataclass_fields__}

    assert '"_system", "cfg_system", "system_directory"' in builder_src
    assert "self._system = system" in analysis_init
    assert "self.cfg_system = " in system_init
    assert "system_directory" in system_config.model_fields

    assert '"cfg_hpc_system", "default_account"' in builder_src
    assert "self.cfg_hpc_system = " in analysis_init
    assert "default_account" in hpc_system_config.model_fields
