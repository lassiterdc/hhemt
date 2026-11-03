"""Zip reading and writing for bundles, so every bundle zip is made and opened one way.

A bundle is shipped as one zip, and the zip is the artifact people copy between
machines. Three things about it have to hold wherever it is written or read, and they
used to be re-decided at each call site:

* Bytes must be reproducible. Two writes of the same entries give the same entries in
  the same order with the same stamps and the same content bytes; the archive bytes
  themselves match whenever the same zlib build compresses them (DEFLATE output is not
  specified across zlib implementations, so a byte comparison between machines with
  different zlib builds can differ while every entry's content is identical).
* Compression must be moderate: DEFLATE at zlib level 6 for every entry except zarr
  chunk files, which are STORED. The toolkit already compresses each zarr chunk before
  it is written, so deflating it again spends CPU for about 2% of size.
* Reading a zip with tens of thousands of entries must not be one file create after
  another. ``extract_zip`` spreads the creates over threads.

Everything here uses the standard library only, so importing it costs nothing and it
can sit below every other bundle module without creating an import cycle.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event

__all__ = [
    "MODERATE",
    "CompressionPolicy",
    "copy_into_zip",
    "extract_zip",
    "is_zarr_chunk",
    "render_set",
    "write_zip",
]

# 1 MiB blocks: large enough that per-call overhead is invisible against a file read,
# small enough that no entry is ever held whole in memory.
_BLOCK_BYTES = 1 << 20

# Every entry carries the zipfile module's minimum valid date and one fixed mode, so the
# real file timestamps and permissions of the machine that wrote the zip never reach it.
_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_FILE_MODE = 0o644

# The filenames that are zarr METADATA documents (v3 `zarr.json`; v2 `.zarray`, `.zgroup`,
# `.zattrs` and the consolidated `.zmetadata`). Written out rather than imported from zarr
# so this module stays standard-library-only; a test compares this set with zarr's own
# constants, so a zarr rename turns that test red instead of silently storing a document.
_ZARR_METADATA_NAMES = frozenset({"zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata"})

# A zip source is a file path, the bytes themselves, or one entry of an already open zip.
ZipSource = Path | bytes | tuple[zipfile.ZipFile, str]


def is_zarr_chunk(name: str) -> bool:
    """Whether the archive entry ``name`` is a chunk file of a zarr store.

    A chunk file is any file inside a directory whose name ends in ``.zarr`` that is not
    one of the store's metadata documents. The check is on the entry's path alone, so
    the same answer comes from a name read out of a zip and from a name about to be
    written. A file that is itself called ``x.zarr`` (a zipped store, say) is not a
    chunk, because only the directories ABOVE the last path segment count.
    """
    *directories, filename = name.split("/")
    if filename in _ZARR_METADATA_NAMES:
        return False
    return any(segment.endswith(".zarr") for segment in directories)


@dataclass(frozen=True)
class CompressionPolicy:
    """How a zip's entries are compressed: one DEFLATE level, and whether zarr chunks skip it.

    Attributes:
        level: zlib level used for every entry that is deflated.
        store_zarr_chunks: when true, entries for which ``is_zarr_chunk`` holds are
            written STORED instead of deflated.
    """

    level: int
    store_zarr_chunks: bool

    def compress_type_for(self, arcname: str) -> int:
        """The zip compression method this policy gives the entry named ``arcname``."""
        if self.store_zarr_chunks and is_zarr_chunk(arcname):
            return zipfile.ZIP_STORED
        return zipfile.ZIP_DEFLATED


# Level 6 is zlib's own default. Levels 6 and 9 gave the same size to three figures on a
# measured 41,449-entry bundle while 9 cost 48% more CPU, and level 1 was 0.9% larger for
# 23% less CPU, so the conventional middle is the moderate choice.
MODERATE = CompressionPolicy(level=6, store_zarr_chunks=True)


def _source_size(source: ZipSource) -> int:
    """The uncompressed size in bytes of ``source``, known before any byte is read."""
    if isinstance(source, bytes | bytearray):
        return len(source)
    if isinstance(source, tuple):
        source_zip, member_name = source
        return source_zip.getinfo(member_name).file_size
    return Path(source).stat().st_size


def _stream_source(source: ZipSource, destination) -> None:
    """Copy ``source`` into the open entry ``destination`` in 1 MiB blocks."""
    if isinstance(source, bytes | bytearray):
        destination.write(source)
    elif isinstance(source, tuple):
        source_zip, member_name = source
        with source_zip.open(member_name) as reader:
            shutil.copyfileobj(reader, destination, _BLOCK_BYTES)
    else:
        with Path(source).open("rb") as reader:
            shutil.copyfileobj(reader, destination, _BLOCK_BYTES)


def _write_entry(
    zf: zipfile.ZipFile,
    arcname: str,
    source: ZipSource,
    compress_type: int,
    level: int,
) -> None:
    """Write one entry with the fixed date and mode, streaming ``source`` into it."""
    info = zipfile.ZipInfo(filename=arcname, date_time=_FIXED_DATE_TIME)
    info.compress_type = compress_type
    # ZipFile.open() applies the ZipFile's own `compresslevel` only to a name it builds
    # itself; a ZipInfo passed in keeps `_compresslevel` unset, which silently means
    # zlib's default (-1) whatever level the caller asked for.
    info._compresslevel = level  # noqa: SLF001
    info.external_attr = _FILE_MODE << 16
    # file_size set from the source up front is what lets zipfile stream the entry and
    # still choose the zip64 form correctly for a file over 2 GiB.
    info.file_size = _source_size(source)
    with zf.open(info, "w") as destination:
        _stream_source(source, destination)


def write_zip(
    entries: Iterable[tuple[str, ZipSource]],
    dest: Path,
    *,
    policy: CompressionPolicy = MODERATE,
) -> Path:
    """Write ``entries`` as one reproducible zip at ``dest`` and return ``dest``.

    Each entry is an ``(arcname, source)`` pair, where ``source`` is a file path, the
    entry's bytes, or a ``(ZipFile, name)`` pair naming an entry of an open zip. Entries
    are written in the order of their path PARTS (``"a/b"`` before ``"a-b/x"``, which is
    the order a directory walk gives), each with the date 1980-01-01 and mode 0o644, and
    each compressed as ``policy`` says.

    The zip is built at ``dest`` plus a ``.partial`` suffix and renamed over ``dest``
    only after it is complete, so a failure or an interrupt leaves ``dest`` exactly as
    it was and removes the partial file. Two arcnames that are equal raise
    ``ValueError`` rather than producing a zip with two entries of one name.
    """
    dest = Path(dest)
    pairs = list(entries)
    names = [arcname for arcname, _ in pairs]
    if len(set(names)) != len(names):
        duplicated = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"write_zip got the same archive name more than once: {duplicated[:5]}")
    ordered = sorted(pairs, key=lambda pair: pair[0].split("/"))
    partial = dest.with_name(dest.name + ".partial")
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as zf:
            for arcname, source in ordered:
                _write_entry(zf, arcname, source, policy.compress_type_for(arcname), policy.level)
        os.replace(partial, dest)
    except BaseException:
        partial.unlink(missing_ok=True)  # EXEMPT-DU: transient-intermediate
        raise
    return dest


def copy_into_zip(zf: zipfile.ZipFile, arcname: str, src_path: Path) -> None:
    """Add the file ``src_path`` to the open zip ``zf`` as ``arcname``, STORED.

    The file is copied whole and never compressed, whatever it is. Use it for a file
    that is already an archive: compressing a zip inside a zip costs CPU for nothing
    and makes the inner file's bytes differ from the original, so its checksum no
    longer matches the one recorded for it. The entry carries the same fixed date and
    mode as ``write_zip`` gives, and the caller owns the order of calls.
    """
    _write_entry(zf, arcname, Path(src_path), zipfile.ZIP_STORED, MODERATE.level)


def _is_render_excluded(name: str) -> bool:
    """Whether ``name`` sits under an ``out_*`` directory's ``bin/`` directory."""
    parts = name.split("/")
    # `out_*` starts at index 1 because the raw solver directories always sit under a
    # scenario folder, so a name that begins with one is not one of them. It stops at
    # len(parts) - 3 because `bin` must be a directory (something sits beneath it); a
    # FILE called `bin` inside an `out_*` directory is kept.
    return any(parts[index].startswith("out_") and parts[index + 1] == "bin" for index in range(1, len(parts) - 2))


def render_set(names: Iterable[str]) -> list[str]:
    """The entries of ``names`` that a re-render can read, in their given order.

    That is every name except those inside a raw solver output directory
    (``.../out_*/bin/...``). Those files are most of a bundle's bytes and no renderer
    opens them, so extracting them only spends file creates. Everything else, including
    every zarr document and chunk and every text file, is kept.
    """
    return [name for name in names if not _is_render_excluded(name)]


def _confined_target(dest: Path, name: str) -> Path:
    """Where ``zipfile`` would put the entry ``name`` under ``dest``.

    This mirrors the sanitising ``ZipFile._extract_member`` applies (drive and leading
    separators dropped, ``..`` and ``.`` segments dropped), so a name that climbs out of
    the destination lands inside it, as it does under ``extractall``. It is used to
    create directories and to test existing files before any thread runs; the bytes
    themselves are always written by ``ZipFile.extract``.
    """
    arcname = name.replace("/", os.path.sep)
    if os.path.altsep:
        arcname = arcname.replace(os.path.altsep, os.path.sep)
    arcname = os.path.splitdrive(arcname)[1]
    arcname = os.path.sep.join(x for x in arcname.split(os.path.sep) if x not in ("", os.path.curdir, os.path.pardir))
    return Path(os.path.normpath(os.path.join(dest, arcname)))


def _extract_slice(src: Path, dest: Path, infos: list[zipfile.ZipInfo], stop: Event) -> None:
    """One worker: extract ``infos`` through this worker's own ``ZipFile``.

    A ``ZipFile`` shares one file position between its readers, so concurrent reads
    through one object interleave and fail CRC checks; each worker therefore opens the
    zip itself.
    """
    with zipfile.ZipFile(src) as zf:
        for info in infos:
            if stop.is_set():
                return
            zf.extract(info, dest)


def extract_zip(
    src: Path,
    dest: Path,
    *,
    threads: int = 8,
    members: Iterable[str] | None = None,
    only_missing_or_resized: bool = False,
) -> None:
    """Extract the zip ``src`` into the directory ``dest``, creating files on ``threads`` threads.

    Directories are created first and serially, then the files are split into
    contiguous runs, one per thread, each read through its own ``ZipFile``. The
    resulting files and bytes are those ``ZipFile.extractall`` gives, and so is the
    confinement: an entry whose name climbs out of ``dest`` is written inside it. An
    error raised while extracting (a bad CRC, an unreadable entry) is raised to the
    caller unchanged once the other threads have stopped.

    Args:
        src: the zip to read.
        dest: the directory to extract into, created if absent.
        threads: how many threads create files; 1 extracts in the calling thread.
        members: the entry names to extract; ``None`` extracts every entry. A name
            absent from the zip raises ``KeyError`` before anything is written.
        only_missing_or_resized: skip an entry whose file already exists at the size
            the zip records for it. This makes a repeat extraction into the same
            directory cheap; it trusts size alone, so a same-size edit is kept.

    Raises:
        ValueError: ``threads`` is below 1.
    """
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    src = Path(src)
    dest = Path(dest)
    with zipfile.ZipFile(src) as zf:
        infos = list(zf.infolist()) if members is None else [zf.getinfo(name) for name in members]
    dest.mkdir(parents=True, exist_ok=True)

    # Entries whose names confine to one target are grouped, so one thread writes them in
    # archive order. Spread over threads they race, and the entry left on disk is whichever
    # finished last rather than the later one, which is what `extractall` leaves.
    groups: dict[Path, list[zipfile.ZipInfo]] = {}
    directories: set[Path] = set()
    for info in infos:
        target = _confined_target(dest, info.filename)
        if info.is_dir():
            directories.add(target)
            continue
        directories.add(target.parent)
        if only_missing_or_resized and target.is_file() and target.stat().st_size == info.file_size:
            continue
        groups.setdefault(target, []).append(info)
    for directory in sorted(directories):
        directory.mkdir(parents=True, exist_ok=True)

    group_list = list(groups.values())
    worker_count = min(threads, len(group_list))
    if worker_count <= 1:
        _extract_slice(src, dest, [info for group in group_list for info in group], Event())
        return
    run_length = -(-len(group_list) // worker_count)
    slices = [
        [info for group in group_list[start : start + run_length] for info in group]
        for start in range(0, len(group_list), run_length)
    ]
    stop = Event()
    with ThreadPoolExecutor(max_workers=len(slices)) as pool:
        futures = [pool.submit(_extract_slice, src, dest, run, stop) for run in slices]
        try:
            for future in futures:
                future.result()
        except BaseException:
            stop.set()
            raise
