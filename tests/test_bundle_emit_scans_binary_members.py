"""The emit-time zero-user-info check reads what a binary member HOLDS, not only its text members.

``emit_bundle`` scans text members with three layers and refuses the archive on a hit. Two kinds of
member used to be skipped: a zarr array of text values (its chunk files are classed binary by name, and
a text value lives inside them after the array's codecs have run), and every other member with a NUL
byte in its first block (a NetCDF file, a zip, a gzip stream). This module pins the repair: a zarr array
of text values is DECODED and its values scanned, a container the standard library can open is scanned
through its decompressor, and any other binary member is scanned in overlapping blocks. A private value
found in such a member REFUSES the emit (it is never rewritten), naming the member, layer and class and
never the value.

SOLVER-FREE: every arm writes real zarr stores, NetCDF files and archives over ``tmp_path`` and runs the
real ``_emit_bundle_zip`` (the function ``emit_bundle`` calls with its scanner) or the scanner itself. No
fixture here compiles or runs a solver, and the module is not ``compile_tier`` or ``shared_runs_root``.

Producer-looking values are BUILT FROM FRAGMENTS below, never written as literals: this file is itself
scanned by the repository's anonymization guard, which refuses a literal that matches its patterns.

Venue of the setup-table chain (open item O2 of the approved shape): Test 3 drives ``_emit_bundle_zip``
over a store written by the same expression that ``sensitivity_analysis.py`` uses for the ``parameters``
node. It needs no ``SimpleNamespace`` analysis (``tests.test_publish_deposit_paths._world``), because the
check lives in ``_emit_bundle_zip``, which takes a staging directory; it is a laptop test.

Each test docstring states Property, Class, Reachability, Both-states anchor, Invocation isolated, What
kills it, and a second correct implementation under which it still passes.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import random
import tarfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from hhemt.bundle import _reprex_gate as G
from hhemt.bundle._emit import _emit_bundle_zip
from hhemt.cf_conventions import stamp_parameter_units
from hhemt.exceptions import ProcessingError

# --------------------------------------------------------------------------------------
# Producer-shaped values, built from fragments (see module docstring)
# --------------------------------------------------------------------------------------


def _unix(root: str, user: str, rest: str = "x") -> str:
    """A path under a per-user root, assembled so no literal of the form appears in this file."""
    return "/" + root + "/" + user + "/" + rest


PLANTED = _unix("home", "someone", "data")
PLANTED_BYTES = PLANTED.encode()


@pytest.fixture(autouse=True)
def _no_private_list(monkeypatch):
    """Layer (c) off: every arm here is decided by layer (a), the per-user path class."""
    monkeypatch.delenv("HHEMT_REPREX_BLOCKLIST", raising=False)


def _scanner() -> G.ZeroUserInfoScanner:
    return G.ZeroUserInfoScanner.for_consume()


def _emit(staging: Path, tmp_path: Path, name: str = "out.zip") -> None:
    _emit_bundle_zip(staging, tmp_path / name, scanner=_scanner())


def _refusal(staging: Path, tmp_path: Path) -> str:
    """The message of the refusal ``_emit_bundle_zip`` raises over ``staging``.

    That the refusal leaves nothing at the OUTPUT path is ``emit_bundle``'s guarantee (it hands this function
    a ``.partial`` path and renames after the call returns) and is pinned by the existing emit tests; this
    function writes to the path it is given, so the archive exists here after a refusal.
    """
    out = tmp_path / "refused.zip"
    with pytest.raises(ProcessingError) as info:
        _emit_bundle_zip(staging, out, scanner=_scanner())
    assert PLANTED not in str(info.value), "the refusal must name the member and class, never the value"
    return str(info.value)


def _chunk_bytes(array_dir: Path) -> bytes:
    """Every byte of the array's chunk files (its metadata document excluded)."""
    return b"".join(f.read_bytes() for f in sorted(array_dir.rglob("*")) if f.is_file() and f.name != "zarr.json")


def _quiet_write(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zarr warns that some data types have no v3 specification
        return fn()


def _text_array(directory: Path, values, dtype) -> Path:
    """A zarr v3 array of text ``values`` at ``directory``; ``dtype`` is a numpy dtype or ``"str"``."""

    def write():
        if dtype == "str":
            arr = zarr.create_array(store=str(directory), shape=(len(values),), dtype="str", zarr_format=3)
            arr[:] = np.array(values, dtype=object)
        else:
            zarr.create_array(store=str(directory), data=np.array(values, dtype=dtype), zarr_format=3)

    _quiet_write(write)
    return directory


def _filler(n: int, seed: int = 7) -> str:
    """Deterministic lowercase filler that matches no layer-(a) class and does not compress to nothing."""
    rng = random.Random(seed)
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz \n") for _ in range(n))


def _payload() -> bytes:
    """About 2 KB of filler with the planted path in the middle."""
    return (_filler(1000, 1) + " " + PLANTED + " " + _filler(1000, 2)).encode()


# --------------------------------------------------------------------------------------
# Test 1: text arrays of a zarr store are decoded
# --------------------------------------------------------------------------------------

#: (arm, builder, expectation about the planted path in the raw chunk bytes). ABSENT arms prove decoding;
#: the PRESENT arm is a control showing that a raw byte scan alone would also have found it.
_ARRAYS = [
    ("fixed_length_utf32", lambda d: _text_array(d, [PLANTED, "cpu", "gpu"], "U60"), False),
    ("string_short", lambda d: _text_array(d, [PLANTED, "cpu", "cpu", "cpu"], "str"), True),
    (
        "string_1000",
        lambda d: _text_array(
            d,
            [f"filler-{i:04d}-{i * 7919 % 10007}" for i in range(500)]
            + [PLANTED]
            + [f"tail-{i:04d}" for i in range(499)],
            "str",
        ),
        False,
    ),
]


@pytest.mark.parametrize(("arm", "build", "raw_present"), _ARRAYS, ids=[a[0] for a in _ARRAYS])
def test_a_planted_path_in_a_text_array_refuses_the_emit(tmp_path, arm, build, raw_present):
    """Property: a per-user path stored as a value of a zarr text array refuses the emit, naming the array, in
        every text data type; and an array with the same shape but benign values passes.

    Class: REGRESSION - red on the parent, which classes every chunk file binary by name and scans none.
    Reachability: ``emit_bundle`` carries consolidated trees whose ``parameters/`` and identifier arrays are
        text; the planted value is the producer value the setup table or an identifier holds.
    Both-states anchor: each arm measures, in the test, whether the planted bytes occur in the raw chunk
        files and asserts the arm's declared expectation (ABSENT for ``fixed_length_utf32`` and the
        1000-value ``string`` array, which are the arms that prove DECODING; PRESENT for the short
        ``string`` array, a control that a raw byte scan would also find it).
    Invocation isolated: ``_emit_bundle_zip`` reads only its arguments; layer (c) is off.
    What kills it: a raw byte scan of chunk files (blind in the ABSENT arms); skipping text arrays.
    Second correct implementation: decoding through ``zarr.open_group`` and scanning every string leaf.
    """
    staging = tmp_path / "staging"
    array_dir = build(staging / "s.zarr" / "arr")
    assert (PLANTED_BYTES in _chunk_bytes(array_dir)) is raw_present, "unit: raw presence of the planted bytes"
    message = _refusal(staging, tmp_path)
    assert (
        "s.zarr/arr (decoded values)" in message and "pattern_rules" in message and "per-user absolute path" in message
    )


def test_benign_text_arrays_and_a_numeric_array_of_arbitrary_bytes_pass(tmp_path):
    """Property: the controls pass - an identifier array (``event_index.0``), a text array of short benign
        values, and a numeric array holding arbitrary bytes.

    Class: CONTROL for the refusal above (varies the property under suspicion: the content, not the dtype).
    Reachability: the same emit path as the refusal arms.
    Both-states anchor: these arrays differ from the refusing ones only in carrying no planted path.
    Invocation isolated: ``_emit_bundle_zip`` over a fresh staging tree.
    What kills it: a decoder that refuses every text array; a raw scan that fires on numeric chunk bytes.
    Second correct implementation: skipping arrays whose values match no pattern before decoding.
    """
    staging = tmp_path / "staging"
    _text_array(staging / "s.zarr" / "ids", ["event_index.0", "event_index.1"], "U20")
    _text_array(staging / "s.zarr" / "names", ["cpu", "gpu", "cpu"], "str")
    rng = np.random.default_rng(1234)
    _quiet_write(
        lambda: zarr.create_array(
            store=str(staging / "s.zarr" / "blob"), data=rng.integers(0, 256, 5000, dtype=np.uint8), zarr_format=3
        )
    )
    scanner = _scanner()
    _emit_bundle_zip(staging, tmp_path / "ok.zip", scanner=scanner)
    assert scanner.hits == [] and (tmp_path / "ok.zip").exists()


def test_the_decoder_finds_the_text_in_a_structured_array_field_and_leaves_numeric_fields(tmp_path):
    """Property: a ``structured`` array whose text field is ``U`` holds the planted path and refuses, asserting
        the path ABSENT from its raw chunk bytes (the arm proves decoding); a companion whose text field is
        ``S`` also refuses, with the path PRESENT in its raw bytes (a control, not a proof of decoding).

    Class: REGRESSION - red on the parent.
    Reachability: a structured array is the registry's one compound text-bearing type.
    Both-states anchor: the same structured shape with a benign value passes.
    Invocation isolated: ``_emit_bundle_zip`` over one store per case.
    What kills it: decoding only top-level text dtypes; scanning the numeric field as text.
    Second correct implementation: flattening the structured array to a record list before scanning.
    """
    for letter, raw_present in (("U", False), ("S", True)):
        staging = tmp_path / letter / "staging"
        rows = np.zeros(3, dtype=[("t", f"{letter}60"), ("n", "i4")])
        rows["t"][0] = PLANTED if letter == "U" else PLANTED_BYTES
        array_dir = staging / "s.zarr" / "rec"
        _quiet_write(
            lambda rows=rows, array_dir=array_dir: zarr.create_array(store=str(array_dir), data=rows, zarr_format=3)
        )
        assert (PLANTED_BYTES in _chunk_bytes(array_dir)) is raw_present, f"unit: raw presence, field {letter}"
        assert "s.zarr/rec (decoded values)" in _refusal(staging, tmp_path / letter)
    benign = tmp_path / "benign" / "staging"
    benign_rows = np.zeros(3, dtype=[("t", "U60"), ("n", "i4")])
    _quiet_write(lambda: zarr.create_array(store=str(benign / "s.zarr" / "rec"), data=benign_rows, zarr_format=3))
    _emit(benign, tmp_path / "benign")


def test_a_data_type_with_no_row_in_the_table_raises_instead_of_being_skipped(tmp_path):
    """Property: a zarr v3 array whose ``data_type`` name is in neither the decode rows nor the skip rows
        raises ``ProcessingError`` naming the type; it is never skipped.

    Class: REGRESSION for the table's closure rule (a type a later zarr adds has no row).
    Reachability: a future zarr registry entry; reproduced by editing the metadata document's name.
    Both-states anchor: the same store passes with a known name and raises with the unknown one.
    Invocation isolated: ``classify_zarr_array`` reads only the metadata document.
    What kills it: a classifier whose default branch is "skip".
    Second correct implementation: validating the name against the live registry instead of a table.
    """
    import json

    array_dir = _text_array(tmp_path / "a.zarr" / "arr", ["x"], "U4")
    assert G.classify_zarr_array(array_dir) == G.ZARR_DECODE
    doc = array_dir / "zarr.json"
    meta = json.loads(doc.read_text())
    meta["data_type"] = {"name": "zz_future_type"}
    doc.write_text(json.dumps(meta))
    with pytest.raises(ProcessingError, match="zz_future_type"):
        G.classify_zarr_array(array_dir)


# --------------------------------------------------------------------------------------
# Test 2: a NetCDF declared input
# --------------------------------------------------------------------------------------


def test_a_planted_history_attribute_in_a_netcdf_input_refuses_the_emit(tmp_path):
    """Property: a per-user path held in a NetCDF ``history`` attribute refuses the emit, naming the member;
        the same file without the attribute passes.

    Class: REGRESSION - red on the parent, which carries the NUL-prefixed file unscanned.
    Reachability: a declared input can be any format; NetCDF tools write ``history`` with the producer's path.
    Both-states anchor: two files, with and without the attribute.
    Invocation isolated: ``_emit_bundle_zip`` over a fresh staging tree per case.
    What kills it: skipping every NUL-prefixed member; scanning only the first block.
    Second correct implementation: reading global attributes through ``xarray`` for a known format.
    """
    planted_dir = tmp_path / "planted" / "staging" / "external"
    planted_dir.mkdir(parents=True)
    xr.Dataset({"v": ("x", np.arange(4.0))}, attrs={"history": "created in " + PLANTED}).to_netcdf(
        planted_dir / "weather.nc"
    )
    assert b"\x00" in (planted_dir / "weather.nc").read_bytes()[:8192], "the file must sniff as binary"
    assert "external/weather.nc" in _refusal(tmp_path / "planted" / "staging", tmp_path / "planted")

    clean_dir = tmp_path / "clean" / "staging" / "external"
    clean_dir.mkdir(parents=True)
    xr.Dataset({"v": ("x", np.arange(4.0))}).to_netcdf(clean_dir / "weather.nc")
    _emit(tmp_path / "clean" / "staging", tmp_path / "clean")


# --------------------------------------------------------------------------------------
# Test 3: the setup-table chain
# --------------------------------------------------------------------------------------


def test_a_producer_path_in_the_parameters_node_refuses_with_the_setup_csv_excluded(tmp_path):
    """Property: a ``parameters`` node written by the expression ``sensitivity_analysis.py`` uses for the
        setup table (``stamp_parameter_units(xr.Dataset.from_dataframe(df_setup), ...)``) refuses the emit
        when a row holds a producer path, although no text member of the staging tree carries it (the setup
        CSV is left out).

    Class: REGRESSION - red on the parent, which scans none of the node's chunk files.
    Reachability: ``df_setup`` rows hold per-member paths and the node is written into the consolidated tree
        that every sensitivity bundle carries.
    Both-states anchor: the same table with a benign row passes.
    Invocation isolated: ``_emit_bundle_zip`` over a tree written by ``DataTree.to_zarr``.
    What kills it: any scan that reads the CSV only; a scan that skips ``parameters/``.
    Second correct implementation: refusing at ``df_setup`` build time instead of at emit.
    """

    def tree(path_value: str) -> Path:
        staging = tmp_path / ("p" if path_value == PLANTED else "b") / "staging"
        df = pd.DataFrame(
            {"weather_dir": [path_value, "cpu"], "partition": ["a", "b"]}, index=pd.Index(["0", "1"], name="sa_id")
        )
        params = stamp_parameter_units(xr.Dataset.from_dataframe(df))
        store = staging / "sensitivity_datatree.zarr"
        store.parent.mkdir(parents=True)
        _quiet_write(lambda: xr.DataTree.from_dict({"/": xr.Dataset(), "parameters": params}).to_zarr(str(store)))
        return staging

    planted = tree(PLANTED)
    assert not list(planted.rglob("*.csv")), "the setup CSV is excluded from the staging tree"
    assert "sensitivity_datatree.zarr/parameters/weather_dir (decoded values)" in _refusal(planted, tmp_path / "p")
    _emit(tree("benign"), tmp_path / "b")


# --------------------------------------------------------------------------------------
# Test 4 and 5: containers
# --------------------------------------------------------------------------------------


def _zip_bytes(entries: dict[str, bytes], compress: int = zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compress) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _tar_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


#: (container, builder of its bytes around a payload, raw presence of the planted path). ABSENT arms prove
#: decompression; the tar arm is a control that the entry is read (a tar stores its entries uncompressed).
_CONTAINERS = [
    ("zip", lambda p: _zip_bytes({"inner/notes.txt": p}), False),
    ("gzip", lambda p: gzip.compress(p, mtime=0), False),
    ("bzip2", lambda p: bz2.compress(p), False),
    ("xz", lambda p: lzma.compress(p), False),
    ("tar", lambda p: _tar_bytes({"inner/notes.txt": p}), True),
    ("zip_in_zip", lambda p: _zip_bytes({"inner.zip": _zip_bytes({"deep/notes.txt": p})}), False),
]


@pytest.mark.parametrize(("name", "build", "raw_present"), _CONTAINERS, ids=[c[0] for c in _CONTAINERS])
def test_a_planted_path_inside_a_container_member_refuses_the_emit(tmp_path, name, build, raw_present):
    """Property: a per-user path inside a container a declared input is (zip, gzip, bzip2, xz, tar, a zip
        inside a zip) refuses the emit, naming the member; the same container around benign content passes.

    Class: REGRESSION - red on the parent, which carries the member unscanned (a deflated entry is not
        readable from the container's bytes).
    Reachability: a declared input can be any file; a bundle's own baseline used to carry a zip.
    Both-states anchor: each arm measures, in the test, whether the planted bytes occur in the container's
        raw bytes and asserts the declared expectation (ABSENT for zip, gzip, bzip2, xz and the nested zip,
        which prove DECOMPRESSION; PRESENT for tar, a control); a benign container passes.
    Invocation isolated: ``_emit_bundle_zip`` over a fresh staging tree per case.
    What kills it: a raw-block scan of container bytes (a false negative in every ABSENT arm).
    Second correct implementation: extracting the container to a temporary directory and scanning the tree.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    member = staging / f"input.{name}"
    member.write_bytes(build(_payload()))
    assert (PLANTED_BYTES in member.read_bytes()) is raw_present, "unit: raw presence of the planted bytes"
    message = _refusal(staging, tmp_path)
    assert f"input.{name}" in message and "pattern_rules" in message

    member.write_bytes(build(_filler(2000).encode()))
    _emit(staging, tmp_path)


def test_a_file_that_begins_like_a_zip_but_is_not_one_refuses_naming_the_member(tmp_path):
    """Property: a member that begins ``PK`` and the zip local-header signature but is not a valid zip refuses
        the emit, naming the member; it is never scanned as its raw bytes.

    Class: REGRESSION - the parent carries it silently.
    Reachability: a truncated or corrupt zip declared as an input.
    Both-states anchor: a valid zip of the same name passes (previous test); this one is corrupt.
    Invocation isolated: ``_emit_bundle_zip`` over one staging tree.
    What kills it: falling back to a raw scan when the container does not open.
    Second correct implementation: probing with ``zipfile.is_zipfile`` and refusing when it is False.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "broken.zip").write_bytes(b"PK\x03\x04\x00" + b"not a zip archive " * 20)
    out = tmp_path / "refused.zip"
    with pytest.raises(ProcessingError) as info:
        _emit_bundle_zip(staging, out, scanner=_scanner())
    assert "broken.zip" in str(info.value) and "zip" in str(info.value)


# --------------------------------------------------------------------------------------
# The opaque-binary rule and the zarr-chunk exclusion
# --------------------------------------------------------------------------------------


def test_an_opaque_binary_member_is_scanned_in_blocks_across_a_block_boundary(tmp_path, monkeypatch):
    """Property: a planted path in an opaque binary member (no container magic) is found wherever it sits,
        including straddling a block boundary; a member of the same size with no path passes.

    Class: REGRESSION - red on the parent.
    Reachability: any declared input that is neither text nor a container.
    Both-states anchor: the path is placed across the boundary of two blocks (block size shrunk for the test).
    Invocation isolated: ``_emit_bundle_zip`` over one staging tree; the block size is monkeypatched.
    What kills it: blocks that do not overlap; scanning only the first block.
    Second correct implementation: memory-mapping the member and scanning it whole.
    """
    monkeypatch.setattr(G, "_BLOCK_BYTES", 4096)
    staging = tmp_path / "staging"
    staging.mkdir()
    filler = b"\x00" + bytes(random.Random(3).choices(b"abcdefghijklmnopqrstuvwxyz", k=4096 * 3))
    offset = 4096 - len(PLANTED_BYTES) // 2  # the path straddles the end of the first block
    straddling = filler[:offset] + PLANTED_BYTES + filler[offset + len(PLANTED_BYTES) :]
    (staging / "opaque.bin").write_bytes(straddling)
    assert "opaque.bin" in _refusal(staging, tmp_path)
    (staging / "opaque.bin").write_bytes(filler)
    _emit(staging, tmp_path)


def test_the_chunks_of_a_numeric_array_in_a_store_not_named_zarr_are_not_raw_scanned(tmp_path, monkeypatch):
    """Property: the chunk files of a zarr array that sits in a directory not named ``*.zarr`` are not
        raw-scanned (the array is classified from its metadata), so a numeric chunk whose compressed bytes
        happen to spell a pattern does not refuse the emit.

    Class: REGRESSION guard for the false-positive mechanism the shape measured on compressed bytes.
    Reachability: a store without the ``.zarr`` suffix is not classed a chunk by name.
    Both-states anchor: the same bytes planted in a plain binary member refuse; in a numeric array chunk they do not.
    Invocation isolated: ``_emit_bundle_zip`` over one staging tree.
    What kills it: raw-scanning every NUL-prefixed member including chunks of known numeric arrays.
    Second correct implementation: naming the store ``*.zarr``, which the by-name rule already skips.
    """
    staging = tmp_path / "staging"
    array_dir = staging / "store" / "nums"
    values = np.frombuffer((b"\x00" + PLANTED_BYTES + b"\x00" * 3).ljust(64, b"\x01"), dtype=np.uint8)
    _quiet_write(
        lambda: zarr.create_array(store=str(array_dir), data=values, chunks=(64,), compressors=None, zarr_format=3)
    )
    assert PLANTED_BYTES in _chunk_bytes(array_dir), "the planted bytes must be raw in the numeric chunk"
    _emit(staging, tmp_path)
    (staging / "plain.bin").write_bytes(b"\x00" + PLANTED_BYTES)
    (tmp_path / "second").mkdir()
    assert "plain.bin" in _refusal(staging, tmp_path / "second")


# --------------------------------------------------------------------------------------
# Test 6 (F5-1): a store carried shallowly is not read
# --------------------------------------------------------------------------------------


def test_a_store_carried_shallowly_is_not_decoded_and_the_same_store_with_its_chunk_is(tmp_path):
    """Property: an array whose metadata document is among the archive entries but none of whose chunk files is
        (a store carried shallowly) passes although its SOURCE chunk holds a planted path; the same store with
        its chunk file carried refuses.

    Class: REGRESSION for the F5-1 ruling - the emit must not be refused for a value it does not ship.
    Reachability: ``metadata_only_stores`` and ``--omit-output-zarrs`` carry stores this way.
    Both-states anchor: one source store, two entry sets that differ only by the chunk file.
    Invocation isolated: ``_emit_bundle_zip`` with ``streamed`` entries pointing into a source tree.
    What kills it: decoding from the source directory whenever a metadata document is present.
    Second correct implementation: decoding from the carried bytes only (reading the chunk from the archive).
    """
    src = _text_array(tmp_path / "src" / "s.zarr" / "arr", [PLANTED, "cpu"], "U60")
    chunk = next(f for f in sorted(src.rglob("*")) if f.is_file() and f.name != "zarr.json")
    staging = tmp_path / "staging"
    staging.mkdir()
    shallow = {"s.zarr/arr/zarr.json": src / "zarr.json"}
    scanner = _scanner()
    _emit_bundle_zip(staging, tmp_path / "ok.zip", streamed=shallow, scanner=scanner)
    assert scanner.hits == []

    full = {**shallow, "s.zarr/arr/" + chunk.relative_to(src).as_posix(): chunk}
    with pytest.raises(ProcessingError) as info:
        _emit_bundle_zip(staging, tmp_path / "bad.zip", streamed=full, scanner=_scanner())
    assert "s.zarr/arr (decoded values)" in str(info.value)


# --------------------------------------------------------------------------------------
# Test 7 (NEST-1): the nesting bound
# --------------------------------------------------------------------------------------


def _nested_zip(depth: int, payload: bytes) -> bytes:
    """``depth`` zips, each holding the next; the innermost holds ``payload`` as one entry."""
    data = _zip_bytes({"leaf.txt": payload})
    for level in range(depth - 1):
        data = _zip_bytes({f"z{level}.zip": data})
    return data


def test_a_chain_of_eight_containers_is_read_and_a_ninth_refuses(tmp_path):
    """Property: a chain of 8 nested zips is read to its content (a planted path refuses on its pattern class);
        a chain of 9 refuses with the nesting message naming the member, planted or benign.

    Class: REGRESSION for the NEST-1 ruling (a container with 8 or more containers around it refuses).
    Reachability: a declared input can be an archive of archives.
    Both-states anchor: 8 with the path (refuses on the content), 9 with the path and 9 with benign bytes (both
        refuse on the bound - the benign arm shows the refusal is the bound and not the content).
    Invocation isolated: ``_emit_bundle_zip`` over one staging tree per arm.
    What kills it: a bound off by one in either direction; a bound that fires only on planted content.
    Second correct implementation: counting depth from the innermost container outward.
    """
    for depth, payload, expect in (
        (8, _payload(), "pattern_rules"),
        (9, _payload(), "container with 8 or more containers around it"),
        (9, _filler(300).encode(), "container with 8 or more containers around it"),
    ):
        staging = tmp_path / f"d{depth}_{len(payload)}" / "staging"
        staging.mkdir(parents=True)
        (staging / "chain.zip").write_bytes(_nested_zip(depth, payload))
        message = _refusal(staging, staging.parent)
        assert "chain.zip" in message and expect in message, (depth, message[:200])
    benign = tmp_path / "ok" / "staging"
    benign.mkdir(parents=True)
    (benign / "chain.zip").write_bytes(_nested_zip(8, _filler(300).encode()))
    _emit(benign, benign.parent)


# --------------------------------------------------------------------------------------
# Test 8 (F4-1): damaged compression magic is raw-scanned
# --------------------------------------------------------------------------------------


def test_damaged_compression_magic_is_an_opaque_binary_and_intact_magic_is_a_container(tmp_path):
    """Property: a member beginning ``\\x1f\\x8b\\x07`` or ``BZh9`` without the block magic is raw-scanned as an
        opaque binary - benign bytes pass, a planted path refuses on the raw scan; the intact signatures
        (``\\x1f\\x8b\\x08``, ``BZh91AY&SY``) followed by junk refuse as containers that fail to open.

    Class: REGRESSION for the F4-1 ruling.
    Reachability: any opaque input that happens to begin with a compression magic.
    Both-states anchor: the damaged and the intact signature, each with benign and with junk bodies.
    Invocation isolated: ``_emit_bundle_zip`` over one staging tree per arm.
    What kills it: matching two-byte or three-byte magic alone; skipping a damaged-magic member unscanned.
    Second correct implementation: probing with ``gzip``/``bz2`` and falling back to a raw scan on failure.
    """
    benign_body = b"\x00" + _filler(300).encode()
    planted_body = b"\x00" + _filler(100).encode() + PLANTED_BYTES + _filler(100).encode()
    for label, magic in (("gz07", b"\x1f\x8b\x07"), ("bz_no_block", b"BZh9")):
        ok = tmp_path / label / "ok" / "staging"
        ok.mkdir(parents=True)
        (ok / "m.bin").write_bytes(magic + benign_body)
        _emit(ok, ok.parent)
        bad = tmp_path / label / "bad" / "staging"
        bad.mkdir(parents=True)
        (bad / "m.bin").write_bytes(magic + planted_body)
        message = _refusal(bad, bad.parent)
        assert "m.bin" in message and "pattern_rules" in message and "begins like" not in message
    for label, magic, kind in (("gz08", b"\x1f\x8b\x08", "gzip"), ("bz_block", b"BZh91AY&SY", "bzip2")):
        staging = tmp_path / label / "staging"
        staging.mkdir(parents=True)
        (staging / "m.bin").write_bytes(magic + benign_body)
        message = _refusal(staging, staging.parent)
        assert "m.bin" in message and f"begins like a {kind} container" in message


# --------------------------------------------------------------------------------------
# Test 9 (F8-1): byte-order-mark encodings are decoded before they are scanned
# --------------------------------------------------------------------------------------

_BOM_TEXT = "x " + PLANTED + " y"
_BENIGN_TEXT = "x " + "cpu gpu" + " y"


def _bom_bytes(encoding: str, text: str) -> bytes:
    import codecs

    marks = {
        "utf32le": (codecs.BOM_UTF32_LE, "utf-32-le"),
        "utf32be": (codecs.BOM_UTF32_BE, "utf-32-be"),
        "utf16le": (codecs.BOM_UTF16_LE, "utf-16-le"),
        "utf16be": (codecs.BOM_UTF16_BE, "utf-16-be"),
    }
    mark, codec = marks[encoding]
    return mark + text.encode(codec)


@pytest.mark.parametrize("encoding", ["utf32le", "utf32be", "utf16le", "utf16be"])
def test_a_planted_path_in_a_bom_encoded_input_refuses_and_benign_text_passes(tmp_path, encoding):
    """Property: a declared input in UTF-32 LE/BE or UTF-16 LE/BE (with its byte-order mark) holding a planted
        path refuses the emit, naming the member; the same encoding with benign text passes.

    Class: REGRESSION - red on the parent, where the NULs interleave every pattern.
    Reachability: Windows-origin text files and some tools write UTF-16 with a mark.
    Both-states anchor: each arm asserts the planted path's UTF-8 bytes ABSENT from the raw bytes (so the arm
        proves decoding), and the UTF-32 LE arm also shows the row order: decoded as UTF-16 its text does not
        contain the path, so only the UTF-32 row finds it.
    Invocation isolated: ``_emit_bundle_zip`` over one staging tree per case.
    What kills it: testing the UTF-16 mark before the UTF-32 marks; raw-scanning the bytes.
    Second correct implementation: trying every codec and scanning the first that decodes.
    """
    data = _bom_bytes(encoding, _BOM_TEXT)
    assert PLANTED_BYTES not in data, "unit: raw presence of the planted bytes"
    if encoding == "utf32le":
        assert PLANTED not in data.decode("utf-16", errors="replace"), "read as UTF-16 the text interleaves NULs"
    staging = tmp_path / "bad" / "staging"
    staging.mkdir(parents=True)
    (staging / "input.txt").write_bytes(data)
    message = _refusal(staging, staging.parent)
    assert "input.txt" in message and "pattern_rules" in message
    ok = tmp_path / "ok" / "staging"
    ok.mkdir(parents=True)
    (ok / "input.txt").write_bytes(_bom_bytes(encoding, _BENIGN_TEXT))
    _emit(ok, ok.parent)


def test_utf16_without_a_byte_order_mark_is_the_named_residual_and_passes(tmp_path):
    """Property: UTF-16 LE WITHOUT a byte-order mark holding a planted path passes the check (a false negative).

    Class: PIN of a named residual - ``--exclude-config`` cannot correct it, and the shape claims no cover. A
        change to this test is a decision (the shape's F8-1 residual), not an accident.
    Reachability: any BOM-less UTF-16 input.
    Both-states anchor: the same text with the mark refuses (previous test).
    Invocation isolated: ``_emit_bundle_zip`` over one staging tree.
    What kills it: nothing it should be killed by; a future BOM-less detector would turn it red on purpose.
    Second correct implementation: none intended.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "input.txt").write_bytes(_BOM_TEXT.encode("utf-16-le"))
    assert b"\x00" in (staging / "input.txt").read_bytes()[:8192]
    _emit(staging, tmp_path)


def test_a_member_that_merely_begins_with_a_mark_is_raw_scanned_and_does_not_refuse(tmp_path):
    """Property: a member of ``\\xff\\xfe`` and 4 KB of arbitrary bytes (which fails a strict UTF-16 decode) passes;
        the same member with a planted path appended in UTF-8 refuses on the raw scan.

    Class: REGRESSION for the S10 policy (a decode failure is an opaque binary, not a refusal).
    Reachability: any opaque binary beginning with a mark.
    Both-states anchor: the appended path shows the member was raw-scanned, not skipped.
    Invocation isolated: ``_emit_bundle_zip`` over one staging tree per arm.
    What kills it: refusing on a decode failure; skipping an undecodable member unscanned.
    Second correct implementation: decoding with ``errors="replace"`` and scanning the result.
    """
    body = bytes(random.Random(11).choices(range(256), k=4096))
    member = b"\xff\xfe" + body
    with pytest.raises(UnicodeDecodeError):
        member.decode("utf-16")
    ok = tmp_path / "ok" / "staging"
    ok.mkdir(parents=True)
    (ok / "m.bin").write_bytes(member)
    _emit(ok, ok.parent)
    bad = tmp_path / "bad" / "staging"
    bad.mkdir(parents=True)
    (bad / "m.bin").write_bytes(member + b" " + PLANTED_BYTES + b" ")
    message = _refusal(bad, bad.parent)
    assert "m.bin" in message and "pattern_rules" in message


# --------------------------------------------------------------------------------------
# Test 9, NUL-free-prefix arm (P2-2): the mark is tested before the NUL sniff
# --------------------------------------------------------------------------------------

_CJK = "あ" * 6000


def _marked_member(encoding: str, text: str) -> bytes:
    return _bom_bytes(encoding, text)


@pytest.mark.parametrize("encoding", ["utf16le", "utf16be"])
def test_a_marked_member_whose_first_block_holds_no_nul_byte_is_still_decoded(tmp_path, encoding):
    """Property: a marked UTF-16 member of the mark, 6000 CJK characters (U+3042), a space and a planted path -
        whose first 8192 bytes hold NO NUL byte, so the NUL sniff classes it text - refuses the emit; the controls
        that vary the position of the first NUL also refuse (the path FIRST; ASCII filler before the path); the
        same CJK member with no planted path passes.

    Class: REGRESSION for the P2-2 ruling - red on the tree before the mark test, where the member is read as
        UTF-8 text and its decoded-nothing bytes pass.
    Reachability: any UTF-16 file of non-Latin text longer than a block puts its first NUL beyond 8192 bytes.
    Both-states anchor: each arm asserts, in the test, whether its first 8192 bytes hold a NUL byte (none for the
        CJK-first arms, one for the two controls), so the arm proves which side of the sniff it is on.
    Invocation isolated: ``_emit_bundle_zip`` over a fresh staging tree per arm.
    What kills it: testing the mark after the NUL sniff; reading a marked member as UTF-8 text or rewriting it.
    Second correct implementation: routing every member to the binary path when its first bytes are a mark,
        by a prefix compare in the classifier rather than a predicate.
    """
    arms = {
        "cjk_first": (_CJK + " " + PLANTED, False, True),
        "path_first": (PLANTED + " " + _CJK, True, True),
        "ascii_filler": ("some ascii filler words " + PLANTED + " " + _CJK, True, True),
        "no_path": (_CJK, False, False),
    }
    for label, (text, nul_in_first_block, refuses) in arms.items():
        data = _marked_member(encoding, text)
        assert (b"\x00" in data[:8192]) is nul_in_first_block, f"unit: NUL in first 8192 bytes, arm {label}"
        assert PLANTED_BYTES not in data
        staging = tmp_path / label / "staging"
        staging.mkdir(parents=True)
        (staging / "input.txt").write_bytes(data)
        if refuses:
            message = _refusal(staging, staging.parent)
            assert "input.txt" in message and "pattern_rules" in message, label
        else:
            _emit(staging, staging.parent)


# --------------------------------------------------------------------------------------
# A byte that is not text breaks adjacency: unrelated text on either side must not glue into a match
# --------------------------------------------------------------------------------------

_BAD = b"\xff\xfe\x80"  # bytes that are never valid UTF-8
# Fake producer address, built from fragments (this file is scanned by the repository's anonymization guard).
_ADDRESS = ("someone" + "@" + "example" + "." + "invalid").encode()


_GLUED = "abc" + "@" + "example" + ".com"  # what the pieces form when the invalid bytes are dropped


def _glued_binary() -> bytes:
    """Binary bytes whose text pieces form an address ONLY when the invalid bytes between them are dropped."""
    return (
        b"\x00\x01"
        + _filler(300, 3).encode()
        + b" abc"
        + _BAD
        + b"@"
        + _BAD
        + b"example"
        + _BAD
        + b".com "
        + _filler(300, 4).encode()
        + b"\x00"
    )


def _old_decode_scanner(monkeypatch) -> G.ZeroUserInfoScanner:
    """A scanner whose member decode is the one that dropped invalid bytes (errors="ignore")."""
    scanner = _scanner()

    def old_scan_member(member: str, data: bytes):
        text = data.decode("utf-8", errors="ignore")
        found = []
        for match in G.EMAIL_PATTERN.finditer(text):
            if match.group(0).lower() not in scanner._exempt_emails:
                found.append(G.MemberHit(member, G.LAYER_PATTERNS, "email-address pattern (value not shown)"))
                break
        return found

    monkeypatch.setattr(scanner, "scan_member", old_scan_member)
    return scanner


def test_invalid_bytes_do_not_glue_unrelated_text_into_an_address(monkeypatch):
    """Property: a binary member whose invalid bytes sit between text pieces that would form an address if the
    bytes were dropped produces NO hit, scanned through `scan_binary_member` as the emit does.

    Class: synthetic binary bytes, no real data. Must-fire control: the same bytes through the decode that drops
    invalid bytes (errors="ignore") DO produce an email hit, and the glued string exists only after dropping.
    Kills it: decoding with errors="ignore". A second correct implementation splits the text at each invalid
    byte instead of replacing it.
    """
    data = _glued_binary()
    assert data.decode("utf-8", errors="ignore").count(_GLUED) == 1, "control: the pieces glue when dropped"
    assert _GLUED.encode() not in data
    clean = _scanner()
    G.scan_binary_member(clean, "x.bin", io.BytesIO(data))
    assert clean.hits == []
    old = _old_decode_scanner(monkeypatch)
    G.scan_binary_member(old, "x.bin", io.BytesIO(data))
    assert old.hits, "control: the decode that drops invalid bytes reports the glued string"


@pytest.mark.parametrize("pad", [b"", _BAD])
def test_a_real_address_in_a_binary_member_between_invalid_bytes_is_still_refused(pad):
    """Property: a real address-shaped value, a per-user path and a configured value each embedded in a binary
    member between invalid bytes are still found.

    Class: the three layers through the same record path. Must-fire control: the `pad` parameter empty and non-empty;
    both fire. Kills it: a replace decode that also hides real values.
    """
    for value, layer in ((_ADDRESS, G.LAYER_PATTERNS), (PLANTED_BYTES, G.LAYER_PATTERNS)):
        scanner = _scanner()
        data = b"\x00" + _BAD + pad + value + pad + _BAD + b"\x00"
        G.scan_binary_member(scanner, "x.bin", io.BytesIO(data))
        assert [h.layer for h in scanner.hits] == [layer], value
    configured = b"fake-configured-value-0123456789"
    scanner = _scanner()
    scanner._config_checks.append(("configured path", len(configured), lambda t: configured.decode() in t))
    G.scan_binary_member(scanner, "x.bin", io.BytesIO(b"\x00" + _BAD + pad + configured + pad + _BAD + b"\x00"))
    assert [h.layer for h in scanner.hits] == [G.LAYER_CONFIG_VALUES]


def test_a_text_member_with_a_real_address_is_still_refused():
    """Property: a valid UTF-8 text member holding an address is refused as before (decode unchanged for text)."""
    scanner = _scanner()
    hits = scanner.record("note.txt", b"contact " + _ADDRESS + b" today")
    assert [h.layer for h in hits] == [G.LAYER_PATTERNS]
