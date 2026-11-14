"""Zero-user-info emit-time gate for reprex bundles.

Scans the members of a bundle for producer-specific values with three layers, so a bundle
that names the producer is refused instead of published. A text member is scanned as it is; a zarr
array of text values is DECODED and its values scanned; any other binary member is scanned through
its decompressor when the standard library has one, and in overlapping blocks otherwise.

Layer (a) is two generic pattern classes (an email address, and an absolute path under a per-user
root) held in this file. Layer (b) is built per emit from the emitted analysis's own configs and the
running login name. Layer (c) is an
optional private list of specific identifiers; it is reached only through the environment variable
``HHEMT_REPREX_BLOCKLIST`` and through no path inside the source tree, so the toolkit repository
and its wheel carry no producer-specific value. A variable that is set to something unusable
refuses the emit; a variable that is not configured leaves layers (a) and (b) running and is
recorded as such in the bundle manifest.

Independence invariant (load-bearing): the private list is NOT derived from the scrub or the
placeholder table, else it weakens with each change to them.
"""

from __future__ import annotations

import bz2
import codecs
import getpass
import gzip
import json
import logging
import lzma
import os
import re
import tarfile
import warnings
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from email.utils import getaddresses
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from hhemt.exceptions import ProcessingError

logger = logging.getLogger(__name__)

# Text detection is a CONTENT sniff, never a suffix guess. A suffix allowlist reports
# "clean in N extensions" while reading as "clean". Measured on a real Norfolk emit: a
# NUL-byte sniff classifies 4,691 of 14,528 members as text; the prior eight-suffix
# allowlist reached roughly 3,300. The ~1,400-member gap included `Snakefile.source`
# (extensionless, and a TRUE positive carrying an absolute producer path) and all 438
# carried `.py` files under hhemt_src/.
_SNIFF_BYTES = 8192
_MIN_EXPECTED_TOKENS = 4
_ABSOLUTE_MIN_TOKENS = 1
# A carrier may declare its own expected count in a `# min-tokens: N` header line, so the
# floor travels WITH the list it describes: the list lives in the producer's private estate,
# outside this repository, and a floor held in this source tree would drift from it.
_MIN_TOKENS_HEADER = re.compile(r"^#\s*min-tokens:\s*(\d+)\s*$")
_ENV_OVERRIDE = "HHEMT_REPREX_BLOCKLIST"


def _refuse_carrier(path: Path | None, reason: str) -> ProcessingError:
    """The ``ProcessingError`` for an unusable carrier, never carrying a token.

    The reason text names the variable and the path, because a caller that returns only ``reason``
    (the consume-side scan does) must still name both.
    """
    where = f" ({path})" if path is not None else ""
    return ProcessingError(
        operation="reprex zero-user-info gate",
        filepath=path,
        reason=(f"${_ENV_OVERRIDE}{where} {reason}. The gate cannot certify a bundle against a carrier it cannot use."),
    )


def _resolve_blocklist_path() -> Path | None:
    """The carrier file named by ``$HHEMT_REPREX_BLOCKLIST``, or None when the variable is not configured.

    Unset and empty both mean not configured. A configured value is the carrier and the only one:
    it is never skipped in favour of another location, and a value that cannot be used raises
    ``ProcessingError`` instead of returning None, because None would read as "no list" and let a
    mistyped path pass as a clean bundle. A leading ``~`` is expanded and the result must be
    absolute, so the carrier never depends on the working directory; a symlink to a regular file
    and a trailing slash on one are both usable.

    There is deliberately NO source-checkout tier and NO packaged-data tier, and neither may be
    added. A checked-in list puts private values in a public repository, and a packaged copy would
    ship them inside the wheel (the carrier class that already put one on PyPI permanently) or,
    if it held synthetic or hashed values, make this gate report a pass it never performed. The
    real list lives in the producer's private estate and is pointed at by the variable.
    """
    raw = os.environ.get(_ENV_OVERRIDE)
    if not raw:
        return None
    try:
        path = Path(raw).expanduser()
    except RuntimeError as exc:
        raise _refuse_carrier(Path(raw), "names a home directory that cannot be resolved") from exc
    if not path.is_absolute():
        raise _refuse_carrier(path, "is not an absolute path")
    try:
        is_regular_file = path.is_file()
        is_directory = path.is_dir()
    except OSError as exc:
        raise _refuse_carrier(path, f"cannot be examined ({exc.strerror or type(exc).__name__})") from exc
    if is_regular_file:
        return path
    if is_directory:
        raise _refuse_carrier(path, "names a directory, not a file")
    raise _refuse_carrier(path, "does not name an existing regular file")


def _load_blocklist() -> list[str]:
    """The carrier's entries, or a diagnosed ``ProcessingError``.

    ``_scan_zero_user_info`` wraps this in ``except ProcessingError``, so a bare ``OSError`` or
    ``UnicodeDecodeError`` would escape and turn a consume-side informational scan into a crash.
    Every way the carrier can be unusable is therefore a ``ProcessingError`` naming the variable
    and the path, with no entry in the message: no variable configured, unreadable, not UTF-8,
    or fewer distinct entries than the floor. The file is read as ``utf-8-sig`` so a leading
    byte-order mark does not make the first entry unmatchable. The floor counts DISTINCT entries
    compared case-insensitively, because the patterns match case-insensitively: a file of
    duplicates or of case variants of one token would otherwise meet the floor with one real
    entry.
    """
    path = _resolve_blocklist_path()
    if path is None:
        raise ProcessingError(
            operation="reprex zero-user-info gate",
            filepath=None,
            reason=f"${_ENV_OVERRIDE} is not configured, so there is no private list to load.",
        )
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise _refuse_carrier(path, "names a file that is not valid UTF-8") from exc
    except OSError as exc:
        raise _refuse_carrier(path, f"names a file that cannot be read ({exc.strerror or type(exc).__name__})") from exc
    tokens: list[str] = []
    declared: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            m = _MIN_TOKENS_HEADER.match(line)
            if m:
                declared = int(m.group(1))
            continue
        if line:
            tokens.append(line)
    floor = max(declared, _ABSOLUTE_MIN_TOKENS) if declared is not None else _MIN_EXPECTED_TOKENS
    distinct = len({t.casefold() for t in tokens})
    if distinct < floor:
        raise _refuse_carrier(
            path,
            f"yielded {distinct} distinct entry(ies) (compared without case), below the floor of {floor}",
        )
    return tokens


def _is_text(path: Path) -> bool:
    """Content sniff: a file is text iff its first _SNIFF_BYTES carry no NUL byte.

    Replaces the retired _TEXT_SUFFIXES allowlist. An unreadable file is treated as
    non-text rather than raising -- the gate must not crash on a permission or race
    error in a tree it is only auditing.
    """
    try:
        with path.open("rb") as fh:
            return b"\x00" not in fh.read(_SNIFF_BYTES)
    except OSError:
        return False


#: The zarr metadata documents (v3 ``zarr.json``; v2 ``.zarray``, ``.zgroup``, ``.zattrs``, ``.zmetadata``).
_ZARR_METADATA_NAMES = frozenset({"zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata"})


def assert_bundle_zero_user_info(bundle_root: Path) -> None:
    """Raise ProcessingError if layer (a) or the private list matches any member under ``bundle_root``.

    The consume-side form of the emit-time check: a consumer holds no producer configs, so layer (b)
    has nothing to read here and only layers (a) and (c) run. The members are read as the emit reads them: a
    zarr array of text values is decoded (only when at least one of its chunk files is under ``bundle_root``),
    a member that begins with a UTF-16 or UTF-32 byte-order mark, or that the NUL sniff classes binary, goes to
    ``scan_binary_member`` (containers through their decompressors, marked text decoded, anything else in
    overlapping blocks), and every other member is scanned as text. The mark is tested BEFORE the NUL sniff, so
    a marked member that is not a zarr chunk file is never read as UTF-8 text. A zarr chunk file is not tested
    for a mark: it is skipped when the NUL sniff classes it binary and scanned as text otherwise; it is never
    raw-scanned.

    A member the scan cannot read (a damaged or over-nested container, an undecodable array) does not discard
    the verdicts given for other members: its refusal is carried beside the recorded hits and the single
    ``ProcessingError`` names both.
    """
    scanner = ZeroUserInfoScanner.for_consume()
    files = [f for f in sorted(bundle_root.rglob("*")) if f.is_file()]
    array_dirs: dict[Path, str] = {}
    refusals: list[ProcessingError] = []
    for f in files:
        if f.name in ("zarr.json", ".zarray"):
            try:
                kind = classify_zarr_array(f.parent)
            except ProcessingError as exc:
                refusals.append(exc)
                continue
            if kind is not None:
                array_dirs[f.parent] = kind
    carried: set[Path] = set()
    for f in files:
        if f.name in _ZARR_METADATA_NAMES:
            continue
        for parent in f.parents:
            if parent in array_dirs:
                carried.add(parent)
                break
            if parent == bundle_root:
                break
    for directory in sorted(carried):
        if array_dirs[directory] == ZARR_DECODE:
            member = directory.relative_to(bundle_root).as_posix()
            try:
                scanner.record(f"{member} (decoded values)", decode_zarr_array_text(directory, consumer_member=member))
            except ProcessingError as exc:
                refusals.append(exc)
    for f in files:
        is_chunk = f.name not in _ZARR_METADATA_NAMES and (
            any(parent in array_dirs for parent in f.parents)
            or any(part.endswith(".zarr") for part in f.relative_to(bundle_root).parts[:-1])
        )
        member = f.relative_to(bundle_root).as_posix()
        try:
            with f.open("rb") as fh:
                head = fh.read(_SNIFF_BYTES)
                if is_chunk:
                    # C1-2: a binary chunk is never scanned (its compressed bytes are the false-positive class
                    # the skip exists for; a text array is decoded above); a NUL-free chunk falls through to the
                    # text scan below exactly as the pre-change loop scanned it, with no mark test.
                    if b"\x00" in head:
                        continue
                elif has_byte_order_mark(head) or b"\x00" in head:
                    try:
                        scan_binary_member(scanner, member, fh, consumer=True)
                    except ProcessingError as exc:
                        refusals.append(exc)
                    continue
        except OSError:
            continue  # unreadable: treated as non-text and skipped, as `_is_text` treats it
        scanner.record(member, f.read_bytes())
    if refusals:
        reasons = [exc.reason for exc in refusals]
        if scanner.hits:
            try:
                scanner.raise_if_hits(bundle_root, "reprex zero-user-info gate")
            except ProcessingError as hits:
                reasons.insert(0, hits.reason)
        raise ProcessingError(operation="reprex zero-user-info gate", filepath=bundle_root, reason=" | ".join(reasons))
    scanner.raise_if_hits(bundle_root, "reprex zero-user-info gate")


# ---------------------------------------------------------------------------------------------
# The member scanner: the three layers over the bytes of one archive member.
# ---------------------------------------------------------------------------------------------

LAYER_PATTERNS = "pattern_rules"
LAYER_CONFIG_VALUES = "config_values"
LAYER_PRIVATE_LIST = "private_list"

PRIVATE_LIST_CHECKED = "checked"
PRIVATE_LIST_NOT_CONFIGURED = "not configured"
PRIVATE_LIST_SET_BUT_EMPTY = "set but empty"

#: A configured value shorter than this is not searched for: a three-letter account or login name
#: would match unrelated words. Each skipped value is logged by length, never by content.
_MIN_VALUE_CHARS = 4

# Layer (a) is built from fragments so this file does not itself contain a literal the path class
# fires on: the toolkit source travels inside bundles and is scanned like any other text member.
_BS = re.escape(chr(92))  # one literal backslash, as it must appear inside a pattern
_UNIX_ROOTS = (
    "home",
    "Users",
    "scratch",
    "sfs/weka/scratch",
    "project",
    "ccs/home",
    "ccs/proj",
    "lustre/orion",
    "gpfs",
)
# The directory after a per-user root is any single path segment that does not begin with a
# brace: the toolkit's own template field for an allocation name sits in that position and names
# no user. The repository guard reads the same position as any segment and covers the template
# with rows of its exemption table, so the two definitions agree on the Windows alternative below
# and differ here by design.
_UNIX_PATH_FRAGMENT = "(?:" + "|".join("/" + r + "/" for r in _UNIX_ROOTS) + ")[^/\\s{][^/\\s]*/"
_SEP = "[" + _BS + "/]{1,2}"
#: The Windows alternative: case-insensitive (drive letter and directory name vary in case), read
#: with a backslash, a forward slash, or the doubled backslash a JSON string escapes it to. Equal,
#: character for character, to the repository guard's fragment.
WINDOWS_PATH_FRAGMENT = "(?i:C:" + _SEP + "Users" + _SEP + "[^" + _BS + "/\\s]+" + _SEP + ")"
PATH_PATTERN = re.compile(_UNIX_PATH_FRAGMENT + "|" + WINDOWS_PATH_FRAGMENT)

# The local part is matched POSSESSIVELY and only from the start of a run of local-part characters
# (the look-behind), so the scan is linear in the member length: the non-possessive form
# backtracks quadratically over a long alphanumeric run such as base64 in an html page. Equal to
# the repository guard's pattern string. The local part and the domain labels are read with Unicode
# `\w` and the top-level label with `[^\W\d_]` (Unicode letters only, two or more), so an address
# written in another script is read whole rather than missed or matched by its ASCII tail.
EMAIL_PATTERN = re.compile(r"(?<![\w.%+-])[\w.%+-]++@[\w.-]+\.[^\W\d_]{2,}")

# Third-party addresses that Snakemake's report inlines into the baseline html page (license
# headers of vendored JavaScript), measured as nine distinct addresses. They are exempt BY VALUE
# and the list is closed: an address that is not on it, or not the toolkit's own published author
# address, refuses the emit, so a new vendored library that adds a header address is a loud
# refusal with a one-line fix. Stored as (local part, domain) pairs and joined at import so no
# address appears as a literal in this file.
_THIRD_PARTY_ADDRESS_PARTS = (
    ("ariya.hidayat", "gmail.com"),
    ("utatane.tea", "gmail.com"),
    ("thaddee.tyl", "gmail.com"),
    ("mathias", "qiwi.be"),
    ("paul", "vorba.ch"),
    ("me", "tomalexander.co.nz"),
    ("kris.kowal", "cixar.com"),
    ("joost-wim", "boekesteijn.nl"),
    ("arpad.borsos", "googlemail.com"),
)
THIRD_PARTY_ADDRESSES: frozenset[str] = frozenset("@".join(parts).lower() for parts in _THIRD_PARTY_ADDRESS_PARTS)

# The segment directly under a per-user root, taken from a configured path so the user directory is
# searched for on its own as well as inside the full path.
_SEGMENT_AFTER_ROOT = re.compile(
    "(?:" + "|".join("/" + r + "/" for r in _UNIX_ROOTS) + "|(?i:C:" + _SEP + "Users" + _SEP + "))"
    "([^" + _BS + "/\\s{][^" + _BS + "/\\s]*)"
)
_DRIVE_PREFIX = re.compile(r"[A-Za-z]:[\\/]")
_STANDS_ALONE_AS_SEGMENT = r"(?<=[/\\])%s(?=[/\\]|$|[\s\"'<>,;:)\]])"


#: A configured path with fewer non-empty segments than this is a mount root, not a producer path, and
#: layer (b) does not search for it.
_MIN_PATH_SEGMENTS = 2


def _non_empty_segment_count(path: str) -> int:
    """The number of non-empty segments of ``path``, splitting on either separator.

    ``/sfs`` has one and ``/sfs/x`` has two; a drive path such as ``C:/Users/x`` counts its drive as a
    segment, so it has three.
    """
    return len([segment for segment in re.split(r"[\\/]", path) if segment])


def _is_absolute_path_leaf(value: str) -> bool:
    """True when ``value`` begins with ``/`` or ``{drive}:{separator}`` and has a non-separator character.

    A string of slashes (the report's default hatch pattern) or a bare drive letter is not a path.
    """
    begins_like_a_path = value.startswith("/") or _DRIVE_PREFIX.match(value) is not None
    return begins_like_a_path and any(ch not in "/\\" for ch in value)


def _string_leaves(obj: Any) -> Iterable[str]:
    """Every string held anywhere in a nested dict/list structure, values only."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for value in obj.values():
            yield from _string_leaves(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _string_leaves(item)


def config_path_values(dumps: Iterable[Mapping[str, Any]]) -> list[str]:
    """The absolute paths held by the string leaves of the given raw config dumps, first occurrence first.

    Takes the PRE-rewrite dumps (``model_dump(mode="json")``) because the emitted config pair holds
    relative paths and would leave nothing to find.
    """
    seen: dict[str, None] = {}
    for dump in dumps:
        for leaf in _string_leaves(dump):
            if _is_absolute_path_leaf(leaf):
                seen.setdefault(leaf, None)
    return list(seen)


def per_user_segments(paths: Iterable[str]) -> list[str]:
    """The directory directly under a per-user or per-project root, for each path that has one."""
    seen: dict[str, None] = {}
    for path in paths:
        for match in _SEGMENT_AFTER_ROOT.finditer(path):
            seen.setdefault(match.group(1), None)
    return list(seen)


def _published_author_addresses() -> frozenset[str]:
    """Lower-cased author and maintainer addresses of the installed ``hhemt`` distribution's metadata.

    Read at run time and never listed in a file. An absent distribution yields the empty set, which
    makes the toolkit's own address fire like any other: loud rather than silent.
    """
    try:
        meta = importlib_metadata.metadata("hhemt")
    except importlib_metadata.PackageNotFoundError:
        return frozenset()
    fields = (meta.get_all("Author-email") or []) + (meta.get_all("Maintainer-email") or [])
    return frozenset(addr.lower() for _name, addr in getaddresses(fields) if addr)


def _private_list_from_environment() -> tuple[str, list[str]]:
    """The private list's state and tokens, read from ``$HHEMT_REPREX_BLOCKLIST`` and nowhere else.

    Unset and empty are distinct states and both mean no list is searched. A set, non-empty value
    must name a usable carrier: ``_load_blocklist`` raises ``ProcessingError`` for a missing,
    short or unreadable one, so a mistyped path refuses instead of degrading to a pass.
    """
    raw = os.environ.get(_ENV_OVERRIDE)
    if raw is None:
        return PRIVATE_LIST_NOT_CONFIGURED, []
    if raw == "":
        return PRIVATE_LIST_SET_BUT_EMPTY, []
    return PRIVATE_LIST_CHECKED, _load_blocklist()


@dataclass(frozen=True)
class MemberHit:
    """One layer firing on one archive member. ``detail`` is a class label or a value index, never a value."""

    member: str
    layer: str
    detail: str

    def render(self) -> str:
        """The hit as one line: member, layer, then the class label or value index."""
        return f"{self.member}: {self.layer}: {self.detail}"


@dataclass
class ZeroUserInfoScanner:
    """The three layers over the bytes of archive members, accumulating every hit.

    Build with ``for_emit`` (all three layers, from the analysis being emitted) or ``for_consume``
    (layers (a) and (c)). ``record`` scans one member's bytes and keeps its hits; ``raise_if_hits``
    turns the accumulated hits into one ``ProcessingError``. A hit never carries a matched value: a
    layer-(a) hit names its class, a layer-(b) or (c) hit names its 1-based index and length.
    """

    private_list_state: str = PRIVATE_LIST_NOT_CONFIGURED
    _config_checks: list[tuple[str, int, Callable[[str], bool]]] = field(default_factory=list)
    _private_checks: list[tuple[int, re.Pattern[str]]] = field(default_factory=list)
    _exempt_emails: frozenset[str] = frozenset()
    hits: list[MemberHit] = field(default_factory=list)

    @classmethod
    def for_consume(cls) -> ZeroUserInfoScanner:
        """Layers (a) and (c): a consumer holds no producer configs, so layer (b) has nothing to read."""
        state, tokens = _private_list_from_environment()
        return cls(
            private_list_state=state,
            _private_checks=_private_patterns(tokens),
            _exempt_emails=THIRD_PARTY_ADDRESSES | _published_author_addresses(),
        )

    @classmethod
    def for_emit(cls, analysis: Any) -> ZeroUserInfoScanner:
        """All three layers, layer (b) read from the RAW configs of ``analysis`` and the login name.

        A stand-in that lacks a config contributes nothing for it. With no private list configured
        the emit still runs layers (a) and (b); that is reported once as a warning and recorded in
        the manifest as ``not configured``.
        """
        state, tokens = _private_list_from_environment()
        if state != PRIVATE_LIST_CHECKED:
            warnings.warn(
                f"zero-user-info check: ${_ENV_OVERRIDE} is "
                f"{'set but empty' if state == PRIVATE_LIST_SET_BUT_EMPTY else 'not set'}, so the private "
                "list was not searched; the generic patterns and the configs' own values were.",
                UserWarning,
                stacklevel=3,
            )
        system = getattr(getattr(analysis, "_system", None), "cfg_system", None)
        cfgs = [c for c in (system, getattr(analysis, "cfg_analysis", None)) if c is not None]
        hpc = getattr(analysis, "cfg_hpc_system", None)
        # A stand-in analysis may hold a config object that cannot be dumped; it contributes nothing.
        dumps = [c.model_dump(mode="json") for c in cfgs if callable(getattr(c, "model_dump", None))]
        hpc_dump = hpc.model_dump(mode="json") if callable(getattr(hpc, "model_dump", None)) else {}
        paths = config_path_values([*dumps, hpc_dump])
        checks: list[tuple[str, int, Callable[[str], bool]]] = []

        def add(label: str, value: str, matcher: Callable[[str], bool]) -> None:
            """Register one layer-(b) check; ``value`` contributes only its length, which is what a hit reports."""
            checks.append((label, len(value), matcher))

        def add_pattern(label: str, value: str, pattern: str) -> None:
            """Register a case-insensitive regular-expression check for ``value``."""
            compiled = re.compile(pattern, re.IGNORECASE)
            add(label, value, lambda text, c=compiled: c.search(text) is not None)

        def usable(label: str, value: str | None) -> bool:
            """True when ``value`` is long enough to search for; a shorter one is logged by length and skipped."""
            if not value:
                return False
            if len(value) < _MIN_VALUE_CHARS:
                logger.warning(
                    "zero-user-info check: a %s of %d character(s) was not searched for (shorter than %d)",
                    label,
                    len(value),
                    _MIN_VALUE_CHARS,
                )
                return False
            return True

        for path in paths:
            if not usable("configured path", path):
                continue
            if _non_empty_segment_count(path) < _MIN_PATH_SEGMENTS:
                # A bare mount root such as a container bind source names no user, and searching for it
                # as a substring fires on every path beneath it. The path stays in `paths`, so
                # `per_user_segments` still reads it and `config_path_values` still returns it.
                logger.warning(
                    "zero-user-info check: a configured path of %d character(s) with fewer than %d segments "
                    "was not searched for",
                    len(path),
                    _MIN_PATH_SEGMENTS,
                )
                continue
            add("configured path", path, lambda text, v=path: v in text)
        for segment in per_user_segments(paths):
            if usable("per-user directory", segment):
                add_pattern("per-user directory", segment, _STANDS_ALONE_AS_SEGMENT % re.escape(segment))
        for key in ("default_account", "login_node"):
            value = hpc_dump.get(key)
            if usable(key, value):
                add_pattern(key, value, rf"(?<!\w){re.escape(value)}(?!\w)")
        try:
            login = getpass.getuser()
        except (OSError, KeyError, ImportError):
            login = None
        if usable("login name", login):
            name = re.escape(login)
            # The login name can be an ordinary word, so it is searched for only where it is plainly
            # a user name: a whole path segment, the local part of an address, or the value of
            # --user / -u / user=.
            for pattern in (
                _STANDS_ALONE_AS_SEGMENT % name,
                r"(?<![A-Za-z0-9._%+-])" + name + r"(?=@)",
                r"(?:--user[ =]|-u |user=)" + name + r"(?![\w.\-])",
            ):
                add_pattern("login name", login, pattern)
        return cls(
            private_list_state=state,
            _config_checks=checks,
            _private_checks=_private_patterns(tokens),
            _exempt_emails=THIRD_PARTY_ADDRESSES | _published_author_addresses(),
        )

    def manifest_record(self) -> dict[str, str]:
        """The ``zero_user_info`` value of the bundle manifest: each layer's state, as it ran."""
        return {
            LAYER_PATTERNS: "checked",
            LAYER_CONFIG_VALUES: "checked",
            LAYER_PRIVATE_LIST: self.private_list_state,
        }

    def scan_member(self, member: str, data: bytes) -> list[MemberHit]:
        """The hits of one member's bytes, without recording them."""
        # A byte that is not valid UTF-8 becomes U+FFFD, which no pattern here treats as part of an address, a path
        # or a configured value, so it BREAKS adjacency. Dropping it (errors="ignore") glued the text on either side
        # of every such byte, and a binary member (a GeoTIFF, a NetCDF) then formed address-shaped strings that no
        # byte of it holds. Valid UTF-8 decodes to exactly the same text either way.
        text = data.decode("utf-8", errors="replace")
        found: list[MemberHit] = []
        for match in EMAIL_PATTERN.finditer(text):
            if match.group(0).lower() not in self._exempt_emails:
                found.append(MemberHit(member, LAYER_PATTERNS, "email-address pattern (value not shown)"))
                break
        if PATH_PATTERN.search(text):
            found.append(MemberHit(member, LAYER_PATTERNS, "per-user absolute path pattern (value not shown)"))
        for index, (label, length, matcher) in enumerate(self._config_checks, start=1):
            if matcher(text):
                found.append(MemberHit(member, LAYER_CONFIG_VALUES, f"{label} #{index} ({length} chars)"))
        for index, pattern in self._private_checks:
            if pattern.search(text):
                found.append(MemberHit(member, LAYER_PRIVATE_LIST, f"private list token #{index}"))
        return found

    def record(self, member: str, data: bytes) -> list[MemberHit]:
        """Scan one member's bytes and keep its hits; returns them."""
        found = self.scan_member(member, data)
        self.hits.extend(found)
        return found

    def overlap_chars(self) -> int:
        """The overlap, in bytes, between consecutive blocks of one member: the longest configured pattern.

        A value that straddles a block boundary lies whole inside one of the two blocks only if the
        blocks share at least its length. The floor of ``_MIN_BLOCK_OVERLAP`` covers the two layer-(a)
        classes, whose matches are short in practice (an email address, a per-user path prefix).
        """
        lengths = [length for _label, length, _matcher in self._config_checks]
        lengths += [len(pattern.pattern) for _index, pattern in self._private_checks]
        return max([_MIN_BLOCK_OVERLAP, *lengths])

    def record_blocks(self, member: str, blocks: Iterable[bytes]) -> list[MemberHit]:
        """Scan the blocks of ONE member and keep each distinct hit once; returns the kept hits.

        Consecutive blocks overlap, so one value can fire in two of them; a hit is kept once per
        (layer, detail), which is what one scan of the whole member would have reported.
        """
        found: dict[tuple[str, str], MemberHit] = {}
        for block in blocks:
            for hit in self.scan_member(member, block):
                found.setdefault((hit.layer, hit.detail), hit)
        kept = list(found.values())
        self.hits.extend(kept)
        return kept

    def raise_if_hits(self, filepath: Path | None, operation: str) -> None:
        """Raise one ``ProcessingError`` naming every hit (member, layer, class or index) when any was recorded."""
        if not self.hits:
            return
        shown = [hit.render() for hit in self.hits[:_MAX_HITS_SHOWN]]
        more = len(self.hits) - len(shown)
        tail = f" (+{more} more)" if more > 0 else ""
        raise ProcessingError(
            operation=operation,
            filepath=filepath,
            reason=(
                f"{len(self.hits)} producer-specific match(es) in the bundle's members; no archive was "
                f"written. {'; '.join(shown)}{tail}"
            ),
        )


_MAX_HITS_SHOWN = 40


def _private_patterns(tokens: list[str]) -> list[tuple[int, re.Pattern[str]]]:
    """(1-based index, whole-word case-insensitive literal pattern) per private-list token."""
    return [(i, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)) for i, t in enumerate(tokens, start=1)]


# ---------------------------------------------------------------------------------------------
# Binary members: zarr arrays of text values are decoded; any other binary member is scanned
# through its decompressor when the standard library has one, and in overlapping blocks otherwise.
# ---------------------------------------------------------------------------------------------

_MIN_BLOCK_OVERLAP = 1024
_BLOCK_BYTES = 1 << 20
#: A container inside a container is opened the same way, down to this depth; a member nested deeper
#: refuses the emit rather than recursing without bound on a hostile or accidental archive.
_MAX_CONTAINER_DEPTH = 8
_SCAN_OPERATION = "bundle zero-user-info check"

#: The zarr 3.1.5 data-type registry, CLOSED at shape time (regenerate the names with
#: ``sorted(zarr.core.dtype.data_type_registry.contents.keys())``). A type the registry gains later
#: has no row here and ``classify_zarr_array`` refuses it loudly rather than skipping it.
ZARR_TEXT_DATA_TYPES = frozenset(
    {"string", "fixed_length_utf32", "null_terminated_bytes", "variable_length_bytes", "raw_bytes", "structured"}
)
ZARR_SKIPPED_DATA_TYPES = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "numpy.datetime64",
        "numpy.timedelta64",
    }
)
#: The numpy kinds of a zarr v2 array's ``dtype`` string, which names no registry entry. ``O`` is a
#: vlen string or bytes array (object codec) and is decoded; ``V`` is raw or structured.
_V2_TEXT_KINDS = frozenset("USOV")
_V2_SKIPPED_KINDS = frozenset("biufcMm")

ZARR_ARRAY_METADATA_NAMES = ("zarr.json", ".zarray")

ZARR_DECODE = "decode"
ZARR_SKIP = "skip"


def _zarr_array_metadata(array_dir: Path) -> dict[str, Any] | None:
    """The array metadata document under ``array_dir``, or None when it holds a group or no document."""
    for name in ZARR_ARRAY_METADATA_NAMES:
        doc = array_dir / name
        if not doc.is_file():
            continue
        try:
            meta = json.loads(doc.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProcessingError(
                operation=_SCAN_OPERATION,
                filepath=doc,
                reason=(
                    f"the zarr metadata document cannot be read ({type(exc).__name__}), "
                    "so its array cannot be classified"
                ),
            ) from exc
        if not isinstance(meta, dict):
            return None  # valid JSON that is not an object cannot be an array document: not an array
        if name == "zarr.json" and meta.get("node_type") != "array":
            return None
        return meta
    return None


def classify_zarr_array(array_dir: Path) -> str | None:
    """``"decode"`` or ``"skip"`` for the zarr array whose metadata sits in ``array_dir``; None for a group or no array.

    Reads only the metadata document. A v3 array is classified by its registry name (the table in the
    chunk specification), a v2 array by the numpy kind of its ``dtype`` string. A data type that is in
    neither table raises ``ProcessingError`` naming it: the table is closed over the registry that was
    installed when it was written, and an unlisted type must not be skipped silently.
    """
    meta = _zarr_array_metadata(array_dir)
    if meta is None:
        return None
    if "data_type" in meta:
        raw = meta["data_type"]
        name = raw.get("name") if isinstance(raw, Mapping) else raw
        if isinstance(name, str) and name in ZARR_TEXT_DATA_TYPES:
            return ZARR_DECODE
        if isinstance(name, str) and name in ZARR_SKIPPED_DATA_TYPES:
            return ZARR_SKIP
        raise ProcessingError(
            operation=_SCAN_OPERATION,
            filepath=array_dir,
            reason=(
                f"the zarr data type {name!r} has no row in the zero-user-info data-type table, so its values "
                "cannot be classified as text or not; add a row for it in hhemt.bundle._reprex_gate"
            ),
        )
    dtype = meta.get("dtype")
    if isinstance(dtype, list):
        return ZARR_DECODE  # a v2 structured dtype is a list of (name, dtype) pairs
    try:
        import numpy as np

        kind = np.dtype(dtype).kind
    except (TypeError, ValueError) as exc:
        raise ProcessingError(
            operation=_SCAN_OPERATION,
            filepath=array_dir,
            reason=f"the zarr v2 dtype {dtype!r} is not a numpy dtype string, so it cannot be classified",
        ) from exc
    if kind in _V2_TEXT_KINDS:
        return ZARR_DECODE
    if kind in _V2_SKIPPED_KINDS:
        return ZARR_SKIP
    raise ProcessingError(
        operation=_SCAN_OPERATION,
        filepath=array_dir,
        reason=f"the zarr v2 dtype kind {kind!r} has no row in the zero-user-info data-type table",
    )


def _values_as_bytes(values: Any) -> Iterator[bytes]:
    """The scannable bytes of every element of the numpy array ``values``, by its numpy kind."""
    kind = values.dtype.kind
    if kind == "U":
        for item in values.ravel().tolist():
            yield str(item).encode("utf-8", errors="replace")
    elif kind == "S":
        for item in values.ravel().tolist():
            yield bytes(item)
    elif kind == "O":
        for item in values.ravel().tolist():
            if isinstance(item, str):
                yield item.encode("utf-8", errors="replace")
            elif isinstance(item, bytes | bytearray):
                yield bytes(item)
    elif kind == "V":
        if values.dtype.names:
            for field_name in values.dtype.names:
                yield from _values_as_bytes(values[field_name])
        else:
            for item in values.ravel():
                yield item.tobytes()
    # every other kind is numeric, boolean or temporal and holds no text


def decode_zarr_array_text(array_dir: Path, *, consumer_member: str | None = None) -> bytes:
    """The text-bearing values of the zarr array in ``array_dir``, decoded and joined by newlines.

    It DECODES; it never byte-scans the chunk files, because a chunk file holds the values after the
    array's codecs have run (measured: a raw byte scan found a planted path in a short ``string`` array
    among repeated fillers and missed it in a ``fixed_length_utf32`` array and in a 1000-value
    ``string`` array). The payload is returned WHOLE as bytes (the interface the shape fixes): rows are read
    ``chunks[0]`` at a time, but every block is kept and joined, so peak memory is about twice the payload
    (the list of blocks and the joined bytes), not bounded by one chunk. A store that cannot be opened raises
    ``ProcessingError`` naming the directory.
    """
    import zarr

    try:
        array = zarr.open_array(str(array_dir), mode="r")
        shape = tuple(array.shape)
        pieces: list[bytes] = []
        if not shape:
            pieces.extend(_values_as_bytes(_as_ndarray(array[()])))
        else:
            step = max(int(array.chunks[0]), 1)
            for start in range(0, shape[0], step):
                block = array[start : start + step]
                pieces.extend(_values_as_bytes(_as_ndarray(block)))
    except ProcessingError:
        raise
    except Exception as exc:  # a zarr store fails with a family of types; none may escape as a crash
        raise ProcessingError(
            operation=_SCAN_OPERATION,
            filepath=array_dir,
            reason=(
                f"the zarr array {consumer_member} cannot be decoded for the check ({type(exc).__name__})"
                if consumer_member is not None
                else f"the zarr array cannot be decoded for the check ({type(exc).__name__}); no archive was written"
            ),
        ) from exc
    return b"\n".join(pieces)


def _as_ndarray(value: Any) -> Any:
    """``value`` as a numpy array (a 0-d read returns a numpy scalar)."""
    import numpy as np

    return np.asarray(value)


def zarr_array_text_payload(array_dir: Path) -> bytes | None:
    """The decoded text of the zarr array in ``array_dir``, or None when it has no text-typed values or is no array."""
    if classify_zarr_array(array_dir) != ZARR_DECODE:
        return None
    return decode_zarr_array_text(array_dir)


def _refuse_container(member: str, kind: str, exc: BaseException, *, consumer: bool = False) -> ProcessingError:
    """The refusal for a member that begins like a container the standard library then cannot open."""
    if consumer:
        return ProcessingError(
            operation=_SCAN_OPERATION,
            filepath=None,
            reason=(
                f"{member} begins like a {kind} container but cannot be opened ({type(exc).__name__}), so its "
                "content was not checked"
            ),
        )
    return ProcessingError(
        operation=_SCAN_OPERATION,
        filepath=None,
        reason=(
            f"{member} begins like a {kind} container but cannot be opened ({type(exc).__name__}), so its content "
            "was not checked; no archive was written. Fix or remove the file, or leave it out of the bundle "
            "and reference it through --exclude-config"
        ),
    )


def container_kind(head: bytes) -> str | None:
    """The container the leading bytes name, or None. Detection is by bytes, never by file name.

    The signatures are chosen so that a TEXT file does not carry one: a gzip stream's three-byte start
    includes a control character, and a bzip2 stream is recognised by its block magic after the ``BZh``
    and a level digit, not by ``BZh`` alone.
    """
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head[257:262] == b"ustar":
        return "tar"
    if head.startswith(b"\x1f\x8b\x08"):
        return "gzip"
    if re.match(rb"BZh[1-9]1AY&SY", head):
        return "bzip2"
    if head.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    return None


_STREAM_OPENERS: dict[str, Callable[[Any], Any]] = {
    "gzip": lambda fh: gzip.GzipFile(fileobj=fh, mode="rb"),
    "bzip2": lambda fh: bz2.BZ2File(fh),
    "xz": lambda fh: lzma.LZMAFile(fh),
}
_CONTAINER_ERRORS = (OSError, EOFError, ValueError, RuntimeError, NotImplementedError, zlib.error, lzma.LZMAError)


def _blocks(handle: Any, overlap: int) -> Iterator[bytes]:
    """The rest of ``handle`` in blocks, each starting with the last ``overlap`` bytes of the one before."""
    carry = b""
    while True:
        chunk = handle.read(_BLOCK_BYTES)
        if not chunk:
            return
        block = carry + chunk
        yield block
        carry = block[-overlap:] if overlap else b""


#: The byte-order marks of UTF-16 and UTF-32 (``codecs.BOM_UTF32_LE``, ``BOM_UTF32_BE``, ``BOM_UTF16_LE``,
#: ``BOM_UTF16_BE``), in the order they are tested: the UTF-32 LE mark begins with the UTF-16 LE mark, so the
#: two UTF-32 rows come first. Each is decoded with the codec of that width, which consumes the mark.
_BOM_CODECS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def _bom_codec(head: bytes) -> str | None:
    """The codec for the byte-order mark ``head`` begins with, or None."""
    for mark, codec in _BOM_CODECS:
        if head.startswith(mark):
            return codec
    return None


def has_byte_order_mark(head: bytes) -> bool:
    """True when ``head`` begins with one of the four UTF-16 / UTF-32 byte-order marks.

    The test the emit and the consume-side scan make BEFORE the NUL sniff: a marked member is never read
    as UTF-8 text and never rewritten, because a UTF-16 file of non-Latin text can hold no NUL byte in its first
    ``_SNIFF_BYTES`` and the sniff would then class it text. It compares the member's first 2 to 4 bytes against
    ``_BOM_CODECS`` in table order.
    """
    return _bom_codec(head) is not None


def _decoded_text(handle: Any, codec: str) -> Iterator[str]:
    """The strictly decoded text of ``handle`` a block at a time; ``UnicodeDecodeError`` on bad bytes."""
    decoder = codecs.getincrementaldecoder(codec)(errors="strict")
    while True:
        chunk = handle.read(_BLOCK_BYTES)
        text = decoder.decode(chunk, final=not chunk)
        if text:
            yield text
        if not chunk:
            return


def _decodes_cleanly(handle: Any, codec: str) -> bool:
    """True when the whole of ``handle`` decodes strictly in ``codec``; the handle is left anywhere."""
    handle.seek(0)
    try:
        for _ in _decoded_text(handle, codec):
            pass
    except UnicodeDecodeError:
        return False
    return True


def _decoded_blocks(handle: Any, codec: str, overlap: int) -> Iterator[bytes]:
    """The decoded text as UTF-8 blocks, each starting with the last ``overlap`` characters of the one before."""
    carry = ""
    for text in _decoded_text(handle, codec):
        block = carry + text
        yield block.encode("utf-8")
        carry = block[-overlap:] if overlap else ""


def scan_binary_member(
    scanner: ZeroUserInfoScanner, member: str, handle: Any, *, _depth: int = 0, consumer: bool = False
) -> None:
    """Scan one binary member's content and record the hits under ``member``.

    A container the standard library can open is scanned through its decompressor, never as its
    compressed bytes (a zip, tar, gzip, bzip2 or xz stream; the entries of a zip or a tar are recorded as
    ``{member}!{entry}``, a decompressed stream as ``{member}!(decompressed)``), and a container inside a
    container is opened the same way. A member that begins with a UTF-16 or UTF-32 byte-order mark and decodes
    strictly is scanned as the decoded text (a mark whose body does not decode is an opaque binary). Any other
    binary member is scanned in blocks that overlap by the longest configured pattern, so no member is held
    whole. A container that fails to open refuses with
    ``ProcessingError`` naming the member; it is never scanned as its compressed bytes.

    ``handle`` must be a seekable binary file object positioned anywhere; it is rewound first. ``consumer`` words
    the two refusal reasons for the consume side (no archive is being written there, and no ``--exclude-config``
    applies); it defaults to the emit's wording.
    """
    handle.seek(0)
    head = handle.read(512)
    handle.seek(0)
    kind = container_kind(head)
    if kind is None:
        codec = _bom_codec(head)
        if codec is not None and _decodes_cleanly(handle, codec):
            # A byte-order mark names an encoding the NUL sniff cannot read: decode, then scan the text.
            handle.seek(0)
            scanner.record_blocks(member, _decoded_blocks(handle, codec, scanner.overlap_chars()))
            return
        # No container and no decodable mark (a mark whose body does not decode is an opaque binary).
        handle.seek(0)
        scanner.record_blocks(member, _blocks(handle, scanner.overlap_chars()))
        return
    if _depth >= _MAX_CONTAINER_DEPTH and consumer:
        raise ProcessingError(
            operation=_SCAN_OPERATION,
            filepath=None,
            reason=(
                f"{member} is a container with {_MAX_CONTAINER_DEPTH} or more containers around it, "
                "so its content was not read"
            ),
        )
    if _depth >= _MAX_CONTAINER_DEPTH:
        raise ProcessingError(
            operation=_SCAN_OPERATION,
            filepath=None,
            reason=(
                f"{member} is a container with {_MAX_CONTAINER_DEPTH} or more containers around it, "
                "so its content was not read; no archive was written. Leave the file out of the bundle and "
                "reference it through --exclude-config"
            ),
        )
    try:
        if kind == "zip":
            with zipfile.ZipFile(handle) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    with zf.open(info) as entry:
                        scan_binary_member(
                            scanner, f"{member}!{info.filename}", entry, _depth=_depth + 1, consumer=consumer
                        )
        elif kind == "tar":
            with tarfile.open(fileobj=handle, mode="r:") as tf:
                for tinfo in tf:
                    if not tinfo.isreg():
                        continue
                    entry = tf.extractfile(tinfo)
                    if entry is not None:
                        scan_binary_member(
                            scanner, f"{member}!{tinfo.name}", entry, _depth=_depth + 1, consumer=consumer
                        )
        else:
            with _STREAM_OPENERS[kind](handle) as stream:
                scan_binary_member(scanner, f"{member}!(decompressed)", stream, _depth=_depth + 1, consumer=consumer)
    except ProcessingError:
        raise
    except (zipfile.BadZipFile, tarfile.TarError, *_CONTAINER_ERRORS) as exc:
        raise _refuse_container(member, kind, exc, consumer=consumer) from exc
