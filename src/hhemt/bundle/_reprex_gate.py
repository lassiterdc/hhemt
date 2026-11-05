"""Zero-user-info emit-time gate for reprex bundles.

Scans the text members of a bundle for producer-specific values with three layers, so a bundle
that names the producer is refused instead of published. Layer (a) is two generic pattern classes
(an email address, and an absolute path under a per-user root) held in this file. Layer (b) is
built per emit from the emitted analysis's own configs and the running login name. Layer (c) is an
optional private list of specific identifiers; it is reached only through the environment variable
``HHEMT_REPREX_BLOCKLIST`` and through no path inside the source tree, so the toolkit repository
and its wheel carry no producer-specific value. A variable that is set to something unusable
refuses the emit; a variable that is not configured leaves layers (a) and (b) running and is
recorded as such in the bundle manifest.

Independence invariant (load-bearing): the private list is NOT derived from the scrub or the
placeholder table, else it weakens with each change to them.
"""

from __future__ import annotations

import getpass
import logging
import os
import re
import warnings
from collections.abc import Callable, Iterable, Mapping
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


def assert_bundle_zero_user_info(bundle_root: Path) -> None:
    """Raise ProcessingError if layer (a) or the private list matches any text member under ``bundle_root``.

    The consume-side form of the emit-time check: a consumer holds no producer configs, so layer (b)
    has nothing to read here and only layers (a) and (c) run.
    """
    scanner = ZeroUserInfoScanner.for_consume()
    for f in sorted(bundle_root.rglob("*")):
        if not f.is_file() or not _is_text(f):
            continue
        scanner.record(f.relative_to(bundle_root).as_posix(), f.read_bytes())
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
# the repository guard's pattern string.
EMAIL_PATTERN = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]++@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

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
        text = data.decode("utf-8", errors="ignore")
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
                f"{len(self.hits)} producer-specific match(es) in the bundle's text members; no archive was "
                f"written. {'; '.join(shown)}{tail}"
            ),
        )


_MAX_HITS_SHOWN = 40


def _private_patterns(tokens: list[str]) -> list[tuple[int, re.Pattern[str]]]:
    """(1-based index, whole-word case-insensitive literal pattern) per private-list token."""
    return [(i, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)) for i, t in enumerate(tokens, start=1)]
