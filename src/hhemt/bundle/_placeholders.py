"""Placeholders that stand in for producer-specific strings inside a bundle's text members.

A bundle is a public, immutable deposit, but its generated text members (the bundled
Snakefile, the baseline report page, ``scenario_status.csv``, SWMM ``.inp`` copies, zarr
metadata documents) were written on the producer's machine and carry its directories, its
interpreter, its login name and its SLURM account. This module owns three things the emit needs
and nothing else:

* the CLOSED set of ``{hhemt_*}`` tokens and the table that maps each producer value to its
  token, built per emit from the analysis being emitted and from the running environment;
* the per-member decision, made from a single ``open``, of whether an archive member is text
  (rewritten) or binary (copied through untouched);
* the whole-path rule, ``rewrite_producer_paths``, which runs after the table and replaces each
  remaining absolute path that carries the rewritten login segment, and each relative path that climbs
  three or more levels, with one token that records neither a directory nor a file name.

The substitution itself runs at archive-write time inside ``_emit.py::_emit_bundle_zip``,
which is why the table works on BYTES: a text member is read whole as bytes, rewritten and
written back, with no decode step that could reject a member the NUL-byte sniff called text.

Tokens are closed on purpose. They share their brace syntax with Snakemake (``{input}``,
``{wildcards.event_id}``) and Python format strings, and a reverse mapping that touched
anything but these ten exact names would corrupt those. A new token needs a new row here,
never a pattern.
"""

from __future__ import annotations

import getpass
import logging
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hhemt.bundle._reprex_gate import _SNIFF_BYTES

logger = logging.getLogger(__name__)

TOKEN_ANALYSIS_DIR = "{hhemt_analysis_dir}"
TOKEN_SYSTEM_DIR = "{hhemt_system_dir}"
TOKEN_TOOLKIT_DIR = "{hhemt_toolkit_dir}"
TOKEN_PACKAGE_DIR = "{hhemt_package_dir}"
TOKEN_PYTHON = "{hhemt_python}"
TOKEN_CONDA_ENV = "{hhemt_conda_env}"
TOKEN_HOME = "{hhemt_home}"
TOKEN_USER = "{hhemt_user}"
TOKEN_SLURM_ACCOUNT = "{hhemt_slurm_account}"
TOKEN_PRODUCER_PATH = "{hhemt_producer_path}"

#: The closed token set, in table order. Order is the tie-break when two rows hold the SAME
#: value: the earlier row wins, so a bundle is rewritten the same way on every emit.
PLACEHOLDER_TOKENS: tuple[str, ...] = (
    TOKEN_ANALYSIS_DIR,
    TOKEN_SYSTEM_DIR,
    TOKEN_TOOLKIT_DIR,
    TOKEN_PACKAGE_DIR,
    TOKEN_PYTHON,
    TOKEN_CONDA_ENV,
    TOKEN_HOME,
    TOKEN_USER,
    TOKEN_SLURM_ACCOUNT,
    TOKEN_PRODUCER_PATH,
)

#: The rows whose value is a filesystem path. Only these are subject to the absolute-path guard;
#: ``{hhemt_user}`` and ``{hhemt_slurm_account}`` hold names, and a name is not an absolute path.
_PATH_VALUED_TOKENS = frozenset(
    {
        TOKEN_ANALYSIS_DIR,
        TOKEN_SYSTEM_DIR,
        TOKEN_TOOLKIT_DIR,
        TOKEN_PACKAGE_DIR,
        TOKEN_PYTHON,
        TOKEN_CONDA_ENV,
        TOKEN_HOME,
    }
)

#: A value shorter than this is never substituted: a three-letter login name would otherwise
#: rewrite unrelated words wherever it occurs as a substring.
MIN_VALUE_CHARS = 4

#: Number of leading bytes that decide text versus binary. Imported, not restated, so the emit's
#: classification and the scan gate's ``_is_text`` cannot drift to different prefix lengths.
SNIFF_BYTES = _SNIFF_BYTES

_DRIVE_PREFIX = re.compile(r"[A-Za-z]:[\\/]")

#: The front of a path value written relative to a directory that lies below it: one or more ``../``
#: segments. The chain must start where no word character, dot or slash stands to its left, so it is
#: the whole chain and never the tail of a longer one (``a/../x``) or of a longer dot run (``.../x``).
#: The renderers record a figure's source files relative to the analysis directory, so a source that
#: lies outside that tree is written ``../../..{value}`` and the value's own leading slash follows a dot,
#: which the absolute-path boundary above declines.
_PARENT_CHAIN = rb"(?<![\w./])(?:\.\./)+"
_LEADING_PARENT_CHAIN = re.compile(rb"^(?:\.\./)+")

#: The characters that end a producer path: whitespace and the quote, bracket, separator and backtick
#: characters a generated text member puts straight after one (a quoted shell word, a CSV cell, a
#: Python argument list, an HTML attribute). A path is NOT ended by ``.``, ``:``, ``}`` or ``{``, so a
#: trailing ``.`` stays inside the match rather than leaking the directory in front of it.
_PATH_END_CLASS = rb"\s\"'<>,;)\]`"

#: One run of characters that no producer path crosses: a path is a maximal stretch of non-end characters.
_PATH_RUN = re.compile(rb"[^" + _PATH_END_CLASS + rb"]+")

#: Where a producer path may begin: a ``/`` or a ``../`` chain with no word character, ``:``, ``/``, ``.`` or ``}``
#: on its left. So a URL (``https://h/{hhemt_user}/repo``) is skipped, because the ``/`` after the host follows a word
#: character and the two slashes of ``//`` follow ``:`` and ``/``; a drive path (``C:/Users/...``) is skipped, because
#: its first ``/`` follows ``:``; and a path that follows another token (``{hhemt_home}/x/{hhemt_user}/z``) is skipped,
#: because its first ``/`` follows ``}``.
_PATH_START = re.compile(rb"(?<![\w:/.}])(?:\.\./|/)")

#: A relative path that climbs three or more levels and then names something: a figure sidecar, the baseline
#: report's configuration table and an SVG's embedded source text all spell a file outside the analysis
#: directory this way, and none of them holds the login segment, so the login rule cannot see it. It starts
#: where a path may start and runs to the first end character, which covers a JSON string's closing quote,
#: the ``<`` of a closing ``</code>`` or XML text end, the ``;`` that joins the sources in an image's source
#: text, whitespace, and a closing bracket. Fewer than three levels stay: they name a neighbour of the analysis
#: directory, which the rule has no ground to call a producer path.
#: The lookahead keeps a chain with no name after it (a bare run of ``../``) as it is, instead of letting the
#: tail swallow the last of its own levels.
_PARENT_CHAIN_PATH = re.compile(rb"(?<![\w:/.}])(?:\.\./){3,}(?!\.\./)[^" + _PATH_END_CLASS + rb"]+")

#: A matched chain whose first name, right after its levels, is already a token.
_FIRST_SEGMENT_IS_TOKEN = re.compile(rb"(?:\.\./)+\{hhemt_")

#: The login segment as it must appear inside a producer path, slashes included, so a path that still holds
#: a raw login never matches.
_LOGIN_SEGMENT = b"/" + TOKEN_USER.encode() + b"/"


def is_absolute_path_value(value: str) -> bool:
    """True when ``value`` begins with ``/`` or ``{drive}:{separator}`` and has a non-separator character.

    ``////`` and ``/`` are not absolute by this rule (a default report hatch value is the first and
    would otherwise be "a producer path"), and a bare ``python`` or a drive with no separator is not
    either. The rule is the one the zero-user-info scan applies to config values, restated here
    because the substitution table and the scan must agree on what counts as a path.
    """
    begins_like_a_path = value.startswith("/") or _DRIVE_PREFIX.match(value) is not None
    return begins_like_a_path and any(ch not in "/\\" for ch in value)


def prefix_is_text(prefix: bytes) -> bool:
    """The text sniff applied to bytes already in hand: no NUL byte in the first ``SNIFF_BYTES``.

    Takes bytes rather than a path so the emit can classify a member from the handle it is about to
    copy from. ``_reprex_gate._is_text`` opens the file itself, which would be a second open per
    member on a bundle with hundreds of thousands of them.
    """
    return b"\x00" not in prefix[:SNIFF_BYTES]


def is_zarr_chunk_member_by_name(arcname: str, metadata_names: frozenset[str]) -> bool:
    """True for a member under a ``*.zarr`` directory that is not one of the store's metadata documents.

    Such a member is a chunk and is copied through without reading its prefix: a zarr store holds
    hundreds of thousands of them, and reading each one's first 8,192 bytes to learn what its
    directory already says would re-introduce the per-file cost the streamed emit exists to avoid.
    The store's metadata documents (``zarr.json`` and the v2 equivalents) are NOT chunks; they are
    text, they carry source paths, and they are classified by the sniff like any other member.
    """
    parts = arcname.split("/")
    return parts[-1] not in metadata_names and any(part.endswith(".zarr") for part in parts[:-1])


@dataclass(frozen=True)
class SkippedValue:
    """One table value that was NOT substituted, and why. ``length`` is reported, never the value."""

    token: str
    reason: str
    length: int


class PlaceholderTable:
    """The producer values of ONE emit, ready to rewrite bytes.

    Build it with ``from_values`` (a plain mapping, for tests and callers that already hold the
    values) or ``build_placeholder_table`` (from an analysis and the running environment).

    Substitution is a single left-to-right pass over the bytes with the values tried longest first,
    so a directory nested inside another is replaced by its own token and a token that has been
    written is never rewritten again. Deliberately NOT one ``bytes.replace`` per row: replacing the
    shorter of two nested values first leaves the longer one as a mangled suffix of a token, and a
    second pass can match inside a token the first pass wrote.
    """

    def __init__(self, values: Mapping[str, Iterable[str | None]]) -> None:
        unknown = set(values) - set(PLACEHOLDER_TOKENS)
        if unknown:
            raise ValueError(f"unknown placeholder token(s) {sorted(unknown)}; the token set is closed")

        skipped: list[SkippedValue] = []
        accepted: list[tuple[str, str]] = []  # (token, value), table order then candidate order
        # Keyed by (is the login-name row, value): the login name is matched only in its three
        # contexts, so an account that happens to spell the same word must still get its own row.
        seen: set[tuple[bool, str]] = set()
        for token in PLACEHOLDER_TOKENS:
            for value in values.get(token, ()):
                if not value or (token == TOKEN_USER, value) in seen:
                    continue
                if len(value) < MIN_VALUE_CHARS:
                    skipped.append(SkippedValue(token, f"shorter than {MIN_VALUE_CHARS} characters", len(value)))
                    continue
                if token in _PATH_VALUED_TOKENS and not is_absolute_path_value(value):
                    skipped.append(SkippedValue(token, "not an absolute path", len(value)))
                    continue
                seen.add((token == TOKEN_USER, value))
                accepted.append((token, value))
        self.skipped: tuple[SkippedValue, ...] = tuple(skipped)
        for item in skipped:
            logger.warning(
                "bundle placeholder %s: a value of %d character(s) was not substituted (%s); "
                "if it occurs in the bundle the zero-user-info check will refuse the emit",
                item.token,
                item.length,
                item.reason,
            )

        self._tokens_by_value: dict[bytes, bytes] = {}
        ordered = sorted(
            (item for item in accepted if item[0] != TOKEN_USER),
            key=lambda item: (-len(item[1]), PLACEHOLDER_TOKENS.index(item[0]), item[1]),
        )
        alternatives: list[bytes] = []
        for token, value in ordered:
            raw = value.encode()
            self._tokens_by_value[raw] = token.encode()
            # A path must not begin in the middle of a longer one (`/mnt/home/u`) nor continue into
            # a longer sibling name (`/home/user2`); a name must stand alone.
            if token in _PATH_VALUED_TOKENS:
                alternatives.append(rb"(?<![\w.])" + re.escape(raw) + rb"(?![\w-])")
                # The same value reached through a `../` chain is replaced WHOLE, chain included, so
                # the result is the token and not a token with a dangling `../` in front of it. It is
                # added beside the absolute form so longest-first order still decides nested values.
                if raw.startswith(b"/"):
                    alternatives.append(_PARENT_CHAIN + re.escape(raw[1:]) + rb"(?![\w-])")
            else:
                alternatives.append(rb"(?<![\w.-])" + re.escape(raw) + rb"(?![\w.-])")
        self._value_pattern: re.Pattern[bytes] | None = re.compile(b"|".join(alternatives)) if alternatives else None

        self._user_rewrites: list[tuple[re.Pattern[bytes], bytes]] = []
        user_token = TOKEN_USER.encode()
        for token, value in accepted:
            if token != TOKEN_USER:
                continue
            name = re.escape(value.encode())
            # The login name is rewritten only where it is plainly a user name, never as a bare
            # substring: a whole path segment, the local part of an e-mail address, or the value
            # of --user / -u / user=.
            self._user_rewrites.extend(
                [
                    (re.compile(rb"(?<=[/\\])" + name + rb"(?=[/\\]|$|[\s\"'<>,;:)\]])"), user_token),
                    (re.compile(rb"(?<![A-Za-z0-9._%+-])" + name + rb"(?=@)"), user_token),
                    (
                        re.compile(rb"(--user[ =]|-u |user=)" + name + rb"(?![\w.\-])"),
                        rb"\g<1>" + user_token,
                    ),
                ]
            )

    @classmethod
    def from_values(cls, values: Mapping[str, Iterable[str | None]]) -> PlaceholderTable:
        """Build a table from ``{token: [candidate values]}``; a token may be absent or empty."""
        return cls(values)

    @property
    def is_empty(self) -> bool:
        """True when no value survived the guards, so ``apply`` returns its input unchanged."""
        return self._value_pattern is None and not self._user_rewrites

    def _token_for_match(self, match: re.Match[bytes]) -> bytes:
        """The token for one value match: the matched text is the value, or a `../` chain followed by it.

        A chain is stripped to the absolute spelling before the lookup, because the table is keyed by the
        absolute value and a chain never begins an absolute value.
        """
        matched = match.group(0)
        token = self._tokens_by_value.get(matched)
        if token is None:
            token = self._tokens_by_value[_LEADING_PARENT_CHAIN.sub(b"/", matched)]
        return token

    def apply(self, data: bytes) -> bytes:
        """Return ``data`` with every producer value replaced by its token."""
        if self._value_pattern is not None:
            data = self._value_pattern.sub(self._token_for_match, data)
        for pattern, replacement in self._user_rewrites:
            data = pattern.sub(replacement, data)
        return data


def rewrite_producer_paths(data: bytes) -> bytes:
    """Replace each producer path with ``{hhemt_producer_path}``: an absolute one that carries the login segment,
    or a relative one that climbs three or more levels.

    Runs on the output of ``PlaceholderTable.apply``, by which point every producer directory that has a
    row of its own is already a token, so what is left under a ``/{hhemt_user}/`` segment is a path no row
    covers (the producer's build estate, an experiment's input files, a copy of the analysis directory).
    The whole path is replaced, so the result names neither the directory nor the file.

    Kept out of ``PlaceholderTable.apply`` on purpose: that method turns ``/data/{login}/x`` into
    ``/data/{hhemt_user}/x`` and ``/d/abcd/x`` into ``/d/{hhemt_user}/x``, and a whole-path rule inside it
    would turn both into the producer-path token and change what the table's own contract says.

    Matches only the literal ``/{hhemt_user}/`` form, so a path that still holds a raw login is not its
    input. A URL carrying the login segment, a Windows drive path and a path that follows another token
    are left as they are (see ``_PATH_START``); the drive path is refused later by the scan.

    Then every ``../`` chain of three or more levels with a name after it is replaced whole as well (see
    ``_PARENT_CHAIN_PATH``), because such a path leaves the analysis directory by a producer-specific route
    without ever holding the login segment. An in-tree upward reference of that depth in rendered text is
    replaced too; one or two levels are not.
    """
    return _tokenise_parent_chains(_tokenise_login_paths(data))


def _tokenise_parent_chains(data: bytes) -> bytes:
    """Replace each ``../`` chain of three or more levels, with its tail, by the producer-path token.

    A chain whose first name after the levels is already a ``{hhemt_`` token is left as it is, so a string a
    table row has rewritten is not rewritten twice: that token stands for everything it covers. A token
    further along does not cover the folders in front of it, so such a chain is replaced whole.
    """
    if b"../../../" not in data:
        return data
    token = TOKEN_PRODUCER_PATH.encode()

    def replace(match: re.Match[bytes]) -> bytes:
        # Only a token standing where the first name after the chain would be already covers everything after
        # it. A token further along does not: the folder names in front of it would otherwise ship.
        return match.group(0) if _FIRST_SEGMENT_IS_TOKEN.match(match.group(0)) else token

    return _PARENT_CHAIN_PATH.sub(replace, data)


def _tokenise_login_paths(data: bytes) -> bytes:
    """Replace each path that carries the login segment; the first half of ``rewrite_producer_paths``."""
    marker_at = data.find(_LOGIN_SEGMENT)
    if marker_at < 0:
        return data
    token = TOKEN_PRODUCER_PATH.encode()
    pieces: list[bytes] = []
    copied = 0
    for run in _PATH_RUN.finditer(data):
        run_start, run_end = run.span()
        # The path of a run is the whole run from its leftmost valid start, provided a login segment lies at
        # or after the slash that ends the start (the search stops after the last login segment's slash, so a
        # ``../`` chain is kept only when its final slash is not past it). One pass per run keeps the cost linear:
        # a retry of the start from every candidate position would be quadratic in a long run of path starts.
        last_login = data.rfind(_LOGIN_SEGMENT, run_start, run_end)
        if last_login < 0:
            continue
        start = _PATH_START.search(data, run_start, last_login + 1)
        if start is None:
            continue
        pieces.append(data[copied : start.start()])
        pieces.append(token)
        copied = run_end
    pieces.append(data[copied:])
    return b"".join(pieces)


def _spellings(path: Any) -> list[str]:
    """The path as given and, when different, as resolved: a symlinked scratch directory is written either way."""
    if path is None:
        return []
    given = Path(path)
    spellings = [str(given)]
    try:
        resolved = str(given.resolve())
    except OSError:
        return spellings
    if resolved != spellings[0]:
        spellings.append(resolved)
    return spellings


def _chain(obj: Any, *names: str) -> Any:
    """``obj.a.b.c`` or None at the first missing link, so a partial analysis yields a partial table."""
    for name in names:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def build_placeholder_table(analysis: Any) -> PlaceholderTable:
    """The table for ONE emit: ``analysis``'s directories and the running environment's identity.

    Missing pieces (no system config, no HPC config) leave their rows empty rather than raising: a
    row with no value substitutes nothing, which is the truthful result for an analysis that has
    no such producer string.

    ``{hhemt_python}`` follows the base workflow builder's resolution: the configured interpreter
    when it is an absolute path, else ``sys.executable``. It is NOT read from a builder, because the
    sensitivity builder holds the bare word ``python`` and a table row of ``python`` would rewrite
    every occurrence of that word in the bundle's prose and shell.
    """
    import hhemt

    package_dir = Path(hhemt.__file__).resolve().parent
    package_roots = _spellings(package_dir) + _spellings(Path(hhemt.__file__).parent)
    # The checkout root is a separate directory from the package only when the package sits under
    # `src/`. It gets its own token, because one token for both directories cannot be inverted:
    # `{checkout}/src/hhemt/x` would be written `{toolkit}/x` and read back as `{checkout}/x`.
    toolkit_roots: list[str] = []
    for package in (package_dir, Path(hhemt.__file__).parent):
        if package.parent.name == "src":
            toolkit_roots += _spellings(package.parent.parent)

    configured_python = str(getattr(analysis, "_python_executable", "python"))
    python = configured_python if is_absolute_path_value(configured_python) else sys.executable

    builder = getattr(analysis, "_workflow_builder", None)
    conda_env_getter = getattr(builder, "_get_conda_env_path", None)
    if callable(conda_env_getter):
        conda_env = str(conda_env_getter())
    else:
        conda_env = str(package_dir.parent.parent / "workflow" / "envs" / "hhemt.yaml")

    try:
        home = _spellings(Path.home())
    except (RuntimeError, KeyError):
        home = []
    try:
        login = getpass.getuser()
    except (OSError, KeyError, ImportError):
        login = None

    return PlaceholderTable(
        {
            TOKEN_ANALYSIS_DIR: _spellings(_chain(analysis, "analysis_paths", "analysis_dir")),
            TOKEN_SYSTEM_DIR: _spellings(_chain(analysis, "_system", "cfg_system", "system_directory")),
            TOKEN_TOOLKIT_DIR: toolkit_roots,
            TOKEN_PACKAGE_DIR: package_roots,
            TOKEN_PYTHON: [python],
            TOKEN_CONDA_ENV: [conda_env],
            TOKEN_HOME: home,
            TOKEN_USER: [login],
            TOKEN_SLURM_ACCOUNT: [_chain(analysis, "cfg_hpc_system", "default_account")],
        }
    )
