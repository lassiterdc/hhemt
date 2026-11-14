"""The GENERATED carrier state space of the carrier-resolution rule (CR).

The state space is GENERATED rather than hand-listed, so a state cannot be built for one
carrier and missed for another.
This module is not a test. It holds three things a carrier test imports:

* ``PER_TIER_STATES`` -- a closed tuple of twenty-four states a single tier's value can be in,
  and ``KEY_TIER_ONLY_STATES`` -- the two states only the git-config tier has. One state per
  distinct way a carrier loader reads, fails, skips or normalises its input.
* ``build_value`` -- a builder that creates each state under a directory, from SYNTHETIC
  tokens only, and returns the tier's value (a string, or ``None`` for unset).
* ``evaluate`` -- the reference evaluator of CR rules 1 to 7: given the state of each tier it
  returns ``degrade``, ``usable`` or ``refuse`` and the expected record. It is about twenty
  lines and knows nothing of the guard's code, so a defect in the guard is not inherited.

A carrier's state space is the Cartesian product of its tiers' state sets. The repository
guard has three tiers (flag, variable, key): 24 x 24 x 26 = 14,976 cells. The bundle gate has
the variable tier only: 24 cells (the bundle gate's carrier test may import this module).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Tier names, as the guard's refusal messages name them.
TIER_FLAG = "--blocklist"
TIER_ENV = "$HHEMT_ANONYMIZATION_BLOCKLIST"
TIER_KEY = "git config hhemt.blocklistPath"
TIERS = (TIER_FLAG, TIER_ENV, TIER_KEY)

MODULE_DEFAULT_FLOOR = 8

PER_TIER_STATES: tuple[str, ...] = (
    # not configured
    "unset",
    "empty",
    # path syntax
    "mistyped_path",
    "directory",
    "relative_path",
    "tilde_path",
    "symlink_to_well_formed",
    "trailing_slash_on_well_formed",
    # a path on which the filesystem call itself raises (pathlib swallows only some errnos)
    "name_too_long_path",
    "unsearchable_parent",
    # content
    "unreadable_file",
    "zero_entries",
    "comments_only",
    "below_the_floor",
    "duplicates_only",
    "case_variants_of_one_token_only",
    "not_utf8",
    "byte_order_mark",
    "crlf_line_endings",
    "trailing_whitespace_entry",
    "blank_lines_among_entries",
    "declared_floor_above_entries",
    "declared_floor_below_module_default",
    "well_formed",
)
KEY_TIER_ONLY_STATES: tuple[str, ...] = (
    "git_exits_other_than_0_or_1",  # a corrupt config: `git config --get` exits 128
    "git_unavailable",
)
KEY_TIER_STATES = PER_TIER_STATES + KEY_TIER_ONLY_STATES

# Usable states and the number of tokens the loader must return for each.
USABLE_TOKEN_COUNT = {
    "tilde_path": MODULE_DEFAULT_FLOOR,
    "symlink_to_well_formed": MODULE_DEFAULT_FLOOR,
    "trailing_slash_on_well_formed": MODULE_DEFAULT_FLOOR,
    "byte_order_mark": MODULE_DEFAULT_FLOOR,
    "crlf_line_endings": MODULE_DEFAULT_FLOOR,
    "trailing_whitespace_entry": MODULE_DEFAULT_FLOOR,
    "blank_lines_among_entries": MODULE_DEFAULT_FLOOR,
    "declared_floor_below_module_default": 3,
    "well_formed": MODULE_DEFAULT_FLOOR,
}
# Read as unconfigured by rule 1 ("empty" is REPORTED, never read as unset).
NOT_CONFIGURED = ("unset", "empty")


def _tokens(n: int, prefix: str = "ZZCARRIERTOKEN") -> list[str]:
    return [f"{prefix}{i:02d}" for i in range(1, n + 1)]


def _body(n: int) -> str:
    return "# synthetic carrier\n" + "\n".join(_tokens(n)) + "\n"


# ---------------------------------------------------------------------------------------
# The builder.
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BuiltValue:
    """A tier value: ``value`` is None for unset; ``buildable`` is False when this process
    cannot create the state (an unreadable file as the superuser)."""

    value: str | None
    buildable: bool = True


def build_value(state: str, tier_dir: Path, home: Path) -> BuiltValue:
    """Create ``state`` under ``tier_dir`` (HOME is ``home``) and return the tier's value."""
    tier_dir.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)

    def write(name: str, data: bytes | str, where: Path = tier_dir) -> Path:
        """Write one synthetic carrier file under ``where`` and return its path."""
        p = where / name
        p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        return p

    good = write("well_formed.txt", _body(MODULE_DEFAULT_FLOOR))
    if state == "unset":
        return BuiltValue(None)
    if state == "empty":
        return BuiltValue("")
    if state == "mistyped_path":
        return BuiltValue(str(tier_dir / "no_such_carrier.txt"))
    if state == "directory":
        d = tier_dir / "a_directory"
        d.mkdir(exist_ok=True)
        return BuiltValue(str(d))
    if state == "relative_path":
        # A file of this name also exists in the working directory the test uses, so an
        # implementation that resolves against the working directory would accept it.
        return BuiltValue("relative_carrier.txt")
    if state == "tilde_path":
        write(f"{tier_dir.name}_tilde.txt", _body(MODULE_DEFAULT_FLOOR), where=home)
        return BuiltValue(f"~/{tier_dir.name}_tilde.txt")
    if state == "symlink_to_well_formed":
        link = tier_dir / "link.txt"
        if not link.is_symlink():
            link.symlink_to(good)
        return BuiltValue(str(link))
    if state == "trailing_slash_on_well_formed":
        return BuiltValue(str(good) + "/")
    if state == "name_too_long_path":
        return BuiltValue(str(tier_dir / ("a" * 300) / "carrier.txt"))
    if state == "unsearchable_parent":
        locked = tier_dir / "locked_dir"
        locked.mkdir(exist_ok=True)
        locked.chmod(0o700)
        (locked / "carrier.txt").write_bytes(_body(MODULE_DEFAULT_FLOOR).encode())
        locked.chmod(0o000)
        if os.geteuid() == 0:  # root searches every directory: the state cannot be built
            locked.chmod(0o700)
            return BuiltValue(str(locked / "carrier.txt"), buildable=False)
        return BuiltValue(str(locked / "carrier.txt"))
    if state == "unreadable_file":
        p = write("unreadable.txt", _body(MODULE_DEFAULT_FLOOR))
        p.chmod(0o000)
        if os.geteuid() == 0:  # root reads it anyway: the state cannot be built
            p.chmod(0o644)
            return BuiltValue(str(p), buildable=False)
        return BuiltValue(str(p))
    if state == "zero_entries":
        return BuiltValue(str(write("zero.txt", "")))
    if state == "comments_only":
        return BuiltValue(str(write("comments.txt", "# one\n# two\n# three\n")))
    if state == "below_the_floor":
        return BuiltValue(str(write("below.txt", _body(3))))
    if state == "duplicates_only":
        return BuiltValue(str(write("dupes.txt", "ZZDUPTOKEN\n" * MODULE_DEFAULT_FLOOR)))
    if state == "case_variants_of_one_token_only":
        variants = ["zzcasetoken", "ZZCASETOKEN", "ZzCaseToken", "zZcAsEtOkEn"] * 2
        return BuiltValue(str(write("case.txt", "\n".join(variants) + "\n")))
    if state == "not_utf8":
        data = b"".join(b"ZZBADTOKEN%02d\xff\n" % i for i in range(MODULE_DEFAULT_FLOOR))
        return BuiltValue(str(write("not_utf8.txt", data)))
    if state == "byte_order_mark":
        return BuiltValue(str(write("bom.txt", b"\xef\xbb\xbf" + _body(MODULE_DEFAULT_FLOOR).encode())))
    if state == "crlf_line_endings":
        return BuiltValue(str(write("crlf.txt", _body(MODULE_DEFAULT_FLOOR).replace("\n", "\r\n"))))
    if state == "trailing_whitespace_entry":
        lines = [t + ("   " if i % 2 else "\t") for i, t in enumerate(_tokens(MODULE_DEFAULT_FLOOR))]
        return BuiltValue(str(write("trailing.txt", "# synthetic\n" + "\n".join(lines) + "\n")))
    if state == "blank_lines_among_entries":
        text = "# synthetic\n\n" + "\n\n".join(_tokens(MODULE_DEFAULT_FLOOR)) + "\n\n"
        return BuiltValue(str(write("blank.txt", text)))
    if state == "declared_floor_above_entries":
        return BuiltValue(str(write("floor_above.txt", "# min-tokens: 12\n" + _body(MODULE_DEFAULT_FLOOR))))
    if state == "declared_floor_below_module_default":
        return BuiltValue(str(write("floor_below.txt", "# min-tokens: 3\n" + "\n".join(_tokens(3)) + "\n")))
    if state == "well_formed":
        return BuiltValue(str(good))
    raise KeyError(f"no builder for carrier state {state!r}")


def build_key_config(state: str, tier_dir: Path, home: Path) -> tuple[Path, bool, bool, str | None]:
    """A git-config FILE for the key tier (``GIT_CONFIG_GLOBAL``), plus whether the state is
    buildable, whether git must be made unavailable for the call, and the value the key
    carries (None when the key is unset or the config is corrupt)."""
    cfg = tier_dir / f"gitconfig_{state}"
    tier_dir.mkdir(parents=True, exist_ok=True)
    if state == "git_exits_other_than_0_or_1":
        cfg.write_text("[hhemt\n\tblocklistPath = broken\n", encoding="utf-8")
        return cfg, True, False, None
    if state == "git_unavailable":
        cfg.write_text("", encoding="utf-8")
        return cfg, True, True, None
    built = build_value(state, tier_dir / "files", home)
    if built.value is None:
        cfg.write_text("", encoding="utf-8")
    else:
        cfg.write_text(f'[hhemt]\n\tblocklistPath = "{built.value}"\n', encoding="utf-8")
    return cfg, built.buildable, False, built.value


# ---------------------------------------------------------------------------------------
# The reference evaluator of CR rules 1 to 7.
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Expected:
    """What the reference evaluator predicts for one cell: the outcome class, the deciding tier, the number
    of entries a usable carrier yields, and the tiers reported as set but empty.
    """

    outcome: str  # "degrade" | "usable" | "refuse"
    tier: str | None  # the deciding tier (usable, refuse)
    tokens: int | None  # usable: the number of entries the loader returns
    empty_tiers: tuple[str, ...]  # degrade: tiers reported as "set but empty"


def evaluate(flag: str, variable: str, key: str) -> Expected:
    """CR: the FIRST configured tier decides; unset and empty are both unconfigured (rule 1),
    a configured tier is never skipped (rule 2), a usable one loads (rule 3), and every other
    state refuses (rules 3, 5, 6, 7); only no configured tier degrades (rule 4)."""
    empties: list[str] = []
    for tier, state in zip(TIERS, (flag, variable, key), strict=True):
        if state == "unset":
            continue
        if state == "empty":
            empties.append(tier)
            continue
        if state in USABLE_TOKEN_COUNT:
            return Expected("usable", tier, USABLE_TOKEN_COUNT[state], ())
        return Expected("refuse", tier, None, ())
    return Expected("degrade", None, None, tuple(empties))
