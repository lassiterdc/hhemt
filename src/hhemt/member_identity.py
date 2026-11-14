"""Single source of truth for the member-identity column across produced artifacts.

WHY THIS MODULE EXISTS. The `sa_id` -> `member_id` rename moved the PRODUCER
(`sensitivity_analysis.py` emits `member_id` into `df_status`) and left seven
consumers naming the old column as a bare string literal. Every one degraded
SILENTLY -- `.get("sa_id")` returns None, the caller substitutes "" and continues --
so the loss surfaced as an empty join, a skipped row, or an "indeterminate" verdict
rather than as an error. Each was found by a human-directed round.

The repair is to stop spelling the column at the call site. A consumer asks this
module which column the artifact it is HOLDING actually carries, so the next rename
is one edit here rather than a search for string literals.

WHY THE LEGACY SPELLING IS ACCEPTED RATHER THAN REMOVED. `sa_id` is still the
on-disk spelling of already-written artifacts, and its REMOVAL is a separately
deferred public-API change (the `override_force_rerun` key). Accepting it on READ is
correct and is NOT the same decision as emitting it on WRITE.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

_LOG = logging.getLogger(__name__)

#: The column the producer emits today.
MEMBER_ID_COLUMN = "member_id"

#: Spellings accepted when READING an artifact written by an older revision.
#: Ordered: the canonical name wins if an artifact somehow carries both.
LEGACY_MEMBER_ID_COLUMNS: tuple[str, ...] = ("sa_id",)

ACCEPTED_MEMBER_ID_COLUMNS: tuple[str, ...] = (MEMBER_ID_COLUMN, *LEGACY_MEMBER_ID_COLUMNS)


def resolve_member_id_column(columns: Iterable[str]) -> str | None:
    """Which accepted identity column `columns` carries, or None if it carries none.

    None is a LEGITIMATE answer -- a non-sensitivity analysis has no member axis --
    which is why this returns rather than raises. It is the CALLER's job to
    distinguish "this artifact has no member axis" from "every row's identity is
    blank", and returning the column NAME rather than the value is what makes that
    distinction available at all. The pre-repair code could not express it: both
    cases arrived as the empty string, which is why a total identity loss was
    reportable only as a low join rate.
    """
    present = set(columns)
    for name in ACCEPTED_MEMBER_ID_COLUMNS:
        if name in present:
            return name
    return None


def member_id_from(record: Mapping[str, object], column: str | None) -> str:
    """The stripped identity value for one record, or "" when there is no member axis."""
    if column is None:
        return ""
    return str(record.get(column) or "").strip()


def member_id_from_mapping(mapping: Mapping[str, object], default: str = "") -> str:
    """The identity value from a free-form mapping -- zarr node attrs, a JSON payload.

    This is the `.get(key, default)` form, internalized. Two call sites each chose a
    DIFFERENT fallback for the same miss (`_config_diff` stripped a prefix and got the
    bare id; `_dem_resolution_plots` kept the node name and got a PREFIXED one), and a
    default is precisely what converts a missing column into plausible-looking output
    rather than an error. Owning it here makes that one decision instead of N.
    """
    for name in ACCEPTED_MEMBER_ID_COLUMNS:
        value = mapping.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def warn_missing_member_id_column(artifact: str, columns: Iterable[str]) -> None:
    """Disclose that a member axis was EXPECTED here and no identity column resolved.

    Call this ONLY where members were expected. `resolve_member_id_column` stays
    silent by design: a non-sensitivity analysis legitimately has no identity column,
    so warning there would fire on every correct non-sensitivity run -- the same
    asymmetry that makes the resolver return rather than raise. The caller is the only
    party that knows which case it is in.
    """
    _LOG.warning(
        "no member-identity column in %s; expected one of %s, found %s. "
        "Member-keyed results for this artifact are EMPTY, not zero.",
        artifact,
        list(ACCEPTED_MEMBER_ID_COLUMNS),
        sorted(str(c) for c in columns),
    )
