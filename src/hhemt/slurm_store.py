"""The consolidated SLURM efficiency store: ONE dataset, one provenance.

WHY THIS EXISTS. Three artifacts fed the efficiency table and were joined at RENDER time:
the executor plugin's globbed `slurm_efficiency_report_*.csv`, this toolkit's sacct-derived
`job_level_recovery.csv`, and `_status/_job_index.json`. Each carried a different subset of
the same jobs, each went stale independently, and the join happened in the renderer -- so a
disagreement between them surfaced as a wrong cell rather than as an error, and no single
file could be pointed at as the source of truth. This module is that single file.

WHAT IS DECIDED HERE, AND WHAT IS NOT. The six engineering decisions D1-D6 are ratified and
are implemented below; each is named at the code that carries it. What this module
deliberately does NOT decide is the COLUMN SET. Nothing here enumerates the columns a store
must have: `write_store` accepts whatever columns its rows carry, `PROVENANCE_CLASS` returns
a documented fallback for a column it does not know, and the reducers live in the renderer.
That is not laziness -- the derived-column set is under concurrent ruling, and a store that
named its columns would force a schema fight every time that ruling moved.

THE STORE INVARIANT (D4), which every future edit is measured against:

    Every field a classification might key on is RETAINED per row and classified at READ
    time, never collapsed at write time.

This is what makes the user's own requirement true -- "fix any bugs in our aggregation
without having to recapture everything". A classifier bug is fixed by editing a reducer and
re-rendering; the store is not touched, because the store never made the classification. The
measured instance that motivated the rule: a reducer keyed the wrapper/solver distinction on
a step's numeric SUFFIX, which is wrong on 18 of 28 solver steps in three campaigns, and it
was fixable at read time only because `JobName` had been retained rather than collapsed into
a role at capture.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

#: The store, beside the artifacts it consolidates. A zarr GROUP (D1) rather than a CSV:
#: CSV has no per-COLUMN metadata channel, and the three-way availability the store must
#: carry has one state -- "this column was never gathered" -- that is a property of the
#: column rather than of any value in it, so no per-value null can express it.
STORE_DIRNAME = "slurm_jobs.zarr"

_EFF_RELDIR = ("logs", "slurm_efficiency_report")

#: The row key (D6). NOT `JobID` alone: `JobRequeue = 1` on this cluster, so a NODE_FAIL
#: requeues the SAME id and that id then names two distinct executions. Measured: job
#: 18583265 ran 07:54:22 on 8 CPUs, hit NODE_FAIL, and was requeued into an instance that
#: never ran; the two differ on `Submit` and on nothing else that identifies them.
KEY_FIELDS = ("JobID", "Submit")

#: Fixed-width unicode for every string column (D5). `object` dtype is what makes a store
#: hostile to its readers: `chunks='auto'` raises `NotImplementedError: Can not use auto
#: rechunking with object dtype`, so every downstream consumer would have to know to pass
#: `chunks={}` forever, and the one that forgets raises. Fixed-width opens under the
#: project-standard `chunks='auto'` and costs only that the width must be asserted rather
#: than assumed -- which `_encode_string_column` does, loudly.
STRING_WIDTH_FLOOR = 64

#: Widths are rounded UP to a multiple of this, giving headroom so a slightly longer value
#: on the next capture does not change the dtype and force a needless full rewrite.
_STRING_WIDTH_QUANTUM = 32

#: Zarr v2, deliberately, and it is what makes D5's `<U` choice spec-compliant rather than
#: merely convenient. Measured: writing `<U64` under `zarr_format=3` emits
#: `UnstableSpecificationWarning` -- "FixedLengthUTF32 does not have a Zarr V3
#: specification ... may be unreadable by other Zarr libraries" -- while the same array
#: under `zarr_format=2` writes silently and reads back as `<U64` under `chunks='auto'`.
#: This is not a retreat from D5: D5 governs the DTYPE (fixed-width, never object) and the
#: chunking, and v2 is the format version in which that dtype is specified. Revisit when
#: zarr-extensions specifies a fixed-length UTF string for v3.
ZARR_FORMAT = 2

#: Columns that are IDENTIFIERS rather than quantities, and must never be numeric-inferred.
#: This is not a style preference -- it is a measured data-loss bug caught by the requeue
#: test. `MainJobID` is all-digits, so a naive numeric inference stores `18583265` as a
#: float64 and any float-to-string rendering narrows it (`%g` yields `1.85833e+07`). An id
#: is a label that happens to be spelled with digits; arithmetic on it is meaningless and
#: rounding it is destruction. Kept minimal ON PURPOSE so the store stays column-agnostic:
#: a future DERIVED NUMERIC column (a percentage, a count) still infers numeric correctly,
#: because this set names the four identifiers rather than a class.
_IDENTIFIER_COLUMNS = frozenset({*KEY_FIELDS, "MainJobID", "StepKind"})

#: SLURM states that are FINAL. A re-capture returning a non-terminal state must never
#: overwrite a stored terminal row (D3's terminality rule), because sacct will happily
#: report a fresher-but-emptier row for a job whose accounting has already settled, and
#: recency-wins would regress a completed measurement to a running one. Sourced from
#: sacct(1) JOB STATE CODES; the `CANCELLED by {uid}` form is matched by prefix because
#: SLURM appends the cancelling uid to the state string.
_TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)

#: D2, half one: the provenance CLASS of each field is a fixed property of the field, so it
#: lives in CODE and is never written per capture. The classes are SLURM's own, and the
#: distinction they encode is load-bearing rather than descriptive: only a MEASURED field
#: can be `not_gathered`, because `not_gathered` is a statement about an `acct_gather`
#: plugin. Stamping it on `Partition` would assert that a plugin sampled the experiment's
#: hardware axis -- false about how the value exists, and unfalsifiable in the direction
#: that matters, since no plugin gathers it either way.
PROVENANCE_CLASS: dict[str, str] = {
    # REQUESTED -- what the user asked for. A request always exists.
    "ReqMem": "requested",
    "Timelimit": "requested",
    # GRANTED -- what the scheduler allocated. A grant always exists once the job starts.
    "AllocTRES": "granted",
    "NCPUS": "granted",
    "NNodes": "granted",
    "NodeList": "granted",
    "Partition": "granted",
    # SCHEDULED -- clock facts the controller records. Never "not gathered", though they
    # can be undefined for an instance that never ran (`Start = None` on a requeue).
    "Elapsed": "scheduled",
    "End": "scheduled",
    "Planned": "scheduled",
    "Start": "scheduled",
    "Submit": "scheduled",
    # MEASURED -- sampled by an acct_gather plugin. THIS CLASS ALONE may be `not_gathered`.
    "MaxRSS": "measured",
    "NTasks": "measured",
    "TRESUsageInTot": "measured",
    "TotalCPU": "measured",
    # DERIVED -- a controller verdict or a toolkit-side label, not a measurement.
    "JobID": "derived",
    "JobName": "derived",
    "MainJobID": "derived",
    "RuleName": "derived",
    "RunMethod": "derived",
    "State": "derived",
    "StepKind": "derived",
}

#: The provenance class assigned to a column this module does not know. `derived` is the
#: deliberate choice rather than a sentinel: it is the only class that constrains nothing
#: downstream, so an unrecognised column is carried and rendered without inheriting a claim
#: about how it was produced. Choosing `measured` would be actively wrong -- it would make
#: `not_gathered` admissible on a column nobody has established is sampled.
DEFAULT_PROVENANCE_CLASS = "derived"

#: D2, half two: availability is OBSERVED per capture and lives in the STORE.
_AVAILABILITY_VALUES = frozenset(
    {
        "available",
        "not_gathered",
        "not_derivable",
        "undefined_at_this_grain",
        "undefined_for_this_instance",
    }
)

#: The two CLASS-RESTRICTED availability values, and the restriction is the whole reason the
#: two-field taxonomy beats the flat three-way it replaced: each is checkable at WRITE time,
#: so a wrong stamp raises instead of rendering as a plausible sentence.
#:
#: `not_derivable` is the deliberate MIRROR of `not_gathered`. It exists because D2 as first
#: ratified had no true thing to say about a derived column whose input was never sampled:
#: `available` asserts a number exists, `undefined_at_this_grain` is false because the grain
#: is fine, and `undefined_for_this_instance` is false because it is EVERY instance rather
#: than this one. Two independent rounds reached that gap from opposite directions -- from
#: the derived-output side and from the constraint side -- and the remedy is a fifth term
#: rather than a relaxed constraint, so checkability is preserved on BOTH sides rather than
#: traded away on one.
_CLASS_RESTRICTED_AVAILABILITY = {
    "not_gathered": "measured",
    "not_derivable": "derived",
}


class StoreSchemaError(ValueError):
    """A store write that would violate the D2 column-metadata contract."""


def provenance_class(column: str) -> str:
    """The fixed provenance class of `column`, or the documented fallback."""
    return PROVENANCE_CLASS.get(column, DEFAULT_PROVENANCE_CLASS)


def is_terminal(state: str) -> bool:
    """Whether a SLURM state string is final.

    Prefix-matched on the first token because SLURM appends context to some states
    (`CANCELLED by 554635`). The full string is what the store RETAINS -- three distinct
    cancel causes were measured on one campaign (`CANCELLED by 554635`, `CANCELLED by 0`,
    bare `CANCELLED`) and a reducer that needs them apart must have them -- so this
    function reads the state and never rewrites it.
    """
    token = (state or "").strip().split(" ", 1)[0].upper()
    return token in _TERMINAL_STATES


def validate_availability(column: str, availability: str) -> None:
    """Raise unless `availability` is admissible for `column`'s provenance class (D2).

    This is the write-time check that makes the two-field split earn more than tidiness: a
    flat one-field taxonomy accepts a wrong stamp silently and renders it as a plausible
    sentence, and there is no later point at which the wrongness becomes visible. Here it
    is an exception at the moment of writing.
    """
    if availability not in _AVAILABILITY_VALUES:
        raise StoreSchemaError(
            f"availability {availability!r} for column {column!r} is not one of {sorted(_AVAILABILITY_VALUES)}"
        )
    cls = provenance_class(column)
    required = _CLASS_RESTRICTED_AVAILABILITY.get(availability)
    if required is not None and cls != required:
        why = {
            "not_gathered": (
                "that state is a statement about an acct_gather plugin and is admissible only for a MEASURED field"
            ),
            "not_derivable": (
                "that state says a reduction could not be computed and is admissible only for a DERIVED field"
            ),
        }[availability]
        raise StoreSchemaError(
            f"column {column!r} has provenance_class {cls!r}, so it can never be "
            f"{availability!r} -- {why}. Use 'undefined_at_this_grain' for a field absent at "
            f"this row class, or 'undefined_for_this_instance' for one undefined for this "
            f"execution."
        )


def derive_availability(input_availabilities: list[str] | tuple[str, ...]) -> str:
    """The `availability` a DERIVED column takes, PROPAGATED from its inputs (D2 rider).

    A derived column's availability is COMPUTED FROM ITS INPUTS, never observed on its own
    output. Observing the output cannot separate two states D2 exists to keep apart: a
    GPU-derived cell is blank on a CPU-only row because the column does not apply, and blank
    on a GPU row whose sampler failed because the measurement was never gathered. The cell is
    empty either way, so a post-hoc inspection collapses `undefined_at_this_grain` into
    `not_gathered` -- the same conflation D2 forbids for captured columns, reappearing one
    layer up where the ratified rule did not reach.

    The rule is uniform and deliberately blunt: **a derived column is `not_derivable` iff ANY
    input is anything other than `available`.** It does NOT inherit the input's own value.
    Inheritance was the first remedy proposed and it is false -- an energy-derived column
    whose input carries `not_gathered` cannot take that value (the class constraint forbids
    it), and every other existing term is untrue of it: `available` asserts a number exists,
    `undefined_at_this_grain` is false because the grain is fine, and
    `undefined_for_this_instance` is false because it is EVERY instance rather than this one.
    `not_derivable` is the term that is true, and it is admissible exactly where it applies.

    Propagation rather than a hand-stamp is the load-bearing half: propagation is
    unit-testable and a hand-stamp is not, which is the same reason D2's original constraint
    was worth having.
    """
    values = list(input_availabilities)
    for value in values:
        if value not in _AVAILABILITY_VALUES:
            raise StoreSchemaError(f"input availability {value!r} is not one of {sorted(_AVAILABILITY_VALUES)}")
    return "available" if all(v == "available" for v in values) else "not_derivable"


def _row_key(row: dict[str, str]) -> tuple[str, str]:
    return tuple((row.get(f) or "").strip() for f in KEY_FIELDS)  # type: ignore[return-value]


def merge_rows(
    stored: list[dict[str, str]],
    incoming: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Merge `incoming` over `stored` on `(JobID, Submit)` (D3). Pure; no I/O.

    Three rules, and each closes a measured failure:

    1. MERGE, NEVER REPLACE. A key in `stored` and absent from `incoming` is RETAINED. A
       capture is a partial view -- `recover_rows` continues past a timed-out chunk, and
       sacct silently omits ids it has purged -- so replacing would make the store a
       snapshot of sacct's current retention window rather than a data product, and would
       discard measurements already safely recorded.
    2. FIELD-WISE, NON-EMPTY WINS. A re-captured row legitimately updates its own fields as
       a job settles; an empty new value never erases a stored one.
    3. TERMINALITY, NOT RECENCY. A non-terminal incoming row does NOT overwrite a stored
       terminal one. Recency-wins is the intuitive rule and is wrong in exactly one
       direction: a job whose accounting has settled can still be re-queried, and the
       fresher read is the less informative one.

    Rule 3 gates the ROW; rule 2 then applies within it. Keeping them separate matters --
    a terminal-vs-non-terminal conflict is about which EXECUTION STATE is authoritative,
    not about which of two values for one field is fuller.
    """
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for row in stored:
        key = _row_key(row)
        if key[0]:
            merged[key] = dict(row)
    for row in incoming:
        key = _row_key(row)
        if not key[0]:
            continue
        target = merged.get(key)
        if target is None:
            merged[key] = dict(row)
            continue
        # Rule 3: refuse a regression from settled to unsettled.
        if is_terminal(target.get("State", "")) and not is_terminal(row.get("State", "")):
            continue
        for field, value in row.items():
            if (value or "").strip():
                target[field] = value
    return sorted(
        merged.values(),
        key=lambda r: (
            int(r["MainJobID"]) if str(r.get("MainJobID", "")).isdigit() else 0,
            str(r.get("StepKind", "")),
            str(r.get("Submit", "")),
        ),
    )


def string_width_for(values: list[str]) -> int:
    """The fixed width a string column takes: the longest value, with headroom.

    Sized PER COLUMN from the data rather than from one global constant, and that is a
    correction to a measurement rather than a preference. The constant this replaces was 64,
    justified as covering "every observed value with ~3x headroom" -- a claim measured on
    `JobName` (36) and `State` (19) and never checked against `TRESUsageInTot`, which is the
    widest column in a real store at **105** characters. A single global width is a number
    that must be re-verified against every column of every future capture, and the failure
    mode when it is not is a raise on real data at capture time.

    Per-column sizing keeps D5 exactly as ratified -- fixed-width, never object, one chunk
    per variable -- while removing the constant that has to be maintained. The floor keeps
    narrow columns at a stable dtype so a one-character growth does not churn the schema.
    """
    longest = max((len(v) for v in values), default=0)
    if longest <= STRING_WIDTH_FLOOR:
        return STRING_WIDTH_FLOOR
    quanta = -(-longest // _STRING_WIDTH_QUANTUM)  # ceil-divide
    return quanta * _STRING_WIDTH_QUANTUM


def _encode_string_column(name: str, values: list[str], width: int | None = None):
    """Fixed-width `<U{width}` array, refusing a silent truncation (D5).

    numpy truncates an over-long value to the dtype width WITHOUT warning, which would make
    the store quietly lossy in exactly the columns whose verbatim content the reducers
    depend on -- `TRESUsageInTot` is where every per-step CPU and GPU figure lives, and a
    truncation there would silently drop the trailing keys. The width is derived from the
    data by default, so this raise is reachable only when a caller pins a width explicitly;
    it is retained because a pinned width is exactly the case that needs the guard.
    """
    import numpy as np

    width = string_width_for(values) if width is None else width
    longest = max((len(v) for v in values), default=0)
    if longest > width:
        raise StoreSchemaError(
            f"column {name!r} carries a value of {longest} characters, which exceeds the "
            f"store's <U{width} width; widen the width rather than truncating, because "
            f"numpy would truncate this silently."
        )
    return np.array(values, dtype=f"<U{width}")


def _is_numeric_column(values: list[str]) -> bool:
    """Whether every non-empty value parses as a float.

    Used only to decide dtype. A column that is numeric on this capture and not on the next
    is stored as a string on the next, which is correct -- the store follows the data rather
    than asserting a schema over it, and a column-agnostic store cannot do otherwise.
    """
    seen = False
    for v in values:
        s = (v or "").strip()
        if not s:
            continue
        seen = True
        try:
            float(s)
        except ValueError:
            return False
    return seen


def rows_to_dataset(
    rows: list[dict[str, str]],
    *,
    availability: dict[str, str] | None = None,
    derived_from: dict[str, tuple[str, ...]] | None = None,
    schema_version: int = 1,
) -> xr.Dataset:
    """Build the store Dataset from row dicts. Column-agnostic by construction.

    Every column present in ANY row becomes a variable; no column list is consulted, so a
    derived column added by a reducer's owner needs no edit here. Numeric columns widen to
    `float64` with NaN for absent (D5) rather than taking a sentinel -- a sentinel integer
    is the same conflation of "not measured" with a measured value that eliminated CSV.
    """
    import numpy as np
    import xarray as xr

    availability = dict(availability or {})
    derived_from = {k: tuple(v) for k, v in (derived_from or {}).items()}
    columns: list[str] = []
    for row in rows:
        for col in row:
            if col not in columns:
                columns.append(col)
    columns.sort()

    n = len(rows)
    data_vars: dict[str, Any] = {}
    for col in columns:
        raw = [str(row.get(col, "") or "") for row in rows]
        inputs = derived_from.get(col)
        if inputs is not None:
            # PROPAGATED, never hand-stamped. A caller-supplied availability for a column
            # that declares `derived_from` is refused rather than silently ignored: the two
            # would disagree exactly when the hand-stamp is wrong, which is the case the
            # propagation exists to catch.
            if col in availability:
                raise StoreSchemaError(
                    f"column {col!r} declares derived_from={list(inputs)!r}, so its "
                    f"availability is PROPAGATED from those inputs and must not also be "
                    f"supplied by hand (got {availability[col]!r})."
                )
            missing = [f for f in inputs if f not in columns and f not in availability]
            avail = derive_availability([availability.get(f, "available") for f in inputs])
            if missing:
                # An input the store does not carry cannot be `available` by omission.
                avail = "not_derivable"
        else:
            avail = availability.get(col, "available")
        validate_availability(col, avail)
        if col not in _IDENTIFIER_COLUMNS and _is_numeric_column(raw):
            arr = np.array(
                [float(v.strip()) if v.strip() else np.nan for v in raw],
                dtype="float64",
            )
        else:
            arr = _encode_string_column(col, raw)
        var = xr.DataArray(arr, dims=("row",))
        var.attrs["provenance_class"] = provenance_class(col)
        var.attrs["availability"] = avail
        if inputs is not None:
            # Recorded so the propagation is auditable from the store itself rather than
            # only from the code that produced it.
            var.attrs["derived_from"] = list(inputs)
        data_vars[col] = var

    ds = xr.Dataset(data_vars, coords={"row": np.arange(n)})
    # A store that widens its column set over time must say which schema it is. Measured on
    # this corpus: an 11-column generation and an 18-column generation of the predecessor
    # artifact were live on disk simultaneously, with nothing in either file recording which
    # it was -- so the same report rendered from two bundles had a column with a numerator
    # in one and no such column in the other. That is a schema-level absence, a fourth kind
    # no per-column attr can express, because the column is not there to carry one.
    ds.attrs["schema_version"] = schema_version
    ds.attrs["key_fields"] = list(KEY_FIELDS)
    ds.attrs["store_invariant"] = (
        "Every field a classification might key on is retained per row and classified at "
        "read time, never collapsed at write time."
    )
    return ds


def store_path(analysis_dir: Path) -> Path:
    return Path(analysis_dir).joinpath(*_EFF_RELDIR, STORE_DIRNAME)


def read_rows(analysis_dir: Path) -> list[dict[str, str]]:
    """Every stored row as a dict of strings, or [] when the store is absent.

    Graceful-absent by contract, mirroring `_load_job_recovery`: an analysis whose store has
    not been written yields [] and the caller renders em-dashes rather than failing.
    """
    path = store_path(analysis_dir)
    if not path.exists():
        return []
    import numpy as np
    import xarray as xr

    with xr.open_dataset(path, engine="zarr", chunks="auto", consolidated=False) as ds:
        ds = ds.load()
    out: list[dict[str, str]] = []
    n = ds.sizes.get("row", 0)
    for i in range(n):
        row: dict[str, str] = {}
        for col in ds.data_vars:
            value = ds[col].values[i]
            if isinstance(value, float | np.floating):
                # NOT `%g`: it renders to 6 significant digits, so a value that survived
                # storage as float64 is narrowed on the way out. `format_float_positional`
                # with `trim='-'` round-trips an integral float as its integer spelling and
                # a fractional one at full precision.
                row[str(col)] = "" if np.isnan(value) else np.format_float_positional(value, trim="-")
            else:
                row[str(col)] = str(value)
        out.append(row)
    return out


def write_store(
    analysis_dir: Path,
    rows: list[dict[str, str]],
    *,
    availability: dict[str, str] | None = None,
    derived_from: dict[str, tuple[str, ...]] | None = None,
) -> Path | None:
    """Merge `rows` into the store and rewrite it atomically (D3). Returns the path.

    Staged through a `.tmp` sibling and `os.replace`d, so a crash mid-write leaves the
    PREVIOUS store intact. That is strictly better than the artifact this replaces, where a
    crash inside `write_text` left a truncated CSV that `csv.DictReader` parses without
    error into a short store -- a silent partial that reads as a complete one.

    Merge-then-rewrite rather than an in-place update: a full rewrite measures ~203 ms at
    this grain, which is below any threshold at which correctness should be traded for
    speed, and the two in-place alternatives each cost real correctness (a positional region
    write needs a stable key-to-offset map, making row order load-bearing; an append-only
    log puts a de-duplication obligation on every reader, and a reader that forgets it
    double-counts a summed figure).
    """
    path = store_path(analysis_dir)
    merged = merge_rows(read_rows(analysis_dir), rows)
    if not merged:
        return None
    ds = rows_to_dataset(merged, availability=availability, derived_from=derived_from)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(tmp, mode="w", consolidated=False, zarr_format=ZARR_FORMAT)
    if path.exists():
        shutil.rmtree(path)
    os.replace(tmp, path)
    return path


def load_for_renderer(
    analysis_dir: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, str]]]]:
    """The store, reshaped into exactly what the efficiency renderer consumes today.

    Returns `(merged, recovery)`:

    - `merged` -- `{step JobID: row}`, the step-grained population `_aggregate_jobs` groups.
    - `recovery` -- `{MainJobID: {StepKind: row}}`, the shape `_load_job_recovery` returns.

    THIS IS THE CUTOVER SEAM, and it is deliberately additive. The renderer today joins
    three sources at render time; this function serves both of its inputs from the ONE
    store, so the cutover is a call-site swap rather than a rewrite of
    `_build_slurm_efficiency_html`. It is provided rather than applied because the derived
    COLUMN SET is under concurrent ruling and lands in that same function -- swapping the
    call site now would collide with work that has not been ruled on, and the store was
    built column-agnostic precisely so the two can land in either order.

    Note the one shape difference a cutover must handle, because it is a REAL improvement
    rather than an incompatibility: `recovery`'s inner dict is keyed on `StepKind`, which
    admits at most one row per `(MainJobID, StepKind)`. A requeued job legitimately has two
    `job`-kind rows, so this collapses them by taking the GREATEST `Submit` -- a stated
    choice rather than a file-order accident, which is what `_load_job_recovery` does today.
    The full row set stays reachable via `read_rows`, so a reducer that wants both instances
    has them; only this compatibility view narrows.
    """
    rows = read_rows(analysis_dir)
    merged: dict[str, dict[str, str]] = {}
    recovery: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        job_id = (row.get("JobID") or "").strip()
        if not job_id:
            continue
        merged[job_id] = row
        main = (row.get("MainJobID") or job_id.split(".", 1)[0]).strip()
        kind = (row.get("StepKind") or "").strip()
        if not (main and kind):
            continue
        prior = recovery.setdefault(main, {}).get(kind)
        if prior is None or (row.get("Submit") or "") >= (prior.get("Submit") or ""):
            recovery[main][kind] = row
    return merged, recovery


def consolidate(
    analysis_dir: Path,
    *,
    recovery_rows: list[dict[str, str]] | None = None,
    availability: dict[str, str] | None = None,
) -> Path | None:
    """Fold every SLURM-efficiency source into the ONE store, then write it.

    The three sources this replaces, and what each contributes:

    - `job_level_recovery.csv` (or `recovery_rows` passed directly) -- the sacct-derived
      per-step rows, which are the store's spine.
    - the executor plugin's `slurm_efficiency_report_*.csv` -- step rows the plugin kept,
      merged in for the columns sacct does not carry. Merged BY KEY onto the spine, never
      appended: appending would put each solver step in the store twice and double any
      summed figure computed over it.
    - `_status/_job_index.json` -- the job-to-rule label, joined onto every row of a job.

    Sources absent from disk contribute nothing and are not an error; a store built from
    the spine alone is a correct store with fewer columns.
    """
    from hhemt import slurm_job_recovery as _sjr

    analysis_dir = Path(analysis_dir)
    rows: list[dict[str, str]] = []

    if recovery_rows is None:
        rec_path = analysis_dir.joinpath(*_EFF_RELDIR, _sjr.RECOVERY_FILENAME)
        if rec_path.is_file():
            import csv as _csv
            import io as _io

            rows = list(_csv.DictReader(_io.StringIO(rec_path.read_text())))
    else:
        rows = [dict(r) for r in recovery_rows]

    by_key = {_row_key(r): r for r in rows if _row_key(r)[0]}

    # Plugin CSV columns, merged onto the spine by step JobID. The plugin does not carry
    # `Submit`, so its rows key on JobID alone and are applied to every stored instance of
    # that id -- which is correct for the columns it supplies (they are properties of the
    # step, not of the requeue instance) and is why they are merged rather than keyed.
    plugin_by_jobid: dict[str, dict[str, str]] = {}
    eff_dir = analysis_dir.joinpath(*_EFF_RELDIR)
    if eff_dir.is_dir():
        import csv as _csv
        import io as _io

        for match in sorted(eff_dir.glob(_sjr._EFF_GLOB)):
            files = [match] if match.is_file() else sorted(match.glob(_sjr._EFF_INNER_GLOB))
            for f in files:
                if not f.is_file():
                    continue
                try:
                    text = f.read_text()
                except OSError:
                    continue
                for prow in _csv.DictReader(_io.StringIO(text)):
                    jid = (prow.get("JobID") or "").strip()
                    if jid:
                        plugin_by_jobid[jid] = prow

    job_index: dict[str, str] = {}
    idx_path = analysis_dir / "_status" / "_job_index.json"
    if idx_path.is_file():
        try:
            loaded = json.loads(idx_path.read_text())
            if isinstance(loaded, dict):
                job_index = {str(k): str(v) for k, v in loaded.items()}
        except (OSError, ValueError):
            job_index = {}

    for key, row in by_key.items():
        jid = key[0]
        plugin = plugin_by_jobid.get(jid)
        if plugin:
            for col, value in plugin.items():
                # Never let the plugin overwrite a field the spine already carries: sacct is
                # the authority on its own fields, and the plugin's parse drops the rows
                # that make several of them meaningful.
                if col not in row or not (row.get(col) or "").strip():
                    if (value or "").strip():
                        row[col] = value
        rule = job_index.get(str(row.get("MainJobID", "")).strip())
        if rule and not (row.get("RuleName") or "").strip():
            row["RuleName"] = rule

    return write_store(analysis_dir, list(by_key.values()), availability=availability)
