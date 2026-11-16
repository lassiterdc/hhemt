"""The external-boundary layer, the outfall decisions of the coupled model, and the SWMM series files.

Three things the toolkit must know about a coupled run's outer edge live here, each at one site:

1. **The boundary layer.** A line layer whose records each describe one straight TRITON boundary segment: a
   ``water_level`` segment (a level-versus-time file), or a ``free_outflow``, ``normal_slope`` or
   ``froude_number`` segment. A ``water_level`` record also names the SWMM outfalls that sit beside its line.
2. **The outfall decisions.** Every outfall of the SWMM model is converted into a coupled junction, kept as an
   outfall, or matched to a ``water_level`` record. Nothing defaults, so a model's outfall is never silently
   coupled or silently left to drain.
3. **The series files.** No SWMM model the toolkit writes holds time-series rows. Each series the base model
   defines inline is written once to a file and every derived model points to it.

The functions are pure text and data in, problems or values out, so each check runs without a solver. A check
returns :class:`Problem` records, one per problem kind with every offender named, and the preflight pass turns
each into a ``ValidationIssue``.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from hhemt.exceptions import ProcessingError
from hhemt.swmm_runnable import (
    WATER_LEVEL_SERIES_PREFIX,
    _all,
    _fold,
    _is_data_row,
    _node_rows,
    _parse,
    _tokens,
)

# Boundary record types and the TRITON ``extbc`` type code each writes (``extbc.h``).
BOUNDARY_TYPES: tuple[str, ...] = ("water_level", "free_outflow", "normal_slope", "froude_number")
EXTBC_TYPE_CODE: dict[str, int] = {"free_outflow": 0, "water_level": 1, "normal_slope": 2, "froude_number": 3}
_VALUE_TYPES = ("normal_slope", "froude_number")

# SWMM keeps a series' date at its engine default until a row names one (``input.c``).
_ENGINE_DEFAULT_START = datetime(2004, 1, 1)
_DATE_FORMATS = ("%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d")

SERIES_DIR_NAME = "swmm_model_series"
COMPLETE_MARKER = "COMPLETE"
_VARIABLE_OUTFALL_KINDS = ("TIMESERIES", "TIDAL")


@dataclass(frozen=True)
class Problem:
    """One problem kind with every offender named, in the fields a ``ValidationIssue`` carries."""

    code: str
    field: str
    message: str
    current_value: Any = None
    fix_hint: str | None = None
    config_line: str | None = None
    layer_edit: str | None = None


@dataclass(frozen=True)
class BoundaryRecord:
    """One record of the external-boundary layer, already reprojected to the system's CRS."""

    index: int
    kind: str
    value: float | None
    outfalls: tuple[str, ...]
    vertices: tuple[tuple[float, float], ...]
    outfall_problem: str | None = None

    @property
    def series_name(self) -> str:
        return f"{WATER_LEVEL_SERIES_PREFIX}{self.index}"

    @property
    def file_stem(self) -> str:
        return f"{self.index:03d}"


# ----------------------------------------------------------------------------------------------------------------
# The layer
# ----------------------------------------------------------------------------------------------------------------


def parse_outfall_names(value: Any) -> tuple[list[str], str | None]:
    """The outfall names a layer cell holds, whitespace-separated, and the reason it is unusable or ``None``.

    A SWMM name holds no whitespace unless it is quoted, and a quoted name is refused, so a name never
    contains a separator.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return [], None
    text = str(value).strip()
    if '"' in text:
        return [], f"{text!r} holds a quote, and a quoted SWMM name is not accepted"
    return text.split(), None


def read_boundary_records(
    layer_path: Path,
    horizontal_epsg: int,
    type_attribute: str,
    value_attribute: str | None,
    outfall_attribute: str | None,
) -> tuple[list[BoundaryRecord], list[Problem]]:
    """Read the layer, reproject it to the system CRS, and parse each record (V1, V2, V3, V17, V18).

    Problems of kind V1 and V2 stop the read, because no record can be built without the layer or its columns;
    the record-level problems are collected over every record.
    """
    import geopandas as gpd

    field_ = "analysis.external_boundary_gis"
    if not Path(layer_path).is_file():
        return [], [Problem("V1", field_, f"the layer {layer_path} does not exist", str(layer_path))]
    try:
        gdf = gpd.read_file(layer_path)
    except Exception as exc:  # noqa: BLE001 - any reader failure is the layer being unreadable
        return [], [Problem("V1", field_, f"the layer {layer_path} cannot be read: {exc}", str(layer_path))]
    if len(gdf) == 0:
        return [], [Problem("V1", field_, f"the layer {layer_path} has no record", str(layer_path))]
    if gdf.crs is None:
        return [], [
            Problem(
                "V1",
                field_,
                f"the layer {layer_path} has no CRS, so its coordinates cannot be placed on the DEM",
                str(layer_path),
                fix_hint="Assign the layer's CRS in your GIS and save it again",
            )
        ]
    wanted = {
        "analysis.external_boundary_type_attribute": type_attribute,
        "analysis.external_boundary_value_attribute": value_attribute,
        "analysis.water_level_boundary_outfall_attribute": outfall_attribute,
    }
    missing = {f: c for f, c in wanted.items() if c is not None and c not in gdf.columns}
    if missing:
        return [], [
            Problem(
                "V2",
                f,
                f"{c!r} is not a column of the layer; its columns are {[c2 for c2 in gdf.columns if c2 != 'geometry']}",
                c,
                fix_hint=f"Set {f.split('.')[1]} to a column the layer has",
            )
            for f, c in missing.items()
        ]
    gdf = gdf.to_crs(epsg=int(horizontal_epsg))
    records: list[BoundaryRecord] = []
    bad_geometry: list[int] = []
    for k, (_, row) in enumerate(gdf.iterrows()):
        geom = row.geometry
        if geom is None or geom.geom_type != "LineString" or len(geom.coords) < 2:
            bad_geometry.append(k)
            continue
        names, reason = parse_outfall_names(row[outfall_attribute]) if outfall_attribute else ([], None)
        raw_value = row[value_attribute] if value_attribute else None
        try:
            value = (
                None
                if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value))
                else float(raw_value)
            )
        except (TypeError, ValueError):
            value = math.nan
        records.append(
            BoundaryRecord(
                index=k,
                kind=str(row[type_attribute]).strip(),
                value=value,
                outfalls=tuple(names),
                vertices=tuple((float(x), float(y)) for x, y, *_ in geom.coords),
                outfall_problem=reason,
            )
        )
    problems: list[Problem] = []
    if bad_geometry:
        problems.append(
            Problem(
                "V1",
                field_,
                f"record(s) {bad_geometry} of the layer are not a LineString of two or more vertices",
                str(layer_path),
                fix_hint="Each record must be one straight line along a DEM edge",
            )
        )
    return records, problems


def record_problems(
    records: Sequence[BoundaryRecord],
    outfall_attribute: str | None,
    value_attribute: str | None,
    layer_name: str,
    type_attribute: str | None = None,
) -> list[Problem]:
    """V3, V17 and V18: each record's type, value column and outfall column agree with its type."""
    type_column = f"column {type_attribute}" if type_attribute else "the type column"
    value_column = f"column {value_attribute}" if value_attribute else "the value column"
    problems: list[Problem] = []
    bad_type = [(r.index, r.kind) for r in records if r.kind not in BOUNDARY_TYPES]
    if bad_type:
        problems.append(
            Problem(
                "V17",
                "analysis.external_boundary_type_attribute",
                f"record type(s) {bad_type} are not one of {list(BOUNDARY_TYPES)}",
                bad_type,
                layer_edit=f"{type_column} of {layer_name}, records {[i for i, _ in bad_type]}: "
                f"one of {list(BOUNDARY_TYPES)}",
            )
        )
    no_value = [
        r.index
        for r in records
        if r.kind in _VALUE_TYPES and not (r.value is not None and math.isfinite(r.value) and r.value > 0)
    ]
    stray_value = [
        r.index
        for r in records
        if r.kind in BOUNDARY_TYPES and r.kind not in _VALUE_TYPES and r.value is not None and math.isfinite(r.value)
    ]
    if no_value:
        problems.append(
            Problem(
                "V18",
                "analysis.external_boundary_value_attribute",
                f"record(s) {no_value} are normal_slope or froude_number and hold no finite number above 0 in the value column",  # noqa: E501
                no_value,
                layer_edit=f"{value_column} of {layer_name}, records {no_value}: a number above 0",
            )
        )
    if stray_value:
        problems.append(
            Problem(
                "V18",
                "analysis.external_boundary_value_attribute",
                f"record(s) {stray_value} are not normal_slope or froude_number and hold a value",
                stray_value,
                layer_edit=f"{value_column} of {layer_name}, records {stray_value}: leave empty",
            )
        )
    stray_outfall = [r.index for r in records if r.kind != "water_level" and r.outfalls]
    if stray_outfall:
        problems.append(
            Problem(
                "V18",
                "analysis.water_level_boundary_outfall_attribute",
                f"record(s) {stray_outfall} are not water_level and name outfalls",
                stray_outfall,
                layer_edit=f"column {outfall_attribute} of {layer_name}, records {stray_outfall}: leave empty",
            )
        )
    empty = [r.index for r in records if r.kind == "water_level" and not r.outfalls]
    unusable = {r.index: r.outfall_problem for r in records if r.outfall_problem}
    if empty or unusable:
        problems.append(
            Problem(
                "V3",
                "analysis.water_level_boundary_outfall_attribute",
                f"water_level record(s) {empty} name no outfall" + (f"; unusable cells {unusable}" if unusable else ""),
                empty,
                layer_edit=f"column {outfall_attribute} of {layer_name}, records {empty}: the SWMM outfall name(s)",
            )
        )
    return problems


# ----------------------------------------------------------------------------------------------------------------
# What the SWMM model says
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OutfallRow:
    name: str
    invert: str
    kind: str
    stage: str | None


def model_outfalls(sections) -> dict[str, OutfallRow]:
    """The ``[OUTFALLS]`` rows of a parsed model, keyed by SWMM's case-folded name."""
    out: dict[str, OutfallRow] = {}
    for toks in _node_rows(sections, "OUTFALLS"):
        if len(toks) >= 3:
            stage = toks[3] if len(toks) > 3 and _fold(toks[2]) in ("TIMESERIES", "TIDAL", "FIXED") else None
            out[_fold(toks[0])] = OutfallRow(toks[0], toks[1], _fold(toks[2]), stage)
    return out


def model_coordinates(sections) -> dict[str, tuple[float, float]]:
    """The ``[COORDINATES]`` rows of a parsed model, keyed by case-folded node name."""
    out: dict[str, tuple[float, float]] = {}
    for toks in _node_rows(sections, "COORDINATES"):
        if len(toks) >= 3:
            try:
                out[_fold(toks[0])] = (float(toks[1]), float(toks[2]))
            except ValueError:
                continue
    return out


def _parse_date(token: str) -> datetime | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            continue
    return None


def _time_hours(token: str) -> float | None:
    if ":" in token:
        parts = token.split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        return sum(n / 60**i for i, n in enumerate(nums))
    try:
        return float(token)
    except ValueError:
        return None


def resolve_series_rows(rows: Iterable[Sequence[str]]) -> list[tuple[datetime, float]]:
    """The ``(abscissa, value)`` pairs of a series, resolved as SWMM does (``input.c``, ``table.c``).

    A row that begins with a date sets the series' current date, which is the engine default before any
    dated row. Each ``time value`` pair then lands at the current date plus the time, read as decimal hours
    or ``hh:mm``; the value is parsed as a float. Names, inline or ``FILE`` form, repeated or carried dates and
    ``1.5`` against ``01:30`` therefore do not change the result.
    """
    current = _ENGINE_DEFAULT_START
    pairs: list[tuple[datetime, float]] = []
    for toks in rows:
        toks = list(toks)
        if toks and (d := _parse_date(toks[0])) is not None:
            current, toks = d, toks[1:]
        for t, v in zip(toks[0::2], toks[1::2], strict=False):
            hours = _time_hours(t)
            try:
                value = float(v)
            except ValueError:
                continue
            if hours is not None:
                pairs.append((current + timedelta(hours=hours), value))
    return pairs


@dataclass(frozen=True)
class ModelSeries:
    """One ``[TIMESERIES]`` series of the base model: inline rows, or the file a ``FILE`` row points to."""

    name: str
    rows: tuple[tuple[str, ...], ...]
    file: Path | None


def model_series(sections, model_dir: Path) -> dict[str, ModelSeries]:
    """The series of a parsed model in order of first appearance, keyed by case-folded name.

    A relative ``FILE`` name is joined to the directory of the base model, as SWMM does when it opens the
    model (``swmm5.c`` ``addAbsolutePath``).
    """
    grouped: dict[str, tuple[str, list[tuple[str, ...]]]] = {}
    for section in _all(sections, "TIMESERIES"):
        for line in section.lines:
            if not _is_data_row(line):
                continue
            toks = _tokens(line)
            if len(toks) < 2:
                continue
            grouped.setdefault(_fold(toks[0]), (toks[0], []))[1].append(tuple(toks[1:]))
    out: dict[str, ModelSeries] = {}
    for key, (name, rows) in grouped.items():
        file: Path | None = None
        if len(rows) == 1 and len(rows[0]) >= 2 and _fold(rows[0][0]) == "FILE":
            file = Path(rows[0][1])
            if not file.is_absolute():
                file = Path(model_dir) / file
        out[key] = ModelSeries(name, tuple(rows), file)
    return out


def series_pairs(series: ModelSeries) -> list[tuple[datetime, float]]:
    """The resolved ``(abscissa, value)`` pairs of a series, reading its file when it is a ``FILE`` series."""
    if series.file is None:
        return resolve_series_rows(series.rows)
    rows: list[list[str]] = []
    with open(series.file, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip() and not line.lstrip().startswith(";"):
                rows.append(_tokens(line))
    return resolve_series_rows(rows)


def missing_series_files(catalog: Mapping[str, ModelSeries]) -> list[tuple[str, Path]]:
    """V21: ``(series, path)`` of every ``FILE`` series whose file does not exist."""
    return [(s.name, s.file) for s in catalog.values() if s.file is not None and not s.file.is_file()]


def reserved_prefix_names(sections, catalog: Mapping[str, ModelSeries]) -> list[str]:
    """V11: every node name and series name of the model that begins with the toolkit's reserved prefix ``hhemt_``."""
    names = [s.name for s in catalog.values()]
    for sec in ("JUNCTIONS", "OUTFALLS", "STORAGE", "DIVIDERS"):
        names += [t[0] for t in _node_rows(sections, sec) if t]
    return sorted({n for n in names if _fold(n).startswith("HHEMT_")})


@lru_cache(maxsize=8)
def _cached_records(
    layer: str, mtime_ns: int, epsg: int, type_col: str, value_col: str | None, outfall_col: str | None
):
    records, problems = read_boundary_records(Path(layer), epsg, type_col, value_col, outfall_col)
    return tuple(records), tuple(problems)


def boundary_records_for(cfg_analysis, cfg_system) -> tuple[tuple[BoundaryRecord, ...], tuple[Problem, ...]]:
    """The records of the analysis' boundary layer, cached on the layer's path and mtime.

    Empty when the external boundaries are off. The layer is reprojected to the system's horizontal CRS.
    """
    if not getattr(cfg_analysis, "toggle_external_boundaries", False) or cfg_analysis.external_boundary_gis is None:
        return (), ()
    layer = Path(cfg_analysis.external_boundary_gis)
    mtime = layer.stat().st_mtime_ns if layer.exists() else 0
    return _cached_records(
        str(layer),
        mtime,
        int(cfg_system.crs.horizontal_epsg),
        cfg_analysis.external_boundary_type_attribute,
        cfg_analysis.external_boundary_value_attribute,
        cfg_analysis.water_level_boundary_outfall_attribute,
    )


def matched_outfall_names(records: Iterable[BoundaryRecord]) -> tuple[str, ...]:
    """The outfalls the ``water_level`` records name, in record order."""
    return tuple(n for r in records if r.kind == "water_level" for n in r.outfalls)


# ----------------------------------------------------------------------------------------------------------------
# One extraction per base model (no SWMM model the toolkit writes holds time-series data)
# ----------------------------------------------------------------------------------------------------------------


def extracted_series_dir(system_dir: Path, model_path: Path, model_bytes: bytes) -> Path:
    sha12 = hashlib.sha256(model_bytes).hexdigest()[:12]
    return Path(system_dir) / SERIES_DIR_NAME / sha12


def _extracted_text(series: ModelSeries, model_path: Path) -> str:
    """The extracted file of an inline series: a ``;;`` header, then ``{date}\\t{time}\\t{value}`` per pair.

    The header's first three whitespace tokens are ``;;``, ``hhemt`` and ``extracted`` whatever the name's
    length, because SWMM's file parser reads three tokens into 50-character buffers before it tests for a
    comment (``table_parseFileLine``).
    """
    lines = [f";; hhemt extracted series {series.name} from {model_path}"]
    current = _ENGINE_DEFAULT_START
    for toks in series.rows:
        toks_l = list(toks)
        if toks_l and (d := _parse_date(toks_l[0])) is not None:
            current, toks_l = d, toks_l[1:]
        for t, v in zip(toks_l[0::2], toks_l[1::2], strict=False):
            lines.append(f"{current.strftime('%m/%d/%Y')}\t{t}\t{v}")
    return "\n".join(lines) + "\n"


def extract_model_series(model_path: Path, system_dir: Path, model_bytes: bytes | None = None) -> dict[str, str]:
    """Write each inline series of the base model once, and return every series' pointer path by folded name.

    The directory ``{system_dir}/swmm_model_series/{model_sha12}/`` is written to a temporary sibling,
    completed with a ``COMPLETE`` marker that lists every file and its sha256, and moved into place; a reader
    treats a directory without the marker as absent and rebuilds it. An inline series becomes
    ``{index:03d}.dat`` by order of first appearance among inline series. A ``FILE`` series keeps its file, and
    its pointer is the absolute path SWMM would have resolved from the base model's directory.
    """
    model_path = Path(model_path)
    data = model_bytes if model_bytes is not None else model_path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    catalog = model_series(_parse(text), model_path.parent)
    target = extracted_series_dir(system_dir, model_path, data)
    inline = [s for s in catalog.values() if s.file is None]
    if inline and not (target / COMPLETE_MARKER).is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        listing = []
        for i, series in enumerate(inline):
            body = _extracted_text(series, model_path).encode("utf-8")
            (tmp / f"{i:03d}.dat").write_bytes(body)
            listing.append(f"{i:03d}.dat {hashlib.sha256(body).hexdigest()}")
        (tmp / COMPLETE_MARKER).write_text("\n".join(listing) + "\n", encoding="utf-8")
        shutil.rmtree(target, ignore_errors=True)
        try:
            os.replace(tmp, target)
        except OSError:
            # A competing preparation completed the directory between the clear and the rename. Its marker says it
            # is whole, so keep it and drop this copy; without the marker the failure is real.
            if not (target / COMPLETE_MARKER).is_file():
                raise
            shutil.rmtree(tmp, ignore_errors=True)
    pointers: dict[str, str] = {}
    inline_index = {s.name: i for i, s in enumerate(inline)}
    for key, series in catalog.items():
        pointers[key] = (
            str(series.file) if series.file is not None else str(target / f"{inline_index[series.name]:03d}.dat")
        )
    return pointers


# ----------------------------------------------------------------------------------------------------------------
# The outfall decisions
# ----------------------------------------------------------------------------------------------------------------


def _name_list(names: Iterable[str]) -> str:
    return ", ".join(names)


def outfall_decision_problems(
    outfalls: Mapping[str, OutfallRow],
    matched: Sequence[str],
    to_couple: Sequence[str],
    to_keep: Sequence[str],
) -> list[Problem]:
    """V6, V9 and V13: every outfall of the coupled model is matched, coupled or kept, in exactly one place.

    ``matched`` are the outfalls the ``water_level`` records name, ``to_couple`` is ``swmm_outfalls_to_couple`` of
    the system config and ``to_keep`` is ``swmm_outfalls_to_keep`` of the analysis config.
    """
    problems: list[Problem] = []
    matched_f, couple_f, keep_f = ({_fold(n) for n in group} for group in (matched, to_couple, to_keep))
    both = sorted(n for n in matched if _fold(n) in couple_f)
    if both:
        problems.append(
            Problem(
                "V6",
                "system.swmm_outfalls_to_couple",
                f"outfall(s) {_name_list(both)} are named by a water_level record and also listed in swmm_outfalls_to_couple; "  # noqa: E501
                "a record converts its outfalls, so a name is stated once",
                both,
                fix_hint="Remove the name from swmm_outfalls_to_couple, or from the record",
            )
        )
    variable_listed = sorted(o.name for k, o in outfalls.items() if k in couple_f and o.kind in _VARIABLE_OUTFALL_KINDS)
    if variable_listed:
        problems.append(
            Problem(
                "V13",
                "system.swmm_outfalls_to_couple",
                f"outfall(s) {_name_list(variable_listed)} have a variable water level ({', '.join(sorted({outfalls[_fold(n)].kind for n in variable_listed}))}) "  # noqa: E501
                "and cannot be listed in swmm_outfalls_to_couple",
                variable_listed,
                fix_hint="Name each in the outfall column of an external-boundary water_level record, which converts it, "  # noqa: E501
                "or list it in swmm_outfalls_to_keep to keep it an outfall that reads the model's series",
                config_line=f"swmm_outfalls_to_keep: [{_name_list(variable_listed)}]",
            )
        )
    unknown_keep = sorted(n for n in to_keep if _fold(n) not in outfalls)
    if unknown_keep:
        problems.append(
            Problem(
                "V9",
                "analysis.swmm_outfalls_to_keep",
                f"{unknown_keep} {'is' if len(unknown_keep) == 1 else 'are'} not an [OUTFALLS] row of SWMM_model",
                list(to_keep),
                fix_hint="List only names that appear in the [OUTFALLS] section of SWMM_model",
            )
        )
    kept_and_other = sorted(n for n in to_keep if _fold(n) in matched_f | couple_f)
    if kept_and_other:
        problems.append(
            Problem(
                "V9",
                "analysis.swmm_outfalls_to_keep",
                f"outfall(s) {_name_list(kept_and_other)} are in swmm_outfalls_to_keep and also matched or coupled; "
                "an outfall is stated in exactly one place",
                kept_and_other,
                fix_hint="Remove the name from one of the places",
            )
        )
    undecided = [o for k, o in outfalls.items() if k not in matched_f | couple_f | keep_f]
    variable = [o for o in undecided if o.kind in _VARIABLE_OUTFALL_KINDS]
    plain = [o for o in undecided if o.kind not in _VARIABLE_OUTFALL_KINDS]
    if variable:
        listing = ", ".join(f"{o.name} ({o.kind}{' ' + o.stage if o.stage else ''})" for o in variable)
        problems.append(
            Problem(
                "V9",
                "analysis.swmm_outfalls_to_keep",
                f"{len(variable)} outfalls of SWMM_model have a variable water level and no decision: {listing}.",
                list(to_keep),
                fix_hint="Name each in the water_level_boundary_outfall_attribute column of the external-boundary record that "  # noqa: E501
                "replaces it, so it becomes a coupled node and the water level is applied on the TRITON surface; "
                "or keep it as an outfall that reads the model's series.",
                config_line=f"swmm_outfalls_to_keep: [{_name_list(o.name for o in variable)}]",
            )
        )
    if plain:
        listing = ", ".join(f"{o.name} ({o.kind})" for o in plain)
        problems.append(
            Problem(
                "V9",
                "system.swmm_outfalls_to_couple",
                f"{len(plain)} outfalls of SWMM_model are neither converted nor kept: {listing}.",
                list(to_couple),
                fix_hint="List each in swmm_outfalls_to_couple (system config) to turn it into a node that exchanges "
                "water with the TRITON surface, or in swmm_outfalls_to_keep (analysis config) to keep it draining as the model says.",  # noqa: E501
                config_line=f"swmm_outfalls_to_couple: [{_name_list(o.name for o in plain)}]\n"
                f"# or, to keep them as outfalls:\nswmm_outfalls_to_keep: [{_name_list(o.name for o in plain)}]",
            )
        )
    return problems


def matched_outfall_problems(
    records: Sequence[BoundaryRecord],
    outfalls: Mapping[str, OutfallRow],
    coordinates: Mapping[str, tuple[float, float]],
    max_distance_m: float | None,
) -> tuple[list[Problem], dict[int, dict[str, float]]]:
    """V4, V5, V6 (two records), V19, V20: the outfalls the records name exist, are variable, are placed and are near.

    Returns the problems and the planar distance, in the system CRS, from each matched outfall to its record's
    line, keyed by record index (computed for every placed outfall whatever the maximum is).
    """
    from shapely.geometry import LineString, Point

    problems: list[Problem] = []
    distances: dict[int, dict[str, float]] = {}
    wl = [r for r in records if r.kind == "water_level"]
    absent, not_outfall_kind, no_coord, far = [], [], [], []
    seen: dict[str, int] = {}
    twice: list[str] = []
    for r in wl:
        for n in r.outfalls:
            key = _fold(n)
            if key in seen and seen[key] != r.index:
                twice.append(n)
            seen[key] = r.index
            if key not in outfalls:
                absent.append((r.index, n))
                continue
            if outfalls[key].kind != "TIMESERIES":
                not_outfall_kind.append((r.index, n, outfalls[key].kind))
            if key not in coordinates:
                no_coord.append((r.index, n))
                continue
            d = LineString(r.vertices).distance(Point(*coordinates[key]))
            distances.setdefault(r.index, {})[n] = d
            if max_distance_m is not None and d > max_distance_m:
                far.append((r.index, n, round(d, 3)))
    if absent:
        problems.append(
            Problem(
                "V4",
                "analysis.water_level_boundary_outfall_attribute",
                f"named outfall(s) {absent} are not [OUTFALLS] rows of SWMM_model; only outfalls can be matched to an external boundary",  # noqa: E501
                absent,
                fix_hint="Name an [OUTFALLS] row of SWMM_model (SWMM folds case); a junction, storage or divider cannot be matched",  # noqa: E501
            )
        )
    if not_outfall_kind:
        problems.append(
            Problem(
                "V5",
                "analysis.water_level_boundary_outfall_attribute",
                f"named outfall(s) {not_outfall_kind} are not TIMESERIES outfalls; a water_level record takes over an outfall that reads a series",  # noqa: E501
                not_outfall_kind,
                fix_hint="Name only TIMESERIES outfalls in the outfall column, or list the others in swmm_outfalls_to_couple",  # noqa: E501
            )
        )
    if twice:
        problems.append(
            Problem(
                "V6",
                "analysis.water_level_boundary_outfall_attribute",
                f"outfall(s) {sorted(set(twice))} are named by more than one record; each outfall is named by at most one record",  # noqa: E501
                sorted(set(twice)),
            )
        )
    if no_coord:
        problems.append(
            Problem(
                "V20",
                "system.SWMM_model",
                f"matched outfall(s) {no_coord} have no [COORDINATES] row, so their distance to the boundary line and their DEM cell are unknown",  # noqa: E501
                no_coord,
                fix_hint="Add a [COORDINATES] row for each matched outfall",
            )
        )
    if far:
        problems.append(
            Problem(
                "V19",
                "analysis.water_level_boundary_outfall_max_distance_m",
                f"matched outfall(s) {far} (record, outfall, distance in meters) are farther from their record's line than the stated maximum {max_distance_m} m",  # noqa: E501
                max_distance_m,
                fix_hint="Name the outfall beside the line, or raise water_level_boundary_outfall_max_distance_m",
            )
        )
    return problems, distances


def distance_lines(distances: Mapping[int, Mapping[str, float]]) -> list[str]:
    """One printed line per record: ``record 0: E143250 15.3 m, E147007 10.5 m``."""
    return [
        f"record {k}: " + ", ".join(f"{n} {d:.1f} m" for n, d in sorted(v.items()))
        for k, v in sorted(distances.items())
    ]


def series_identity_problems(
    records: Sequence[BoundaryRecord],
    catalog: Mapping[str, ModelSeries],
    outfalls: Mapping[str, OutfallRow],
    source: str | None,
    unequal_policy: str,
) -> list[Problem]:
    """V12: the outfalls one record names read identical resolved series, element for element."""
    problems: list[Problem] = []
    for r in records:
        if r.kind != "water_level" or len(r.outfalls) < 2:
            continue
        resolved = {}
        for n in r.outfalls:
            row = outfalls.get(_fold(n))
            series = catalog.get(_fold(row.stage)) if row is not None and row.stage else None
            if series is not None:
                resolved[n] = (series.name, series_pairs(series))
        names = list(resolved)
        for other in names[1:]:
            a, b = resolved[names[0]][1], resolved[other][1]
            if a == b:
                continue
            first = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
            if source == "weather_time_series":
                way = (
                    "put each outfall in its own record, or make the model's series equal. The averaging option does "
                    "not apply here, because under the weather source no run reads the outfalls' own series"
                )
            elif source == "swmm_model":
                way = (
                    "put each outfall in its own record, make the model's series equal, or set "
                    "water_level_boundary_unequal_outfall_series: average"
                )
            else:
                way = (
                    "put each outfall in its own record, or make the model's series equal. Under the swmm_model source "
                    "water_level_boundary_unequal_outfall_series: average is also available, and it does not apply "
                    "under the weather source"
                )
            if source == "swmm_model" and unequal_policy == "average":
                break
            problems.append(
                Problem(
                    "V12",
                    "analysis.water_level_boundary_outfall_attribute",
                    f"record {r.index} names outfalls {names[0]} (series {resolved[names[0]][0]}) and {other} "
                    f"(series {resolved[other][0]}) whose series differ from row {first}: "
                    f"{a[first] if first < len(a) else 'no row'} against {b[first] if first < len(b) else 'no row'}",
                    r.index,
                    fix_hint=way[0].upper() + way[1:] + ".",
                )
            )
            break
    return problems


def averaged_pairs(pair_lists: Sequence[Sequence[tuple[datetime, float]]]) -> list[tuple[datetime, float]]:
    """The mean of several series at each abscissa; the series must share their abscissas."""
    first = pair_lists[0]
    for other in pair_lists[1:]:
        if [t for t, _ in other] != [t for t, _ in first]:
            raise ProcessingError(
                operation="water_level_boundary_unequal_outfall_series",
                reason="average requires the outfalls' series to share their abscissas, and they do not",
            )
    return [(t, sum(p[i][1] for p in pair_lists) / len(pair_lists)) for i, (t, _) in enumerate(first)]


# ----------------------------------------------------------------------------------------------------------------
# Levels in meters
# ----------------------------------------------------------------------------------------------------------------


def to_meters(values, unit: str, offset: float):
    """Level values in ``unit``, as meters at the datum: multiplied by the unit's factor, then offset added."""
    from hhemt.triton_cfg import LEVEL_UNIT_FACTORS

    return values * LEVEL_UNIT_FACTORS[unit] + float(offset)


def variable_problems(
    outfall_variables: Mapping[str, str] | None,
    matched: Sequence[str],
    records: Sequence[BoundaryRecord],
    dataset_variables: Iterable[str] | None,
    units: Mapping[str, str] | None,
    offsets: Mapping[str, float] | None,
) -> list[Problem]:
    """V14, V15, V16, V22 and the unit and offset entries: the weather source's outfall-to-variable map."""
    from hhemt.triton_cfg import LEVEL_UNIT_FACTORS

    problems: list[Problem] = []
    mapping = dict(outfall_variables or {})
    keys = {_fold(k): k for k in mapping}
    matched_f = {_fold(n): n for n in matched}
    missing = sorted(n for k, n in matched_f.items() if k not in keys)
    extra = sorted(k for f, k in keys.items() if f not in matched_f)
    if missing or extra:
        lines = ", ".join(f"{n}: VARIABLE" for n in missing)
        problems.append(
            Problem(
                "V14",
                "analysis.water_level_boundary_outfall_variables",
                f"the map's keys must equal the outfalls the water_level records name; missing {missing}, not matched {extra}",  # noqa: E501
                mapping,
                config_line=f"water_level_boundary_outfall_variables: {{{lines}}}" if missing else None,
            )
        )
    if dataset_variables is not None:
        avail = set(dataset_variables)
        absent = sorted({v for v in mapping.values() if v not in avail})
        if absent:
            problems.append(
                Problem(
                    "V15",
                    "analysis.water_level_boundary_outfall_variables",
                    f"variable(s) {absent} are not data variables of the event dataset; it has {sorted(avail)}",
                    mapping,
                    fix_hint="Name a data variable of the weather time series",
                )
            )
    split = [
        r.index
        for r in records
        if r.kind == "water_level"
        and len({mapping.get(next((k for k in mapping if _fold(k) == _fold(n)), n)) for n in r.outfalls}) > 1
    ]
    if split:
        problems.append(
            Problem(
                "V16",
                "analysis.water_level_boundary_outfall_variables",
                f"record(s) {split} name outfalls that map to different variables; one record carries one variable",
                mapping,
                fix_hint="Put each outfall in its own record, one per variable",
            )
        )
    used = sorted(set(mapping.values()))
    bad_units = {v: (units or {}).get(v) for v in used if (units or {}).get(v) not in LEVEL_UNIT_FACTORS}
    if bad_units:
        problems.append(
            Problem(
                "V22",
                "analysis.water_level_boundary_variable_units",
                f"unit(s) {bad_units} are not one of {list(LEVEL_UNIT_FACTORS)}",
                units,
                config_line="water_level_boundary_variable_units: {" + ", ".join(f"{v}: m" for v in bad_units) + "}",
            )
        )
    no_offset = [v for v in used if (offsets or {}).get(v) is None]
    if no_offset:
        problems.append(
            Problem(
                "V22",
                "analysis.water_level_boundary_variable_datum_offsets",
                f"variable(s) {no_offset} have no datum offset, in meters, and there is no default",
                offsets,
                config_line="water_level_boundary_variable_datum_offsets: {"
                + ", ".join(f"{v}: 0" for v in no_offset)
                + "}",
            )
        )
    return problems


# ----------------------------------------------------------------------------------------------------------------
# Snapping and DEM depths
# ----------------------------------------------------------------------------------------------------------------


def snap_segments(
    records: Sequence[BoundaryRecord], rds_dem
) -> tuple[list[tuple[BoundaryRecord, tuple]], list[Problem]]:
    """Snap each record to one DEM edge by the existing rule, and refuse a cell two segments share (V10).

    Returns ``(record, (side, x1, y1, x2, y2))`` pairs. A segment with one cell, or two segments that share a
    cell, are refused: a shared corner cell would be written twice in one kernel launch, so TRITON's result would
    depend on thread order.
    """
    from hhemt.scenario import find_closest_dem_coord, infer_side

    snapped: list[tuple[BoundaryRecord, tuple]] = []
    cells_of: dict[int, set[tuple[float, float]]] = {}
    one_cell: list[int] = []
    xs, ys = rds_dem.x.values, rds_dem.y.values
    for r in records:
        lst_x = [v[0] for v in r.vertices]
        lst_y = [v[1] for v in r.vertices]
        side = infer_side(rds_dem, min(lst_x), max(lst_x), min(lst_y), max(lst_y))
        x1, y1 = find_closest_dem_coord(min(lst_x), min(lst_y), side, rds_dem)
        x2, y2 = find_closest_dem_coord(max(lst_x), max(lst_y), side, rds_dem)
        if (x1, y1) == (x2, y2):
            one_cell.append(r.index)
        if side in ("left", "right"):
            lo, hi = sorted((y1, y2))
            cells = {(float(x1), float(y)) for y in ys if lo <= y <= hi}
        else:
            lo, hi = sorted((x1, x2))
            cells = {(float(x), float(y1)) for x in xs if lo <= x <= hi}
        cells_of[r.index] = cells
        snapped.append((r, (side, float(x1), float(y1), float(x2), float(y2))))
    problems: list[Problem] = []
    if one_cell:
        problems.append(
            Problem(
                "V10",
                "analysis.external_boundary_gis",
                f"record(s) {one_cell} snap to a single DEM cell; a segment needs two distinct cells on one edge",
                one_cell,
                fix_hint="Lengthen the line along the DEM edge",
            )
        )
    shared = sorted({(a, b) for a in cells_of for b in cells_of if a < b and cells_of[a] & cells_of[b]})
    if shared:
        problems.append(
            Problem(
                "V10",
                "analysis.external_boundary_gis",
                f"record pair(s) {shared} share a DEM cell after snapping; a segment must stop one cell short of a corner another segment uses",  # noqa: E501
                shared,
                fix_hint="Shorten one of the lines so the snapped segments share no cell",
            )
        )
    return snapped, problems


def converted_node_depths(
    rds_dem, nodes: Mapping[str, tuple[float, float, float]]
) -> tuple[dict[str, float], list[Problem]]:
    """The SWMM ``MaxDepth`` of each converted node: ground at its cell minus its invert (DR-35, V23).

    ``nodes`` maps a name to ``(x, y, invert)``. The cell is TRITON's own: ``col = ceil((x - xll)/cell +
    1e-16) - 1`` and ``row = ceil((nrows - (y - yll)/cell) + 1e-16) - 1`` (``nodes_outside_triton_grid``).
    A node whose cell is outside the grid, no-data, not finite or at or below the invert is refused with its
    invert and cell elevation.
    """
    left, bottom, right, top = rds_dem.rio.bounds()
    nrows, ncols = int(rds_dem.sizes["y"]), int(rds_dem.sizes["x"])
    cell = (right - left) / ncols
    nodata = rds_dem.rio.nodata
    values = rds_dem.squeeze().values
    depths: dict[str, float] = {}
    refused: list[tuple[str, float, float | None]] = []
    for name, (x, y, invert) in nodes.items():
        col = math.ceil((x - left) / cell + 1e-16) - 1
        row = math.ceil((nrows - (y - bottom) / cell) + 1e-16) - 1
        if not (0 <= col < ncols and 0 <= row < nrows):
            refused.append((name, invert, None))
            continue
        z = float(values[row, col])
        if not math.isfinite(z) or (nodata is not None and z == float(nodata)) or z <= invert:
            refused.append((name, invert, z))
            continue
        depths[name] = z - invert
    problems = []
    if refused:
        problems.append(
            Problem(
                "V23",
                "analysis.dem_rim_resolution",
                "the DEM gives no usable depth for node(s) "
                + ", ".join(
                    f"{n} (invert {i:.3f} m, cell elevation {z if z is None else format(z, '.3f')} m)"
                    for n, i, z in refused
                )
                + ": each needs a finite cell elevation, not no-data, above its invert",
                refused,
                fix_hint="Raise the DEM there, lower the invert, or choose the other dem_rim_resolution",
            )
        )
    return depths, problems


FULL_RESOLUTION_NOTICE = (
    "dem_rim_resolution: full writes each coupled node's SWMM depth from the full-resolution ground at its point, "
    "and TRITON then takes the node's rim as the larger of that depth and the pipe crown; TRITON exchanges each node "
    "with the coarse cell it lies in, whose ground is the cell's average, so a rim from a point can sit up to the "
    "cell's relief away from the surface the exchange uses (for example 1.4 m between the point and the cell at "
    "Norfolk's E143250), and every coupled junction's SWMM depth is rewritten. A node whose full-resolution ground is "
    "at or below its invert is refused (V23), as Norfolk's E143250 is (ground -1.500 m, invert -1.079 m)."
)


# ----------------------------------------------------------------------------------------------------------------
# The level series each record carries, in meters, for both the TRITON file and the SWMM file
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelSeries:
    """One record's level series: ``times`` and ``meters`` of equal length, and the words the file headers print."""

    times: tuple[datetime, ...]
    meters: tuple[float, ...]
    source: str


def _folded_lookup(mapping: Mapping[str, Any], name: str):
    key = _fold(name)
    for k, v in mapping.items():
        if _fold(k) == key:
            return v
    raise KeyError(name)


def _clip_to_window(
    pairs: Sequence[tuple[datetime, float]], start: datetime, end: datetime
) -> list[tuple[datetime, float]]:
    """The pairs inside ``[start, end]``, with the window's two ends interpolated linearly from their neighbours.

    TRITON reads the first column as hours since the event start, so no row may sit before the start.
    """
    first, last = pairs[0][0], pairs[-1][0]
    if first > start or last < end:
        raise ProcessingError(
            operation="boundary_level_series",
            reason=f"the series covers {first} to {last}, which does not cover the event's simulation window "
            f"{start} to {end}",
        )
    seconds = [(t - start).total_seconds() for t, _ in pairs]
    values = [v for _, v in pairs]
    total = (end - start).total_seconds()

    def interp(x: float) -> float:
        for i in range(1, len(seconds)):
            if seconds[i] >= x:
                x0, x1, y0, y1 = seconds[i - 1], seconds[i], values[i - 1], values[i]
                return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return values[-1]

    inner = [(t, v) for (t, v), s in zip(pairs, seconds, strict=True) if 0 < s < total]
    return [(start, interp(0.0)), *inner, (end, interp(total))]


def boundary_level_series(scenario, record: BoundaryRecord) -> LevelSeries:
    """The level, in meters, of one ``water_level`` record, as the one function both level files are written from.

    Under the weather source the record's weather variable is read, multiplied by its unit's factor, and its datum
    offset added. Under the SWMM-model source the series its outfalls read in the base model is resolved (the mean of
    the outfalls' series when ``water_level_boundary_unequal_outfall_series`` is ``average``), checked to cover the
    event's simulation window, and clipped to it; the model's own units are meters, so no unit or offset applies.
    """
    cfg_analysis = scenario._analysis.cfg_analysis
    source = cfg_analysis.water_level_boundary_series_source
    if source == "weather_time_series":
        variable = _folded_lookup(cfg_analysis.water_level_boundary_outfall_variables, record.outfalls[0])
        unit = cfg_analysis.water_level_boundary_variable_units[variable]
        offset = cfg_analysis.water_level_boundary_variable_datum_offsets[variable]
        frame = scenario.ds_event_ts[variable].reset_coords(drop=True).to_dataframe()[variable]
        times = tuple(t.to_pydatetime() for t in frame.index)
        meters = tuple(float(v) for v in to_meters(frame.to_numpy(dtype=float), unit, offset))
        return LevelSeries(times, meters, f"from {variable} in {unit} with a datum offset of {float(offset):g} m")
    cfg_system = scenario._system.cfg_system
    model_path = Path(cfg_system.SWMM_model)
    sections = _parse(model_path.read_text(encoding="utf-8", errors="replace"))
    catalog = model_series(sections, model_path.parent)
    outfalls = model_outfalls(sections)
    pair_lists = [series_pairs(catalog[_fold(outfalls[_fold(n)].stage)]) for n in record.outfalls]
    averaged = any(p != pair_lists[0] for p in pair_lists[1:])
    if averaged and getattr(cfg_analysis, "water_level_boundary_unequal_outfall_series", "refuse") != "average":
        raise ProcessingError(
            operation="boundary_level_series",
            reason=f"the outfalls {', '.join(record.outfalls)} of record {record.index} read different series in "
            f"{model_path.name} and water_level_boundary_unequal_outfall_series is refuse; set it to average to use "
            "their mean",
        )
    pairs = averaged_pairs(pair_lists) if averaged else pair_lists[0]
    start, end = scenario._resolve_event_window()
    import pandas as pd

    clipped = _clip_to_window(pairs, pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime())
    words = f"from the series of outfall(s) {', '.join(record.outfalls)} in {model_path.name}"
    return LevelSeries(
        tuple(t for t, _ in clipped), tuple(v for _, v in clipped), words + (", averaged" if averaged else "")
    )


# ----------------------------------------------------------------------------------------------------------------
# The preflight pass
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class PreflightReport:
    """What the boundary pass found: refusals, warnings, and the lines it prints for the user."""

    problems: list[Problem]
    warnings: list[Problem]
    lines: list[str]


def _event_count(cfg_analysis) -> int | None:
    csv = getattr(cfg_analysis, "weather_events_to_simulate", None)
    if csv is None or not Path(csv).is_file():
        return None
    import pandas as pd

    return len(pd.read_csv(csv))


def _weather_variables(cfg_analysis) -> list[str] | None:
    path = getattr(cfg_analysis, "weather_timeseries", None)
    if path is None or not Path(path).is_file():
        return None
    try:
        import xarray as xr

        # FORCING-READ: preflight
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            return list(ds.data_vars)
    except Exception:  # noqa: BLE001 - an unreadable weather file is reported by the weather checks
        return None


def _stated(config_line: str, field_: str, message: str, current: Any = None, fix: str | None = None) -> Problem:
    return Problem("CFG", field_, message, current, fix_hint=fix, config_line=config_line)


def preflight_problems(cfg_system, cfg_analysis) -> PreflightReport:
    """Every refusal the external-boundary layer, the outfall decisions and the DEM depths can raise, in one pass.

    Runs the validators of package 1h in the order of their table (V1 to V25, ``swmm_threads`` excepted, which its own
    pass owns), one problem per kind with every offender named. A check whose input is absent is skipped with that
    absence as its own problem, and the checks that do not need it still run.
    """
    from hhemt.swmm_utils import _read_model_text, processed_dem_grid

    problems: list[Problem] = []
    warnings: list[Problem] = []
    lines: list[str] = []
    model_path = Path(cfg_system.SWMM_model)
    try:
        sections = _parse(_read_model_text(model_path))
    except (OSError, ProcessingError):
        return PreflightReport(problems, warnings, lines)
    outfalls = model_outfalls(sections)
    coordinates = model_coordinates(sections)
    catalog = model_series(sections, model_path.parent)

    records: tuple[BoundaryRecord, ...] = ()
    toggle = bool(cfg_analysis.toggle_external_boundaries)
    if toggle and cfg_analysis.external_boundary_gis is not None:
        read_records, read_problems = boundary_records_for(cfg_analysis, cfg_system)
        records = tuple(read_records)
        problems.extend(read_problems)
        layer = Path(cfg_analysis.external_boundary_gis)
        problems.extend(
            record_problems(
                records,
                cfg_analysis.water_level_boundary_outfall_attribute,
                cfg_analysis.external_boundary_value_attribute,
                layer.name,
                cfg_analysis.external_boundary_type_attribute,
            )
        )
    wl = [r for r in records if r.kind == "water_level"]
    matched = matched_outfall_names(records)
    if toggle:
        value_records = [r for r in records if r.kind in _VALUE_TYPES]
        if cfg_analysis.external_boundary_type_attribute is None:
            problems.append(
                _stated(
                    "external_boundary_type_attribute: type",
                    "analysis.external_boundary_type_attribute",
                    "the column that names each record's type is not set",
                    fix="Name the layer column that holds water_level, free_outflow, normal_slope or froude_number",
                )
            )
        if wl and cfg_analysis.water_level_boundary_outfall_attribute is None:
            problems.append(
                _stated(
                    "water_level_boundary_outfall_attribute: outfalls",
                    "analysis.water_level_boundary_outfall_attribute",
                    "the layer has a water_level record and the column that names its outfalls is not set",
                    fix="Name the column of the layer that holds the SWMM outfall names of each water_level record",
                )
            )
        if value_records and cfg_analysis.external_boundary_value_attribute is None:
            problems.append(
                _stated(
                    "external_boundary_value_attribute: value",
                    "analysis.external_boundary_value_attribute",
                    f"record(s) {[r.index for r in value_records]} are normal_slope or froude_number and the column that holds their number is not set",  # noqa: E501
                    fix="Name the column of the layer that holds the slope or the Froude number",
                )
            )
        if not value_records and records and cfg_analysis.external_boundary_value_attribute is not None:
            problems.append(
                Problem(
                    "CFG",
                    "analysis.external_boundary_value_attribute",
                    "no record is normal_slope or froude_number, so there is no number to read",
                    cfg_analysis.external_boundary_value_attribute,
                    fix_hint="Remove external_boundary_value_attribute",
                )
            )
        if wl and cfg_analysis.water_level_boundary_series_source is None:
            problems.append(
                _stated(
                    "water_level_boundary_series_source: weather_time_series",
                    "analysis.water_level_boundary_series_source",
                    "the layer has a water_level record and the source of its water level is not set",
                    fix="Choose weather_time_series or swmm_model",
                )
            )
        problems.extend(
            matched_outfall_problems(
                records, outfalls, coordinates, cfg_analysis.water_level_boundary_outfall_max_distance_m
            )[0]
        )
        source = cfg_analysis.water_level_boundary_series_source
        if wl:
            problems.extend(
                series_identity_problems(
                    records, catalog, outfalls, source, cfg_analysis.water_level_boundary_unequal_outfall_series
                )
            )
        if source == "weather_time_series":
            problems.extend(
                variable_problems(
                    cfg_analysis.water_level_boundary_outfall_variables,
                    matched,
                    records,
                    _weather_variables(cfg_analysis),
                    cfg_analysis.water_level_boundary_variable_units,
                    cfg_analysis.water_level_boundary_variable_datum_offsets,
                )
            )
        if source == "swmm_model":
            n_events = _event_count(cfg_analysis)
            if n_events is not None and n_events > 1:
                problems.append(
                    Problem(
                        "CFG",
                        "analysis.water_level_boundary_series_source",
                        f"the analysis simulates {n_events} events and the swmm_model source holds one series for all of them",  # noqa: E501
                        source,
                        fix_hint="Use weather_time_series, or simulate one event",
                    )
                )
    elif cfg_analysis.water_level_boundary_series_source is not None:
        warnings.append(
            Problem(
                "CFG",
                "analysis.water_level_boundary_series_source",
                "a water-level source is set and toggle_external_boundaries is false",
                cfg_analysis.water_level_boundary_series_source,
                fix_hint="Remove the water_level_boundary_* settings or set toggle_external_boundaries to true",
            )
        )

    keep = list(cfg_analysis.swmm_outfalls_to_keep)
    coupled = bool(getattr(cfg_system, "toggle_tritonswmm_model", True))
    # V9 is asked of a coupled model only, because only a coupled model reads the disposition; V6 and V13 stay always.
    problems.extend(
        p
        for p in outfall_decision_problems(outfalls, matched, list(cfg_system.swmm_outfalls_to_couple), keep)
        if coupled or p.code != "V9"
    )
    if toggle and records:
        _, distances = matched_outfall_problems(records, outfalls, coordinates, None)
        lines.extend(distance_lines(distances))

    converted = [*cfg_system.swmm_outfalls_to_couple, *matched]
    uses_dem = coupled and (bool(converted) or bool(cfg_analysis.swmm_node_rim_from_dem))
    if uses_dem and cfg_analysis.dem_rim_resolution is None:
        problems.append(
            _stated(
                "dem_rim_resolution: processed",
                "analysis.dem_rim_resolution",
                "a node depth or rim is taken from the DEM (an outfall is converted or swmm_node_rim_from_dem is true) "
                "and the DEM that gives it is not chosen",
                fix="Choose processed, the coarsened DEM TRITON loads, or full, the full-resolution DEM",
            )
        )
    if not uses_dem and cfg_analysis.dem_rim_resolution is not None:
        problems.append(
            Problem(
                "CFG",
                "analysis.dem_rim_resolution",
                "no outfall is converted and swmm_node_rim_from_dem is false, so no depth is taken from the DEM",
                cfg_analysis.dem_rim_resolution,
                fix_hint="Remove dem_rim_resolution",
            )
        )
    if cfg_analysis.dem_rim_resolution == "full" and (converted or cfg_analysis.swmm_node_rim_from_dem):
        lines.append(FULL_RESOLUTION_NOTICE)

    grid = None
    if toggle and records and Path(cfg_system.DEM_fullres).exists():
        grid, provisional = processed_dem_grid(cfg_system)
        snapped, snap_problems = snap_segments(records, grid)
        problems.extend(snap_problems)
        for record, (side, x1, y1, x2, y2) in snapped:
            lines.append(
                f"record {record.index} ({record.kind}): endpoints {record.vertices[0]} to {record.vertices[-1]}, "
                f"snapped to {side} edge cells ({x1}, {y1}) to ({x2}, {y2})"
                + (" (provisional grid)" if provisional else "")
            )
    return PreflightReport(problems, warnings, lines)
