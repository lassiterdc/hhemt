"""Per-artist data-provenance log for report-renderer figures.

Records, for every visible feature in a rendered matplotlib figure: the data
variable name, the variable's metadata (xarray attrs — units, long_name, etc.),
the visual channel it drives (X / Y / Z / color / size / linewidth / alpha /
hatch / gid), and any selection or transform applied. The serialized payload
lands in the figure's sibling `<stem>.manifest.json` under a top-level
`artists:` array, alongside the existing manifest schema produced by
`_figure_emission.emit_plot_with_sources`.

The discipline is enforced by `tests/test_provenance_discipline.py` (AST lint —
every artist-creating matplotlib call inside a renderer module must be enclosed
in a `with prov.artist(...)` block) and by the `## Provenance log contract`
section of the `report renderers accept uniform signature` stipulation.

Usage pattern in a renderer's `render()`:

    prov = ProvenanceLog()
    with prov.artist(axes_id="ax_depth", kind="image",
                     note="peak flood depth") as a:
        a.add_xarray_channel("z", da_masked)
        a.add_xarray_channel("color", da_masked,
                             transform="masked to watershed",
                             cmap=cfg.cmap, vmin=cfg.vmin, vmax=cfg.vmax)
        ax.imshow(da_masked.values, ...)
    return emit_plot_with_sources(fig, output_path, source_paths,
                                  analysis_dir=analysis_dir,
                                  dpi=dpi,
                                  provenance=prov)

The additivity contract: this module does not perturb pixel output. Provenance
binding is a metadata-only discipline.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config.viz_vocabulary import EncodingChannel as _Channel

if TYPE_CHECKING:
    import xarray as xr


@dataclass(frozen=True, slots=True)
class ProvenanceRef:
    """A reference to a data variable that drives a visual channel."""

    source_path: str
    variable: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] | None = None
    transform: str | None = None


@dataclass(frozen=True, slots=True)
class ChannelEntry:
    channel: _Channel | str
    ref: ProvenanceRef
    encoding: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtistRecord:
    axes_id: str
    kind: str
    channels: list[ChannelEntry] = field(default_factory=list)
    note: str | None = None


class ProvenanceLog:
    """Collects per-artist channel-level provenance for a single figure."""

    def __init__(self) -> None:
        self._artists: list[ArtistRecord] = []

    @contextmanager
    def artist(
        self,
        axes_id: str,
        kind: str,
        *,
        note: str | None = None,
    ) -> Iterator[_ArtistBuilder]:
        """Yield an `_ArtistBuilder`; on exit, append the record to the log."""
        record = ArtistRecord(axes_id=axes_id, kind=kind, note=note)
        builder = _ArtistBuilder(record)
        try:
            yield builder
        finally:
            self._artists.append(record)

    def serialize(self) -> list[dict[str, Any]]:
        """Emit a JSON-safe payload for manifest embedding."""
        return [_artist_to_dict(a) for a in self._artists]


class _ArtistBuilder:
    """Per-artist channel accumulator returned by `ProvenanceLog.artist(...)`."""

    def __init__(self, record: ArtistRecord) -> None:
        self._record = record

    def add_channel(
        self,
        channel: _Channel | str,
        ref: ProvenanceRef,
        **encoding: Any,
    ) -> None:
        self._record.channels.append(ChannelEntry(channel=channel, ref=ref, encoding=dict(encoding)))

    def add_xarray_channel(
        self,
        channel: _Channel | str,
        da: xr.DataArray,
        *,
        selection: dict[str, Any] | None = None,
        transform: str | None = None,
        source_path: str | None = None,
        **encoding: Any,
    ) -> None:
        """Capture a channel driven by an xarray DataArray.

        `source_path` defaults to `da.encoding.get("source")` when omitted.
        """
        if source_path is None:
            source_path = str(da.encoding.get("source", "")) if da.encoding else ""
        ref = ProvenanceRef(
            source_path=source_path,
            variable=str(da.name) if da.name is not None else None,
            attrs=dict(da.attrs),
            selection=selection,
            transform=transform,
        )
        self.add_channel(channel, ref, **encoding)

    def add_swmm_channel(
        self,
        channel: _Channel | str,
        *,
        swmm_inp: Path | str,
        kind: str,
        link_id: str | None = None,
        node_id: str | None = None,
        **encoding: Any,
    ) -> None:
        """Capture a channel driven by swmmio-parsed `.inp` coordinates."""
        sel: dict[str, Any] = {}
        if link_id is not None:
            sel["link_id"] = link_id
        if node_id is not None:
            sel["node_id"] = node_id
        ref = ProvenanceRef(
            source_path=str(swmm_inp),
            variable=kind,
            attrs={},
            selection=sel or None,
        )
        self.add_channel(channel, ref, **encoding)


def _artist_to_dict(record: ArtistRecord) -> dict[str, Any]:
    return {
        "axes_id": record.axes_id,
        "kind": record.kind,
        "note": record.note,
        "channels": [
            {
                "channel": c.channel,
                "ref": asdict(c.ref),
                "encoding": c.encoding,
            }
            for c in record.channels
        ],
    }


def format_manifest_artists(artists: list[dict[str, Any]]) -> str:
    """Pretty-print a serialized `manifest["artists"]` array.

    One section per artist record, channels listed under each. Designed for
    the Phase 4 STOP-gate manifest review:

        python -c "from TRITON_SWMM_toolkit.report_renderers._provenance \\
                   import format_manifest_artists as f; \\
                   import json; \\
                   print(f(json.load(open('manifest.json'))['artists']))"
    """
    lines: list[str] = []
    for i, artist in enumerate(artists):
        header = f"[{artist.get('axes_id', '?')}:{artist.get('kind', '?')}]"
        if artist.get("note"):
            header += f' — "{artist["note"]}"'
        lines.append(header)
        for ch in artist.get("channels", []):
            lines.extend(_format_channel_lines(ch))
        if i < len(artists) - 1:
            lines.append("")
    return "\n".join(lines)


def _format_channel_lines(channel: dict[str, Any]) -> list[str]:
    name = channel.get("channel", "?")
    ref = channel.get("ref", {}) or {}
    var = ref.get("variable") or "—"
    attrs = ref.get("attrs", {}) or {}
    units = attrs.get("units", "—")
    long_name = attrs.get("long_name")
    src = ref.get("source_path") or "—"
    sel = ref.get("selection")
    transform = ref.get("transform")
    enc = channel.get("encoding", {}) or {}

    head = f"  {name:<10} {var} ({units}) ← {src}"
    if sel:
        sel_str = ", ".join(f"{k}={v!r}" for k, v in sel.items())
        head += f"  [{sel_str}]"
    if transform:
        head += f"  ({transform})"
    out = [head]
    sub_parts: list[str] = []
    if long_name:
        sub_parts.append(f'long_name="{long_name}"')
    if enc:
        sub_parts.extend(f"{k}={v}" for k, v in enc.items())
    if sub_parts:
        out.append(f"             {' '.join(sub_parts)}")
    return out
