"""Short, stable labels for the device models TRITON reports at runtime.

LEAF MODULE BY DESIGN: it imports nothing from hhemt, so `report_renderers/`,
`eda/` and `analysis.py` can all read one map rather than growing three. This
corpus has already paid for the alternative — `_hardware_family`,
`_hw_family_key` and `_b4b_family_key` are three hardware keys in three modules,
and the claim that they agreed was false for several phases before anyone checked.

The map is ORDERED and matched by case-insensitive SUBSTRING, not by exact key,
because vendor strings carry SKU-varying noise ("64-Core Processor", "with Radeon
Graphics"). First match wins, so more specific needles must precede less specific
ones.

An unmapped name is returned UNCHANGED. It is never collapsed to "unknown": a
shared sentinel would merge two genuinely different unmapped devices into one
apparent hardware, re-creating exactly the blindness this map exists to remove —
and it would do so first on the AMD arm, whose runtime string is the one nobody
has yet observed.
"""

from __future__ import annotations

#: (needle, label) in match order. GPU labels use the arch-key vocabulary of
#: `system.py::_resolve_cuda_arch_flags` (rtx3090/a6000/a100/h100/h200/v100) so a
#: legend and a compile flag name the same device. Note that the Rivanna PARTITION
#: suffix for the same card is `a100-80`; the two spellings are deliberate and the
#: arch key is the one used here.
_DEVICE_LABELS: tuple[tuple[str, str], ...] = (
    # GPU — NVIDIA. "rtx a6000" precedes "a6000" only for readability; both hit.
    ("rtx a6000", "a6000"),
    ("a100", "a100"),
    ("h200", "h200"),
    ("h100", "h100"),
    ("v100", "v100"),
    ("rtx 3090", "rtx3090"),
    # GPU — AMD is deliberately UNMAPPED. No AMD GPU is reachable from this
    # campaign's cluster, so any needle here would guess at an unobserved string:
    # it would either fail to match (indistinguishable from the fallback) or match
    # something it should not. The raw-name passthrough is the correct handling
    # until a real Frontier log supplies the string.
    # CPU. Same map, same contract — the CPU axis is the one that currently has no
    # hardware channel at all in the benchmarking figure.
    ("epyc 7742", "epyc-7742"),
    ("epyc 7763", "epyc-7763"),
    ("xeon gold 6248", "xeon-6248"),
)


def hardware_label(raw: str | None) -> str | None:
    """Return a short label for a runtime-reported device name.

    None (or blank) in -> None out: a log predating the GPU emission carries no
    value, and an absent measurement must stay absent rather than becoming a
    device. An unmapped name is returned unchanged.
    """
    if raw is None:
        return None
    name = " ".join(raw.split())
    if not name:
        return None
    low = name.lower()
    for needle, label in _DEVICE_LABELS:
        if needle in low:
            return label
    return name
