"""``BundleableAnalysis`` Protocol — duck-typed contract for
``emit_bundle``.

Both ``TRITONSWMM_analysis`` and ``TRITONSWMM_sensitivity_analysis``
expose this surface (``sensitivity_analysis.py`` delegates the
attributes to its wrapped master analysis). ``emit_bundle`` reads only
these attributes — no class-specific code paths.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BundleableAnalysis(Protocol):
    """Attributes ``emit_bundle()`` requires from its input.

    Conforming types are duck-typed; no explicit ``register`` call is
    needed. The Protocol exists for documentation, IDE hinting, and
    optional ``isinstance`` checks during emit-time invariant tests.
    """

    _system: Any  # exposes ``cfg_system`` (with ``system_directory``)
    analysis_paths: Any  # exposes ``analysis_dir``
    cfg_analysis: Any  # exposes ``analysis_id``, ``weather_events_to_simulate``
