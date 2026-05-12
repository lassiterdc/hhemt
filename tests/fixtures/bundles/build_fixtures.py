"""One-shot fixture-build script for bundle tests.

Produces two committed-not-built fixture bundles under
``tests/fixtures/bundles/``:

  - ``multi_sim/`` — from the cached
    ``Local_TestCases.retrieve_synth_multi_sim_test_case`` analysis
  - ``sensitivity_master/`` — from the cached
    ``Local_TestCases.retrieve_synth_sensitivity_analysis_test_case``
    (master) analysis

The script is **idempotent**: re-running it overwrites the fixtures in
place and produces byte-identical output modulo deterministic
timestamps in ``bundle_manifest.json``.

Source data: the cached synth-analysis state under
``$HOME/.cache/TRITON_SWMM_toolkit/synthetic_test_runs/``. The script
does NOT call ``analysis.run()`` — the cached state must already be
populated (run ``pytest tests/test_synth_04_multisim_with_snakemake.py
tests/test_synth_05_sensitivity_analysis_with_snakemake.py`` once if
the cache is empty).

For each case the script:

  1. Materializes the analysis object via the cached-test-case helper.
  2. Writes ``cfg_system.yaml`` + ``cfg_analysis.yaml`` via the actual
     ``_copy_configs_with_relative_paths`` helper from
     ``TRITON_SWMM_toolkit.bundle._emit`` — so the fixture exercises
     the real path-rewriter code path.
  3. Writes a minimal ``bundle_manifest.json`` containing the
     ``bundle_root_invariants`` dict (Phase 3 forward-compat).
  4. If a ``plots/`` directory exists in the analysis dir, mirrors its
     ``*.manifest.json`` sidecar layout (without copying large binary
     PNG/HTML payloads — those are unnecessary for the Phase 1 tests).

Run:

    python -m tests.fixtures.bundles.build_fixtures
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import the test-case catalog from the toolkit's test fixtures package.
TESTS_ROOT = Path(__file__).resolve().parents[2]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT.parent))

from tests.fixtures import test_case_catalog as cases  # noqa: E402

from TRITON_SWMM_toolkit.bundle._emit import (  # noqa: E402
    _copy_configs_with_relative_paths,
)
from TRITON_SWMM_toolkit.bundle._path_policy import (  # noqa: E402
    _PATH_FIELD_POLICY,
    enumerate_path_fields,
)
from TRITON_SWMM_toolkit.version_migration.constants import (  # noqa: E402
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_SCHEMA_VERSION,
    LAYOUT_VERSION,
)


FIXTURES_DIR = Path(__file__).resolve().parent


def _build_one(case_name: str, fixture_subdir: str, analysis) -> Path:
    out = FIXTURES_DIR / fixture_subdir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    _copy_configs_with_relative_paths(analysis, out)

    # Compute bundle_root_invariants by walking both cfg models.
    invariants: dict[str, list[str]] = {}
    for cfg in (
        analysis._system.cfg_system,
        analysis.cfg_analysis,
    ):
        for name in enumerate_path_fields(type(cfg)):
            policy = _PATH_FIELD_POLICY[name]
            invariants.setdefault(policy.value, []).append(name)

    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "layout_version": LAYOUT_VERSION,
        "toolkit_git_sha": "fixture",
        "analysis_id": analysis.cfg_analysis.analysis_id,
        "created_at_utc": "2026-05-11T00:00:00+00:00",  # deterministic
        "source_paths_by_renderer": {},
        "bundle_root_invariants": invariants,
    }
    (out / BUNDLE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    # Minimal plots/ mirror — copy only *.manifest.json sidecars if present.
    src_plots = analysis.analysis_paths.analysis_dir / "plots"
    if src_plots.exists():
        dest_plots = out / "plots"
        dest_plots.mkdir()
        for sidecar in sorted(src_plots.rglob("*.manifest.json")):
            rel = sidecar.relative_to(src_plots)
            (dest_plots / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sidecar, dest_plots / rel)

    return out


def main() -> int:
    case_multi = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False
    )
    out_multi = _build_one(
        "synth_multi_sim", "multi_sim", case_multi.analysis
    )
    print(f"Built {out_multi}")

    case_sens = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(
        start_from_scratch=False
    )
    # The case's .analysis is the TRITONSWMM_analysis configured with
    # toggle_sensitivity_analysis=True; bundle_report_data on the master
    # is what produces the sensitivity-master bundle.
    out_sens = _build_one(
        "synth_sensitivity_master",
        "sensitivity_master",
        case_sens.analysis,
    )
    print(f"Built {out_sens}")
    print(f"Fixtures regenerated at {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
