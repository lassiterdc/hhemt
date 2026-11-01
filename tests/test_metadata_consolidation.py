"""Embedded provenance core + RO-Crate sidecar at consolidation (regular + sensitivity master/sub), idempotently.

Both rows run on a committed GOLDEN (``tests/fixtures/golden_trees.py``), not on a freshly run synthetic pipeline: the
golden carries the summary layer and the completion records, so ``consolidate_to_datatree`` and
``consolidate_sensitivity_datatree`` (the operations under test) run for real against it and no solver is compiled or
executed. A golden's log carries no consolidation build stamp, and an absent stamp is always a mismatch, so the first
consolidation REBUILDS the stores. The rebuild re-emits the embedded provenance core (the tree's ``ro_crate_metadata``
root attribute, a JSON string) and writes the RO-Crate sidecar (``ro-crate-metadata.json`` in the analysis directory).
The golden carries the core but not the sidecar (``tests/goldens/SPEC.md`` section 9), so the rows assert only that the
sidecar is absent beforehand. Not established: that the rebuilt core differs from the golden's, since the rows assert
only that the attribute exists afterward.

No reprocess prerequisite is needed here, measured. These rows call the consolidation methods in-process, so
no Snakefile (no dry run), no ``python`` on ``PATH`` and no child-interpreter identity shim is required, and no
Snakemake process runs (so no ``requires_snakemake_subprocess``). One declaration is used (the
``HHEMT_DECLARE_STALE_BUILD`` environment variable), scoped to the second call of the idempotence assertion only
(see there).
"""

from __future__ import annotations

import json

import xarray as xr


def test_consolidation_emits_core_and_sidecar(golden_multisim_analysis, monkeypatch):
    # Regular analysis. Consolidate the golden, then assert the embedded core + co-located
    # sidecar, and that a re-consolidation is mtime-stable.
    a = golden_multisim_analysis
    sidecar = a.analysis_paths.analysis_dir / "ro-crate-metadata.json"
    assert not sidecar.exists(), "precondition: the golden carries no sidecar, so emission is the consolidation's"

    a.process.consolidate_to_datatree()

    tree_path = a.analysis_paths.analysis_datatree_zarr
    assert tree_path is not None and tree_path.exists()
    tree = xr.open_datatree(tree_path, engine="zarr", consolidated=False)
    assert "ro_crate_metadata" in tree.attrs
    json.loads(tree.attrs["ro_crate_metadata"])  # the core parses as JSON

    assert sidecar.exists()

    # Re-consolidation does not bump the sidecar mtime: the early return in ``consolidate_to_datatree`` skips the
    # rewrite. That return needs the store to exist AND the log to be complete AND the inputs fingerprint to match
    # AND no build mismatch (measured here; the sidecar's compare-and-write is a second guard this row does not
    # isolate).
    m1 = sidecar.stat().st_mtime_ns
    # The build gate treats a DIRTY running checkout as a mismatch (``provenance.store_build_mismatch``), which a
    # tree under edit always is. Setting the variable acknowledges the stale build: the gate then prints the reason
    # and reuses the store exactly as it would on a match, so the row measures the early return it names rather
    # than the state of the working tree.
    monkeypatch.setenv("HHEMT_DECLARE_STALE_BUILD", "1")
    a.process.consolidate_to_datatree()
    assert sidecar.stat().st_mtime_ns == m1


def test_sensitivity_consolidation_emits_master_and_sub_provenance(golden_sensitivity_analysis):
    # Sensitivity master carries the embedded core + co-located sidecar, and each
    # consolidated member independently carries its own sidecar (produced by the
    # per-sub consolidate_to_datatree wiring — no sub-loop in the master path).
    sensitivity = golden_sensitivity_analysis.sensitivity
    master_sidecar = sensitivity.experiment.analysis_paths.analysis_dir / "ro-crate-metadata.json"
    assert not master_sidecar.exists(), "precondition: the golden carries no master sidecar"

    sensitivity.consolidate_sensitivity_datatree()

    master_zarr = sensitivity.analysis_paths.sensitivity_datatree_zarr
    assert master_zarr is not None and master_zarr.exists()
    master_tree = xr.open_datatree(master_zarr, engine="zarr", consolidated=False)
    assert "ro_crate_metadata" in master_tree.attrs
    json.loads(master_tree.attrs["ro_crate_metadata"])  # the core parses as JSON
    assert master_sidecar.exists()

    # The golden's four members each carry an ``analysis_datatree.zarr`` (``tests/goldens/SPEC.md`` section 3), so
    # the skip below is never taken on the golden.
    consolidated_subs = 0
    for _member_id, sub in sensitivity.members.items():
        sub_zarr = sub.analysis_paths.analysis_datatree_zarr
        if sub_zarr is not None and sub_zarr.exists():
            consolidated_subs += 1
            assert (sub.analysis_paths.analysis_dir / "ro-crate-metadata.json").exists()
    assert consolidated_subs > 0  # at least one sub consolidated + carried a sidecar
