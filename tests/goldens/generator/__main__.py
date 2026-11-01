"""Entry point: write the multi-sim and sensitivity goldens into a directory.

A golden is a finished analysis tree, with the files hhemt needs to construct an analysis object from it.
This package writes one with synthetic writers that run no solver. Among the tests that read the goldens,
``tests/test_golden_conformance.py`` asks hhemt to build an analysis from each extracted golden and
``tests/test_golden_regeneration.py`` compares the committed zips with a fresh run of this generator.

``python -m tests.goldens.generator OUT_DIR [--which multisim|sensitivity|both]`` writes the extracted
golden trees to ``OUT_DIR/multisim`` and ``OUT_DIR/sensitivity`` (the zip root of each is that directory).
It does not zip them: ``python -m tests.test_golden_regeneration`` runs this generator into a temporary directory and
writes ``tests/fixtures/goldens/multisim.zip`` and ``sensitivity.zip``. Specification: ``tests/goldens/SPEC.md``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import raw, records, summary

# An arm is one model type (triton, tritonswmm or swmm); the multi-sim golden carries all three.
ARMS = ("triton", "tritonswmm", "swmm")
# hhemt raises ValueError for a sensitivity analysis that enables more than one model type
# (SensitivityAnalysisWorkflowBuilder.generate_master_snakefile_content), so the sensitivity golden has one arm.
SENSITIVITY_ARMS = ("tritonswmm",)
# Placeholder written as ``consolidation_inputs_fingerprint`` in every analysis ``log.json`` (the multi-sim analysis,
# each sensitivity member and the experiment root) and in every ``analysis_datatree.zarr`` (the multi-sim analysis and
# each member), and in neither ``system_log.json`` nor ``experiment_datatree.zarr``. It is 40 zeros and never computed:
# hhemt derives the real value with its own rule (``_consolidation_inputs_fingerprint`` in ``processing_analysis.py``)
# and this package imports nothing from hhemt, so it cannot call that rule. Not established: whether hhemt accepts the
# zeros as the value it recomputes from the inputs (``tests/goldens/SPEC.md`` section 4.4; OI-F1 in its section 10).
FINGERPRINT = "0" * 40


def _event_ids(n: int) -> list[str]:
    return [f"event_index.{i}" for i in range(n)]


# Writes one event's raw, processed and record files. ``event_id`` is the event's directory name
# (``event_index.{i}``) and ``event_iloc`` is the integer i it stands for. ``model_log_dir`` receives the per-model run
# logs and ``log_analysis_id`` is put in their file names: empty for the multi-sim golden, ``member_{id}`` for a member
# (``tests/goldens/SPEC.md`` section 4.7). ``compute`` is the member's row of ``records.SENSITIVITY_MEMBERS``, whose
# ``n_mpi_procs`` and ``n_omp_threads`` are the task and thread counts the raw log states (1 and 1 when absent).
def _write_event(
    root: Path,
    analysis_rel: str,
    sims_rel: str,
    event_id: str,
    event_iloc: int,
    net: raw.Network,
    dims: raw.Dimensions,
    *,
    model_log_dir: Path,
    log_analysis_id: str = "",
    arms: tuple[str, ...] = ARMS,
    compute: dict | None = None,
) -> Path:
    event_dir = root / sims_rel / event_id
    compute = compute or {}
    raw.write_event_raw(
        event_dir,
        arms=arms,
        network=net,
        dims=dims,
        n_tasks=compute.get("n_mpi_procs", 1),
        omp_threads=compute.get("n_omp_threads", 1),
    )
    stores = summary.write_event_processed(event_dir, event_iloc=event_iloc, arms=arms, network=net, dims=dims)
    records.write_event_records(
        root,
        analysis_rel,
        sims_rel,
        event_id,
        event_iloc,
        arms,
        stores,
        net,
        dims,
        model_log_dir=model_log_dir,
        log_analysis_id=log_analysis_id,
    )
    return event_dir


def generate_multisim(root: Path) -> None:
    """The multi-sim golden: one regular analysis, 3 events, the three model arms (SPEC.md section 1).

    A regular analysis is one that is not a sensitivity analysis, so it has no members.
    """
    net, dims = raw.default_network(), raw.Dimensions()
    aid = "synth_multi_sim"
    ids = _event_ids(3)
    # The layout stamp is written in the system directory (root) and again in the analysis directory
    # (SPEC.md section 4.1).
    records.write_version_stamp(root)
    records.write_configs(root, aid, sensitivity=False)
    records.write_system_inputs(root, net, dims, event_indices=list(range(3)))
    records.write_json(root / "system_log.json", records.system_log(root, dims))
    adir = root / aid
    records.write_version_stamp(adir)
    events = []
    for i, eid in enumerate(ids):
        edir = _write_event(root, aid, f"{aid}/sims", eid, i, net, dims, model_log_dir=adir / "logs" / "sims")
        events.append((eid, edir))
    tree = adir / "analysis_datatree.zarr"
    summary.write_analysis_datatree(tree, events, ARMS, analysis_id=aid, fingerprint=FINGERPRINT)
    # analysis_log writes size_MiB as 0.0; the size is measured on the finished tree, so it is patched in afterwards.
    log = records.analysis_log(aid, "analysis_datatree.zarr", experiment=False, fingerprint=FINGERPRINT)
    log["processing_log"]["outputs"]["analysis_datatree.zarr"]["size_MiB"] = records.dir_size_mib(tree)
    records.write_json(adir / "log.json", log)
    records.write_multisim_flags(adir / "_status", ids, ARMS)


def generate_sensitivity(root: Path) -> None:
    """The sensitivity golden: 4 members, 1 event, the coupled tritonswmm arm only (SPEC.md section 1).

    A member is one row of the sensitivity CSV, a compute configuration with its own ``members/member_{id}/``
    directory (SPEC.md sections 5.1 and 3).
    """
    net, dims = raw.default_network(), raw.Dimensions()
    aid = "synth_sensitivity"
    ids = _event_ids(1)
    members = sorted(records.SENSITIVITY_MEMBERS)
    # Stamped in the system directory and the analysis directory, as in generate_multisim (SPEC.md section 4.1).
    records.write_version_stamp(root)
    records.write_configs(root, aid, sensitivity=True, arms=SENSITIVITY_ARMS)
    records.write_system_inputs(root, net, dims, event_indices=[0])
    records.write_json(root / "system_log.json", records.system_log(root, dims))
    adir = root / aid
    records.write_version_stamp(adir)
    member_events: dict[int, list[tuple[str, Path]]] = {}
    for m in members:
        mrel = f"{aid}/members/member_{m}"
        mdir = root / mrel
        records.write_text(mdir / f"member_{m}.yaml", records.yaml_text(records.member_config_dict(m)))
        (mdir / "_status").mkdir(parents=True, exist_ok=True)
        events = []
        for i, eid in enumerate(ids):
            # A member's run logs go in the experiment root's logs/sims and their names carry the member's analysis id
            # (SPEC.md section 4.7), unlike a regular analysis whose logs go in its own logs/sims.
            edir = _write_event(
                root,
                mrel,
                f"{mrel}/sims",
                eid,
                i,
                net,
                dims,
                model_log_dir=adir / "logs" / "sims",
                log_analysis_id=f"member_{m}",
                arms=SENSITIVITY_ARMS,
                compute=records.SENSITIVITY_MEMBERS[m],
            )
            events.append((eid, edir))
        member_events[m] = events
        tree = mdir / "analysis_datatree.zarr"
        summary.write_analysis_datatree(
            tree, events, SENSITIVITY_ARMS, analysis_id=f"member_{m}", fingerprint=FINGERPRINT
        )
        # size_MiB is patched in after the store is written, as in generate_multisim.
        log = records.analysis_log(mrel, "analysis_datatree.zarr", experiment=False, fingerprint=FINGERPRINT)
        log["processing_log"]["outputs"]["analysis_datatree.zarr"]["size_MiB"] = records.dir_size_mib(tree)
        records.write_json(mdir / "log.json", log)
    etree = adir / "experiment_datatree.zarr"
    summary.write_experiment_datatree(
        etree, {m: records.SENSITIVITY_MEMBERS[m] for m in members}, member_events, SENSITIVITY_ARMS, analysis_id=aid
    )
    elog = records.analysis_log(aid, "experiment_datatree.zarr", experiment=True, fingerprint=FINGERPRINT)
    elog["processing_log"]["outputs"]["experiment_datatree.zarr"]["size_MiB"] = records.dir_size_mib(etree)
    records.write_json(adir / "log.json", elog)
    records.write_sensitivity_flags(adir / "_status", members, ids, SENSITIVITY_ARMS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tests.goldens.generator", description=__doc__)
    ap.add_argument("out_dir", type=Path, help="directory to write the golden trees into")
    ap.add_argument("--which", choices=("multisim", "sensitivity", "both"), default="both")
    ap.add_argument("--clean", action="store_true", help="remove an existing target directory first")
    args = ap.parse_args(argv)
    todo = {"multisim": generate_multisim, "sensitivity": generate_sensitivity}
    for name in ("multisim", "sensitivity"):
        if args.which in (name, "both"):
            target = args.out_dir / name
            if target.exists():
                if not args.clean:
                    ap.error(f"{target} exists; pass --clean to replace it")
                shutil.rmtree(target)
            target.mkdir(parents=True)
            todo[name](target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
