import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def setup():
    import sys

    sys.path.insert(
        0,
        "/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/globus-auto-transfer-and-debug-restructuring/src",
    )

    from pathlib import Path

    from TRITON_SWMM_toolkit.config.globus import (
        PostRunTransferConfig,
        _get_endpoint_uuids,
    )
    from TRITON_SWMM_toolkit.globus_transfer import GlobusTransferManager

    return GlobusTransferManager, Path, PostRunTransferConfig, _get_endpoint_uuids


@app.cell
def config(Path, PostRunTransferConfig, _get_endpoint_uuids):
    config = PostRunTransferConfig(
        destination_root=r"D:\Dropbox\_GradSchool\repos\TRITON-SWMM_toolkit\frontier",
        system="frontier",
    )

    spec = config.to_transfer_spec(
        analysis_dir=Path(
            "/lustre/orion/cli190/proj-shared/dcl3nd/TRITON-SWMM_toolkit/"
            "test_data/norfolk_coastal_flooding/cases/frontier_sensitivity_suite"
        ),
        analysis_id="frontier_sensitivity_suite",
    )

    print(f"Label: {spec.label}")
    print(f"Source: {spec.endpoints.source_uuid}")
    print(f"Dest:   {spec.endpoints.destination_uuid}")
    for item in spec.items:
        print(f"  {item.source_path} → {item.destination_path}")
    print(f"Exclude dirs: {config.exclude_patterns}")

    # Resolve data_access consent requirement
    _uuid, _base, needs_data_access = _get_endpoint_uuids(config.system)
    consent_uuids = [spec.endpoints.source_uuid] if needs_data_access else []

    return config, consent_uuids, spec


@app.cell
def submit(GlobusTransferManager, config, consent_uuids, spec):
    manager = GlobusTransferManager(collection_uuids=consent_uuids)
    task_id = manager.transfer(spec, exclude_dirs=config.exclude_patterns)
    print(f"Task ID: {task_id}")
    print(f"Monitor: https://app.globus.org/activity/{task_id}")
    return manager, task_id


@app.cell
def wait(manager, task_id):
    manager.wait(task_id, timeout_minutes=60)
    return


@app.cell
def verify():
    from pathlib import Path

    dest = Path("/mnt/d/Dropbox/_GradSchool/repos/TRITON-SWMM_toolkit/" "frontier/frontier_sensitivity_suite")
    print(f"Exists: {dest.exists()}")
    if dest.exists():
        print(f"Contents: {[p.name for p in sorted(dest.iterdir())]}")
        sims = dest / "sims"
        if sims.exists():
            for d in ["out_triton", "out_tritonswmm", "out_swmm"]:
                excluded = any((s / d).exists() for s in sims.iterdir() if s.is_dir())
                print(f"  sims/*/{d}/ excluded: {not excluded}")
        subanalyses = dest / "subanalyses"
        print(f"  subanalyses/ excluded: {not subanalyses.exists()}")
    return


if __name__ == "__main__":
    app.run()
