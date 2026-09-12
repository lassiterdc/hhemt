"""The build-DAG driver: `python -m hhemt.sif.driver --snakefile S --sif-root R --keys k1,k2` runs the
DAG in the FOREGROUND of whatever process hosts it, holding a `_status/_queued/{key}.json` sentinel
per planned identity for the DAG's lifetime (the PENDING-recovery shape of toolkit Gotcha 46).
`detach(...)` is the ONE way a login-node command hosts that foreground: a setsid-detached
`bash -c` chain writing to a log, so no login shell is ever held open ([Q315] 6)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def snakemake_argv(snakefile: Path, sif_root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "snakemake",
        "--snakefile",
        str(snakefile),
        "--profile",
        str(Path(sif_root) / "profile"),
        "--executor",
        "slurm",
        "--keep-going",
    ]


def run_build_dag(snakefile: Path, *, sif_root: Path, keys: set[str]) -> int:
    status = Path(sif_root) / "_status" / "_queued"
    status.mkdir(parents=True, exist_ok=True)
    held = []
    for key in sorted(keys):
        p = status / f"{key}.json"
        p.write_text(json.dumps({"key": key, "driver_pid": os.getpid(), "host": os.uname().nodename}))
        held.append(p)
    try:
        return subprocess.run(snakemake_argv(snakefile, sif_root), cwd=str(sif_root)).returncode
    finally:
        for p in held:
            # EXEMPT-DU: outside-analysis-tree
            p.unlink(missing_ok=True)


def driver_command(snakefile: Path, sif_root: Path, keys: set[str]) -> str:
    return shlex.join(
        [
            sys.executable,
            "-m",
            "hhemt.sif.driver",
            "--snakefile",
            str(snakefile),
            "--sif-root",
            str(sif_root),
            "--keys",
            ",".join(sorted(keys)),
        ]
    )


def detach(commands: list[str], *, log: Path) -> int:
    """Run `cmd1 && cmd2 && ...` in a new session, stdout+stderr to `log`; return the pid.
    The caller returns immediately — the chain outlives the caller's shell."""
    log.parent.mkdir(parents=True, exist_ok=True)
    script = " && ".join(commands)
    with log.open("ab") as fh:
        proc = subprocess.Popen(
            ["bash", "-c", f"set -e; {script}"],
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return proc.pid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snakefile", required=True)
    ap.add_argument("--sif-root", required=True)
    ap.add_argument("--keys", required=True, help="comma-separated identity keys the DAG builds")
    a = ap.parse_args(argv)
    return run_build_dag(Path(a.snakefile), sif_root=Path(a.sif_root), keys={k for k in a.keys.split(",") if k})


if __name__ == "__main__":
    sys.exit(main())
