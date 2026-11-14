"""emit_runner_flag must carry the runner's --member-id into the sidecar's ``sa_id`` key.

Every runner defines ``--member-id`` (argparse dest ``member_id``); none defines
``--sa-id``. ``emit_runner_flag`` once read ``args.sa_id`` so no runner-written
sidecar carried a member id. The payload KEY stays ``sa_id`` (no on-disk format change).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hhemt.status_flags import emit_runner_flag


def _runner_namespace(argv: list[str]) -> argparse.Namespace:
    """Parse the argument surface every runner shares (names as in the runners' parsers)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--member-id", type=str, default=None)
    parser.add_argument("--flag-output", type=str, default=None)
    parser.add_argument("--rule-name", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    return parser.parse_args(argv)


def _read_sidecar(flag: Path) -> dict:
    return json.loads(flag.with_suffix(flag.suffix + ".json").read_text())


def test_member_id_argument_lands_in_sa_id_key(tmp_path: Path) -> None:
    flag = tmp_path / "_status" / "run_sim_32.flag"
    args = _runner_namespace(["--member-id", "32", "--flag-output", str(flag), "--rule-name", "run_sim"])
    assert not hasattr(args, "sa_id")  # the runner namespace carries only member_id

    emit_runner_flag(args)

    assert _read_sidecar(flag)["sa_id"] == "32"


def test_absent_member_id_writes_no_sa_id_key(tmp_path: Path) -> None:
    flag = tmp_path / "_status" / "consolidate.flag"
    args = _runner_namespace(["--flag-output", str(flag), "--rule-name", "consolidate"])

    emit_runner_flag(args)

    assert "sa_id" not in _read_sidecar(flag)
