"""The container-target preflight accepts a SIF file OR an apptainer sandbox directory.

WHY THE SANDBOX ARM EXISTS. `apptainer exec` runs a sandbox directory exactly as it runs a SIF, and
on a cluster with no ``/etc/subuid`` mapping and ptrace restricted (``yama.ptrace_scope=3``)
apptainer falls back to proot for root emulation and cannot produce a SIF at all -- both ``%post``
execution and squashfs packing fail, while ``apptainer build --sandbox`` from a docker-archive
succeeds. A file-only existence test rejects a working container on such a cluster.

WHY IT IS NOT SIMPLY ``exists()``. The check exists to catch a declared-but-absent container at
login-node preflight rather than as an opaque ``apptainer exec`` failure inside a SLURM allocation.
Widening it to any path would retire that guarantee. The sandbox arm is therefore keyed on the
``.singularity.d/`` marker directory apptainer writes at every sandbox root, so a typo'd path that
happens to name an existing directory still fails.
"""

from __future__ import annotations

import pytest

from hhemt.config.hpc_system import ContainerSpec
from hhemt.validation import ValidationResult


class _Analysis:
    """Minimal stand-in: the validator reads execution_environment via getattr only."""

    execution_environment = "container"


class _HpcSystem:
    def __init__(self, container):
        self.container = container


def _run_check(sif_path):
    """Invoke the real preflight validator, not a re-implementation of its rule.

    Calls _validate_container_config directly. Do NOT substitute a local copy of the
    file-or-sandbox predicate here: a test that restates the rule it is checking passes
    whether or not the shipped code agrees with it.
    """
    from hhemt.validation import _validate_container_config

    result = ValidationResult()
    cspec = ContainerSpec(sif_path=str(sif_path))
    _validate_container_config(_Analysis(), _HpcSystem(cspec), result)
    return result


@pytest.fixture
def sif_file(tmp_path):
    p = tmp_path / "image.sif"
    p.write_bytes(b"\x00")
    return p


@pytest.fixture
def sandbox_dir(tmp_path):
    p = tmp_path / "image_sandbox"
    (p / ".singularity.d").mkdir(parents=True)
    return p


def test_sif_file_accepted(sif_file):
    assert _run_check(sif_file).is_valid


def test_sandbox_directory_accepted(sandbox_dir):
    """The regression this test exists for: a sandbox is a valid container target."""
    assert _run_check(sandbox_dir).is_valid


def test_bare_directory_rejected(tmp_path):
    """A directory WITHOUT the marker must still fail -- the guard is an existence proof,
    not a widening to 'any path that happens to exist'."""
    plain = tmp_path / "not_a_sandbox"
    plain.mkdir()
    assert not _run_check(plain).is_valid


def test_absent_path_rejected(tmp_path):
    assert not _run_check(tmp_path / "nothing_here").is_valid
