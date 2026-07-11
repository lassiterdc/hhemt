"""Unit tests for the ADR-10 field_bucket classifier.

Totality proof composes two set-equality guards:
  (1) test_all_path_fields_have_policy (tests/test_bundle.py): every config
      Path field is a _PATH_FIELD_POLICY key.
  (2) test_policy_to_bucket_is_total (below): every PathPolicy member is a
      _POLICY_TO_BUCKET key.
(1) composed with (2) proves field_bucket is total over config Path fields.
"""
from __future__ import annotations

import pytest

from hhemt.bundle._path_policy import _PATH_FIELD_POLICY, PathPolicy
from hhemt.config.reprex_taxonomy import (
    _FIELD_BUCKET,
    _POLICY_TO_BUCKET,
    all_field_bucket,
    column_bucket,
    field_bucket,
)

_VALID_BUCKETS = {"user", "hpc", "experiment"}

def test_policy_to_bucket_is_total() -> None:
    """Every PathPolicy member maps to a bucket (the enum-coverage guard).

    This is half (2) of the totality proof; half (1) is
    test_all_path_fields_have_policy in tests/test_bundle.py.
    """
    assert set(_POLICY_TO_BUCKET) == set(PathPolicy)
    assert set(_POLICY_TO_BUCKET.values()) <= _VALID_BUCKETS

@pytest.mark.parametrize("field_name", sorted(_PATH_FIELD_POLICY))
def test_field_bucket_returns_valid_bucket(field_name: str) -> None:
    """Behavioral: every config Path field classifies to a valid bucket."""
    assert field_bucket(field_name) in _VALID_BUCKETS

def test_field_bucket_raises_on_non_path_field() -> None:
    """Contract: a non-path / unknown field raises KeyError."""
    with pytest.raises(KeyError):
        field_bucket("case_name")


def test_field_bucket_is_total() -> None:
    """all_field_bucket is provably total over EVERY config field.

    Bidirectional set-equality between the bucketed keys
    (_FIELD_BUCKET | _PATH_FIELD_POLICY) and the live model_fields on both
    config models. A new field with no bucket ("unbucketed") or a removed/
    renamed field left stale in the table ("stale") fails loudly.
    """
    from hhemt.config.analysis import analysis_config
    from hhemt.config.system import system_config

    all_fields = set(system_config.model_fields) | set(analysis_config.model_fields)
    bucketed = set(_FIELD_BUCKET) | set(_PATH_FIELD_POLICY)
    assert bucketed == all_fields, {
        "unbucketed": all_fields - bucketed,  # new field, no bucket -> FAIL loudly
        "stale": bucketed - all_fields,       # removed/renamed field -> FAIL loudly
    }


@pytest.mark.parametrize(
    "column, expected",
    [
        ("hpc.partition", "hpc"),                       # hpc.* root -> hpc
        ("system.target_dem_resolution", "experiment"),  # system.* -> strip -> experiment
        ("analysis.n_gpus", "hpc"),                     # analysis.* -> strip -> hpc
        ("n_gpus", "hpc"),                              # bare HPC field
    ],
)
def test_column_bucket(column: str, expected: str) -> None:
    """Sensitivity-column prefix routing (Gotcha 54)."""
    assert column_bucket(column) == expected


@pytest.mark.parametrize(
    "field_name",
    sorted(
        {
            "execution_environment",
            "run_mode",
            "n_gpus",
            "multi_sim_run_method",
            "hpc_ensemble_partition",
            "local_cpu_cores_for_workflow",
            "local_gpus_for_workflow",
        }
    ),
)
def test_known_hpc_fields_bucket_hpc(field_name: str) -> None:
    """Bucket-CORRECTNESS guard: known-HPC fields must bucket "hpc".

    The totality test only checks completeness; this catches a future HPC
    field silently mis-bucketed "experiment".
    """
    assert all_field_bucket(field_name) == "hpc"


def test_bundle_does_not_import_reprex_taxonomy() -> None:
    """config->bundle acyclicity guard (reprex-foundations C12 / roundtrip C8).

    field_bucket lives in hhemt.config.reprex_taxonomy and imports
    _PATH_FIELD_POLICY/PathPolicy FROM hhemt.bundle._path_policy -- a
    config->bundle edge. That edge stays acyclic only while nothing in the
    hhemt.bundle package imports hhemt.config.reprex_taxonomy back AT
    MODULE-IMPORT TIME. Enforced statically (AST scan of the bundle package
    source) so a future bundle-side top-level import that reintroduces the
    cycle fails CI.

    Scope: MODULE-import-time imports only -- module scope, class body, and
    top-level if/try/with -- because only an import that executes while
    hhemt.bundle is importing can close the cycle. Function/method-body
    (function-local) imports are DEFERRED to call time and are the documented
    sanctioned escape (reprex_taxonomy.py docstring: "route any such call
    through a function-body local import instead"); hhemt.bundle._compatibility
    already uses exactly that pattern for field_bucket, so the scan must NOT
    descend into function bodies (that would flag the sanctioned escape as if
    it were the forbidden top-level import).
    """
    import ast
    import importlib.util
    import pathlib

    class _ModuleLevelImportVisitor(ast.NodeVisitor):
        """Collect imports executed at module-import time. Does NOT descend
        into function/method bodies -- a function-local import is deferred to
        call time and is the sanctioned config->bundle acyclicity escape."""

        def __init__(self) -> None:
            self.imports: list[ast.AST] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            pass  # deliberately do not descend into function bodies

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_Import(self, node: ast.Import) -> None:
            self.imports.append(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            self.imports.append(node)

    spec = importlib.util.find_spec("hhemt.bundle")
    assert spec is not None and spec.origin is not None
    bundle_dir = pathlib.Path(spec.origin).parent

    offenders: list[str] = []
    for py in sorted(bundle_dir.rglob("*.py")):
        visitor = _ModuleLevelImportVisitor()
        visitor.visit(ast.parse(py.read_text(), filename=str(py)))
        for node in visitor.imports:
            if isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "hhemt.config.reprex_taxonomy"
                or node.module.startswith("hhemt.config.reprex_taxonomy.")
            ):
                offenders.append(f"{py.name}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "hhemt.config.reprex_taxonomy" or (
                        alias.name.startswith("hhemt.config.reprex_taxonomy.")
                    ):
                        offenders.append(f"{py.name}: import {alias.name}")

    assert not offenders, (
        "hhemt.bundle must not import hhemt.config.reprex_taxonomy at module "
        "scope (config->bundle acyclicity; reprex_taxonomy imports FROM bundle, "
        "not the reverse; function-local imports are the sanctioned escape): "
        f"{offenders}"
    )
