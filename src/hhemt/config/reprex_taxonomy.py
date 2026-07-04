"""ADR-10 config Path-field -> reprex taxonomy classifier (reproducibility-system C12).

`field_bucket(field_name)` returns the ADR-10 USER / HPC / EXPERIMENT bucket for a
config **Path** field, derived from its `PathPolicy` in the bundle
`_PATH_FIELD_POLICY` table. The classifier is *provably total over the config
Path-field domain* via two composed totalities:
  (1) config Path field -> PathPolicy, enforced by
      tests/test_bundle.py::test_all_path_fields_have_policy (bidirectional
      set-equality between _PATH_FIELD_POLICY keys and
      enumerate_path_fields(system_config) | enumerate_path_fields(analysis_config));
  (2) PathPolicy -> bucket, enforced by
      tests/test_reprex_taxonomy.py::test_policy_to_bucket_is_total
      (set(_POLICY_TO_BUCKET) == set(PathPolicy)).

Domain is config Path fields ONLY. A field_name outside _PATH_FIELD_POLICY raises
KeyError -- the full non-path taxonomy (toggles, case_name, gpu_hardware, ...) is
deferred to bundle-reprex-roundtrip C8. The "hpc" return value is part of the
shared ADR-10 vocabulary but is UNREACHABLE over the Path-field domain: no config
Path field is HPC-bucketed, because HPC identity lives entirely in
hpc_system_config (partition / account / gpu_hardware), which carries zero Path
fields.

No I/O, no runtime state, no Pydantic introspection at classify time -- the
classifier reads only the static policy table (live introspection via
enumerate_path_fields is the exhaustiveness test's job, not the classify path's).

Package-scope import invariant: this module does ``from hhemt.bundle._path_policy
import ...``, which executes ``hhemt.bundle.__init__`` (NOT stdlib-only -- it eagerly
imports ``_emit`` et al.). Acyclicity therefore rests on the invariant that NO module
reachable from ``hhemt.bundle.__init__`` imports ``hhemt.config.reprex_taxonomy``.
Adding a top-level ``from hhemt.config.reprex_taxonomy import field_bucket`` to any
``hhemt.bundle`` module (e.g. an emit-time bucketing call in ``_emit.py``) WOULD form
an import cycle -- route any such call through a function-body local import instead.
"""
from __future__ import annotations

from typing import Literal

from hhemt.bundle._path_policy import _PATH_FIELD_POLICY, PathPolicy

Bucket = Literal["user", "hpc", "experiment"]

# PathPolicy -> ADR-10 bucket. Total over all 6 PathPolicy values (enforced by
# tests/test_reprex_taxonomy.py::test_policy_to_bucket_is_total). No policy maps
# to "hpc": HPC identity lives in hpc_system_config (non-path), so "hpc" is
# unreachable over this domain.
_POLICY_TO_BUCKET: dict[PathPolicy, Bucket] = {
    PathPolicy.FORCED_DOT: "experiment",
    PathPolicy.BUNDLE_RELATIVE: "experiment",
    PathPolicy.BUNDLE_RELATIVE_OR_NONE: "experiment",
    PathPolicy.BUNDLE_RELATIVE_LIST: "experiment",
    PathPolicy.HELPER_RESOLVED: "experiment",  # unused today; total-map filler
    PathPolicy.IS_NONE_ACCEPTABLE: "user",  # nulled at emit = machine-local host path
}

def field_bucket(field_name: str) -> Bucket:
    """Return the ADR-10 USER/HPC/EXPERIMENT bucket for a config Path field.

    Args:
        field_name: A Pydantic Path-field name from system_config or
            analysis_config (a ``_PATH_FIELD_POLICY`` key).

    Returns:
        The reprex bucket: "user" | "hpc" | "experiment".

    Raises:
        KeyError: ``field_name`` is not a config Path field (not in
            ``_PATH_FIELD_POLICY``). Non-path fields (e.g. ``case_name``,
            ``gpu_hardware``) are out of scope for this foundation slice -- see
            bundle-reprex-roundtrip C8 for the full taxonomy.
    """
    try:
        policy = _PATH_FIELD_POLICY[field_name]
    except KeyError:
        raise KeyError(
            f"{field_name!r} is not a config Path field; field_bucket classifies "
            f"only _PATH_FIELD_POLICY keys. Non-path fields are out of scope "
            f"(see bundle-reprex-roundtrip C8)."
        ) from None
    return _POLICY_TO_BUCKET[policy]
