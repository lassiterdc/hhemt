"""The compile venue guard is WIRED IN, and wired in BEFORE any build is dispatched.

WHY THIS EXISTS SEPARATELY FROM test_compile_venue.py. That module proves the PREDICATE
decides correctly. It cannot prove the predicate is CALLED, or called at the right point.
Those are different failures with the same symptom -- a build starts where it should not.

Gate A unit test: no toolchain, no clone, no solver. It drives the real
`compile_TRITON_SWMM` over a SimpleNamespace double whose prelude methods are no-ops,
so `_download_tritonswmm_source` never runs. The sentinel stands in for
`_compile_backend`, and the assertions are about whether it is REACHED.
"""

from types import SimpleNamespace

import pytest

import hhemt.system as sysmod
from hhemt.exceptions import ConfigurationError

# MAIN-AGENT V13 DISCLOSE-AND-PROCEED, 2026-09-07: round 33's File 4b rewired both
# call sites to `pre_arming_entry_point(...)` and imported nothing. Second missing
# binding in the same round; the form follows tests/conftest.py:10's convention.
from tests.fixtures._compile_guard import pre_arming_entry_point


class _Reached(Exception):
    """Raised by the sentinel. Reaching it means no guard fired above it."""


def _system_double(tmp_path):
    """The minimum compile_TRITON_SWMM reads before dispatching a CPU build.

    Enumerated from system.py:617-666 rather than guessed: three prelude methods, the
    gpu toggle, one config field, and the three sys_paths entries evaluated as arguments
    at the call site. The prelude methods are attributes HERE, not class patches -- an
    unbound call with a SimpleNamespace self never consults TRITONSWMM_system.
    """

    def _sentinel(**_kwargs):
        raise _Reached

    return SimpleNamespace(
        gpu_compilation_backend=None,
        cfg_system=SimpleNamespace(TRITONSWMM_software_directory=tmp_path / "software"),
        sys_paths=SimpleNamespace(
            TRITONSWMM_build_dir_cpu=tmp_path / "build_tritonswmm_cpu",
            compilation_script_cpu=tmp_path / "compile.sh",
            compilation_logfile_cpu=tmp_path / "compilation.log",
        ),
        system_config_yaml=tmp_path / "system.yaml",
        _download_tritonswmm_source=lambda **_kw: None,
        _verify_tritonswmm_pin=lambda **_kw: None,
        _capture_tritonswmm_provenance=lambda **_kw: None,
        _compile_backend=_sentinel,
    )


def test_guard_fires_before_any_build_is_dispatched(monkeypatch, tmp_path):
    """Arm 1 -- the guard is called, and called BEFORE _compile_backend."""
    monkeypatch.setattr(sysmod, "refused_in_undeclared_test_venue", lambda: True)

    with pytest.raises(ConfigurationError):
        pre_arming_entry_point("compile_TRITON_SWMM")(_system_double(tmp_path), backends=["cpu"], verbose=False)


def test_build_is_dispatched_when_the_venue_is_declared(monkeypatch, tmp_path):
    """Arm 2 -- the discriminating half. A guard that refuses unconditionally passes
    arm 1 and fails here. Anchored on sentinel-REACHED, a property present in BOTH
    states, so neither arm can be tautologically green."""
    monkeypatch.setattr(sysmod, "refused_in_undeclared_test_venue", lambda: False)

    with pytest.raises(_Reached):
        pre_arming_entry_point("compile_TRITON_SWMM")(_system_double(tmp_path), backends=["cpu"], verbose=False)
