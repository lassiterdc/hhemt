import importlib.metadata

import tests.utils_for_testing as tst_ut


def _patch_versions(monkeypatch, mapping):
    def fake_version(pkg):
        if pkg not in mapping:
            raise importlib.metadata.PackageNotFoundError(pkg)
        return mapping[pkg]

    monkeypatch.setattr(importlib.metadata, "version", fake_version)


def test_swmm_stack_version_mismatch_ok(monkeypatch):
    _patch_versions(monkeypatch, {"pyswmm": "2.0.1", "swmm-toolkit": "0.15.0"})
    assert tst_ut.swmm_stack_version_mismatch() is None


def test_swmm_stack_version_mismatch_pyswmm_downgraded(monkeypatch):
    _patch_versions(monkeypatch, {"pyswmm": "1.5.1", "swmm-toolkit": "0.15.0"})
    msg = tst_ut.swmm_stack_version_mismatch()
    assert msg is not None and "pyswmm==1.5.1" in msg


def test_swmm_stack_version_mismatch_swmm_toolkit_wrong_major(monkeypatch):
    _patch_versions(monkeypatch, {"pyswmm": "2.0.1", "swmm-toolkit": "0.17.0"})
    msg = tst_ut.swmm_stack_version_mismatch()
    assert msg is not None and "swmm-toolkit" in msg and "0.17.0" in msg


def test_swmm_stack_version_mismatch_absent(monkeypatch):
    _patch_versions(monkeypatch, {"swmm-toolkit": "0.15.0"})
    msg = tst_ut.swmm_stack_version_mismatch()
    assert msg is not None and "pyswmm not installed" in msg
