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


def test_pyproject_swmm_toolkit_pin_stays_guard_passing():
    """q8c regression: pyproject MUST cap swmm-toolkit below the abi3 boundary (<0.16) so a
    `uv lock` re-resolve can only land a 0.15.x wheel -- the version _assert_validated_swmm_stack
    accepts (startswith '0.15.'). Before q8c the pin was >=0.15,<0.17, so a re-lock resolved
    0.16.2 (abi3, the flagged free()/SIGABRT teardown lineage) which the guard REFUSES ->
    [Q8] aborts at native scenario prep. The packaging pin and the runtime guard must agree in
    one direction (Option C: the validated 0.15.x engine ships on PyPI, not only conda-forge)."""
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    swmm = [d for d in deps if d.replace(" ", "").startswith("swmm-toolkit")]
    assert len(swmm) == 1, f"expected exactly one swmm-toolkit dependency, found {swmm}"
    spec = swmm[0].replace(" ", "")
    assert "<0.16" in spec, (
        f"swmm-toolkit pin {swmm[0]!r} must cap <0.16 (the guard accepts only 0.15.x); a <0.17 "
        "cap lets a re-lock resolve guard-failing 0.16.2/0.17.0"
    )
