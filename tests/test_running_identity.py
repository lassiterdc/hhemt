"""`hhemt.validation.running_identity` ([Q331]): one identity source per staging shape, no fallback.

Pure unit: every case builds its tree under tmp_path (`git archive` of this repo, a `file://`
clone, or a bare `src/` copy) and runs the function in a SUBPROCESS whose PYTHONPATH is that
tree, so the shape the function sees is the tree's own. No solver is compiled or executed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import hhemt

REPO = Path(hhemt.__file__).resolve().parents[2]
UNSUBSTITUTED = "$Format:%H$"

PROBE = (
    "import sys, pathlib\n"
    "import hhemt.validation as v\n"
    "if len(sys.argv) > 1:\n"
    "    v._LABELS_PATH = pathlib.Path(sys.argv[1])\n"
    "try:\n"
    "    r = v.running_identity(); print('OK', r.sha, r.shape, r.dirty)\n"
    "except Exception as e:\n"
    "    print('REFUSED', type(e).__name__, str(e).replace(chr(10), ' '))\n"
)


def _head_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _archive(dst: Path, *paths: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD", *paths], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dst)], input=tar, check=True)


def _run(tree: Path, labels: Path | None = None, env: dict | None = None) -> str:
    import os

    e = {**os.environ, "PYTHONPATH": str(tree / "src")}
    if env:
        e.update(env)
    argv = [sys.executable, "-c", PROBE] + ([str(labels)] if labels else [])
    return subprocess.run(argv, capture_output=True, text=True, env=e, cwd="/").stdout.strip()


def _labels(tree: Path, sha: str, identity: str | None = None) -> Path:
    p = tree / "labels.json"
    d = {"org.hhemt.hhemt_sha": sha}
    if identity is not None:
        d["org.hhemt.identity"] = identity
    p.write_text(json.dumps(d))
    return p


#: --- image rule -------------------------------------------------------------------------


def test_a_image_archive_with_matching_label(tmp_path):
    t = tmp_path / "img"
    _archive(t, "src", "HHEMT_SHA")
    sha = (t / "HHEMT_SHA").read_text().strip()
    assert sha == _head_sha() and sha != UNSUBSTITUTED
    assert _run(t, _labels(t, sha)) == f"OK {sha} image False"


def test_b_image_label_disagrees(tmp_path):
    t = tmp_path / "img"
    _archive(t, "src", "HHEMT_SHA")
    out = _run(t, _labels(t, "0" * 40))
    assert out.startswith("REFUSED ConfigurationError") and "org.hhemt.hhemt_sha" in out


def test_c_image_with_planted_git_refuses(tmp_path):
    t = tmp_path / "img"
    _archive(t, "src", "HHEMT_SHA")
    (t / ".git").mkdir()
    out = _run(t, _labels(t, _head_sha()))
    assert out.startswith("REFUSED") and "carries a git tree" in out


def test_d_image_unsubstituted_stamp_refuses(tmp_path):
    t = tmp_path / "img"
    _archive(t, "src")
    (t / "HHEMT_SHA").write_text(UNSUBSTITUTED + "\n")
    out = _run(t, _labels(t, _head_sha()))
    assert out.startswith("REFUSED") and "not produced by git archive" in out


#: --- checkout rule ----------------------------------------------------------------------


def _clone(tmp_path: Path) -> Path:
    t = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", f"file://{REPO}", str(t)], check=True)
    return t


def test_e_checkout_reads_git_and_keeps_stamp_unsubstituted(tmp_path):
    t = _clone(tmp_path)
    assert (t / "HHEMT_SHA").read_text().strip() == UNSUBSTITUTED
    head = subprocess.run(["git", "-C", str(t), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert _run(t) == f"OK {head} checkout False"


def test_f_checkout_with_substituted_stamp_is_mixed(tmp_path):
    t = _clone(tmp_path)
    (t / "HHEMT_SHA").write_text(_head_sha() + "\n")
    out = _run(t)
    assert out.startswith("REFUSED") and "mixed shape" in out


def test_g_dirty_checkout_is_flagged(tmp_path):
    t = _clone(tmp_path)
    with (t / "uv.lock").open("a") as f:
        f.write("\n# dirty\n")
    out = _run(t)
    assert out.startswith("OK") and out.endswith("checkout True")


#: --- archive / wheel / shadowing --------------------------------------------------------


def test_h_wheel_staged_tree_refuses(tmp_path):
    t = tmp_path / "wheel"
    _archive(t, "src")
    out = _run(t)
    assert out.startswith("REFUSED") and "has no identity" in out


def test_i_git_dir_env_cannot_shadow_an_archive(tmp_path):
    t = tmp_path / "arch"
    _archive(t, "src", "HHEMT_SHA")
    out = _run(t, env={"GIT_DIR": str(REPO / ".git")})
    assert out == f"OK {_head_sha()} archive False"


#: --- the config field and the construction-time comparison ------------------------------


def test_j_config_field_is_required_and_40_hex():
    from pydantic import ValidationError

    from hhemt.config.analysis import analysis_config

    base = {name: f.default for name, f in analysis_config.model_fields.items() if not f.is_required()}
    with pytest.raises(ValidationError, match="hhemt_sha"):
        analysis_config.model_validate(base)
    for bad in ("v0.1.0", "9eacf5f2", "9EACF5F296B5011FFD9B79D0AEA454C0C32FCBBF", "develop"):
        with pytest.raises(ValidationError, match="hhemt_sha"):
            analysis_config.model_validate({**base, "hhemt_sha": bad})


def test_j2_mismatch_refuses_at_construction_with_exit_2(synth_multi_sim_analysis, tmp_path, monkeypatch):
    """The synth fixture composes hhemt_sha from the tree under test (D3); rewriting it to a
    wrong 40-hex and re-constructing refuses BEFORE the workflow builder, and the CLI maps the
    error to exit 2."""
    import yaml

    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.cli_utils import EXIT_CODE_MAP
    from hhemt.exceptions import ConfigurationError

    a = synth_multi_sim_analysis
    raw = yaml.safe_load(Path(a.analysis_config_yaml).read_text())
    raw["hhemt_sha"] = "0" * 40
    bad = tmp_path / "analysis_config_bad.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ConfigurationError, match="!= running toolkit") as ei:
        TRITONSWMM_analysis(
            analysis_config_yaml=bad, system=a._system, skip_log_update=True, is_main_orchestrator=False
        )
    assert EXIT_CODE_MAP[ConfigurationError] == 2 and "hhemt_sha" in str(ei.value)


def test_k_image_identity_label_mismatch_refuses(tmp_path, monkeypatch):
    """A3: inside an image (labels present) the construction also requires
    org.hhemt.identity == derive_identity(...).key. The synth default carries NO
    hpc_system_config (cfg_hpc_system is None, so A3 is skipped by design), so this case
    builds its own: the test hpc config plus a container: block, threaded through the
    case builder. Simulated in-process: running_identity is patched to an image-shaped
    identity at the case's own sha, and the labels file carries a wrong identity key."""
    import yaml

    from hhemt import analysis as an
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.exceptions import ConfigurationError
    from hhemt.validation import RunningIdentity
    from tests.fixtures.test_case_catalog import Local_TestCases

    hpc = yaml.safe_load((REPO / "tests" / "fixtures" / "hpc_system_config_test.yaml").read_text())
    hpc["container"] = {"sif_root": str(tmp_path / "sifs")}
    hpc_path = tmp_path / "hpc_system_config.yaml"
    hpc_path.write_text(yaml.safe_dump(hpc, sort_keys=False))
    # Isolate the start_from_scratch=True wipe under tmp_path (the conftest fixture's own recipe).
    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    a = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=True, hpc_system_config_yaml=hpc_path
    ).analysis
    assert a.cfg_hpc_system is not None and a.cfg_hpc_system.container is not None
    sha = a.cfg_analysis.hhemt_sha
    monkeypatch.setattr(an, "running_identity", lambda: RunningIdentity(sha=sha, shape="image", dirty=False))
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"org.hhemt.hhemt_sha": sha, "org.hhemt.identity": "not-the-key"}))
    monkeypatch.setattr("hhemt.validation._LABELS_PATH", labels)
    with pytest.raises(ConfigurationError, match="org.hhemt.identity"):
        TRITONSWMM_analysis(
            analysis_config_yaml=a.analysis_config_yaml,
            system=a._system,
            skip_log_update=True,
            is_main_orchestrator=False,
            hpc_system_config_yaml=a.hpc_system_config_yaml,
        )
