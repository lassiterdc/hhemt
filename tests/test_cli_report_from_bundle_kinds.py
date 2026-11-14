"""`hhemt report-from-bundle` renders a single or a combined bundle, and `--declare-stale-plots` reaches both.

The command tells the two kinds apart by the files in the unpacked directory: a combined
bundle (the output of `hhemt combine`) holds `child_crates/` and `bundle_manifest.json` and
no root `cfg_analysis.yaml`; a single-analysis bundle holds the config at its root. Each
kind has its own `regenerate_report`. The tests run the command through Typer's
`CliRunner` with both methods replaced by recorders, so no figure is rendered and no solver
runs, and compare which class was called and with which keywords.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import hhemt.bundle as bundle_module
from hhemt.cli import app
from hhemt.exceptions import CLIValidationError

SINGLE = "single"
COMBINED = "combined"


def _make_bundle(root, kind):
    """Create the files that mark `kind` of bundle under `root` and return `root`.

    `single` writes `bundle_manifest.json` and `cfg_analysis.yaml`; `combined` writes
    `bundle_manifest.json` and a `child_crates/` directory; `neither` writes nothing;
    `both` writes all three markers.
    """
    root.mkdir(parents=True, exist_ok=True)
    if kind in (SINGLE, "both"):
        (root / "cfg_analysis.yaml").write_text("analysis_id: x\n")
    if kind in (SINGLE, COMBINED, "both"):
        (root / "bundle_manifest.json").write_text("{}")
    if kind in (COMBINED, "both"):
        (root / "child_crates").mkdir()
    return root


@pytest.fixture
def recorder(monkeypatch):
    """Replace both classes' `from_directory` and `regenerate_report`; return the list of calls.

    Each call is `(class name, keywords)`. `Bundle.manifest` is replaced by an empty dict
    because the command reads it for the toolkit-sha comparison.
    """
    calls = []

    def _record(self, **kwargs):
        calls.append((type(self).__name__, kwargs))
        return self._root / "analysis_report.html"

    for cls in (bundle_module.Bundle, bundle_module.CombinedBundle):
        monkeypatch.setattr(cls, "from_directory", classmethod(lambda c, p: _stand_in(c, p)))
        monkeypatch.setattr(cls, "regenerate_report", _record)
    monkeypatch.setattr(bundle_module.Bundle, "manifest", {}, raising=False)
    monkeypatch.setattr(bundle_module, "_get_toolkit_git_sha", lambda strict=False: "unknown")
    return calls


def _stand_in(cls, path):
    inst = object.__new__(cls)
    inst._root = path
    return inst


def _invoke(path, *args):
    return CliRunner().invoke(app, ["report-from-bundle", str(path), *args])


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize(("kind", "expected_class"), [(SINGLE, "Bundle"), (COMBINED, "CombinedBundle")])
def test_each_kind_calls_its_own_class_with_the_flag_as_given(tmp_path, recorder, kind, expected_class, flag):
    """Property: the bundle kind selects the class, and the flag reaches `regenerate_report` as `declare_stale_plots`.

    Class: both kinds, each with and without `--declare-stale-plots` (four inputs); class
    REGRESSION for the combined rows, red on the tree before this chunk (the command read
    a combined bundle through `Bundle` and failed), output pasted in the chunk return.
    Why required: a combined bundle's report was reachable through the Python API only, and
    the override for a stale-plots refusal must reach whichever kind the directory holds.
    What kills it: a command that always builds `Bundle`, that drops the keyword, or that
    passes `True` unconditionally. A second correct implementation: tell the kinds apart by
    reading `bundle_manifest.json` keys instead of by the three marker files.
    """
    bundle = _make_bundle(tmp_path / "b", kind)
    args = ["--declare-stale-plots"] if flag else []
    result = _invoke(bundle, *args)
    assert result.exit_code == 0, result.output
    assert recorder == [(expected_class, {"format": None, "declare_stale_plots": flag})]


@pytest.mark.parametrize("kind", [SINGLE, COMBINED])
def test_an_explicit_format_reaches_either_kind(tmp_path, recorder, kind):
    """Property: `--format` is passed to `regenerate_report` unchanged for both kinds.

    Class: both kinds with `--format zip`. Why required: the combined branch must keep
    the option's meaning. What kills it: a combined branch that omits `format`. A second
    correct implementation: pass a dict of keywords built once for both branches.
    """
    bundle = _make_bundle(tmp_path / "b", kind)
    result = _invoke(bundle, "--format", "zip")
    assert result.exit_code == 0, result.output
    assert [kw for _, kw in recorder] == [{"format": "zip", "declare_stale_plots": False}]


def test_a_directory_holding_the_single_bundle_config_is_a_single_bundle_even_with_child_crates(tmp_path, recorder):
    """Property: the root `cfg_analysis.yaml` makes a directory a single-analysis bundle.

    Class: a directory with all three markers. Why required: the rule's third clause (no root
    config) is what keeps a single bundle that happens to hold a `child_crates/` directory on
    the command's original path. What kills it: a rule that tests only the first two markers.
    A second correct implementation: test the config first and return early.
    """
    bundle = _make_bundle(tmp_path / "b", "both")
    result = _invoke(bundle)
    assert result.exit_code == 0, result.output
    assert [name for name, _ in recorder] == ["Bundle"]


def test_control_a_directory_with_neither_kinds_files_exits_through_cli_validation_error(tmp_path):
    """Property: a directory holding neither kind's files is refused with `CLIValidationError`.

    Class: an empty directory and a directory with only `child_crates/`. No recorder is
    installed, so the real `Bundle.from_directory` runs and raises `FileNotFoundError`,
    which the command converts. Why required: the refusal names the argument and the two
    layouts instead of surfacing a bare file-not-found. What kills it: a command that lets
    the `FileNotFoundError` through. A second correct implementation: test the markers before
    constructing either class and raise at once.
    """
    for name, setup in (
        ("empty", lambda r: r.mkdir()),
        ("only_children", lambda r: (r / "child_crates").mkdir(parents=True)),
    ):
        root = tmp_path / name
        setup(root)
        result = _invoke(root)
        assert result.exit_code != 0
        assert isinstance(result.exception, CLIValidationError), (name, result.exception)
        assert "neither a single-analysis bundle" in str(result.exception)


def test_a_manifest_without_child_crates_and_without_config_is_read_as_a_single_bundle(tmp_path, recorder, monkeypatch):
    """Property: a directory holding only `bundle_manifest.json` goes to the single-bundle path.

    Class: directories that hold the manifest and neither `child_crates/` nor a root
    `cfg_analysis.yaml` (an unpacked zip that lost its crates is one). Class NEW CAPABILITY;
    the plausible wrong implementation is a rule that drops the `child_crates/` clause, so the
    manifest alone reads as a combined bundle; the test is red under that rule.
    Why required: the shape defines a combined bundle by all three markers, and the
    manifest alone must keep the command's original path and its refusal message.
    What kills it: the `child_crates/` condition removed from `_is_combined_bundle`.
    A second correct implementation: test the markers in any order or return early on the
    config; the test asserts the class that receives the call, not how the rule is spelled.
    """
    monkeypatch.setattr(bundle_module, "_get_toolkit_git_sha", lambda strict=False: "unknown")
    (tmp_path / "bundle_manifest.json").write_text("{}")
    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    assert [name for name, _ in recorder] == ["Bundle"]


@pytest.mark.parametrize("kind", [SINGLE, COMBINED])
def test_a_toolkit_sha_divergence_is_reported_for_either_kind(tmp_path, recorder, monkeypatch, kind):
    """Property: a manifest that names a toolkit sha other than the local one prints the warning, for both kinds.

    Class: both kinds with a manifest `toolkit_git_sha` of `single-sha` against a local sha of
    `local-sha`. Class NEW CAPABILITY; the plausible wrong implementation is a combined
    branch that ignores the manifest's `toolkit_git_sha`, which is red for the combined row.
    Why required: the combined branch reads the key from the manifest file itself, because a
    combined bundle has no `Bundle.manifest`; a bundle emitted by another toolkit commit must
    still warn that the local templates differ. The text is matched as `SHA divergence`,
    because the bare word `divergence` also appears in the pytest temporary path of this test.
    What kills it: the combined branch reading a constant or the wrong key.
    A second correct implementation: compare through `CombinedBundle.manifest` if that
    property existed, or read the file in a helper shared by both branches.
    """
    monkeypatch.setattr(bundle_module, "_get_toolkit_git_sha", lambda strict=False: "local-sha")
    monkeypatch.setattr(bundle_module.Bundle, "manifest", {"toolkit_git_sha": "single-sha"}, raising=False)
    bundle = _make_bundle(tmp_path / "b", kind)
    (bundle / "bundle_manifest.json").write_text(json.dumps({"toolkit_git_sha": "single-sha"}))
    result = _invoke(bundle)
    assert result.exit_code == 0, result.output
    assert "SHA divergence" in result.output
