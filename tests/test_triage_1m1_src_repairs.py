"""The coder's tests for the source rows of the release-floor triage (package 1m, chunk 1m.1).

One test group per repaired row. Each asserts a property that exists on the pre-change tree and on
the repaired one, so the red-before arm measures behaviour and not new wording. Nothing here compiles
or runs a solver, reaches a network, or needs a case-study dataset.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "hhemt"


# --- row 2: a migration-helper cross-reference in src names a helper that module defines ----------


def _source_files():
    return sorted(_SRC.rglob("*.py"))


def test_every_migration_helper_cross_reference_resolves_to_a_definition():
    """Class: every `V{NNNN}.{name}` reference in `src/hhemt`. A reference to a helper its named
    module does not define sends a reader to the wrong migration, as `utils.py` did when it cited
    `V0023._member_group_count`, which `V0025` defines."""
    versions = _SRC / "version_migration" / "versions"
    modules = {p.name[:5]: p.read_text() for p in versions.glob("V[0-9][0-9][0-9][0-9]__*.py")}
    assert modules, "precondition: the migration modules were found"
    ref = re.compile(r"\b(V\d{4})\.(_?[A-Za-z]\w*)")
    unresolved = []
    for path in _source_files():
        if versions in path.parents:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            for serial, name in ref.findall(line):
                body = modules.get(serial)
                if body is None:
                    continue
                if not re.search(rf"^\s*(?:def|class)\s+{re.escape(name)}\b|^{re.escape(name)}\s*[:=]", body, re.M):
                    unresolved.append(f"{path.relative_to(_SRC)}:{lineno} {serial}.{name}")
    assert not unresolved, f"cross-references to a helper the named migration does not define: {unresolved}"


# --- row 5: preflight reports a sensitivity file it cannot parse -----------------------------------


def _cfg(path, toggle=True):
    return SimpleNamespace(toggle_sensitivity_analysis=toggle, sensitivity_analysis=path)


def _check_readable(cfg):
    from hhemt.validation import ValidationResult, _validate_sensitivity_file_readable

    result = ValidationResult(context="test")
    _validate_sensitivity_file_readable(cfg, result)
    return result


def test_an_unparseable_sensitivity_file_is_a_preflight_error(tmp_path):
    """Class: a file that exists and cannot be parsed, by either reader. A silent return passes
    preflight and fails at workflow build, which is the pre-change behaviour."""
    bad_xlsx = tmp_path / "sens.xlsx"
    bad_xlsx.write_bytes(b"not a workbook")
    bad_csv = tmp_path / "sens.csv"
    bad_csv.write_bytes(b"")  # pandas raises EmptyDataError
    for bad in (bad_xlsx, bad_csv):
        result = _check_readable(_cfg(bad))
        assert [e.field for e in result.errors] == ["analysis.sensitivity_analysis"], bad
        assert str(bad) in str(result.errors[0].current_value)


def test_a_readable_missing_or_unrequested_sensitivity_file_is_not_an_error(tmp_path):
    """The satisfying arms, each a different position: readable, absent, toggle off, no path."""
    good = tmp_path / "sens.csv"
    good.write_text("a,b\n1,2\n")
    bad = tmp_path / "bad.xlsx"
    bad.write_bytes(b"not a workbook")
    assert not _check_readable(_cfg(good)).errors
    assert not _check_readable(_cfg(tmp_path / "absent.csv")).errors
    assert not _check_readable(_cfg(bad, toggle=False)).errors
    assert not _check_readable(_cfg(None)).errors


# --- rows 7 and 8: the renderer package's __init__ stays import-free and declares its surface ------


def test_the_renderer_package_declares_an_explicit_empty_public_surface():
    """Class: the package object. The autodoc gate grounds on `__all__`; an absent one falls back
    to whatever the module happens to define."""
    import hhemt.report_renderers as pkg

    assert pkg.__all__ == []


def test_importing_the_renderer_package_loads_no_config_module():
    """Class: the package `__init__` import path. Held green before and after; the mutation that
    reds it is `from hhemt.config.report import report_config` in the package `__init__`, which
    closes the `config.report -> report_renderers -> config.report` cycle."""
    # `import hhemt` runs hhemt/__init__ first, so isolate the package's own contribution by
    # reading its source: the package module must carry no import statement of its own.
    tree = ast.parse((_SRC / "report_renderers" / "__init__.py").read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.Import | ast.ImportFrom)]
    allowed = [n for n in imports if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    assert imports == allowed, "the renderer package __init__ must import nothing from hhemt"
    proc = subprocess.run(
        [sys.executable, "-c", "import hhemt.report_renderers"], capture_output=True, text=True, cwd=_SRC.parents[1]
    )
    assert proc.returncode == 0, proc.stderr


# --- row 9: a public accessor for the device path token ---------------------------------------------


def test_hardware_path_token_reads_the_label_map_row():
    """Class: every `_DEVICE_LABELS` row. The token an accessor returns is the row's own, for any
    raw string that row's needle matches, so no caller re-slugifies the label."""
    from hhemt.hardware_labels import _DEVICE_LABELS, hardware_path_token

    for needle, _label, token in _DEVICE_LABELS:
        assert hardware_path_token(f"Vendor {needle.upper()} Noise") == token
        assert re.fullmatch(r"[a-z0-9_]+", token), f"{token!r} is not path- and wildcard-safe"


def test_hardware_path_token_keeps_unmapped_devices_apart_and_absent_absent():
    from hhemt.hardware_labels import hardware_path_token

    a = hardware_path_token("AMD Instinct MI250X")
    b = hardware_path_token("AMD Instinct MI300X")
    assert a == "amd_instinct_mi250x" and b == "amd_instinct_mi300x" and a != b
    for absent in (None, "", "   ", "???"):
        assert hardware_path_token(absent) is None


# --- row 19: creators is threaded through every publish facade -------------------------------------

_CREATORS = [{"name": "Doe, Jane", "affiliation": "Example University"}]


def test_the_analysis_facades_forward_creators_to_publish_analysis(monkeypatch):
    """Class: the four publish facades. Each reaches `publish_analysis` with the caller's creators
    unchanged; a facade that does not accept the keyword credits every deposit to the organization."""
    import hhemt.bundle
    import hhemt.publishing as publishing
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis

    seen = []
    monkeypatch.setattr(publishing, "publish_analysis", lambda *a, **kw: seen.append(kw.get("creators")) or {})
    monkeypatch.setattr(hhemt.bundle, "emit_bundle", lambda *a, **kw: Path("bundle.zip"))

    analysis = SimpleNamespace()
    TRITONSWMM_analysis.publish(analysis, "zenodo", creators=_CREATORS)
    TRITONSWMM_analysis.publish_reprex_bundle(analysis, "zenodo", creators=_CREATORS)
    assert seen == [_CREATORS, _CREATORS]

    seen.clear()
    sens = SimpleNamespace(experiment=analysis)
    TRITONSWMM_sensitivity_analysis.publish(sens, "zenodo", creators=_CREATORS)
    sens.experiment = SimpleNamespace(
        publish_reprex_bundle=lambda *a, **kw: seen.append(kw.get("creators")) or {},
    )
    TRITONSWMM_sensitivity_analysis.publish_reprex_bundle(sens, "zenodo", creators=_CREATORS)
    assert seen == [_CREATORS, _CREATORS]


def test_the_facades_default_creators_to_none(monkeypatch):
    """The default-preserving arm: a caller that passes nothing reaches `publish_analysis` with None."""
    import hhemt.publishing as publishing
    from hhemt.analysis import TRITONSWMM_analysis

    seen = []
    monkeypatch.setattr(publishing, "publish_analysis", lambda *a, **kw: seen.append(kw["creators"]) or {})
    TRITONSWMM_analysis.publish(SimpleNamespace(), "zenodo")
    assert seen == [None]


# --- row 23: the HydroShare target does not silently drop creators ---------------------------------


def _hydroshare_publish(monkeypatch, **kw):
    """Call `_HydroShareTarget.publish` against a stubbed hsclient that raises on use, so no
    network is reached and the call ends at the first hsclient touch."""
    import hhemt.publishing as publishing

    def _stub_hydroshare(**_kw):
        raise RuntimeError("stubbed hsclient")

    monkeypatch.setitem(sys.modules, "hsclient", SimpleNamespace(HydroShare=_stub_hydroshare))
    monkeypatch.setenv("HHEMT_HYDROSHARE_USERNAME", "u")
    monkeypatch.setenv("HHEMT_HYDROSHARE_PASSWORD", "p")
    with pytest.raises(RuntimeError, match="stubbed hsclient"):
        publishing._HydroShareTarget().publish(
            deposit=[], license_spdx="CC-BY-4.0", software_doi=None, analysis_dir=Path("."), **kw
        )


def test_hydroshare_warns_when_creators_are_given_and_stays_silent_otherwise(monkeypatch):
    with pytest.warns(UserWarning, match="does not apply `creators`"):
        _hydroshare_publish(monkeypatch, creators=_CREATORS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _hydroshare_publish(monkeypatch)
        _hydroshare_publish(monkeypatch, creators=None)


# --- row 32: no CLI error line interpolates producer text into Rich markup unescaped ---------------


def test_no_cli_error_print_interpolates_unescaped_text_into_markup():
    """Class: every `console_err.print(f"...")` in `cli.py`. A bare interpolated name or call is
    parsed by Rich as markup, so a `[/x]` in an exception message raises MarkupError inside the
    handler, and a `[word]` vanishes from the report."""
    tree = ast.parse((_SRC / "cli.py").read_text())
    unescaped = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "print"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "console_err"
            and node.args
            and isinstance(node.args[0], ast.JoinedStr)
        ):
            continue
        for part in node.args[0].values:
            if not isinstance(part, ast.FormattedValue):
                continue
            expr = part.value
            is_escape = isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "escape"
            is_label = isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "label_for"
            if not (is_escape or is_label):
                unescaped.append(f"cli.py:{node.lineno} {ast.unparse(expr)}")
    assert not unescaped, unescaped


# --- F9: preflight reads the sensitivity file by the extension rule production uses ---------------


@pytest.mark.parametrize("name", ["sens.txt", "sens.xls", "sens"])
def test_preflight_refuses_a_sensitivity_file_whose_extension_production_rejects(tmp_path, name):
    """Class: any extension other than csv and xlsx. A file holding valid CSV text under such a name
    passed preflight and raised ValueError at workflow build."""
    path = tmp_path / name
    path.write_text("a,b\n1,2\n")
    result = _check_readable(_cfg(path))
    assert [e.field for e in result.errors] == ["analysis.sensitivity_analysis"]
    assert "ValueError" in result.errors[0].message


@pytest.mark.parametrize("name", ["sens.csv", "SENS.CSV"])
def test_preflight_accepts_csv_by_the_lowercased_extension(tmp_path, name):
    path = tmp_path / name
    path.write_text("a,b\n1,2\n")
    assert not _check_readable(_cfg(path)).errors
