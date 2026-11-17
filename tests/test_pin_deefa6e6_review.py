"""Reviewer tests for the pin move (the synthetic tier's live solver pin moves to the fork's develop head).

Uncommitted reviewer file. No solver compiles or runs; the one git-reading test skips when no full clone is at hand.
Each test varies the PROPERTY under suspicion (a pin that is registered in one place and missing in another),
not a token.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_NEW = "deefa6e6fcb26184a60eb8242dc82dcc3cab9618"
_OLD = "faba223212408f04b7a3dbf6580adedf4a4e6b13"
_BB13 = "bb13ea28" + "603d50879f8d85cd041c17f440fbdb2b"
_CLONE = Path(os.environ.get("HHEMT_TRITON_CLONE", str(Path.home() / "dev" / "triton-custom")))


def _module_string(rel: str, name: str) -> str:
    for node in ast.parse((_ROOT / rel).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{rel} binds no module-level {name}")


def _norfolk() -> dict:
    return yaml.safe_load((_ROOT / "test_data/norfolk_coastal_flooding/template_system_config.yaml").read_text())


def test_every_registry_entry_resolves_absent_at_the_live_pin_from_the_explicit_set():
    """Property: ALL registry entries read ABSENT from `known_absent_in` at the live pin (no entry left indeterminate).

    Class: the all-entries registration of a pin move. Why: the guard that holds it is a pre-commit hook, which a
    test run does not run, and an entry missing the sha reads indeterminate and a check passes over it.
    Kills it: the new sha dropped from ONE entry's known_absent_in, or moved to also_present_in.
    Control: the number of entries is nonzero and an unregistered sha resolves indeterminate on every entry.
    """
    from hhemt.model_defects import REGISTRY, SHA_TRITON_CUSTOM_DIAGNOSTICS_PIN, resolve

    assert SHA_TRITON_CUSTOM_DIAGNOSTICS_PIN == _NEW
    assert len(REGISTRY) >= 8
    for d in REGISTRY:
        v = resolve(d, _NEW)
        assert (v.status, v.rule) == ("absent", "known_absent_set"), (d.defect_id, v)
    # Control: the same loop over a sha in no set does NOT read absent, so the assertion above can fail.
    assert all(resolve(d, "0123456789abcdef" * 2 + "01234567").status != "absent" for d in REGISTRY)


def test_the_outgoing_and_the_baseline_pin_stay_registered_in_every_entry():
    """Property: `faba2232` and `bb13ea28` still read ABSENT from the explicit set on every entry.

    Class: recorded runs and the neutrality baseline. Kills it: replacing a member instead of adding one.
    """
    from hhemt.model_defects import REGISTRY, SHA_MAIN_SWMM_PATH_REFUSAL_PIN, SHA_ORNL_COUPLING_DECLARATION_PIN, resolve

    assert SHA_ORNL_COUPLING_DECLARATION_PIN == _OLD and SHA_MAIN_SWMM_PATH_REFUSAL_PIN == _BB13
    for sha in (_OLD, _BB13):
        for d in REGISTRY:
            assert resolve(d, sha).status == "absent", (sha[:8], d.defect_id)


def test_the_live_pin_the_registry_the_norfolk_example_and_the_clone_url_agree():
    """Property: one commit and one clone URL across the fixture, the registry constant and the Norfolk example."""
    from hhemt.model_defects import SHA_TRITON_CUSTOM_DIAGNOSTICS_PIN

    fixture = "tests/fixtures/_triton_source_cache.py"
    pin = _module_string(fixture, "TRITON_PIN")
    url = _module_string(fixture, "TRITON_GIT_URL")
    cfg = _norfolk()
    assert pin == SHA_TRITON_CUSTOM_DIAGNOSTICS_PIN == cfg["TRITONSWMM_branch_key"] == _NEW
    assert url == cfg["TRITONSWMM_git_URL"] == "https://github.com/lassiterdc/triton.git"
    # Control: a different sha does not satisfy the chain.
    assert _OLD != cfg["TRITONSWMM_branch_key"]


def test_the_runnable_model_how_to_names_the_commit_the_norfolk_template_pins():
    """Property: the page that states what the Norfolk template pins states what the template pins.

    Class: a prose restatement of a coupled literal that no census reaches (the census is `tests` and `src`).
    Kills it: moving the template's branch key and leaving the page naming the ORNL head.
    Control: the page names a 40-hex commit at all.
    """
    page = (_ROOT / "docs/how-to/using-a-runnable-swmm-model.md").read_text(encoding="utf-8")
    line = next(ln for ln in page.splitlines() if "TRITONSWMM_branch_key" in ln and "Norfolk" in ln)
    named = re.findall(r"\b[0-9a-f]{40}\b", line)
    assert named, line
    assert named == [_norfolk()["TRITONSWMM_branch_key"]], (named, line[-220:])


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(_CLONE), *args], capture_output=True, text=True)


needs_clone = pytest.mark.skipif(
    not (_CLONE / ".git").exists() or _git("cat-file", "-e", f"{_NEW}^{{commit}}").returncode != 0,
    reason="no full clone holding the new pin (set HHEMT_TRITON_CLONE)",
)


def _count(sha: str, path: str, needle: str) -> int:
    out = _git("show", f"{sha}:{path}").stdout
    return len(re.findall(needle, out))


@needs_clone
def test_the_registry_probes_hold_at_the_new_pin():
    """Property: the ancestry and content probes the registry comment records are reproducible at `deefa6e6`.

    Unit: ancestry is the rc of `merge-base --is-ancestor`; content is occurrences of the needle in the file at that
    sha.
    Control: each content needle is nonzero at the commit that carries the fix (the property varies the SHA).
    """
    for anc in ("3a832f7d", "9db367dd", "0cf5faff", "a38338b0", "faba2232"):
        assert _git("merge-base", "--is-ancestor", anc, _NEW).returncode == 0, anc
    for not_anc in ("5d2ad1e8", "681d809", "d456518", "852f10c", "2ecb1cf"):
        assert _git("merge-base", "--is-ancestor", not_anc, _NEW).returncode == 1, not_anc
    assert _git("rev-list", "--count", f"{_OLD}..{_NEW}").stdout.strip() == "10"
    snap, sw, out_h = "external/swmm/src/solver/snapshot.c", "src/swmm_triton.h", "src/output.h"
    probes = (
        (snap, r"s->routeTo >= 0", 1, "681d809", 1, "031ea42b", 0),
        (sw, r"swmm_snapshot_keep_previous", 5, "d456518", 5, "681d809", 0),
        (sw, r"coupled_node_rim\(", 2, "852f10c", 2, "d456518", 0),
        (sw, r"0\.3048", 2, "2ecb1cf", 2, "852f10c", 0),
    )
    for path, needle, want, fix, at_fix, before, at_before in probes:
        assert _count(_NEW, path, needle) == want, (path, needle)
        assert _count(fix, path, needle) == at_fix, (fix, needle)  # control: nonzero where the fix lives
        assert _count(before, path, needle) == at_before, (before, needle)  # control: zero before it
    assert _count(_NEW, "external/swmm/src/solver/snapshot.c", r"oldQual") == 7
    assert _count(_NEW, "src/config_utils.h", r'argsd\("swmm_node_rim_from_dem"') == 1
    assert _count(_NEW, "src/config_utils.h", r'argsd\("swmm_coupling_nodes"') == 1
    assert _count(_NEW, out_h, r"write_output_ghost_ring|read_output_ghost_ring") == 5


@needs_clone
def test_the_golden_spec_byte_layout_claims_hold_at_the_new_pin():
    """Property: the raster writer region of `src/output.h` is byte-identical between the old and the new pin.

    Class: SPEC.md section 7.1's claim that the byte layout follows `src/output.h` of the cited commit. Unit: added or
    removed diff lines that carry a layout needle. Control: the same needles occur in the old file (nonzero).
    """
    diff = _git("diff", "-U0", _OLD, _NEW, "--", "src/output.h").stdout
    body = [ln for ln in diff.splitlines() if re.match(r"^[-+][^-+]", ln)]
    needle = r"get_mat_path|write_output_binary|GHOST_CELL_PADDING|sizeof\(T\)|outfile_pattern|MH_"
    assert sum(len(re.findall(needle, ln)) for ln in body) == 0
    assert len(re.findall(needle, _git("show", f"{_OLD}:src/output.h").stdout)) > 0
    assert _git("diff", "--quiet", _OLD, _NEW, "--", "doc/configuration_reference.rst").returncode == 0
