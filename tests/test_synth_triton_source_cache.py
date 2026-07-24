"""Regression guards for the shared TRITON object store (Phase 1).

The first two tests are cheap and construct no builder — they build tiny
git repos under `tmp_path` and never touch the network or the real cache.
The third constructs a builder with `start_from_scratch=False, skip_run=True`,
which never compiles (the only expensive call, `process_system_level_inputs`,
is gated on `start_from_scratch and not skip_run`).
"""

import subprocess
from pathlib import Path

import pytest

from tests.fixtures._triton_source_cache import (
    TRITON_PIN,
    _reborrow_in_place,
    borrower_is_healthy,
    is_borrowing,
    synthetic_runs_root,
)
from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )


def _init_origin(root: Path) -> Path:
    """A tiny standalone repo with one commit, used as the clone source."""
    origin = root / "origin"
    origin.mkdir(parents=True)
    _git("init", "-q", "-b", "main", cwd=origin)
    _git("config", "user.email", "t@t.invalid", cwd=origin)
    _git("config", "user.name", "t", cwd=origin)
    (origin / "CMakeLists.txt").write_text("# tiny\n")
    _git("add", "CMakeLists.txt", cwd=origin)
    _git("commit", "-q", "-m", "init", cwd=origin)
    return origin


def test_borrower_is_healthy_detects_a_missing_canonical(tmp_path):
    """rev-parse --verify is the ONLY sound probe: a .git-exists check and an
    alternates-target-exists check are both false-clean after a canonical gc."""
    origin = _init_origin(tmp_path)
    canonical = tmp_path / "canonical"
    borrower = tmp_path / "borrower"

    _git("clone", "-q", str(origin), str(canonical), cwd=tmp_path)
    # A `file://` URL is REQUIRED here, and `--no-hardlinks` is not a substitute.
    # Cloning from a bare local PATH triggers git's local-copy optimization: it
    # copies (or hardlinks) the whole object store into the borrower, which then
    # does NOT depend on the alternate at all — while still writing an
    # `objects/info/alternates` file. Measured on git 2.x with this exact fixture:
    # the local-path borrower kept 4 own object files and `rev-parse HEAD` still
    # resolved after the canonical was moved away, so the test passed vacuously.
    # The `file://` transport performs real pack negotiation and skips objects
    # already reachable through the alternate (1 own object), which is what the
    # production https:// clone does.
    _git(
        "-c", "protocol.file.allow=always",
        "clone", "-q",
        "--reference", str(canonical / ".git"),
        f"file://{origin}", str(borrower),
        cwd=tmp_path,
    )
    pin = _git("rev-parse", "HEAD", cwd=borrower).stdout.strip()

    assert borrower_is_healthy(borrower, pin), "freshly provisioned borrower must be healthy"

    # Sever the donor. Everything a naive gate would look at survives.
    (canonical).rename(tmp_path / "canonical_moved")

    assert (borrower / ".git").exists(), "precondition: .git survives — a .git check is false-clean"
    assert (borrower / "CMakeLists.txt").exists(), (
        "precondition: the working tree survives, so system.py's clone gate would skip re-cloning"
    )
    assert not borrower_is_healthy(borrower, pin), (
        "borrower must be reported unhealthy once the borrowed object store is gone"
    )


def test_borrower_is_healthy_requires_checkout_identity(tmp_path):
    """Resolvability is not enough: a borrower can RESOLVE a sha it is not
    CHECKED OUT AT, and system.py::_verify_tritonswmm_pin compares the checkout.
    This is the condition that makes the Phase-2 pin bump visible to the gate."""
    origin = _init_origin(tmp_path)
    (origin / "second.txt").write_text("second\n")
    _git("add", "second.txt", cwd=origin)
    _git("commit", "-q", "-m", "second", cwd=origin)

    borrower = tmp_path / "borrower"
    _git("clone", "-q", str(origin), str(borrower), cwd=tmp_path)

    head = _git("rev-parse", "HEAD", cwd=borrower).stdout.strip()
    first = _git("rev-parse", "HEAD~1", cwd=borrower).stdout.strip()

    assert borrower_is_healthy(borrower, head)
    # `first` resolves fine in this repo, but the tree is not checked out at it.
    assert _git("rev-parse", "--verify", f"{first}^{{commit}}", cwd=borrower).returncode == 0
    assert not borrower_is_healthy(borrower, first), (
        "a resolvability-only gate would return True here; checkout identity must reject it"
    )


def test_standalone_clone_is_reborrowed_in_place_without_losing_the_build(tmp_path):
    """The health gate cannot see a destroyed borrow, and the repair must not
    delete the build tier.

    Two distinct properties, both regression-prone:
      (a) `borrower_is_healthy` returns True for a STANDALONE clone at the pin.
          That is by design (it is a pin gate), and it is exactly why
          `provision_borrower`'s reuse gate needs `is_borrowing` as a SECOND
          conjunct. Measured on the real cache: a 217 MB standalone clone with no
          alternates file passed the health gate and was adopted permanently.
      (b) the repair is IN PLACE. A destructive re-provision would `fast_rmtree`
          the tree, and every TRITON build dir is nested inside it — so the marker
          file below stands in for `build_tritonswmm_cpu/triton.exe`, which a
          destructive self-heal deletes mid-suite.

    This test asserts the STATE TRANSITION, not the size: a local-path clone
    triggers git's hardlink optimisation, so byte counts here are meaningless. The
    size ceiling is asserted against a real provisioned tree by
    `test_software_root_is_real_dir_and_object_store_is_borrowed`.
    """
    origin = _init_origin(tmp_path)
    canonical = tmp_path / "canonical"
    _git("clone", "-q", str(origin), str(canonical), cwd=tmp_path)

    standalone = tmp_path / "standalone"
    _git("clone", "-q", str(origin), str(standalone), cwd=tmp_path)
    pin = _git("rev-parse", "HEAD", cwd=standalone).stdout.strip()

    assert borrower_is_healthy(standalone, pin), (
        "precondition: a standalone clone at the pin PASSES the health gate — "
        "this is why a second predicate is required"
    )
    assert not is_borrowing(standalone), "precondition: it borrows nothing"

    build = standalone / "build_tritonswmm_cpu"
    build.mkdir()
    (build / "triton.exe").write_text("compiled")

    assert _reborrow_in_place(standalone, canonical, pin), "in-place re-borrow failed"

    assert is_borrowing(standalone)
    assert borrower_is_healthy(standalone, pin), "the pin must still resolve after repack"
    assert (build / "triton.exe").read_text() == "compiled", (
        "the re-borrow destroyed the build tier — it must never delete the tree"
    )
    alternates = standalone / ".git" / "objects" / "info" / "alternates"
    assert alternates.read_text().strip() == str(canonical / ".git" / "objects")
    assert _git("config", "--get", "gc.auto", cwd=standalone).stdout.strip() == "0", (
        "config parity with a fresh --reference clone is what makes the two "
        "provisioning paths converge on one state"
    )


def test_bare_canonical_fails_loud_not_silently(tmp_path):
    """A modules-less canonical must fail non-zero, never degrade to an
    undeduped clone. Guards against a future --reference-if-able 'fix'.

    A BARE canonical has no `.git/modules/{name}`, so `submodule.alternateLocation
    =superproject` derives a nonexistent alternate for each submodule and the
    default `die` strategy aborts. `--reference-if-able` would exit 0 here while
    silently leaving the submodule undeduped — the 51 MB/worktree regression.
    """
    sub_origin = _init_origin(tmp_path / "sub")
    super_origin = tmp_path / "super"
    super_origin.mkdir()
    _git("init", "-q", "-b", "main", cwd=super_origin)
    _git("config", "user.email", "t@t.invalid", cwd=super_origin)
    _git("config", "user.name", "t", cwd=super_origin)
    _git("-c", "protocol.file.allow=always", "submodule", "add", "-q",
         str(sub_origin), "external/tiny", cwd=super_origin)
    _git("commit", "-q", "-m", "add submodule", cwd=super_origin)

    bare = tmp_path / "canonical_bare.git"
    _git("clone", "-q", "--bare", str(super_origin), str(bare), cwd=tmp_path)

    borrower = tmp_path / "borrower"
    _git("clone", "-q", "--reference", str(bare), str(super_origin), str(borrower), cwd=tmp_path)
    _git("config", "submodule.alternateLocation", "superproject", cwd=borrower)
    _git("config", "submodule.alternateErrorStrategy", "die", cwd=borrower)

    result = subprocess.run(
        ["git", "-c", "protocol.file.allow=always",
         "submodule", "update", "--init", "--recursive"],
        cwd=str(borrower), capture_output=True, text=True,
    )
    assert result.returncode != 0, (
        "a bare (modules-less) canonical must fail LOUD; it exited 0, which means the "
        "borrow silently degraded to an undeduped clone.\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_single_sourced_runs_root_literal():
    """R12: exactly one `synthetic_test_runs` literal per legitimate site.

    Uses a filesystem scan, NOT `git grep` — `git grep` skips untracked files, so
    it cannot see a newly created module and would under-report.
    """
    tests_dir = Path(__file__).resolve().parent
    self_path = Path(__file__).resolve()
    hits = sorted(
        p.relative_to(tests_dir).as_posix()
        for p in tests_dir.rglob("*.py")
        # Exclude THIS module: its own assertion text contains the literal, which
        # would otherwise make the guard fail on itself.
        if p.resolve() != self_path and '"synthetic_test_runs"' in p.read_text()
    )
    assert hits == [
        "fixtures/_triton_source_cache.py",
        "fixtures/test_case_builder.py",
    ], (
        "expected exactly two sites: the single source in _triton_source_cache.py and "
        f"the deliberately-retained runs_root_override branch in test_case_builder.py; got {hits}"
    )


@pytest.mark.slow
def test_software_root_is_real_dir_and_object_store_is_borrowed():
    """Gotcha 52: the BUILD tier is never shared. The OBJECT STORE is.

    Three assertions on the alternates, not one: existence alone passes on a
    tree whose donor was gc'd or removed.
    """
    builder = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="single_sim", start_from_scratch=False, skip_run=True
    )
    assert not builder._software_root.is_symlink()
    triton = builder._software_root / "triton"
    assert triton.is_dir() and not triton.is_symlink()
    alternates = triton / ".git" / "objects" / "info" / "alternates"
    # (a) the sanctioned share exists at all
    assert alternates.exists(), "expected borrowed objects from the canonical clone"
    # (b) it is not DANGLING — the new corruption class this plan introduces
    donor = Path(alternates.read_text().strip())
    assert donor.is_dir(), f"alternates points at a missing canonical: {donor}"
    # (c) the borrowed objects are REACHABLE — what _verify_tritonswmm_pin demands
    rc = subprocess.run(
        ["git", "-C", str(triton), "rev-parse", "--verify", f"{TRITON_PIN}^{{commit}}"],
        capture_output=True,
    ).returncode
    assert rc == 0, "pinned commit unreachable through borrowed objects"
    # (d) the saving is real — the alternates file is PRESENT in the regressed
    #     state too (a borrower repack silently re-absorbs), so assert a ceiling
    git_bytes = sum(f.stat().st_size for f in (triton / ".git").rglob("*") if f.is_file())
    assert git_bytes < 5 * 1024**2, f".git is {git_bytes} bytes — objects were re-absorbed"


def test_synthetic_runs_root_is_outside_the_canonical():
    """The canonical must never be nested inside the reaper's sweep root."""
    from tests.fixtures._triton_source_cache import canonical_root

    assert not str(canonical_root()).startswith(str(synthetic_runs_root())), (
        "canonical_root() must be a SIBLING of synthetic_test_runs/, never nested inside it, "
        "or the Phase 4 reaper could target it"
    )
