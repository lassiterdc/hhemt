"""Unit tests for the ADR-19 on-ingest SIF builder (`hhemt.container_build`).

Every test here is hermetic: `sbatch` / `apptainer` are never invoked. The point is to pin
the CONTRACT (the resource ask, the preflight, the cache home) rather than to exercise a
real build, which is what the [Q8] live-Rivanna stress test is for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hhemt.container_build import (
    build_sbatch_argv,
    build_sif,
    compute_cache_key,
    get_or_build_sif,
    render_build_script,
    sif_cache_root,
)
from hhemt.exceptions import ConfigurationError, ProcessingError


def _write_def(tmp_path: Path, *, compiling: bool = True) -> Path:
    body = "Bootstrap: docker\nFrom: nvidia/cuda:12.8.0-devel-ubuntu24.04@sha256:deadbeef\n\n%post\n"
    body += "    make -j$(nproc) all\n" if compiling else "    echo hello\n"
    p = tmp_path / "containers" / "test.def"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


class TestCacheKey:
    def test_deterministic_and_input_sensitive(self):
        a = compute_cache_key(b"def", "sha256:x", b"lock", "a100")
        assert a == compute_cache_key(b"def", "sha256:x", b"lock", "a100")
        # Each of the four inputs must move the key — otherwise two different recipes
        # could collide onto one cached SIF, which is the one thing a content-addressed
        # cache must never do.
        assert a != compute_cache_key(b"DEF", "sha256:x", b"lock", "a100")
        assert a != compute_cache_key(b"def", "sha256:y", b"lock", "a100")
        assert a != compute_cache_key(b"def", "sha256:x", b"LOCK", "a100")
        assert a != compute_cache_key(b"def", "sha256:x", b"lock", "mi250")


class TestCacheHome:
    def test_sif_cache_home_is_outside_bundle_root(self, tmp_path, monkeypatch):
        """The OE-2 regression test.

        `extract_reprex_bundle` rmtree's `bundle_root` on every ingest
        (`bundle/_reprex.py:156-159`), so a cache under it can NEVER hit — and the [Q8]
        runbook's Leg 2 depends on the hit. A miss makes Leg 2 submit a second ~1.6 h
        `sbatch --wait` build from inside the GPU allocation. Without this test the hazard
        is re-introducible by a one-line default change.
        """
        monkeypatch.delenv("HHEMT_SIF_CACHE_DIR", raising=False)
        bundle_root = tmp_path / "bundle"
        bundle_root.mkdir()
        assert not sif_cache_root().is_relative_to(bundle_root)

    def test_env_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HHEMT_SIF_CACHE_DIR", str(tmp_path / "mycache"))
        assert sif_cache_root() == tmp_path / "mycache"


class TestSbatchForm:
    def test_emits_the_ratified_resource_ask(self, tmp_path):
        argv = build_sbatch_argv(
            account="myalloc",
            sif_out=tmp_path / "out.sif",
            build_script=tmp_path / "b.sh",
            log_dir=tmp_path / "logs",
        )
        j = " ".join(argv)
        assert "--wait" in j  # from_doi must block on the build
        assert "-p standard" in j  # CPU partition; the build needs no GPU
        assert "-c 16" in j
        assert "--mem=64G" in j
        # defect-9 (2026-07-20): raised from 04:00:00 after a6000 job 17140966 TIMEOUTed at
        # Elapsed 04:00:30 with every compile already done. The build is I/O-bound, so size
        # for the tail of the Weka/node-variance spread, not the median.
        assert "-t 08:00:00" in j
        assert "-A myalloc" in j

    def test_walltime_is_env_overridable(self, tmp_path, monkeypatch):
        """defect-9: the build walltime must be tunable per-cluster without a source edit.

        A reproducer on a slower filesystem than Rivanna's Weka would otherwise have to patch
        library code to avoid the same TIMEOUT. The override is read at CALL time (not import
        time), giving parity with the other four env knobs.
        """
        monkeypatch.delenv("HHEMT_SIF_BUILD_WALLTIME", raising=False)
        base = build_sbatch_argv(
            account="a", sif_out=tmp_path / "o.sif", build_script=tmp_path / "b.sh", log_dir=tmp_path
        )
        assert base[base.index("-t") + 1] == "08:00:00"

        monkeypatch.setenv("HHEMT_SIF_BUILD_WALLTIME", "12:00:00")
        overridden = build_sbatch_argv(
            account="a", sif_out=tmp_path / "o.sif", build_script=tmp_path / "b.sh", log_dir=tmp_path
        )
        assert overridden[overridden.index("-t") + 1] == "12:00:00"

    def test_never_emits_tmp_or_gres(self, tmp_path):
        """`--tmp` is a submission-killer and `--gres` is a billing error.

        Every Rivanna node reports TmpDisk=0, so ANY `--tmp=N` fails submission validation
        outright. A `--gres` ask would pay ~5-10x the billing weight for work that needs no
        device (nvcc cross-compiles for the Kokkos-named arch).
        """
        argv = build_sbatch_argv(
            account="a", sif_out=tmp_path / "o.sif", build_script=tmp_path / "b.sh", log_dir=tmp_path
        )
        j = " ".join(argv)
        assert "--tmp" not in j
        assert "--gres" not in j

    def test_never_defaults_to_the_producers_allocation(self, tmp_path):
        """The ratified spec wrote `-A "${HHEMT_SLURM_ACCOUNT:-quinnlab}"`.

        `quinnlab` is the producer's UVA allocation and is on the zero-user-info blocklist;
        defaulting public library code to it would make a third-party reproducer submit
        against an account they do not belong to. The account comes from the reproducer's own
        hpc_system_config instead.
        """
        argv = build_sbatch_argv(
            account="theirs", sif_out=tmp_path / "o.sif", build_script=tmp_path / "b.sh", log_dir=tmp_path
        )
        assert "quinnlab" not in " ".join(argv)

    def test_batch_mode_requires_an_account(self, tmp_path, monkeypatch):
        d = _write_def(tmp_path)
        with pytest.raises(ConfigurationError) as ei:
            build_sif(def_path=d, sif_out=tmp_path / "o.sif", account=None, mode="batch")
        assert "default_account" in str(ei.value)


class TestBuildScript:
    def test_cds_into_the_def_directory(self, tmp_path):
        d = _write_def(tmp_path)
        body = render_build_script(
            def_path=d,
            sif_out=tmp_path / "o.sif",
            apptainer_module="apptainer/1.5.0",
            tmpdir="/scratch/u/t",
            cachedir="/scratch/u/c",
        )
        assert f'cd "{d.parent}"' in body
        assert "apptainer build --fakeroot" in body
        assert "set -euo pipefail" in body
        assert 'module load "apptainer/1.5.0"' in body

    def test_exports_ignore_proot_for_version_agnostic_mksquashfs(self, tmp_path):
        """APPTAINER_IGNORE_PROOT=1 forces the -all-root mksquashfs branch on ANY apptainer
        version. Without it, Rivanna's 1.5.0 dies at mksquashfs (proot ptrace under
        ptrace_scope=3) after the full ~1.5 h %post. This is the FLOOR fix."""
        body = render_build_script(
            def_path=_write_def(tmp_path), sif_out=tmp_path / "o.sif",
            apptainer_module="apptainer/1.5.0", tmpdir="/t", cachedir="/c",
        )
        assert "export APPTAINER_IGNORE_PROOT=1" in body

    def test_guards_apptainer_on_path(self, tmp_path):
        """A wrong/missing module fails loud on the BUILD host, not as a cryptic execve."""
        body = render_build_script(
            def_path=_write_def(tmp_path), sif_out=tmp_path / "o.sif",
            apptainer_module="apptainer/1.5.0", tmpdir="/t", cachedir="/c",
        )
        assert "command -v apptainer" in body

    def test_no_module_load_when_module_is_none(self, tmp_path):
        """On a PATH-apptainer cluster (Frontier) apptainer_module is None -> no module load."""
        body = render_build_script(
            def_path=_write_def(tmp_path), sif_out=tmp_path / "o.sif",
            apptainer_module=None, tmpdir="/t", cachedir="/c",
        )
        # Match the `module load "<module>"` COMMAND form, not the substring: the build-host
        # guard's error prose legitimately reads "...not on PATH after module load (check ...)".
        assert 'module load "' not in body
        assert "command -v apptainer" in body  # the guard still fires


class TestCacheHitSkipsSubmission:
    def test_cache_hit_returns_without_building(self, tmp_path, monkeypatch):
        d = _write_def(tmp_path)
        lock = tmp_path / "uv.lock"
        lock.write_text("lock")
        cache = tmp_path / "cache"
        monkeypatch.setenv("HHEMT_SIF_CACHE_DIR", str(cache))

        key = compute_cache_key(d.read_bytes(), "sha256:x", lock.read_bytes(), "a100")
        cached = cache / f"hhemt-a100-{key}.sif"
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text("pretend SIF")

        def _explode(*a, **k):  # a cache hit must never reach the builder
            raise AssertionError("build_sif was called on a cache HIT")

        monkeypatch.setattr("hhemt.container_build.build_sif", _explode)
        out = get_or_build_sif(
            def_path=d,
            base_image_digest="sha256:x",
            lock_path=lock,
            target_arch="a100",
            account="acct",
        )
        assert out == cached


class TestDefRewrite:
    """ADR-19's emit-side `%files ../` rewrite (`bundle/_emit.py`).

    These live here rather than in test_bundle.py because they are pure functions over
    .def text — no bundle fixture, no 5-minute render.
    """

    def test_files_rewrite_touches_only_the_payload_line(self):
        """R-FLOOR: a naive str.replace('../') corrupts 4 non-payload lines in
        uva-cuda.def (incl. lines where a prose ELLIPSIS '.../' is a substring match).
        The parser must change exactly one line per .def."""
        from hhemt.bundle._emit import _rewrite_files_section

        for def_path in sorted(Path("containers").glob("*.def")):
            text = def_path.read_text()
            new = _rewrite_files_section(text, "hhemt_src")
            # strict=True is load-bearing, not lint appeasement: the rewrite replaces a
            # line in place and must never change the line COUNT. A length mismatch here
            # is a real defect that a lenient zip would silently truncate away.
            changed = [
                (a, b)
                for a, b in zip(text.splitlines(), new.splitlines(), strict=True)
                if a != b
            ]
            assert len(changed) == 1, f"{def_path.name}: changed {len(changed)} lines"
            assert changed[0][0].strip().split()[0] == "../"
            assert changed[0][1].strip().startswith("hhemt_src ")
            # every comment bearing '../' survives byte-identical
            assert new.count("`../`") == text.count("`../`")

    def test_files_rewrite_fails_closed_on_unexpected_shape(self):
        from hhemt.bundle._emit import _rewrite_files_section

        with pytest.raises(ProcessingError, match="exactly one"):
            _rewrite_files_section("%files\n    src.txt /dst\n%post\n    :\n", "hhemt_src")

    def test_files_rewrite_preserves_a_second_entry(self):
        from hhemt.bundle._emit import _rewrite_files_section

        out = _rewrite_files_section(
            "%files\n    # a ../ comment\n    ../  /opt/hhemt-src\n    extra.txt /opt/x\n"
            "%post\n    # another ../ mention\n",
            "hhemt_src",
        )
        assert "    hhemt_src /opt/hhemt-src" in out
        assert "    extra.txt /opt/x" in out
        assert "# a ../ comment" in out and "# another ../ mention" in out

    def test_base_image_digest_parses_only_a_real_pin(self):
        """A false pin is worse than no pin: frontier-rocm's %labels org.hhemt.base_digest
        is the literal placeholder 'sha256-recorded-in-SIF-lockfile-post-build', so the
        parser reads ONLY the `From:` line and yields "" for a tag-pinned base."""
        from hhemt.bundle._emit import _parse_base_image_digest

        uva = Path("containers/uva-cuda.def").read_text()
        assert _parse_base_image_digest(uva).startswith("sha256:")
        # tag-pinned bases yield "" — never the frontier %labels placeholder
        assert _parse_base_image_digest(Path("containers/dev-cpu.def").read_text()) == ""
        frontier = Path("containers/frontier-rocm.def").read_text()
        assert _parse_base_image_digest(frontier) == ""
        assert "sha256-recorded-in-SIF-lockfile-post-build" not in _parse_base_image_digest(
            frontier
        )

    def test_no_def_uses_apptainer_incompatible_tag_and_digest(self):
        """[Q8] regression (R9b friction): apptainer's docker conveyor REJECTS a `From:`
        reference carrying BOTH a tag and a digest (`repo:tag@sha256:...` -> FATAL "Docker
        references with both a tag and digest are currently not supported"). A digest-pinned
        base MUST use the digest-ONLY form `repo@sha256:...`. This caught a live [Q8] STAGE-1
        build failure (job 17068716); guard against re-introduction."""
        for def_path in sorted(Path("containers").glob("*.def")):
            for line in def_path.read_text().splitlines():
                if not line.strip().startswith("From:"):
                    continue
                ref = line.split("From:", 1)[1].strip()
                if "@sha256:" in ref:
                    repo = ref.split("@sha256:", 1)[0]
                    assert ":" not in repo, (
                        f"{def_path.name}: `From: {ref}` combines a tag AND a digest "
                        f"(apptainer rejects this) -- use the digest-only form repo@sha256:..."
                    )

    def test_producer_defs_carry_the_q8b_build_fixes(self):
        """[Q8] q8b build-reliability regression. The producer's 3 arch .defs must carry the
        fixes that let them build under rootless --fakeroot on Rivanna: (1) apt's Acquire
        sandbox disabled before the first apt-get (no /etc/subuid -> _apt setgroups EPERM,
        exit 100); (2) the two CUDA defs bound the TRITON compile to `make -j8` (not
        $(nproc)=16, which OOMs 16 concurrent nvcc device compiles in the 64 GB cgroup) and
        carry an explicit -DCMAKE_CUDA_ARCHITECTURES (GPU-less arch-probe preempt)."""
        for name in ("uva-cuda.def", "uva-cuda-a6000.def", "uva-cpu.def"):
            body = (Path("containers") / name).read_text()
            assert 'APT::Sandbox::User "root"' in body, (
                f"{name}: missing the apt-sandbox-disable line -> apt-get exits 100 under "
                "rootless --fakeroot with no /etc/subuid"
            )
        for name, arch in (("uva-cuda.def", "80"), ("uva-cuda-a6000.def", "86")):
            body = (Path("containers") / name).read_text()
            assert f"-DCMAKE_CUDA_ARCHITECTURES={arch}" in body, (
                f"{name}: missing the explicit CUDA arch belt flag (GPU-less arch-probe preempt)"
            )
            assert any(ln.strip() == "make -j8" for ln in body.splitlines()), (
                f"{name}: the TRITON compile must be `make -j8` (nvcc OOM preempt); the "
                "memory-light UCX/OpenMPI/SWMM makes stay -j$(nproc)"
            )

    def test_no_def_sets_the_unvalidated_swmm_override(self):
        """[Q8] q8c regression: NO .def may set HHEMT_ALLOW_UNVALIDATED_SWMM_STACK. The guarded
        Python swmm-toolkit runoff runs during NATIVE scenario prep on the HOST, never in the
        SIF, so an in-SIF override is a no-op for the guard; set in the run/bundle env it would
        propagate to every from_doi reproducer, permanently neutering the guard. The correct fix
        is the <0.16 pin (guard passes natively), not the override."""
        for def_path in sorted(Path("containers").glob("*.def")):
            assert "HHEMT_ALLOW_UNVALIDATED_SWMM_STACK" not in def_path.read_text(), (
                f"{def_path.name}: sets HHEMT_ALLOW_UNVALIDATED_SWMM_STACK -- a no-op in-SIF and "
                "a guard-neutering anti-pattern if it reaches the run env (q8c)"
            )


class TestLoginNodeRefusal:
    def test_compiling_def_refused_without_slurm(self, tmp_path, monkeypatch):
        """A compiling .def on a host with no sbatch is refused.

        `make -j$(nproc)` sees no cgroup cap on a login node and forks 40-way on a shared
        frontend (an AUP violation), and the multithreaded mksquashfs finalization is the
        process most likely to be SIGKILLed by RLIMIT_CPU — dying at the last step after an
        hour of work.
        """
        d = _write_def(tmp_path, compiling=True)
        monkeypatch.setattr("hhemt.container_build._slurm_available", lambda: False)
        with pytest.raises(ConfigurationError) as ei:
            build_sif(def_path=d, sif_out=tmp_path / "o.sif", mode="auto")
        assert "compiles in %post" in str(ei.value)
