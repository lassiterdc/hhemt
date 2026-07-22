"""Operator-gated END-TO-END DOI round-trip: deposit -> mint DOI -> fetch back -> RUN.

This is the only test that proves the reproducibility claim against a LIVE deposit
repository. Its network-free twin is ``test_synth_09_from_doi_run_proof.py``, which exercises
the identical emit -> from_doi -> run path with the fetch mocked; this module replaces the
mock with a real ``publish_reprex_bundle`` + a real ``from_doi`` fetch-by-DOI.

It is skipped by default and never runs in CI.

===============================================================================
NO DIRECTORY CONFIG NEEDED — the experiment is the synthetic test corpus
===============================================================================

The analysis this test deposits is the synth-corpus ``rendered_synth_multi_sim`` fixture —
the SAME completed, rendered analysis the rest of the compile-tier suite uses. Its package /
data / scratch directories are derived deterministically from inside the toolkit
(``Local_TestCases.retrieve_synth_multi_sim_test_case``); there is nothing for an operator to
point at. The run leg reuses the pre-compiled ``tritonswmm_cpu_compiled`` toolchain via
symlink, so no git-clone and no recompile fire (the same trick ``test_synth_09`` uses).

Consequence: the ONLY thing this test needs from the operator is Zenodo credentials.

===============================================================================
HOW TO RUN THIS TEST
===============================================================================

1. Create the credential file (once, per machine). It lives OUTSIDE the repo so it can
   never be committed::

       mkdir -p ~/.config/hhemt && touch ~/.config/hhemt/e2e.env && chmod 600 ~/.config/hhemt/e2e.env

   Fill it in with your OWN editor (never paste a value into an AI chat). The variables this
   test reads::

       HHEMT_ZENODO_TOKEN            # personal access token; scopes deposit:write, deposit:actions
       HHEMT_ZENODO_BASE_URL         # https://sandbox.zenodo.org  (see the WARNING below)
       HHEMT_E2E_ALLOW_PRODUCTION    # '1' ONLY when you intend to mint a permanent DOI
       HHEMT_E2E_INPUT_URL           # leg 2 only (optional) — see LEG 2 below

2. Run it::

       source ~/.config/hhemt/e2e.env
       HHEMT_PUBLISH_E2E=1 conda run -n hhemt python -m pytest tests/test_doi_roundtrip_e2e.py -v

   Without ``HHEMT_PUBLISH_E2E=1`` the whole module skips. Without a token, it skips with a
   message naming the exact variable that is missing. It is ``slow`` +
   ``requires_snakemake_subprocess`` (it compiles and launches Snakemake), so it is
   deselected from the default ``-m "not slow"`` fast set.

===============================================================================
WARNING — THIS TEST PUBLISHES. A MINTED DOI IS PERMANENT.
===============================================================================

A Zenodo publish is IRREVERSIBLE: once a DOI is minted it cannot be withdrawn and the record
cannot be deleted. There is no undo.

Two independent locks guard production:

  * ``HHEMT_ZENODO_BASE_URL`` defaults to the SANDBOX (https://sandbox.zenodo.org), which
    mints disposable ``10.5072/...`` DOIs on records you can delete freely.
  * Pointing at production (``zenodo.org``) additionally requires
    ``HHEMT_E2E_ALLOW_PRODUCTION=1``. Without it this module FAILS rather than publishing.

Get a green run on the sandbox first, always. Sandbox and production are separate sites with
separate accounts and separate tokens — a production token 401s on the sandbox and vice-versa.

===============================================================================
CREDENTIAL SECURITY
===============================================================================

* NEVER put a credential in the repo, a planning doc, a scratch file, a commit message, or
  an AI-agent chat transcript. If one is exposed, REVOKE it immediately (Zenodo: delete the
  token in account settings).
* The env file is ``chmod 600`` and lives outside every git repo. Keep it that way.
* A secret manager is better than a plaintext file if you have one::

      op run --env-file=~/.config/hhemt/e2e.env -- pytest tests/test_doi_roundtrip_e2e.py

* No hhemt code path prints a credential VALUE — ``publishing._require_env`` reports only the
  NAME of a missing variable, so a failure cannot leak the secret into logs.

===============================================================================
WHEN THIS TEST FAILS — the likely causes, in order
===============================================================================

1. ``PublishError: missing required credential env var HHEMT_ZENODO_TOKEN``
   -> You did not ``source`` the env file, or the variable is empty. Note the ``HHEMT_``
      prefix: the variable is ``HHEMT_ZENODO_TOKEN``, NOT ``ZENODO_TOKEN``.
2. ``HTTP 401`` / ``HTTP 403`` from Zenodo
   -> The token expired, was revoked, or was minted on the WRONG SITE (a production token
      used against the sandbox, or vice-versa). Mint a fresh one on the site
      ``HHEMT_ZENODO_BASE_URL`` points at, with scopes ``deposit:write`` + ``deposit:actions``.
3. The ``rendered_synth_multi_sim`` fixture errors or skips
   -> This is the compile-tier synth infrastructure, not a credential problem. It skips on a
      scheduler node or when the C++ toolchain is unavailable; run it on a workstation with a
      working compiler in the ``hhemt`` env. See ``test_synth_09_from_doi_run_proof.py`` — if
      that green-passes locally, this fixture is healthy.

===============================================================================
FOR SOMEONE ON ANOTHER MACHINE / A FORK
===============================================================================

You need only your OWN Zenodo (sandbox) account and token — there is nothing machine-specific
to configure, because the deposited experiment is the synthetic fixture, built from source.
Create a free sandbox account, mint a token, fill in the env file, and run.

===============================================================================
LEG 2 (optional) — the ADR-20 by-reference opt-out
===============================================================================

Leg 1 proves the SELF-CONTAINED round-trip (R8) and needs only a token. Leg 2 additionally
proves the governed opt-out (R11): an input is EXCLUDED from the bundle and fetched by
reference on ingest. That requires an input you have ALREADY deposited as its own record and
that has a DIRECT-DOWNLOAD url (the ADR-20 ordering constraint — the toolkit has no per-file
deposit helper). Put that url in ``HHEMT_E2E_INPUT_URL``; leave it blank and leg 2 skips
while leg 1 still runs.

===============================================================================
LEG 3 (optional) — HydroShare authenticated PRIVATE retrieval
===============================================================================

HydroShare's DOI mint is a MANUAL, PERMANENT web-UI "Publish" action that hsclient cannot
perform, so there is no automatable deposit -> DOI -> public-fetch loop like Zenodo's. But
retrieval of a PRIVATE resource works with the OWNER's credentials by resource id — no DOI,
no publication. Leg 3 tests exactly that: it stages a bundle into your existing private
resource (test-side, authenticated), then runs the real ``from_doi(host='hydroshare')``
retrieval + reconstitute + run, and clears the file afterward. The resource is never made
public and never published. Needs::

    HHEMT_HYDROSHARE_USERNAME     # bare id OR institutional email — try the other if auth fails
    HHEMT_HYDROSHARE_PASSWORD     # hsclient v1.1.6 has no token support — this is your password
    HHEMT_E2E_HYDROSHARE_RESOURCE # the 32-hex id of a resource you own (may be empty)

Leave any of the three blank and leg 3 skips while the Zenodo legs still run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import tests.utils_for_testing as tst_ut
from hhemt.experiments import TRITON_SWMM_experiment

E2E_ENV_FILE = "~/.config/hhemt/e2e.env"

pytestmark = [
    pytest.mark.requires_snakemake_subprocess,
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("HHEMT_PUBLISH_E2E"),
        reason=(
            "operator-gated live-deposit e2e. To run: "
            f"`source {E2E_ENV_FILE} && HHEMT_PUBLISH_E2E=1 pytest tests/test_doi_roundtrip_e2e.py -v`. "
            "See this module's docstring for the full runbook."
        ),
    ),
    pytest.mark.skipif(
        tst_ut.is_scheduler_context(),
        reason="live-deposit run-proof; do not launch on an HPC scheduler node.",
    ),
]

_PRODUCTION_HOSTS = ("https://zenodo.org", "http://zenodo.org", "zenodo.org")


def _zenodo_base() -> str:
    return os.environ.get("HHEMT_ZENODO_BASE_URL", "https://zenodo.org").rstrip("/")


@pytest.fixture(scope="module", autouse=True)
def production_guard() -> None:
    """Refuse to publish to PRODUCTION Zenodo without an explicit, separate opt-in.

    A minted DOI is permanent. The default base URL in a fresh shell is production
    (``publishing._ZENODO_DEFAULT_BASE``), so an operator who forgets to source the env file
    would otherwise publish for real on their first run. Production requires BOTH a production
    base URL AND ``HHEMT_E2E_ALLOW_PRODUCTION=1``. This is a hard failure, not a skip — a
    silent skip would let someone believe the test passed against production when it never
    ran at all.
    """
    base = _zenodo_base()
    if base in _PRODUCTION_HOSTS and os.environ.get("HHEMT_E2E_ALLOW_PRODUCTION") != "1":
        pytest.fail(
            "REFUSING to run the live e2e against PRODUCTION Zenodo.\n"
            f"  HHEMT_ZENODO_BASE_URL = {base}\n"
            "  HHEMT_E2E_ALLOW_PRODUCTION is not '1'.\n\n"
            "A published Zenodo record mints a PERMANENT, IRREVOCABLE DOI — there is no undo.\n"
            "Get a green run on the sandbox first:\n"
            "    export HHEMT_ZENODO_BASE_URL='https://sandbox.zenodo.org'\n"
            "(sandbox needs its own account + its own token — a production token will 401 there).\n\n"
            "When you genuinely intend to mint a real DOI, set HHEMT_E2E_ALLOW_PRODUCTION=1."
        )


def _require_token() -> None:
    if not os.environ.get("HHEMT_ZENODO_TOKEN"):
        pytest.skip(
            f"missing HHEMT_ZENODO_TOKEN — set it in {E2E_ENV_FILE} and re-source it. "
            "(Note the HHEMT_ prefix: the variable is HHEMT_ZENODO_TOKEN, not ZENODO_TOKEN.)"
        )


@pytest.fixture
def prebuilt_software_dir(tritonswmm_cpu_compiled, tmp_path) -> Path:
    """A tmp ``software_dir`` whose ``tritonswmm``/``swmm`` subdirs SYMLINK into the shared
    ``tritonswmm_cpu_compiled`` synth tree, so the run leg reuses the pre-built toolchain
    instead of git-cloning + recompiling. Byte-for-byte the pattern
    ``test_synth_09_from_doi_run_proof.py`` established.
    """
    from tests.fixtures.test_case_catalog import Local_TestCases

    case = Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=False)
    cfg = case.analysis._system.cfg_system
    compiled_triton = Path(cfg.TRITONSWMM_software_directory)
    compiled_swmm = Path(cfg.SWMM_software_directory)
    assert (compiled_triton / "build_tritonswmm_cpu" / "compilation.log").exists(), (
        "tritonswmm_cpu_compiled left no CPU compilation.log — the recompile-skip "
        f"precondition is unmet under {compiled_triton}."
    )
    software_dir = tmp_path / "software"
    software_dir.mkdir()
    (software_dir / "tritonswmm").symlink_to(compiled_triton, target_is_directory=True)
    (software_dir / "swmm").symlink_to(compiled_swmm, target_is_directory=True)
    return software_dir


def _assert_reconstituted_run(exp, prebuilt_software_dir: Path) -> None:
    """Guard the network-free precondition, then RUN the reconstituted analysis end to end
    and assert real consolidated outputs (a --dry-run could not satisfy these)."""
    resolved = Path(exp.system.cfg_system.TRITONSWMM_software_directory).resolve()
    assert resolved == Path(prebuilt_software_dir / "tritonswmm").resolve(), (
        f"from_doi did not adopt the pre-built software_dir (got {resolved}); the run would "
        "git-clone + recompile."
    )
    result = exp.analysis.test(execution_mode="local", verbose=False)
    assert (exp.bundle_root / "_test").exists(), "no _test subtree materialized"
    assert result.subanalyses, "analysis.test() produced no _test sub-analyses"
    for sub in result.subanalyses:
        tst_ut.assert_analysis_workflow_completed_successfully(sub.analysis)


def test_leg1_self_contained_roundtrip_runs(
    rendered_synth_multi_sim, prebuilt_software_dir, tmp_path
) -> None:
    """Leg 1 (R8) — SELF-CONTAINED: deposit -> fetch by DOI -> RUN, no exclusions.

    The primary contract (ADR-9): a DOI-downloaded bundle runs from scratch. The assertion
    that matters is not merely that the fetch succeeded — it is that NO input_deposit fetch
    fired, i.e. every input was CARRIED.
    """
    _require_token()

    result = rendered_synth_multi_sim.publish_reprex_bundle(target="zenodo")
    doi = result.get("data_doi")
    assert doi, f"no DOI minted; publish returned {result}"

    ingest_dir = tmp_path / "ingest_leg1"
    exp = TRITON_SWMM_experiment.from_doi(
        doi=doi, host="zenodo", target_dir=ingest_dir, software_dir=prebuilt_software_dir
    )

    manifest = exp.bundle_root / "bundle_manifest.json"
    if manifest.exists():
        assert not json.loads(manifest.read_text()).get("input_deposit"), (
            "a bundle emitted with NO exclude-config must carry every input — an "
            "input_deposit block means the self-contained default regressed"
        )

    _assert_reconstituted_run(exp, prebuilt_software_dir)


def test_leg2_exclude_config_roundtrip_fetches_by_reference(
    rendered_synth_multi_sim, prebuilt_software_dir, tmp_path
) -> None:
    """Leg 2 (R11) — the ADR-20 governed opt-out: an excluded input is fetched BY REFERENCE.

    The bundle omits ``weather_timeseries``, records an input_deposit block for it, and the
    consumer's ingest fetches it through the per-file seam and sha256-verifies it before the
    fail-closed gate.
    """
    _require_token()
    input_url = os.environ.get("HHEMT_E2E_INPUT_URL")
    if not input_url:
        pytest.skip(
            "leg 2 needs HHEMT_E2E_INPUT_URL — a DIRECT-download url for an input you have "
            "ALREADY deposited as its own record (the ADR-20 ordering constraint; the toolkit "
            f"has no per-file deposit helper). Add it to {E2E_ENV_FILE}. Leg 1 still proves R8."
        )

    exclude_yaml = tmp_path / "bundle_exclude.yaml"
    exclude_yaml.write_text(
        "exclusions:\n"
        "  weather_timeseries:\n"
        "    citation: 'Weather forcing, deposited by the operator as its own resource.'\n"
        f"    contentUrl: '{input_url}'\n"
    )

    result = rendered_synth_multi_sim.publish_reprex_bundle(
        target="zenodo", exclude_config=exclude_yaml
    )
    doi = result.get("data_doi")
    assert doi, f"no DOI minted; publish returned {result}"

    ingest_dir = tmp_path / "ingest_leg2"
    exp = TRITON_SWMM_experiment.from_doi(
        doi=doi, host="zenodo", target_dir=ingest_dir, software_dir=prebuilt_software_dir
    )

    deposits = json.loads((exp.bundle_root / "bundle_manifest.json").read_text())["input_deposit"]
    assert deposits, "the excluded input emitted no input_deposit block"
    # The excluded input was NOT carried — it was fetched, and is now on disk.
    for block in deposits:
        assert (exp.bundle_root / block["relpath"]).exists(), (
            f"by-reference input {block['relpath']} was never materialized on ingest"
        )

    _assert_reconstituted_run(exp, prebuilt_software_dir)


def _require_hydroshare() -> str:
    """Skip the HydroShare leg unless credentials + a resource id are all present.

    Returns the resource id. The username may be a bare id or a full institutional email —
    if auth fails, try the other form in ``HHEMT_HYDROSHARE_USERNAME``.
    """
    missing = [
        v
        for v in (
            "HHEMT_HYDROSHARE_USERNAME",
            "HHEMT_HYDROSHARE_PASSWORD",
            "HHEMT_E2E_HYDROSHARE_RESOURCE",
        )
        if not os.environ.get(v)
    ]
    if missing:
        pytest.skip(
            f"HydroShare leg needs {', '.join(missing)} — set them in {E2E_ENV_FILE}. "
            "This leg tests authenticated retrieval of your PRIVATE resource; it never "
            "publishes or mints a DOI."
        )
    return os.environ["HHEMT_E2E_HYDROSHARE_RESOURCE"]


def _clear_resource_files(resource) -> None:
    """Best-effort: remove every file from a HydroShare resource so the 'exactly one bundle
    zip' contract holds on the next run. Idempotent — a resource with no files is fine."""
    for f in resource.files():
        try:
            resource.file_delete(path=f.path)
        except Exception:  # noqa: BLE001 — cleanup is best-effort; a stuck file is not fatal
            pass


def test_leg3_hydroshare_authenticated_private_retrieval(
    rendered_synth_multi_sim, prebuilt_software_dir, tmp_path
) -> None:
    """HydroShare — authenticated round-trip against a PRIVATE (non-published) resource.

    HydroShare DOI minting is a manual, permanent web-UI 'Publish' action that hsclient
    cannot perform, so there is NO automatable deposit -> DOI -> public-fetch loop the way
    there is on Zenodo. But retrieval of a PRIVATE resource works with the OWNER's
    credentials by resource id — no DOI, no publication needed (this is what the tier-2
    credentialed path in ``_connect_to_hydroshare`` enables). This test exercises exactly
    that path:

      setup     : emit a bundle locally and stage the single zip into the operator's
                  existing PRIVATE resource (test-side arrangement, via authenticated
                  hsclient — NOT the code under test);
      under test: ``from_doi(pid=resource_id, host='hydroshare')`` authenticates with the
                  env credentials, downloads the resource, reconstitutes, and RUNS;
      teardown  : clear the staged file so the resource stays clean and reusable.

    The resource is never made public and never published. If authentication fails, check
    that HHEMT_HYDROSHARE_USERNAME is the right form (bare id vs institutional email).
    """
    resource_id = _require_hydroshare()

    bundle_zip = rendered_synth_multi_sim.bundle_report_data(tmp_path / "emit" / "bundle.zip")

    # Authenticated staging (test-side setup, not the code under test). Reuse the toolkit's
    # own connection helper so this uses the exact tier-2 credentialed path.
    hs = TRITON_SWMM_experiment._connect_to_hydroshare(resource_id)
    resource = hs.resource(resource_id)
    _clear_resource_files(resource)  # ensure exactly one zip after upload
    resource.file_upload(str(bundle_zip))

    try:
        exp = TRITON_SWMM_experiment.from_doi(
            pid=resource_id,
            host="hydroshare",
            target_dir=tmp_path / "ingest_hs",
            software_dir=prebuilt_software_dir,
        )
        # Self-contained by default: the staged bundle carried every input (no by-reference
        # fetch), so the retrieval proves the full HydroShare consume path.
        manifest = exp.bundle_root / "bundle_manifest.json"
        if manifest.exists():
            assert not json.loads(manifest.read_text()).get("input_deposit"), (
                "the staged bundle was self-contained; an input_deposit block is unexpected"
            )
        _assert_reconstituted_run(exp, prebuilt_software_dir)
    finally:
        _clear_resource_files(resource)  # leave the private resource clean for the next run
