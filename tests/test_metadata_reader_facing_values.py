"""Reader-facing values on the Metadata page: the digest anchor, the build chain and the id-set label.

Each test names the value a reader acts on and carries a must-fail arm against the code
before these values were printed: a count of zero digests that read as "0 input digest(s)",
a build chain with one row for two stages, and an id set printed as a span plus a count of
how many ids it skips.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from hhemt.report_renderers import metadata


def _input_entity(sha: str | None) -> dict:
    entity: dict = {"@id": "in.csv", "@type": "File", "name": "in.csv"}
    if sha:
        entity["sha256"] = sha
    return entity


def test_zero_input_digests_read_as_not_captured() -> None:
    """NEW CAPABILITY. Plausible wrong implementation: a count is always printed, so zero
    digests read "0 input digest(s)" and a reader takes the run for checkable.

    Differently-positioned input: one digest prints a count with its unit, in the singular.
    """
    app = {"softwareVersion": "abc123"}
    none_recorded = metadata._provenance_bluf(app, None, [_input_entity(None)])
    one_recorded = metadata._provenance_bluf(app, None, [_input_entity("f00dfeed"), _input_entity(None)])
    two_recorded = metadata._provenance_bluf(app, None, [_input_entity("aa"), _input_entity("bb")])

    assert "input digests <em>not captured</em>" in none_recorded
    assert "0 input digest" not in none_recorded
    assert "1 input digest." in one_recorded
    assert "1 input digests" not in one_recorded
    assert "2 input digests" in two_recorded


def test_the_anchor_sha_is_labelled_with_the_stage_that_wrote_it() -> None:
    """NEW CAPABILITY. The SHA in the anchors is the consolidation build's, so the label says
    so. Plausible wrong implementation: the bare label "code git-SHA" suggests the build that
    simulated the data. Both the present and the absent form carry the stage."""
    present = metadata._provenance_bluf({"softwareVersion": "abc123"}, None, [])
    absent = metadata._provenance_bluf({}, None, [])

    assert "consolidation code git-SHA <code>abc123</code>" in present
    assert "consolidation code git-SHA <em>not captured</em>" in absent


def _chain(tmp_path: Path, sim_stamp: dict | None, monkeypatch: pytest.MonkeyPatch) -> str:
    import hhemt.analysis_validation as av

    monkeypatch.setattr(av, "_collect_stage_stamps", lambda analysis: {"sim": sim_stamp})
    analysis = types.SimpleNamespace(analysis_paths=types.SimpleNamespace(analysis_dir=tmp_path))
    return metadata._provenance_chain(tmp_path, analysis, "cafe01")


def test_the_chain_separates_simulated_by_from_processed_by(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NEW CAPABILITY. Plausible wrong implementation: one row "Simulated / processed by"
    that reads the processing stamp and labels it as the simulating build -- red under it
    because there is no row named "Simulated by".

    With a sim stamp the row shows its sha; with none it reads "not captured" and does not
    borrow the processing stamp.
    """
    stamped = _chain(tmp_path, {"hhemt_sha": "5111aa", "hhemt_version": "1.2.3"}, monkeypatch)
    unstamped = _chain(tmp_path, None, monkeypatch)

    assert "<code>5111aa</code> (1.2.3)" in stamped
    for html in (stamped, unstamped):
        assert "Simulated by" in html and "Processed by" in html
        assert "Simulated / processed by" not in html
        assert "Consolidated by" in html and "Rendered by" in html
    sim_row = unstamped[unstamped.index("Simulated by") :].split("</tr>", 1)[0]
    assert "<em>not captured</em>" in sim_row


def test_the_chain_prints_its_four_stage_rows_and_no_note_above_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEW CAPABILITY. The note over the table restated the row labels, so it is cut; what the
    reader acts on is the four rows. The former note said "These three" over what is now four
    rows, so the count text stays absent.

    Property kept from the earlier test: the page names each of the four stages, now by its
    row label. Must-fire: no `note` paragraph precedes the table.
    """
    html = _chain(tmp_path, None, monkeypatch)

    assert "These three" not in html
    assert "class='note'" not in html.split("<table", 1)[0]
    for label in ("Simulated by", "Processed by", "Consolidated by", "Rendered by"):
        assert label in html, label


def _label(ids: list[int]) -> str:
    names = [f"simulation_member_{i}" for i in ids]
    return metadata._sentinel_label(names)


def test_a_gapped_id_set_prints_its_ranges_not_a_count_of_the_missing() -> None:
    """NEW CAPABILITY. The ids 4 to 9, 22 to 29 and 32 print as that set. Plausible wrong
    implementation: `4…32 (15 of 29)`, which names a span and how many ids it holds but not
    which -- red under it."""
    label = _label([*range(4, 10), *range(22, 30), 32])

    assert "i = 4…9, 22…29, 32" in label
    assert "15 of 29" not in label
    assert " × 15," in label


def test_an_unbroken_run_and_short_runs_keep_a_readable_form() -> None:
    """The control for the ranges: an unbroken run is one range, and a run shorter than three
    lists its ids, so the label never prints a range that hides a gap."""
    assert "i = 1…5" in _label([1, 2, 3, 4, 5])
    gapped = _label([1, 2, 4, 5, 6, 7])
    assert "i = 1, 2, 4…7" in gapped
