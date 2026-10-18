"""Reviewer pins for the HPC profile currency gate: two wrong implementations the first suite admits.

These tests PASS against the gate as built. They are here because the gate's own test
module (tests/test_hpc_profile_currency.py) stays green under two plausible wrong
implementations, measured by the reviewer of package 1g:

1. A gate that reads keys off the page TEXT and so counts a commented-out line
   (``# max_concurrent_cpus: null``) as a shown field. A reader copying the block gets no
   such key, so a comment is not a shown field. The first suite never puts a commented
   key on a page.
2. A gate that walks partition entries against a hard-wired ``PartitionSpec`` rather than
   the value type the model declares. The first suite grows the model only at the top
   level and inside ``container``, never inside a partition entry, so a field added to
   ``PartitionSpec`` is unexercised by its field-added-to-the-model test.

Nothing here builds an analysis, compiles, or simulates.
"""

from __future__ import annotations

import re
import types

import pytest
from pydantic import create_model

import scripts.check_hpc_profile_currency as gate  # repo root is on sys.path under pytest
from hhemt.config.hpc_system import ContainerSpec, PartitionSpec, hpc_system_config


def _live_text() -> str:
    return (gate.REPO_ROOT / gate.PAGE_REL).read_text(encoding="utf-8")


def _live_profile() -> dict:
    profile, findings = gate.extract_profile(_live_text())
    assert profile is not None, findings
    return profile


@pytest.mark.parametrize(
    ("line_re", "location"),
    [
        (r"^max_concurrent_cpus:", "max_concurrent_cpus"),
        (r"^  apptainer_module:", "container.apptainer_module"),
    ],
    ids=["top-level-field", "container-field"],
)
def test_field_shown_only_in_a_comment_is_absent(tmp_path, capsys, line_re: str, location: str) -> None:
    """Property: a field whose only appearance in the block is a comment is not shown.

    Class: the live page with exactly one field's line commented out in place, at the top
    level and inside ``container``.
    Why required: the page is copied as yaml; a commented key never reaches the loader, so
    the reader never learns the setting exists, which is the staleness the gate detects.
    What kills it: deciding coverage from the page text (a key-shaped token anywhere in the
    block, comments included) rather than from the parsed mapping.
    Second correct implementation: parsing with ``ruamel.yaml`` instead of PyYAML -- both
    discard comments, so both report the commented field absent.
    """
    text = _live_text()
    pattern = re.compile(line_re, re.M)
    assert len(pattern.findall(text)) == 1, f"expected exactly one {line_re!r} line on the live page"
    page = tmp_path / "page.md"
    # Comment the line out in place, keeping its indentation, so the block still parses.
    page.write_text(pattern.sub(lambda m: m.group(0).replace(m.group(0).lstrip(), "# " + m.group(0).lstrip()), text))
    assert gate.main(["--page", str(page)]) == 1
    err = capsys.readouterr().err
    assert f"{location}: field-absent" in err, err


def test_field_added_to_partition_spec_fails_unchanged_page() -> None:
    """Property: a field added to the partition model fails the unchanged page.

    Class: ``PartitionSpec`` grown by one optional field, wired into a grown profile model's
    ``partitions`` mapping, checked against the live page's profile.
    Why required: partition caps are the fields the page tells a reader to replace for
    their own cluster; a new cap the page never shows is a cap preflight never validates.
    What kills it: walking partition entries against a hard-wired ``PartitionSpec`` (or a
    written-down list of its fields) instead of the value type the model's ``partitions``
    annotation declares.
    Second correct implementation: deriving the nested field set from
    ``model_json_schema()`` following ``$defs`` passes this.
    """
    grown_partition = create_model("GrownPartition", __base__=PartitionSpec, new_cap=(int | None, None))
    model = create_model(
        "GrownProfile", __base__=hpc_system_config, partitions=(types.GenericAlias(dict, (str, grown_partition)), ...)
    )
    model.model_rebuild(_types_namespace={"ContainerSpec": ContainerSpec})
    findings, _ = gate.check_profile(_live_profile(), model=model)
    assert ("partitions.*.new_cap", "field-absent") in {(f.location, f.code) for f in findings}
