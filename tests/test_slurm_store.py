"""Red tests for the consolidated SLURM efficiency store (D1-D6).

Each test names the property it forces and the pre-fix behaviour it fails against. A test
that cannot fail certifies nothing, so the pre-fix comparator is named per test rather than
asserted collectively: two of these fail against SHIPPING code in `write_recovery_csv`, and
the rest fail against the absence of a consolidated store at HEAD.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hhemt.slurm_store import (
    KEY_FIELDS,
    PROVENANCE_CLASS,
    StoreSchemaError,
    is_terminal,
    merge_rows,
    provenance_class,
    read_rows,
    rows_to_dataset,
    store_path,
    validate_availability,
    write_store,
)


def _row(job_id, submit, **kw):
    row = {
        "MainJobID": job_id.split(".", 1)[0],
        "StepKind": job_id.split(".", 1)[1] if "." in job_id else "job",
        "JobID": job_id,
        "Submit": submit,
        "JobName": "triton.exe",
        "Elapsed": "00:01:00",
        "State": "COMPLETED",
    }
    row.update(kw)
    return row


# ---------------------------------------------------------------- D3, property 1


def test_merge_never_replace_retains_a_key_absent_from_this_capture(tmp_path):
    """PROPERTY: a key in the store and absent from this capture is RETAINED.

    Pre-fix comparator: there is no consolidated store at HEAD (`git show
    HEAD:src/hhemt/slurm_store.py` exits 128), so this fails at import.

    A capture is a partial view -- `recover_rows` continues past a timed-out chunk and
    sacct silently omits purged ids -- so a replacing writer makes the store a snapshot of
    sacct's current retention window rather than a data product.
    """
    write_store(tmp_path, [_row("1001", "2026-08-16T10:00:00"), _row("1002", "2026-08-16T10:00:00")])
    write_store(tmp_path, [_row("1002", "2026-08-16T10:00:00")])  # chunk 1 timed out

    got = {r["MainJobID"] for r in read_rows(tmp_path)}
    assert got == {"1001", "1002"}, f"a partial capture dropped a retained key; store holds {sorted(got)}"


# ---------------------------------------------------------------- D3, property 2


def test_a_non_terminal_recapture_does_not_overwrite_a_terminal_row():
    """PROPERTY: last-wins is by TERMINALITY, not recency.

    Pre-fix comparator: SHIPPING `slurm_job_recovery.write_recovery_csv`, whose merge is
    purely field-wise non-empty-wins with no notion of terminality -- so a re-query of a
    settled job regresses its stored State to RUNNING and its Elapsed to the partial value.
    `test_pre_fix_write_recovery_csv_regresses_a_terminal_row` below RUNS that comparator
    and asserts the regression, so the pre-fix behaviour is measured rather than described.
    """
    stored = [_row("2001", "2026-08-16T10:00:00", State="COMPLETED", Elapsed="07:54:22")]
    incoming = [_row("2001", "2026-08-16T10:00:00", State="RUNNING", Elapsed="00:00:30")]

    merged = merge_rows(stored, incoming)

    assert len(merged) == 1
    assert merged[0]["State"] == "COMPLETED", "a RUNNING re-capture regressed a settled row"
    assert merged[0]["Elapsed"] == "07:54:22", "the settled elapsed was overwritten by a partial one"


def test_pre_fix_write_recovery_csv_regresses_a_terminal_row(tmp_path):
    """The pre-fix comparator for the test above, RUN rather than asserted.

    This documents the behaviour the new merge corrects. It is green because it asserts the
    OLD behaviour; if `write_recovery_csv` later grows a terminality rule this test goes red,
    which is the correct signal that the two implementations have converged.
    """
    from hhemt.slurm_job_recovery import RECOVERY_HEADER, write_recovery_csv

    def _full(**kw):
        row = {k: "" for k in RECOVERY_HEADER}
        row.update(MainJobID="2001", StepKind="job", JobID="2001", Submit="2026-08-16T10:00:00")
        row.update(kw)
        return row

    write_recovery_csv(tmp_path, [_full(State="COMPLETED", Elapsed="07:54:22")])
    write_recovery_csv(tmp_path, [_full(State="RUNNING", Elapsed="00:00:30")])

    import csv
    import io

    from hhemt.slurm_job_recovery import RECOVERY_FILENAME

    text = (tmp_path / "logs" / "slurm_efficiency_report" / RECOVERY_FILENAME).read_text()
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 1
    assert rows[0]["State"] == "RUNNING", (
        "pre-fix comparator no longer regresses; the terminality rule may have landed there too"
    )


def test_terminality_recognises_the_cancelled_by_uid_form():
    """PROPERTY: `CANCELLED by 554635` is terminal, and the full string is retained.

    Measured on synth_cc_resume_triton's solver steps: `CANCELLED by 554635` x82,
    `COMPLETED` x28, `CANCELLED by 0` x2. A prefix match is required to classify them, and
    the store must still carry the verbatim string because the two cancel causes are
    different events.
    """
    assert is_terminal("CANCELLED by 554635")
    assert is_terminal("NODE_FAIL")
    assert not is_terminal("RUNNING")
    assert not is_terminal("PENDING")

    stored = [_row("3001", "s", State="CANCELLED by 554635")]
    merged = merge_rows(stored, [_row("3001", "s", State="RUNNING")])
    assert merged[0]["State"] == "CANCELLED by 554635", "the verbatim cancel cause was rewritten"


# ---------------------------------------------------------------- D2, property 3


def test_writer_refuses_not_gathered_on_a_non_measured_column():
    """PROPERTY: `not_gathered` is admissible only when provenance_class == 'measured'.

    Pre-fix comparator: SHIPPING `write_recovery_csv` has no availability concept at all --
    `test_pre_fix_has_no_availability_channel` RUNS that and asserts the absence -- so a
    wrong stamp had no point at which it could be refused.

    `not_gathered` is a statement about an acct_gather plugin. On `Partition` it would
    assert that a plugin sampled the experiment's hardware axis, which is false about how
    that value exists and unfalsifiable in the direction that matters.
    """
    for column in ("Partition", "ReqMem", "Elapsed", "State"):
        with pytest.raises(StoreSchemaError, match="never be 'not_gathered'"):
            validate_availability(column, "not_gathered")

    # The MEASURED class alone admits it, and that is the case that must keep working.
    validate_availability("TRESUsageInTot", "not_gathered")
    validate_availability("MaxRSS", "not_gathered")


def test_writer_refuses_not_gathered_through_the_dataset_builder():
    """The same constraint, reached through the write path rather than the predicate.

    A check that only exists as a standalone function is one an eager writer routes around;
    this asserts the builder itself enforces it.
    """
    rows = [_row("4001", "s", Partition="gpu-a6000")]
    with pytest.raises(StoreSchemaError, match="never be 'not_gathered'"):
        rows_to_dataset(rows, availability={"Partition": "not_gathered"})


def test_pre_fix_has_no_availability_channel():
    """The pre-fix comparator for the two tests above, RUN rather than asserted."""
    from hhemt.slurm_job_recovery import RECOVERY_HEADER

    assert not any("availability" in c.lower() or "provenance" in c.lower() for c in RECOVERY_HEADER), (
        "the pre-fix header grew an availability channel; the comparator is stale"
    )


def test_every_known_column_has_a_provenance_class_and_unknowns_get_the_fallback():
    """PROPERTY: the taxonomy is total, and column-agnostic for what it does not know."""
    for column, cls in PROVENANCE_CLASS.items():
        assert cls in {"requested", "granted", "scheduled", "measured", "derived"}, column
    # A column nobody has declared is carried, and inherits no claim about how it was made.
    assert provenance_class("SomeFutureDerivedColumn") == "derived"
    validate_availability("SomeFutureDerivedColumn", "available")


# ---------------------------------------------------------------- D2 rider, derived columns


def test_a_derived_columns_availability_is_propagated_from_its_inputs():
    """PROPERTY (D2 rider): a derived column's availability is PROPAGATED, never observed.

    The rule is uniform: `not_derivable` iff ANY input is anything other than `available`.
    It does NOT inherit the input's own value -- inheritance was the first remedy proposed
    and is false, because an energy-derived column cannot take its input's `not_gathered`
    (the class constraint forbids it) and no other existing term is true of it.

    Pre-fix comparator: output observation, RUN in
    `test_output_observed_stamping_conflates_the_two_blank_causes` below.
    """
    from hhemt.slurm_store import derive_availability

    assert derive_availability(["available", "available"]) == "available"
    assert derive_availability([]) == "available"

    for unavailable in ("not_gathered", "undefined_at_this_grain", "undefined_for_this_instance"):
        assert derive_availability(["available", unavailable]) == "not_derivable", unavailable

    # It never inherits: the output is `not_derivable` for EVERY unavailable input kind.
    assert derive_availability(["not_gathered"]) != "not_gathered"
    assert derive_availability(["undefined_at_this_grain"]) != "undefined_at_this_grain"


def test_not_derivable_is_admissible_only_on_a_derived_column():
    """PROPERTY: the fifth term is class-restricted, MIRRORING `not_gathered`/`measured`.

    Preserving checkability on BOTH sides is the reason a fifth term beats relaxing the
    constraint: a wrong stamp still raises rather than rendering as a plausible sentence.
    """
    # `derived` accepts it.
    validate_availability("RunMethod", "not_derivable")
    validate_availability("SomeFutureDerivedColumn", "not_derivable")
    # Every other class refuses it.
    for column in ("MaxRSS", "Partition", "ReqMem", "Elapsed"):
        with pytest.raises(StoreSchemaError, match="never be 'not_derivable'"):
            validate_availability(column, "not_derivable")
    # And the mirror still holds in the other direction.
    with pytest.raises(StoreSchemaError, match="never be 'not_gathered'"):
        validate_availability("Partition", "not_gathered")


def test_the_writer_propagates_rather_than_accepting_a_hand_stamp():
    """PROPERTY: propagation is done BY THE WRITER, and a hand-stamp is refused.

    This is the load-bearing half of the rider -- propagation is unit-testable and a
    hand-stamp is not. A caller that supplies both is refused rather than silently
    overridden, because the two disagree exactly when the hand-stamp is wrong.
    """
    rows = [_row("4100", "s", TRESUsageInTot="cpu=00:00:10,energy=0", EnergyPerCpu="")]

    # Input never gathered -> the derived column is not_derivable, computed not declared.
    ds = rows_to_dataset(
        rows,
        availability={"TRESUsageInTot": "not_gathered"},
        derived_from={"EnergyPerCpu": ("TRESUsageInTot",)},
    )
    assert ds["EnergyPerCpu"].attrs["availability"] == "not_derivable"
    assert ds["EnergyPerCpu"].attrs["provenance_class"] == "derived"
    assert list(ds["EnergyPerCpu"].attrs["derived_from"]) == ["TRESUsageInTot"]
    # The input keeps its own, unpropagated, stamp.
    assert ds["TRESUsageInTot"].attrs["availability"] == "not_gathered"

    # Inputs all available -> the derived column is available.
    ds2 = rows_to_dataset(rows, derived_from={"EnergyPerCpu": ("TRESUsageInTot",)})
    assert ds2["EnergyPerCpu"].attrs["availability"] == "available"

    # A hand-stamp alongside derived_from is refused.
    with pytest.raises(StoreSchemaError, match="must not also be supplied by hand"):
        rows_to_dataset(
            rows,
            availability={"EnergyPerCpu": "available"},
            derived_from={"EnergyPerCpu": ("TRESUsageInTot",)},
        )


def test_output_observed_stamping_conflates_the_two_blank_causes():
    """The pre-fix comparator for the rider, RUN rather than described.

    Two rows, two DIFFERENT causes of blankness, one indistinguishable observed result. This
    is why the stamp cannot be read off the output.
    """

    def _observed_stamp(cell: str) -> str:
        """The naive rule the rider forbids: look at the output and guess."""
        return "available" if (cell or "").strip() else "not_gathered"

    cpu_only_row_gpu_cell = ""  # column does not apply -> undefined_at_this_grain upstream
    gpu_row_sampler_failed = ""  # measurement never taken -> not_gathered upstream

    assert _observed_stamp(cpu_only_row_gpu_cell) == _observed_stamp(gpu_row_sampler_failed), (
        "output observation no longer conflates the two causes; this comparator is stale"
    )
    # Observation also produces a stamp the class constraint would REFUSE on a derived
    # column, which is the second, independent reason the naive rule cannot be used.
    with pytest.raises(StoreSchemaError):
        validate_availability("SomeDerivedGpuColumn", _observed_stamp(gpu_row_sampler_failed))


def test_derive_availability_rejects_an_unknown_input_value():
    """A typo'd availability must raise rather than resolve to a default."""
    from hhemt.slurm_store import derive_availability

    with pytest.raises(StoreSchemaError, match="is not one of"):
        derive_availability(["availble"])


# ---------------------------------------------------------------- D5, D6


def test_string_columns_are_fixed_width_and_open_under_auto_chunks(tmp_path):
    """PROPERTY (D5): no object dtype, so `chunks='auto'` works for every future reader.

    Pre-fix comparator: none in-tree -- the predecessor is a CSV with no dtypes at all. The
    trap this forecloses is measured: an object-dtype string column raises
    `NotImplementedError: Can not use auto rechunking with object dtype`, which is why the
    combine path had to special-case `chunks={}` for the sensitivity tree.
    """
    write_store(tmp_path, [_row("5001", "2026-08-16T10:00:00")])
    with xr.open_dataset(store_path(tmp_path), engine="zarr", chunks="auto", consolidated=False) as ds:
        ds.load()
        for name, var in ds.data_vars.items():
            assert var.dtype != np.dtype("O"), f"{name} is object dtype"


def test_string_width_is_sized_per_column_from_the_data():
    """PROPERTY (D5): the width follows the data, so a wide column does not raise.

    Pre-fix comparator: a single global `<U64`, which RAISED on a real campaign store --
    `TRESUsageInTot` carries 105 characters there, against the 36 of `JobName` that the
    64 was sized from. `test_a_global_64_would_have_raised_on_real_data` RUNS that.
    """
    from hhemt.slurm_store import STRING_WIDTH_FLOOR, string_width_for

    assert string_width_for(["short"]) == STRING_WIDTH_FLOOR
    assert string_width_for([]) == STRING_WIDTH_FLOOR
    # 105 -> next multiple of 32 above it.
    assert string_width_for(["x" * 105]) == 128
    assert string_width_for(["x" * 128]) == 128
    assert string_width_for(["x" * 129]) == 160

    ds = rows_to_dataset([_row("6001", "s", TRESUsageInTot="k=v," * 30)])
    assert ds["TRESUsageInTot"].dtype.kind == "U"
    assert ds["TRESUsageInTot"].values[0] == "k=v," * 30, "a wide value was truncated"


def test_a_global_64_would_have_raised_on_real_data():
    """The pre-fix comparator for the test above, RUN rather than described."""
    from hhemt.slurm_store import _encode_string_column

    real_shape = "cpu=00:00:13,energy=0,fs/disk=0,gres/gpumem=0,gres/gpuutil=0,mem=49524K,pages=0,vmem=0"
    assert len(real_shape) > 64
    with pytest.raises(StoreSchemaError, match="exceeds the store's"):
        _encode_string_column("TRESUsageInTot", [real_shape], width=64)


def test_a_pinned_width_still_refuses_a_silent_truncation():
    """PROPERTY (D5): the guard survives per-column sizing for any caller that pins a width."""
    from hhemt.slurm_store import _encode_string_column

    with pytest.raises(StoreSchemaError, match="exceeds the store's"):
        _encode_string_column("NodeList", ["x" * 200], width=64)


def test_a_requeued_job_id_retains_both_instances(tmp_path):
    """PROPERTY (D6): the key is `(JobID, Submit)`, so a requeue keeps both executions.

    Pre-fix comparator: a bare-`JobID` key. The real shape of job 18583265 -- it ran
    07:54:22 on 8 CPUs, hit NODE_FAIL, and was requeued into an instance that never ran --
    fuses under a bare key, and the later instance's non-empty `00:00:00` overwrites the
    real elapsed. That is a wrong VALUE rather than a blank, which is the class that gets
    believed rather than questioned.
    """
    assert KEY_FIELDS == ("JobID", "Submit")
    first = _row("18583265", "2026-08-16T10:43:33", Elapsed="07:54:22", NCPUS="8", State="NODE_FAIL")
    second = _row("18583265", "2026-08-16T18:38:10", Elapsed="00:00:00", NCPUS="0", State="CANCELLED by 554635")

    write_store(tmp_path, [first])
    write_store(tmp_path, [second])

    rows = [r for r in read_rows(tmp_path) if r["MainJobID"] == "18583265"]
    assert len(rows) == 2, f"the requeued instances fused; store holds {len(rows)} row(s)"
    elapsed = {r["Submit"]: r["Elapsed"] for r in rows}
    assert elapsed["2026-08-16T10:43:33"] == "07:54:22", "the NODE_FAIL execution's elapsed was lost"


def test_a_bare_jobid_key_would_fuse_them(tmp_path):
    """The pre-fix comparator for the test above, RUN rather than described."""
    first = _row("18583265", "2026-08-16T10:43:33", Elapsed="07:54:22")
    second = _row("18583265", "2026-08-16T18:38:10", Elapsed="00:00:00")
    fused: dict[str, dict] = {}
    for row in (first, second):
        target = fused.setdefault(row["JobID"], {})
        for k, v in row.items():
            if v.strip():
                target[k] = v
    assert len(fused) == 1
    assert fused["18583265"]["Elapsed"] == "00:00:00", (
        "the bare-key comparator no longer fuses; this test's premise is stale"
    )


# ---------------------------------------------------------------- D4, store contract


def test_the_store_declares_a_schema_version_and_its_invariant(tmp_path):
    """PROPERTY: the store says which schema it is.

    Measured on the predecessor artifact: an 11-column and an 18-column generation were live
    on disk at once with nothing recording which was which, so the same report rendered from
    two bundles disagreed about whether a column existed.
    """
    write_store(tmp_path, [_row("7001", "s")])
    with xr.open_dataset(store_path(tmp_path), engine="zarr", chunks="auto", consolidated=False) as ds:
        assert ds.attrs["schema_version"] == 1
        assert list(ds.attrs["key_fields"]) == list(KEY_FIELDS)
        assert "read time" in ds.attrs["store_invariant"]


def test_the_store_is_column_agnostic(tmp_path):
    """PROPERTY: a column nobody declared is carried, so the derived-column ruling can land
    without a schema fight here."""
    write_store(tmp_path, [_row("8001", "s", SomeDerivedColumn="42", AnotherOne="text")])
    rows = read_rows(tmp_path)
    assert rows[0]["SomeDerivedColumn"] == "42"
    assert rows[0]["AnotherOne"] == "text"


def test_write_is_atomic_leaving_no_tmp_directory(tmp_path):
    """PROPERTY (D3): the staged write cleans up, so a reader never sees a half-store."""
    write_store(tmp_path, [_row("9001", "s")])
    path = store_path(tmp_path)
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


# ---------------------------------------------------------------- the cutover seam


def test_load_for_renderer_serves_both_render_inputs_from_the_one_store(tmp_path):
    """PROPERTY: the store alone serves both structures the renderer joins three sources for.

    Pre-fix comparator: the renderer builds `merged` from the plugin CSVs and `recovery`
    from `job_level_recovery.csv`, joining them (plus `_job_index.json`) at render time.
    This asserts one artifact can supply both, which is the whole consolidation.
    """
    from hhemt.slurm_store import load_for_renderer

    write_store(
        tmp_path,
        [
            _row("500", "2026-08-16T10:00:00", RuleName="run_triton"),
            _row("500.batch", "2026-08-16T10:00:00"),
            _row("500.0", "2026-08-16T10:00:00", JobName="python"),
        ],
    )
    merged, recovery = load_for_renderer(tmp_path)

    assert set(merged) == {"500", "500.batch", "500.0"}
    assert set(recovery["500"]) == {"job", "batch", "0"}
    assert recovery["500"]["job"]["RuleName"] == "run_triton", "the job-to-rule label did not survive"
    assert merged["500.0"]["JobName"] == "python"


def test_the_compatibility_view_collapses_a_requeue_by_greatest_submit_not_file_order(tmp_path):
    """PROPERTY: the narrowing is a STATED choice, and the full set stays reachable.

    `recovery`'s inner dict is keyed on StepKind and so admits one row per (job, kind); a
    requeued job has two `job`-kind rows. Collapsing by greatest Submit is deterministic,
    where taking whichever row the reader yields last is not.
    """
    from hhemt.slurm_store import load_for_renderer

    write_store(
        tmp_path,
        [
            _row("18583265", "2026-08-16T10:43:33", Elapsed="07:54:22", State="NODE_FAIL"),
            _row("18583265", "2026-08-16T18:38:10", Elapsed="00:00:00", State="CANCELLED by 554635"),
        ],
    )
    _merged, recovery = load_for_renderer(tmp_path)

    assert recovery["18583265"]["job"]["Submit"] == "2026-08-16T18:38:10"
    # ...and nothing was destroyed: both executions are still in the store.
    both = [r for r in read_rows(tmp_path) if r["MainJobID"] == "18583265"]
    assert len(both) == 2
    assert {r["Elapsed"] for r in both} == {"07:54:22", "00:00:00"}
