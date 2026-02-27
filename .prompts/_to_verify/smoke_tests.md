# Run Smoke Tests

Run all 4 required smoke tests in the correct order and report results.

## Test Sequence

Run these tests in this exact order:

1. `python -m pytest tests/test_PC_01_singlesim.py -v`
2. `python -m pytest tests/test_PC_02_multisim.py -v`
3. `python -m pytest tests/test_PC_04_multisim_with_snakemake.py -v`
4. `python -m pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py -v`

## Notes

- **Do NOT impose artificial timeouts on PC_05** — it legitimately takes 12-15 minutes
- Report pass/fail counts for each test file
- If any test fails, investigate before proceeding
- All tests must pass before committing significant changes

## Expected Output

Summary table showing:
- Test file name
- Pass/fail/skip counts
- Total duration
- Overall status (ALL PASS / FAILURES DETECTED)
