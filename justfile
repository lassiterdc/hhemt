# Justfile for TRITON-SWMM_toolkit

# Show available commands
list:
    @just --list

# Run all the formatting, linting, and testing commands
qa:
    uv run --python=3.12 --extra test ruff format .
    uv run --python=3.12 --extra test ruff check . --fix
    uv run --python=3.12 --extra test ruff check --select I --fix .
    uv run --python=3.12 --extra test ty check .
    uv run --python=3.12 --extra test python scripts/check_du_sentinel_sites.py
    uv run --python=3.12 --extra test python scripts/check_vocabulary_freeze.py
    # Gate A's population, hosted on the low-barrier uv path so a contributor needs no
    # conda env. The marker is stamped at COLLECTION by tests/conftest.py from each
    # test's fixture closure, and the compile gate's skip fires later at fixture SETUP,
    # so this selects the same tests under uv as under conda. That equality is ASSERTED
    # here and checked by nothing -- do not record a count in this comment; the last one
    # (2996/3195) was stale within weeks. `just test-fast` is the conda-hosted
    # sibling that additionally PROVES the marker did the deselecting rather than PATH.
    uv run --python=3.12 --extra test --extra docs pytest -m "not slow and not compile_tier and not requires_snakemake_subprocess"

# Run all the tests for all the supported Python versions
testall:
    uv run --python=3.11 --extra test --extra docs pytest
    uv run --python=3.12 --extra test --extra docs pytest

# Run all the tests, but allow for arguments to be passed.
# NOTE: the compile-dependent tests (test_synth_04/05, test_workflow_rerun_triggers_empirical)
# build TRITON-SWMM CPU, which needs cmake + mpic++ + mpi.h from the `hhemt` conda env.
# Under this uv `.venv` command they SKIP cleanly (tests/utils_for_testing.py::
# compile_toolchain_unavailable). To actually run them green, invoke under the conda env
# (drop the 3.12 pin; the hhemt env is python 3.11 and uv --active uses it):
#   conda run -n hhemt uv run --active --extra test pytest {{ARGS}}
test *ARGS:
    @echo "Running with arg: {{ARGS}}"
    uv run --python=3.12 --extra test --extra docs pytest {{ARGS}}

# Authoritative gating run: the conda env supplies cmake+mpic++ AND
# HHEMT_REQUIRE_COMPILE_TIER=1 turns any compile-tier skip into a HARD FAILURE,
# so the coupled compile->run->process->report tier is GATED, not silently skipped.
# This is the invocation a toolchain-bearing CI job (or a pre-merge check) should run.
# GATE B. This is the invocation the D70 green-suite claim is made against, UNDESELECTED,
# run detached by an external wrapper (never from this file). It is the only invocation
# with no silent-skip arm, because HHEMT_REQUIRE_COMPILE_TIER=1 converts a compile-tier
# skip into a hard failure. --no-capture-output is load-bearing under detachment: without
# it `conda run` buffers the child's stdout AND stderr, so a killed run leaves an empty
# log and pyproject's faulthandler dumps are discarded (measured: 0 bytes on disk 3s into
# a 6s job). The JSONL ledger survives regardless -- the plugin writes its own fd.
test-gated *ARGS:
    @echo "Running GATED (compile tier required) with arg: {{ARGS}}"
    HHEMT_COMPILE_VENUE=toolchain HHEMT_REQUIRE_COMPILE_TIER=1 PYTHONPATH="${PWD}/src/hhemt/suite" HHEMT_SUITE_LOGREPORT_OUT="${PWD}/suite_ledger_gated.jsonl" env -u VIRTUAL_ENV conda run --no-capture-output -n hhemt uv run --active --extra test --extra docs pytest -p _runner {{ARGS}}

# GATE A -- the fast, waitable contributor gate. NOT the D70 gate.
# Runs under conda ON PURPOSE: the toolchain is present there, so a green result proves
# the MARKER deselected the compile tier rather than the PATH having skipped it.
# HHEMT_FORBID_COMPILE=1 arms the fail-closed guard in tests/conftest.py, so a test that
# compiles here fails loudly instead of costing ten minutes.
test-fast *ARGS:
    @echo "Gate A (fast tier) with arg: {{ARGS}}"
    HHEMT_FORBID_COMPILE=1 PYTHONPATH="${PWD}/src/hhemt/suite" HHEMT_SUITE_LOGREPORT_OUT="${PWD}/suite_ledger_fast.jsonl" env -u VIRTUAL_ENV conda run --no-capture-output -n hhemt uv run --active --extra test --extra docs pytest -p _runner -m "not slow and not compile_tier and not requires_snakemake_subprocess" {{ARGS}}

# LAPTOP TIER -- everything that does not compile or run a solver and is not a Snakemake-subprocess test. Solver
# compiles and runs belong to the Rivanna tier, so `-m "not compile_tier"` leaves them out, and HHEMT_FORBID_COMPILE=1
# makes any test that still reaches a compile fail loudly instead of costing minutes. The 96 tests marked
# `requires_snakemake_subprocess` (they start Snakemake as a child process) are NOT in this tier: they cost about 25
# minutes run serially (1,498.28 s, run laptop-20261104T173100Z-11ac40f5) against 354.57 s for the rest (run
# laptop-20261104T172458Z-11ac40f5). They run in every whole-suite Rivanna array, and in `test-snakemake-subprocess`
# below on a laptop; a `--select-tier compile_tier` Rivanna run excludes them because they are not `compile_tier`.
# ONE invocation: `--dist loadgroup` spreads every selected test over every core (`-n` equal to the core count) and
# keeps each `xdist_group` on one worker, so the tests derived as `shared_runs_root` (tests/conftest.py) run together
# on one worker while the rest are handed out one at a time, which has no consecutive-batch tail. The recipe exports
# nothing for the runs root: an autouse fixture in tests/conftest.py points each unmarked test at an empty session
# directory and leaves it unset for each marked one. `-p _runner` writes the JSONL ledger named by
# HHEMT_SUITE_LOGREPORT_OUT.
test-laptop *ARGS:
    @echo "Laptop tier, with arg: {{ARGS}}"
    HHEMT_FORBID_COMPILE=1 PYTHONPATH="${PWD}/src/hhemt/suite" HHEMT_SUITE_LOGREPORT_OUT="${PWD}/suite_ledger_laptop.jsonl" env -u VIRTUAL_ENV conda run --no-capture-output -n hhemt uv run --active --extra test --extra docs pytest -p _runner -n "$(nproc)" --dist loadgroup -m "not compile_tier and not requires_snakemake_subprocess" {{ARGS}}

# The tests the laptop tier leaves out: `requires_snakemake_subprocess` and not `compile_tier`, run LOCALLY AND SERIALLY
# (no `-n`: a child Snakemake scheduler under xdist is nested parallelism). About 25 minutes: 1,498.28 s for the 96 in
# run laptop-20261104T173100Z-11ac40f5. Same shape as the other pytest recipes.
test-snakemake-subprocess *ARGS:
    @echo "Snakemake-subprocess tests (serial), with arg: {{ARGS}}"
    HHEMT_FORBID_COMPILE=1 PYTHONPATH="${PWD}/src/hhemt/suite" HHEMT_SUITE_LOGREPORT_OUT="${PWD}/suite_ledger_snakemake_subprocess.jsonl" env -u VIRTUAL_ENV conda run --no-capture-output -n hhemt uv run --active --extra test --extra docs pytest -p _runner -m "requires_snakemake_subprocess and not compile_tier" {{ARGS}}

# Run all the tests, but on failure, drop into the debugger
pdb *ARGS:
    @echo "Running with arg: {{ARGS}}"
    uv run --python=3.12  --extra test --extra docs pytest --pdb --maxfail=10 --pdbcls=IPython.terminal.debugger:TerminalPdb {{ARGS}}

# Run coverage, and build to HTML
coverage:
    uv run --python=3.12 --extra test --extra docs coverage run -m pytest .
    uv run --python=3.12 --extra test coverage report -m
    uv run --python=3.12 --extra test coverage html

# The documentation gates: the SAME script CI runs, on the per-tree uv .venv, whose
# editable .pth names THIS checkout. Never `uv run --active` here: under a foreign
# VIRTUAL_ENV it re-points another tree's editable install at this one (measured).
docs-check:
    RUN="uv run --locked --extra docs" bash scripts/docs_gates.sh

# Build the project, useful for checking that packaging is correct
build:
    rm -rf build
    rm -rf dist
    uv build

VERSION := `grep -m1 '^version' pyproject.toml | sed -E 's/version = "(.*)"/\1/'`

# Print the current version of the project
version:
    @echo "Current version is {{VERSION}}"

# Tag the current version in git and put to github
tag:
    git diff --quiet HEAD -- CITATION.cff pyproject.toml
    python scripts/check_release_metadata.py
    echo "Tagging version v{{VERSION}}"
    git tag -a v{{VERSION}} -m "Creating version v{{VERSION}}"
    git push origin v{{VERSION}}

# remove all build, test, coverage and Python artifacts
clean: 
	clean-build
	clean-pyc
	clean-test

# remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

# remove Python file artifacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

# remove test and coverage artifacts
clean-test:
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache