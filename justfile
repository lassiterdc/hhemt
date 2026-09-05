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
    # so this selects the same tests under uv as under conda -- measured, 2996/3195 on
    # both hosts with zero node-id difference. `just test-fast` is the conda-hosted
    # sibling that additionally PROVES the marker did the deselecting rather than PATH.
    uv run --python=3.12 --extra test pytest -m "not slow and not compile_tier and not requires_snakemake_subprocess"

# Run all the tests for all the supported Python versions
testall:
    uv run --python=3.11 --extra test pytest
    uv run --python=3.12 --extra test pytest

# Run all the tests, but allow for arguments to be passed.
# NOTE: the compile-dependent tests (test_synth_04/05, test_workflow_rerun_triggers_empirical)
# build TRITON-SWMM CPU, which needs cmake + mpic++ + mpi.h from the `hhemt` conda env.
# Under this uv `.venv` command they SKIP cleanly (tests/utils_for_testing.py::
# compile_toolchain_unavailable). To actually run them green, invoke under the conda env
# (drop the 3.12 pin; the hhemt env is python 3.11 and uv --active uses it):
#   conda run -n hhemt uv run --active --extra test pytest {{ARGS}}
test *ARGS:
    @echo "Running with arg: {{ARGS}}"
    uv run --python=3.12 --extra test pytest {{ARGS}}

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
    HHEMT_REQUIRE_COMPILE_TIER=1 PYTHONPATH="${PWD}/src/hhemt/suite" HHEMT_SUITE_LOGREPORT_OUT="${PWD}/suite_ledger_gated.jsonl" conda run --no-capture-output -n hhemt uv run --active --extra test pytest -p _runner {{ARGS}}

# GATE A -- the fast, waitable contributor gate. NOT the D70 gate.
# Runs under conda ON PURPOSE: the toolchain is present there, so a green result proves
# the MARKER deselected the compile tier rather than the PATH having skipped it.
# HHEMT_FORBID_COMPILE=1 arms the fail-closed guard in tests/conftest.py, so a test that
# compiles here fails loudly instead of costing ten minutes.
test-fast *ARGS:
    @echo "Gate A (fast tier) with arg: {{ARGS}}"
    HHEMT_FORBID_COMPILE=1 PYTHONPATH="${PWD}/src/hhemt/suite" HHEMT_SUITE_LOGREPORT_OUT="${PWD}/suite_ledger_fast.jsonl" conda run --no-capture-output -n hhemt uv run --active --extra test pytest -p _runner -m "not slow and not compile_tier and not requires_snakemake_subprocess" {{ARGS}}

# Run all the tests, but on failure, drop into the debugger
pdb *ARGS:
    @echo "Running with arg: {{ARGS}}"
    uv run --python=3.12  --extra test pytest --pdb --maxfail=10 --pdbcls=IPython.terminal.debugger:TerminalPdb {{ARGS}}

# Run coverage, and build to HTML
coverage:
    uv run --python=3.12 --extra test coverage run -m pytest .
    uv run --python=3.12 --extra test coverage report -m
    uv run --python=3.12 --extra test coverage html

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