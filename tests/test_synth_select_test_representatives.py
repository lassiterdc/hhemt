"""Pure-unit coverage of ``TRITONSWMM_analysis._select_test_representatives``.

No model runs, no compile, no Snakemake, and no scheduler predicate: this
module consumes the configured-but-not-run ``synth_sensitivity_builder``
fixture and calls the selector directly. That matters beyond speed. The
compile-bearing ``.test()`` mirror and the ``from_doi`` run-proof are both
skipped on a SLURM node, so on an HPC harness this module is the only thing
that exercises the grouping key at all.

The headline assertion is a TOTALITY check, not a count: the set of
compute-configs across the selected representatives must EQUAL the set across
the candidates the selector enumerated. The denominator is derived from the
analysis rather than declared here, so a row added to the sensitivity CSV
strengthens the assertion instead of stranding a hard-coded number.

Residual, named rather than left implicit: the shared fixture calls
``_require_cpu_cores_for_sensitivity()``, which skips below 4 CPUs. That gate
guards the honored-thread validator during a full workflow RUN, which this
module never performs, so it is over-scoped for this consumer. It is inherited
deliberately -- forking the fixture to dodge one skip would create a second
construction path that drifts invisibly from the first.
"""


def _compute_config(analysis):
    """The compute-config component of the selector's group key, read off an
    analysis exactly as ``_group_key`` reads it."""
    cfg = analysis.cfg_analysis
    return (
        cfg.run_mode,
        cfg.n_mpi_procs,
        cfg.n_omp_threads,
        cfg.n_gpus,
        cfg.n_nodes,
    )


#: Shape contract of ``_group_key``'s returned tuple, stated once so both call
#: sites below read the compute-config by contract rather than by bare position.
_KEY_ARITY = 4
_KEY_SHAPE = "(model_toggles, compilation_backend, partition, compute_config)"


def _compute_config_of_key(key):
    """The compute-config member of a representative's group key.

    Checks the key's ARITY before indexing. A regression that changes the key's
    shape -- a dropped or added member -- is then reported as a key-shape
    failure naming the expected shape, instead of surfacing as an ``IndexError``
    raised from inside a set comprehension at the call site. Measured
    2026-08-25: the bare ``key[3]`` form died on ``IndexError: tuple index out
    of range`` under a mutant that dropped ``compute_config``, describing the
    comprehension rather than the key.
    """
    if len(key) != _KEY_ARITY:
        raise AssertionError(f"group key has arity {len(key)}, expected {_KEY_ARITY} {_KEY_SHAPE}; got {key!r}")
    return key[_KEY_ARITY - 1]


def test_selection_covers_every_compute_config(synth_sensitivity_builder):
    """Every distinct compute-config among the candidates is represented exactly once.

    Measured on the synth sensitivity master (2026-08-25): 4 members ->
    4 representatives -> 4 distinct keys, totality holds. Under a mutant
    ``_group_key`` that drops ``compute_config`` the 4 candidates collapse to
    1 group; ``_compute_config_of_key`` reports the arity change as a key-shape
    failure. Under an arity-preserving mutant the totality assertion below
    reports the set difference directly.
    """
    analysis = synth_sensitivity_builder
    candidates = list(analysis.sensitivity.members.values())
    assert len(candidates) >= 2, (
        f"fixture supplied {len(candidates)} member/-es; this test needs "
        "a multi-candidate master to reach the grouping branch at all"
    )

    reps = analysis._select_test_representatives()

    keys = [rep.key for rep in reps]
    assert len(set(keys)) == len(keys), f"duplicate representative keys: {keys}"

    selected = {_compute_config_of_key(rep.key) for rep in reps}
    expected = {_compute_config(candidate) for candidate in candidates}
    assert selected == expected, (
        "representative compute-configs must be exactly the candidates' "
        f"compute-configs; missing={expected - selected}, extra={selected - expected}"
    )


def test_selection_picks_a_real_candidate_per_group(synth_sensitivity_builder):
    """Each representative's ``source_analysis`` is one of the enumerated candidates.

    Guards the ``min(members, key=_device_demand)`` step against returning
    anything the selector did not receive as a candidate.
    """
    analysis = synth_sensitivity_builder
    candidate_ids = {id(sub) for sub in analysis.sensitivity.members.values()}
    reps = analysis._select_test_representatives()
    orphans = [rep.key for rep in reps if id(rep.source_analysis) not in candidate_ids]
    assert not orphans, f"representatives not drawn from the candidate set: {orphans}"


def test_non_sensitivity_master_is_its_own_single_candidate(synth_multi_sim_analysis):
    """The differently-positioned SATISFYING state: exactly one group, totality intact.

    A non-sensitivity analysis takes the ``candidates = [self]`` branch, so one
    representative is CORRECT here. Pinning it keeps the multi-group assertion
    above from being read as 'more groups is always better'.
    """
    reps = synth_multi_sim_analysis._select_test_representatives()
    assert len(reps) == 1, f"expected exactly one representative, got {len(reps)}"
    assert {_compute_config_of_key(rep.key) for rep in reps} == {_compute_config(synth_multi_sim_analysis)}
