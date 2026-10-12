"""Item 2's declared test: the realised retry ceiling is disclosed, not implied.

Work package 1a, chunk 1a.2. The property this file asserts, as the shape's Form A row
states it: *the realised ceiling is disclosed and equals the per-process budget rather
than its multiplied product.*

WHY THE PROPERTY IS REQUIRED, as a consequence a reader can check against the code.
``hpc_restart_times_simulate``'s description claimed a per-rule retry count. That is true
of the Snakemake DIRECTIVE and false of the realised behaviour, because the directive is
compared against a per-PROCESS counter and under ``--executor slurm`` more than one such
process reads it. The description is not merely internal prose: ``report_renderers/
metadata.py`` renders every config field's description into the report's Reproduction
Guide, so the false claim reached a surface a REPRODUCER acts on and cannot re-derive --
they hold the rendered report, not this repository.

THE INVARIANT: an operator or reproducer who reads only the numbers this toolkit presents
can compute the actual bound on solver launches. Two readers exist and both are covered
here -- the one reading the rendered Reproduction Guide, and the one reading submit-time
stdout.

The guide assertion RENDERS the guide rather than reading the source string, which is what
the Definition of Done requires: the rendered artifact is the surface the claim is about,
and a source read cannot show that the description survives the render path (the
descriptions live in the Tabulator FRAGMENTS, not in the guide's own markup, so a reader
checking only ``ReprexGuide.html`` would find nothing and conclude the opposite).
"""

from hhemt.config.analysis import analysis_config


class _StubBuilder:
    """The two surfaces ``_realised_retry_ceiling_disclosure`` reads, and nothing else.

    A stub rather than a real builder because a real ``SnakemakeWorkflowBuilder`` needs an
    analysis, a system and a tree; the disclosure's subject is ARITHMETIC over two numbers
    and a stub keeps the test non-simulation-bearing. The method is called unbound so the
    production implementation is the thing under test rather than a reimplementation.
    """

    def __init__(self, per_process: int, budget: int):
        self._per_process = per_process

        class _Cfg:
            hpc_consecutive_failure_budget_simulate = budget

        self.cfg_analysis = _Cfg()

    def _resolved_simulate_retries(self) -> int:
        return self._per_process


def _disclosure(per_process: int, budget: int) -> str:
    from hhemt.workflow import SnakemakeWorkflowBuilder

    return SnakemakeWorkflowBuilder._realised_retry_ceiling_disclosure(_StubBuilder(per_process, budget))


def test_the_disclosure_states_the_per_process_nature_and_the_multiplication():
    """The declared number is marked PER PROCESS and the multiplication is named.

    Input class: every submit-path invocation, since the guard the disclosure is emitted
    from is on every Snakemake-submitting path. Anchored on the words a reader needs to
    reach the right conclusion rather than on the whole sentence, so the prose may be
    rewritten without reding this test while its CLAIM is pinned.

    Kills: reverting the disclosure to a bare restatement of the configured number, which
    is the pre-change state of the only surface a reproducer holds.
    """
    text = _disclosure(per_process=3, budget=3)
    assert "PER SNAKEMAKE PROCESS" in text
    assert "MULTIPLIES" in text
    assert "hpc_restart_times_simulate=3" in text


def test_the_disclosure_separates_launches_per_process_from_launches_in_total():
    """A budget of N permits N+1 launches per process, and the disclosure says so.

    This is the off-by-one a reader makes unaided: ``retries: 3`` is three RESTARTS, hence
    four launches. Input class: every configured value of the knob; two positions are
    exercised rather than one so an implementation hard-coding either number reds.

    Kills: printing the retries value as the launch count.
    """
    assert "4 solver launches per process" in _disclosure(per_process=3, budget=3)
    assert "1 solver launches per process" in _disclosure(per_process=0, budget=3)


def test_the_disclosure_names_the_durable_total_bound():
    """The per-process number alone is misleading by omission, so both are reported.

    Input class: every analysis with the gate armed (budget > 0). The disclosure must name
    the knob that bounds the TOTAL, because the whole point of stating the per-process
    nature is to send the reader to the number that is additive.

    Kills: disclosing the multiplication without naming the durable budget -- which would
    tell a reader their ceiling is unbounded when item 1 has in fact bounded it.
    """
    text = _disclosure(per_process=3, budget=5)
    assert "hpc_consecutive_failure_budget_simulate" in text
    assert "at most 5 consecutive terminal" in text
    assert "walltime kill is not a terminal failure" in text


def test_the_disabled_arm_discloses_that_the_total_is_unbounded():
    """A DIFFERENTLY-POSITIONED satisfying input: the gate off.

    Input class: every analysis whose operator set the budget to 0. The honest statement
    there is the opposite of the armed one, so a disclosure that reported the armed text
    unconditionally would be false on exactly the configuration that most needs the
    warning.

    Kills: emitting one unconditional string. This is the arm that makes the test a class
    assertion rather than a single-point transcription.
    """
    text = _disclosure(per_process=3, budget=0)
    assert "DISABLES" in text
    assert "UNBOUNDED" in text
    assert "at most" not in text


def test_the_description_reaches_the_rendered_reproduction_guide():
    """The DoD's own clause: verified by RENDERING the guide, not by reading the source.

    Input class: every reproducer who holds the rendered report rather than this
    repository -- which is every reproducer, and the reason the stale claim mattered.

    Kills: two independent regressions that a source-string assertion cannot see --
    reverting the description, AND any change to the guide's field selection that drops
    these fields from the render (a reproducer would then meet no statement at all, which
    is the pre-change condition restored by a different route).
    """
    from hhemt.report_renderers.metadata import _build_reprex_guide_html

    guide = _build_reprex_guide_html()
    rendered = guide.html + "".join(repr(frag) for _, frag in guide.fragments)

    assert "PER SNAKEMAKE PROCESS" in rendered, "the per-process disclosure must survive the render path"
    assert "hpc_consecutive_failure_budget_simulate" in rendered, (
        "the knob that bounds the total must be reachable from the guide a reproducer reads"
    )


def test_both_knobs_are_bucketed_so_the_guide_can_render_at_all():
    """Guard on the producer/consumer coupling this chunk's config field created.

    ``_build_reprex_guide_html`` buckets EVERY ``analysis_config`` field through
    ``reprex_taxonomy.all_field_bucket``, which raises ``KeyError`` on an unbucketed one.
    So adding a config field without a bucket entry does not degrade the Metadata report --
    it crashes the renderer outright, and the test above would red with an error rather
    than an assertion failure, which reads like a test defect rather than a missing table
    entry. This test names the real cause so the next reader is not misdirected.

    Input class: every field on ``analysis_config``; asserted over the whole set rather
    than the two this package added, so the next config field to arrive without a bucket
    reds here too.
    """
    from hhemt.config.reprex_taxonomy import all_field_bucket

    unbucketed = []
    for name in analysis_config.model_fields:
        try:
            all_field_bucket(name)
        except KeyError:
            unbucketed.append(name)
    assert unbucketed == [], f"unbucketed analysis_config fields would crash the Metadata renderer: {unbucketed}"
