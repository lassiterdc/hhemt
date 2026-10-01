"""CF-1.13 section 7.3 cell_methods admissibility, as an ORACLE independent of the code it audits.

This module deliberately imports nothing from `hhemt.cf_conventions`. The tests that consume it
audit what `apply_cf_attributes` stamps; sourcing the admissibility rule from the module under
audit would make a wrong rule agree with itself at both ends, and the disagreement the test
exists to surface becomes structurally unreachable. The oracle must be independent of the subject.
"""

from __future__ import annotations

# CF-1.13 section 7.3.4 licenses a `standard_name` as a cell_methods name and names `time` in its
# own worked example. This repository vendors no CF standard-name table, so the class-3 vocabulary
# is an EXPLICIT enumeration. It is NOT derived from the corpus's declared `standard_name` values:
# measured, `grep '"standard_name": "time"' src/hhemt/cf_conventions.py` returns zero, because a
# `standard_name` attribute names a VARIABLE's quantity while a class-3 cell_methods name names an
# AXIS. Adding a member here is a deliberate act that must record its section 7.3.4 precondition.
CF_STANDARD_NAMES_ADMITTED_AS_CELL_METHODS_NAMES = frozenset({"time"})


def cell_method_pairs(cell_methods: str) -> list[tuple[str, str]]:
    r"""Split a CF cell_methods string into its `(name, method)` pairs.

    CF-1.13 section 7.3 defines the attribute as a list of blank-separated words of the form
    `name: method`, so a MULTI-PAIR string cannot be parsed by splitting on the colon.
    Measured: `"time: sum Rank: maximum".split(":")` is `['time', ' sum Rank', ' maximum']` --
    the middle token carries the first pair's METHOD and the second pair's NAME at once, so the
    colon split yields no pair structure at all rather than the wrong element of one.

    SCOPED, and the scope is stated over the CLASS rather than as a list of known gaps.
    CF-1.13 section 7.3's opening sentence defines the attribute as "a list of blank-separated
    words of the form `name: method`" (library normalized line 2972), and CF's own later
    sections widen BOTH halves of that form. This parser implements the one-word-name /
    one-word-method core and REFUSES every widening -- 7.3.1 (multi-name), 7.3.2
    (parenthetical), 7.3.3 (portion clause), 7.4 (climatological) and the operand form --
    loudly rather than by mis-parsing.

    Illustrative census, and the INSTRUMENT is stated because the figure is only as complete
    as the extractor that built it: matching `cell_methods\s*=\s*"..."` over CF's text as ONE
    STRING (whitespace tolerated around `=`, literals allowed to span a source line break)
    yields 36 distinct worked literals, of which this parser accepts 14 and refuses 22.
    A line-oriented pattern requiring the quote to abut the `=` returns 32/11/21 instead,
    because CF writes most worked examples as CDL declarations. Re-derive with the stated
    instrument before concluding this parser has drifted.

    The 22 refusals are FIVE families. FOUR are false-grammar refusals (17 values):
      * multi-name pair      -- `lat: lon: standard_deviation`      (CF 7.3.1, line 3045)
      * portion clause       -- `area: mean where sea_ice over sea` (CF 7.3.3, line 3150)
      * climatological qual. -- `time: mean within days time: mean over days` (CF 7.4, line 3296)
      * method with operand  -- `area: anomaly_wrt areamin`         (CF 7.5, line 3699)
    The FIFTH is honestly scoped rather than false-grammar (5 values):
      * parenthetical qual.  -- `time: standard_deviation (interval: 1 day)` (CF 7.3.2, line 3061)

    No value this repository stamps is in any of them. A NEW corpus value that is refused here
    is a signal to widen this parser, NOT a signal that the value is malformed.

    EXEMPLAR CONVENTION, stated because a sample is not produced by the instrument above and
    therefore is not reached by re-deriving the census: each exemplar is a VERBATIM member of
    the 36-value corpus, and each cited line is the line of the CORPUS IDENTIFIED BELOW on
    which that literal occurs -- not the section heading, not the generic form, not a range.

    CORPUS, by identifier rather than by description, because a line number into an
    unidentified text is not a citation. In the agentic-workspace repository:
        path    library/knowledge/xarray/references/cf-conventions/
                  .normalized/cf conventions full text normalized.md
        lines   10278
        sha256  ae75a24e2e9dc248...
    A sibling extraction lives beside it at .raw/ and is 10282 lines; every locator here is
    off by +4 against it, and the +4 landing is a well-formed on-topic CF sentence, so a
    wrong-copy resolution reads as five authoring errors rather than as one wrong file.
    CHECK THE CORPUS FIRST -- it is the only step that fails loudly:
        CF=<the path above> ; wc -l "$CF"        # must print 10278
        sha256sum "$CF"                          # must open ae75a24e2e9dc248
    Then verify both clauses of each pair, and re-verify after any corpus re-pin:
        grep -Fc '{exemplar}' "$CF"              # literal is in the corpus
        sed -n  '{line}p'     "$CF"              # cited line carries it
    """
    if "(" in cell_methods:
        raise ValueError(f"qualified cell_methods are outside this parser's scope: {cell_methods!r}")
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for token in cell_methods.split():
        if token.endswith(":"):
            if pending is not None:
                raise ValueError(f"two names with no method between them: {cell_methods!r}")
            pending = token[:-1]
        else:
            if pending is None:
                raise ValueError(f"a method with no name before it: {cell_methods!r}")
            pairs.append((pending, token))
            pending = None
    if pending is not None or not pairs:
        raise ValueError(f"unparseable cell_methods: {cell_methods!r}")
    return pairs


def cell_methods_name_class(name: str, da, ds) -> str | None:
    """Which of CF-1.13 section 7.3's FOUR name classes admits `name`, or None.

    Returns one of "dimension", "scalar_coordinate", "area", "standard_name", or None.

    Classes 1 and 3 are MUTUALLY EXCLUSIVE by section 7.3.4's own prohibition, so the
    standard-name arm asserts the ABSENCE that LICENSES it rather than the presence that would
    forbid it. That re-statement is redundant with the early return above and is kept
    deliberately: the campaign's decisive defect was a guard encoding the NEGATION of this
    precondition, and a guard that states its precondition positively cannot be read that way.

    THE CLASS-3 GUARD IS DELIBERATELY WIDER THAN CF'S NORMATIVE PRECONDITION, AND THE EXTRA
    WIDTH IS THIS MODULE'S CHOICE RATHER THAN A CF-SANCTIONED OPTION. CF states the
    precondition twice at two different widths. The NORMATIVE sentence (corpus line 3164)
    forbids the standard-name form when `name` matches "a dimension or SCALAR coordinate
    variable". The parenthetical gloss (line 3171) states the same assumption WITHOUT
    "scalar" -- "it is assumed here that ... 'time' is not a dimension or coordinate variable"
    -- but it opens "As required by this convention, it is assumed here", so it DEFERS to the
    normative sentence rather than widening it. The `name not in ds.coords` arm below is the
    wider form: it also refuses a 1-d coordinate variable, which 3164 would permit. That is a
    conservative choice, and it happens to coincide with the width at which 3171 states the
    assumption. Narrowing it to `ndim == 0` would be CF-conformant; do not do so without
    recording why, because the width is the whole content of this paragraph.

    TEST-SCOPE CLAUSE. This is a TEST oracle and not a production validator. It is permitted to
    refuse a value CF admits, because a refusal here is a loud red that a maintainer reads,
    whereas a production validator refusing a conformant value would reject a legitimate store.
    The asymmetry is why the extra width above is acceptable at all.
    """
    if name in da.dims:
        return "dimension"
    if name in ds.coords and ds[name].ndim == 0:
        return "scalar_coordinate"
    if name == "area":
        return "area"
    if name in CF_STANDARD_NAMES_ADMITTED_AS_CELL_METHODS_NAMES and name not in da.dims and name not in ds.coords:
        return "standard_name"
    return None


def cell_methods_classes(cell_methods: str, da, ds, *, where: str) -> list[str]:
    """Assert every pair names something CF admits, and RETURN the per-pair class list.

    The return value is what distinguishes a two-class annotation from a renamed dimension. A
    caller that only asserts "it resolved" is satisfied by `"Rank: sum Rank: maximum"`; a caller
    that asserts the class SEQUENCE is not.

    CLASS 2 (scalar coordinate variable) IS TAKEN BY ZERO VALUES IN THIS CORPUS, and its
    presence here is CORRECTNESS, not coverage. Omitting it would make this oracle refuse a
    legitimate CF form. It is also the tripwire for a regression to the scalar carrier: were
    the summary's `Rank` carrier ever changed back to a scalar coordinate, the class-2 arm would
    admit the per-member store while the class-1 arm rejected the consolidated one -- the exact
    asymmetry `test_a_differing_scalar_rank_would_not_have_survived_the_concat` exists to hold.
    That arm and this one are the same argument expressed twice and must not drift apart.
    """
    classes: list[str] = []
    for name, _method in cell_method_pairs(cell_methods):
        klass = cell_methods_name_class(name, da, ds)
        assert klass is not None, (
            f"{where}: cell_methods {cell_methods!r} names {name!r}, which is none of CF-1.13 "
            f"section 7.3's four classes for this variable (dims {da.dims}); CF admits a dimension "
            "of the variable, a scalar coordinate variable, a valid standard name, or 'area'"
        )
        classes.append(klass)
    return classes
