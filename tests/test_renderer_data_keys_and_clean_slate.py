"""The renderers' manifest data keys carry no unit token, and the text this branch adds carries no history.

Two independent properties, each with a control that fires when the property is violated:

(a) Renderer data. A renderer writes its per-figure facts into the plot manifest's ``renderer_data``. A key there
    follows the same convention as a stored variable: the name carries no unit token and the unit sits in a
    ``{key}_units`` companion. The test reads every module under ``src/hhemt/report_renderers`` as source, so it needs
    no render, no store and no solver.

(b) Clean slate. Documentation, docstrings, captions, messages and code comments describe the toolkit as it is. They
    do not say a name was changed, call a name old, say which layout a name belongs to, or cite a chunk or a decision
    of the change that introduced it. The matcher below runs over the lines the working tree adds relative to
    ``origin/develop``, with the versioning system excluded (the one place where history is the subject).

Every word the matcher looks for is assembled from fragments, so this file does not itself hold a word that the
matcher reports. No solver compiles or runs.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RENDERERS = REPO / "src" / "hhemt" / "report_renderers"
BUNDLE_FIXTURES = REPO / "tests" / "fixtures" / "bundles"

#: The keys that carried a unit token, assembled from parts so this file does not hold them whole.
OLD_KEYS = (
    "_".join(("peak", "flow", "max", "cms")),
    "_".join(("wse", "m", "range")),
    "_".join(("rainfall", "max", "mm", "per", "hr")),
    "_".join(("bc", "water", "level", "range", "m")),
    "_".join(("depth", "boundaries", "m")),
)

#: The keys after the change: key -> (companion key, unit value).
NEW_KEYS = {
    "peak_flow_max": ("peak_flow_max_units", "m3 s-1"),
    "wse_range": ("wse_range_units", "m"),
    "rainfall_max": ("rainfall_max_units", "mm h-1"),
    "bc_water_level_range": ("bc_water_level_range_units", "m"),
    "depth_boundaries": ("depth_boundaries_units", "m"),
}

#: The unit-token pattern of the stored-variable naming convention: a whole token of a snake_case name that states a
#: unit.
UNIT_TOKEN = re.compile(r"(^|_)(m|mps|min|cms|m3|ltr|hr|percent|perc|pct|m2ps|s|mm|10e6)(_|$)")

#: The commit before the unit tokens were removed from the renderer data, whose renderers still wrote the five keys.
SOURCE_BEFORE_KEYS_CHANGED = "209d0217"
#: The commit whose added text still held the history that the sweep removed.
TIP_BEFORE_SWEEP = "9b644fc1"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def _git_has(rev: str) -> bool:
    return (
        subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{rev}^{{commit}}"], capture_output=True).returncode
        == 0
    )


# ---------------------------------------------------------------------------------------------------------------
# (a) the renderer data keys
# ---------------------------------------------------------------------------------------------------------------


def _quoted_old_keys(source: str) -> list[str]:
    """The old keys that occur as a quoted string literal in `source`."""
    return [k for k in OLD_KEYS if re.search(r"""["']""" + re.escape(k) + r"""["']""", source)]


def _manifest_data_keys(source: str) -> list[str]:
    """Every string key of a dict literal passed as `manifest_data=`, or built in a `_build_manifest_data` function."""
    keys: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.keyword) and node.arg == "manifest_data" and isinstance(node.value, ast.Dict):
            keys += [k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if isinstance(node, ast.FunctionDef) and node.name == "_build_manifest_data":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Dict):
                    keys += [k.value for k in inner.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    return keys


def _manifest_data_dicts(source: str) -> list[dict[str, object]]:
    """The dict literals passed as `manifest_data=`, each as key -> constant value (None for a non-constant)."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.keyword) and node.arg == "manifest_data" and isinstance(node.value, ast.Dict):
            out.append(
                {
                    k.value: (v.value if isinstance(v, ast.Constant) else None)
                    for k, v in zip(node.value.keys, node.value.values, strict=True)
                    if isinstance(k, ast.Constant)
                }
            )
    return out


def _renderer_sources() -> dict[str, str]:
    return {p.relative_to(REPO).as_posix(): p.read_text(encoding="utf-8") for p in sorted(RENDERERS.rglob("*.py"))}


def test_no_renderer_module_holds_a_unit_token_key_of_the_renderer_data():
    """Property: no renderer module quotes one of the five keys that carried a unit token, and no key of a
    ``manifest_data`` dict matches the unit-token pattern.

    Class: every ``*.py`` under ``src/hhemt/report_renderers`` (recursive); the quoted-literal form of each of the five
    keys, and every string key of a ``manifest_data=`` dict literal or of a ``_build_manifest_data`` function.
    Why required: the plot manifests are published and bundled; a key that names its unit is a name that must change
    when the unit does, and the convention states the unit in a ``{key}_units`` companion.
    Kills it: a renderer that writes a key ending ``_cms`` (or any ``*_m``, ``*_mm_per_hr`` key) into its manifest.
    Passes under: any key without a unit token, whatever its spelling.
    """
    sources = _renderer_sources()
    assert sources, "no renderer module was found, so the scan reads nothing"
    held = {rel: _quoted_old_keys(src) for rel, src in sources.items() if _quoted_old_keys(src)}
    assert held == {}, held
    token_keys = {
        rel: [k for k in _manifest_data_keys(src) if UNIT_TOKEN.search(k)]
        for rel, src in sources.items()
        if any(UNIT_TOKEN.search(k) for k in _manifest_data_keys(src))
    }
    assert token_keys == {}, token_keys


def test_the_scan_fires_on_a_renderer_that_writes_a_unit_token_key():
    """Control for the scan above, varying the PROPERTY (a unit token as a whole token of a key) and not the token.

    Class: planted renderer source holding a key with a unit token at the end, the middle and the start, and keys that
    carry no unit token.
    Why required: a scan that returns an empty list because its pattern or its walk is broken is indistinguishable
    from a clean tree.
    Kills it: a pattern anchored on the end of the key only; a walk that skips ``manifest_data`` dicts.
    """
    end, start, middle = "_".join(("flow", "cms")), "_".join(("cms", "flow")), "_".join(("peak", "cms", "max"))
    planted = (
        "def render(a):\n"
        f"    return emit(manifest_data={{'{end}': 1.0, '{start}': 2.0, '{middle}': 3.0, 'capacity_max': 4.0,\n"
        "                               'hours_flooded': 5.0, 'depth_max': 6.0})\n"
    )
    keys = _manifest_data_keys(planted)
    assert sorted(k for k in keys if UNIT_TOKEN.search(k)) == sorted([end, start, middle])
    assert [k for k in keys if not UNIT_TOKEN.search(k)] == ["capacity_max", "hours_flooded", "depth_max"]
    assert _quoted_old_keys(f"x = {{'{OLD_KEYS[0]}': 1}}") == [OLD_KEYS[0]]
    assert _quoted_old_keys(f"x = cfg.{OLD_KEYS[4]}") == []  # a configuration attribute read is not a manifest key


@pytest.mark.skipif(
    not _git_has(SOURCE_BEFORE_KEYS_CHANGED), reason="the commit before the keys changed is not in this clone"
)
def test_the_scan_finds_the_five_keys_in_the_renderers_before_they_changed():
    """Control: the same scan over the renderers as they were before the change returns all five keys.

    Class: ``git show`` of every module under ``src/hhemt/report_renderers`` at the commit before the change.
    Why required: a zero on the current tree only means something if the same instrument returns non-zero on a tree
    that holds the keys.
    Kills it: a scan that reads the wrong directory or the wrong form of the key.
    """
    names = _git("ls-tree", "-r", "--name-only", SOURCE_BEFORE_KEYS_CHANGED, "--", "src/hhemt/report_renderers").split(
        "\n"
    )
    found: set[str] = set()
    for name in (n for n in names if n.endswith(".py")):
        found |= set(_quoted_old_keys(_git("show", f"{SOURCE_BEFORE_KEYS_CHANGED}:{name}")))
    assert found == set(OLD_KEYS), found ^ set(OLD_KEYS)


def test_each_new_key_has_its_units_companion_with_the_stated_unit():
    """Property: the two renderers that wrote the five keys write each new key beside its ``{key}_units`` companion.

    Class: the ``manifest_data`` dict literals of ``per_sim_conduit_flow`` (one) and ``per_sim_peak_flood_depth``
    (two, one for the static figure and one for the interactive one); the companion's constant value.
    Why required: dropping the unit token from the key moves the unit into the companion, so a key without it loses
    the unit altogether.
    Kills it: a key without its companion; a companion carrying the wrong unit; one of the two peak-flood-depth
    dicts left out.
    Passes under: any value order, any additional key.
    """
    flow = _manifest_data_dicts((RENDERERS / "per_sim_conduit_flow.py").read_text(encoding="utf-8"))
    depth = _manifest_data_dicts((RENDERERS / "per_sim_peak_flood_depth.py").read_text(encoding="utf-8"))
    assert len(flow) == 1 and len(depth) == 2, (len(flow), len(depth))
    for d in flow:
        assert "peak_flow_max" in d and d["peak_flow_max_units"] == "m3 s-1", d
    for d in depth:
        for key in ("wse_range", "rainfall_max", "bc_water_level_range", "depth_boundaries"):
            companion, unit = NEW_KEYS[key]
            assert key in d and d[companion] == unit, (key, d)


def test_the_committed_bundle_fixtures_carry_the_new_keys_and_none_of_the_old():
    """Property: the seven peak-flood-depth manifest fixtures hold the four keys of that renderer under their new
    names, each beside its units companion, and none of the five old keys.

    Class: every ``peak_flood_depth.manifest.json`` under ``tests/fixtures/bundles``.
    Why required: the fixtures are what the bundle tests copy and harvest; a fixture holding a key the renderer no
    longer writes describes an output that cannot occur.
    Kills it: a fixture rewritten in one key only; a fixture with a companion but the old key kept.
    """
    paths = sorted(BUNDLE_FIXTURES.rglob("peak_flood_depth.manifest.json"))
    assert len(paths) == 7, [p.relative_to(REPO).as_posix() for p in paths]
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))["renderer_data"]
        assert not set(OLD_KEYS) & set(data), path
        for key in ("wse_range", "rainfall_max", "bc_water_level_range", "depth_boundaries"):
            companion, unit = NEW_KEYS[key]
            assert key in data and data[companion] == unit, (path, key)


# ---------------------------------------------------------------------------------------------------------------
# (b) the clean slate
# ---------------------------------------------------------------------------------------------------------------

#: Each alternative is assembled from fragments (adjacent string literals), so no line of this file holds a word the
#: matcher reports.
_ALTERNATIVES = (
    r"\b(" "re" "named" "|" "for" "merly" "|" "pre" "viously" "|" "used " "to be" "|" "was " "named" "|" "lega" "cy",
    "|" "re" "tired" "|" "for" "mer" "|" "old " "names?" r")\b",
    "|" "layout[- ]" "2" "6" r"\b",
    "|" r"\b" "1" "a" r"\.[0-9]+" r"\b",
    "|" r"\(" "D" "[0-9]{1,2}" r"\)",
    "|" r"\b" "D" "(1[0-9]|2[0-9]) " "(ruled|option)",
    "|" r"(?-i:\b([0-9][a-z]\.[0-9]+|I[0-9]{1,3}|[EO]-[0-9]{1,3}|D-[A-Z])\b)",
    "|" r"(?-i:(?<![-\w])N[0-9]{1,2}\b|\b" "run " "G" r"-B\b|\b[Bb]inding[ -][0-9]+\b)",
)
MATCHER = re.compile("".join(_ALTERNATIVES), re.IGNORECASE)

#: The roots and the exclusions of the clean-slate census. The versioning system is where history is the subject.
ROOTS = ("src", "docs", "scripts", "tests", "README*", "_layout_relevant_files.yaml", "_caption_pairing_allowlist.yaml")
EXCLUDED = (
    ":!docs/reference/bundles",
    ":!src/hhemt/version_migration",
    ":!scripts/vocabulary_freeze.yaml",
    ":!scripts/vocabulary_residue_exceptions.yaml",
    ":!tests/fixtures",
    ":!tests/goldens/SPEC.md",
    ":!tests/test_version_migration_v0028.py",
    ":!tests/test_review_v0028_migration.py",
)

#: (path, substring) pairs of added lines that the matcher reports and that are NOT history: an identifier or a string
#: that names a name in order to assert it is absent, or a data description a variable's attributes carry. The measured
#: set is the one code value below; every other added line the matcher reports at the working tree is rewritten.
EXCEPTIONS: tuple[tuple[str, str], ...] = (
    ("docs/reference/coupled-inflow-nodes.md", "| `lega" + "cy`, an explicit opt-in |"),
)


def _added_lines(diff_text: str) -> list[tuple[str, int, str]]:
    """The (path, line number, text) of every added line of a ``git diff -U0`` output."""
    path = ""
    lineno = 0
    out: list[tuple[str, int, str]] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else ""
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            lineno = int(m.group(1)) if m else 0
        elif line.startswith("+") and not line.startswith("+++"):
            out.append((path, lineno, line[1:]))
            lineno += 1
    return out


#: A notebook member that carries an embedded image: the key `image/png` or `image/jpeg` and a value that is a
#: single run of at least two hundred base64 characters (with the optional trailing line break the notebook format
#: writes), and nothing else on the line. A person cannot read such a value, and a short run of its characters can
#: match any word pattern by chance, so the line is not text a reader meets. A line that holds anything besides the
#: member, and a member whose value is short or holds a character outside the base64 alphabet, is read as before.
_EMBEDDED_IMAGE = re.compile(r'^\s*"image/(?:png|jpeg)":\s*"[A-Za-z0-9+/=]{200,}(?:\\n)?",?\s*$')


def _hits(diff_text: str) -> list[tuple[str, int, str]]:
    return [h for h in _added_lines(diff_text) if MATCHER.search(h[2]) and not _EMBEDDED_IMAGE.match(h[2])]


def _excepted(hit: tuple[str, int, str]) -> bool:
    return any(hit[0] == p and s in hit[2] for p, s in EXCEPTIONS)


def _diff(base: str, head: str | None, repo: Path = REPO) -> str:
    args = ["diff", "-U0", base] + ([head] if head else []) + ["--", *ROOTS, *EXCLUDED]
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


needs_base = pytest.mark.skipif(
    subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--verify", "-q", "origin/develop"], capture_output=True
    ).returncode
    != 0,
    reason="origin/develop is not a ref of this clone, so the lines this branch adds cannot be computed",
)


@needs_base
def test_the_lines_this_branch_adds_carry_only_the_exceptions_of_the_clean_slate_rule():
    """Property: over the lines the working tree adds relative to ``origin/develop``, the matcher returns only
    exception lines (none are listed).

    Class: the added lines of ``git diff -U0 origin/develop`` under src, docs, scripts, tests and README*, minus the
    versioning system, the vocabulary manifests, the tracked fixtures and goldens.
    Why required: documentation, docstrings, captions, messages and comments describe the toolkit as it is; git is
    the history.
    Kills it: a comment that says a name was changed, calls one old, names a layout number, or cites a chunk or a
    decision.
    Passes under: any rewrite that describes the present state or the thing that is absent.
    """
    hits = [h for h in _hits(_diff("origin/develop", None)) if not _excepted(h)]
    assert hits == [], "\n".join(f"{p}:{n}:{t[:150]}" for p, n, t in hits)


def test_the_matcher_fires_on_each_of_its_alternatives_in_an_included_path_and_on_no_excluded_path():
    """Control for the census above, varying the PROPERTY (a history phrase in an added line of an included path).

    Class: a planted diff with one added line per alternative of the matcher in ``src``, the same line in each
    excluded root, and four present-tense lines.
    Why required: a census that returns nothing because the matcher is broken or the diff walk drops lines is
    indistinguishable from a clean branch.
    Kills it: a matcher missing an alternative; a diff walk that loses the path of the added line; an exclusion list
    that does not exclude.
    """
    phrases = (
        "# " + "re" + "named in chunk " + "1" + "a" + ".4",
        "# the " + "for" + "merly stored name",
        "# " + "pre" + "viously a velocity",
        "# it " + "used " + "to be" + " a depth",
        "# the column " + "was " + "named" + " by the solver",
        "# a " + "lega" + "cy path",
        "# the " + "re" + "tired key",
        "# the " + "for" + "mer key",
        "# the " + "old " + "name",
        "# stored at " + "layout " + "2" + "6",
        "# decided " + "(D" + "17" + ")",
        "# " + "D" + "19 " + "ruled" + " this",
        "# built in chunk " + "6" + "b" + ".1",
        "# the item " + "I" + "62" + " rewrote it",
        "# erratum " + "E" + "-55" + " applies",
        "# ruled at " + "O" + "-28",
        "# decision " + "D" + "-H" + " binds it",
        "# decision row " + "N" + "16 owns it",
        "# the " + "run " + "G" + "-B review found it",
        "# binding " + "13 of the style brief",
    )
    present = (
        "# the depth above the bed",
        "# a name that states no unit",
        "# the layout is 28",
        "# a bold design",
        "# the n16 offset of the grid",
    )
    header = "diff --git a/src/hhemt/x.py b/src/hhemt/x.py\n+++ b/src/hhemt/x.py\n"
    diff = header + f"@@ -0,0 +1,{len(phrases) + len(present)} @@\n"
    diff += "".join(f"+{p}\n" for p in (*phrases, *present))
    hits = _hits(diff)
    assert [t for _, _, t in hits] == list(phrases), [t for _, _, t in hits]
    assert {p for p, _, _ in hits} == {"src/hhemt/x.py"}
    assert [n for _, n, _ in hits] == list(range(1, len(phrases) + 1))


def test_the_exclusions_keep_the_versioning_system_and_the_fixtures_out_of_the_census(tmp_path):
    """Control for the exclusion list: in a scratch repository, the planted phrase is reported in an included path
    and in none of the excluded ones.

    Class: one scratch repository whose second commit adds the same history phrase to ``src/hhemt/x.py`` and to each
    excluded path (the migration tree, the two migration tests, the vocabulary manifests, a fixture, the golden spec,
    the bundle docs), diffed with the census's own pathspecs.
    Why required: the exclusions are git pathspecs, so only a real ``git diff`` shows that they exclude and that they
    exclude nothing else.
    Kills it: a pathspec that is misspelled and excludes nothing; one so wide that it excludes ``src``.
    """
    phrase = "# " + "lega" + "cy path\n"
    included = ["src/hhemt/x.py", "tests/test_x.py"]
    excluded = [
        "src/hhemt/version_migration/versions/V0028__x.py",
        "tests/test_version_migration_v0028.py",
        "tests/test_review_v0028_migration.py",
        "scripts/vocabulary_freeze.yaml",
        "scripts/vocabulary_residue_exceptions.yaml",
        "tests/fixtures/bundles/x.txt",
        "tests/goldens/SPEC.md",
        "docs/reference/bundles/x.md",
    ]

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            capture_output=True,
            text=True,
            check=True,
        )

    git("init", "-q")
    for rel in (*included, *excluded):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x = 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    for rel in (*included, *excluded):
        (tmp_path / rel).write_text("x = 1\n" + phrase, encoding="utf-8")
    reported = sorted({p for p, _, _ in _hits(_diff("HEAD", None, repo=tmp_path))})
    assert reported == sorted(included), reported


def test_the_census_reads_the_two_root_level_allowlists(tmp_path):
    """Control for the roots: in a scratch repository, a history phrase added to each root-level allowlist is reported.

    Class: one scratch repository whose second commit adds the same history phrase to ``_layout_relevant_files.yaml``,
    to ``_caption_pairing_allowlist.yaml`` and to an unlisted root-level file, diffed with the census's own pathspecs.
    Why required: the roots are git pathspecs, so only a real ``git diff`` shows that they take the two files in and
    take nothing else in.
    Kills it: a root that is misspelled; a root so wide that it reads every root-level file.
    """
    phrase = "# " + "lega" + "cy path\n"
    listed = ["_layout_relevant_files.yaml", "_caption_pairing_allowlist.yaml"]

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            capture_output=True,
            text=True,
            check=True,
        )

    git("init", "-q")
    for rel in (*listed, "justfile"):
        (tmp_path / rel).write_text("x: 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    for rel in (*listed, "justfile"):
        (tmp_path / rel).write_text("x: 1\n" + phrase, encoding="utf-8")
    reported = sorted({p for p, _, _ in _hits(_diff("HEAD", None, repo=tmp_path))})
    assert reported == sorted(listed), reported


@pytest.mark.skipif(not _git_has(TIP_BEFORE_SWEEP), reason="the commit before the sweep is not in this clone")
@needs_base
def test_the_matcher_returns_source_lines_over_the_branch_as_it_was_before_the_sweep():
    """Control: the census over the branch tip before the sweep returns the added source lines it was written for.

    Class: ``git diff -U0 origin/develop {tip before the sweep}``, same roots and exclusions.
    Why required: the working-tree census reads empty; the same instrument must read non-empty on a tree that holds
    the history, or an empty result proves nothing.
    Kills it: a diff walk that reads the wrong side; a path filter that drops ``src``.
    """
    hits = _hits(_diff("origin/develop", TIP_BEFORE_SWEEP))
    src = [h for h in hits if h[0].startswith("src/")]
    assert len(src) >= 5, [f"{p}:{n}" for p, n, _ in src]
    assert len(hits) >= 150, len(hits)


def _notebook_diff(*added: str) -> str:
    body = "".join(f"+{line}\n" for line in added)
    return f"+++ b/docs/tutorials/example.ipynb\n@@ -0,0 +1,{len(added)} @@\n{body}"


def test_an_embedded_image_member_is_not_read_and_every_other_notebook_line_still_is():
    """Property: the census skips a diff line that holds only an embedded base64 image member and reads every other
    added line of a notebook, markdown cells and code cells included.

    Class: added lines of a notebook diff. A line is skipped only when it is a member keyed ``image/png`` or
    ``image/jpeg`` whose whole value is a run of at least two hundred base64 characters.
    Why required: a long base64 value holds, by chance, substrings that look like a label; no person reads the value.
    Kills it: a skip keyed on the notebook path (the cells would go unread); a skip keyed on the key alone.
    Passes under: any predicate that skips the same lines and no others.
    Must-fire control: (a) a base64 image member holding a label-shaped run is not reported; (b) the same diff with a
    real label added in a markdown-cell line is reported; (c) a short ``image/png`` value, a value outside the base64
    alphabet and a line that adds text after the member are reported.
    """
    label_shaped = "A" * 120 + "+N" + "12+" + "A" * 120
    member = f'      "image/png": "{label_shaped}",'
    assert MATCHER.search(member), "the planted member must be one the matcher would report without the skip"
    assert _hits(_notebook_diff(member)) == []
    assert _hits(_notebook_diff(member.replace("image/png", "image/jpeg"), member.replace('",', '\\n",'))) == []
    real_label = '    "The choice follows ' + "O" + "-32 and chunk " + "10c" + '.1 of the plan.\\n",'
    reported = _hits(_notebook_diff(member, real_label))
    assert [t for _, _, t in reported] == [real_label]
    short = '      "image/png": "x",'
    non_base64 = f'      "image/png": "{label_shaped[:100]} {label_shaped[100:]}",'
    trailing = member + " N" + "12 trailing text"
    assert MATCHER.search(trailing) and MATCHER.search(non_base64)
    assert [t for _, _, t in _hits(_notebook_diff(short, non_base64, trailing))] == [non_base64, trailing]
