"""Chunk-manifest generation for the hhemt pytest suite.

Partitions a collected node-ID universe into chunks whose boundaries are the
CLOSURES of the suite's expensive session-scoped fixtures, plus duration-agnostic
bins for the remainder. Two files that share a heavy fixture always land in the
same chunk, because a session fixture re-executes once per pytest session and a
split closure pays it twice.

The module raises PartitionDriftError when the assignment does not cover the
collected universe exactly. That check is the point of the module: a collected
test belonging to no chunk would never run while every chunk still reported
success.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

#: Session fixtures whose consumer sets are chunk atoms. `tritonswmm_cpu_compiled`
#: is deliberately absent: it is warmed once before the array and is shared by
#: every chunk, so it constrains no boundary.
HEAVY_FIXTURES: tuple[str, ...] = (
    "synthetic_multisim_completed",
    "synthetic_sensitivity_completed",
    "rendered_synth_multi_sim",
    "rendered_synth_sensitivity",
    "rendered_synth_sensitivity_triton_only",
)

#: Session fixtures a chunk is EXPECTED to set up successfully. Superset of
#: HEAVY_FIXTURES: the compile gate constrains no chunk boundary but a chunk that
#: uses it and does not get it has lost a precondition, not a test.
RECORDED_FIXTURES: tuple[str, ...] = HEAVY_FIXTURES + ("tritonswmm_cpu_compiled",)

_NODE_FILE_RE = re.compile(r"^([^:]+)::")


class PartitionDriftError(RuntimeError):
    """Raised when the chunk assignment does not cover the collected universe."""


def node_file(node_id: str) -> str:
    m = _NODE_FILE_RE.match(node_id)
    if m is None:
        raise PartitionDriftError(f"node id has no file component: {node_id!r}")
    return m.group(1)


def _fixtures_used(path: Path, names: tuple[str, ...] = HEAVY_FIXTURES) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    return {f for f in names if f in text}


#: The marker that CREATES a shared analysis tree. `test_case_builder` computes
#: `system_directory = runs_root / analysis_name`, so a fixture or catalog function that
#: builds a tree CANNOT AVOID naming it -- which is what makes this a structural marker
#: rather than a convention. Measured 2026-08-23: restricting the derivation to the tree
#: names actually present on disk does not change the component shape, so the literals
#: this finds are the real trees and not an over-capture.
_ANALYSIS_NAME_RE = re.compile(r"""analysis_name\s*=\s*['"]([\w.\-]+)['"]""")

#: The SUPPORT layer -- conftest and the fixtures package. The substring scan lives HERE
#: and nowhere else, and that relocation is the whole point of this derivation. It used to
#: run over the 226-file TEST layer, which churns constantly: every new test was a fresh
#: chance to under-connect, and it had already failed on 28 files reaching three trees
#: through a fixture hop no literal could see. The support layer is ~2 files and changes
#: rarely, so the fragile part is now small, stable, and forced -- a tree-building fixture
#: must name its tree to build it.
_SUPPORT_NAMES = ("conftest.py",)
_SUPPORT_DIR = "fixtures"

#: How many hops to follow inside the support layer (fixture -> catalog function -> tree).
#: Bounded rather than a fixpoint: an unbounded closure over every `def` collides on
#: generic names (`case`, `analysis`, `_builder`) and reported half the suite as reaching
#: every tree when this was first measured.
_SUPPORT_HOPS = 2


def _iter_support_files(repo_root: Path):
    for p in (repo_root / "tests").rglob("*.py"):
        if p.name in _SUPPORT_NAMES or _SUPPORT_DIR in p.parts:
            yield p


def _defs_in(src: str) -> dict[str, str]:
    """name -> exact source of each def, via AST.

    AST rather than a regex over `def` lines, and the difference is not cosmetic. The
    regex version bounded a def's body by the next `def` at the SAME OR LESSER indent --
    which never matches when the following siblings are methods at a GREATER indent, so a
    module-level helper sitting above a class swallowed every def below it. Measured
    2026-08-23: `_load_norfolk_example_or_skip` extracted 36,520 chars containing 34 other
    defs (the catalog file is 37,984 chars total), inheriting all 28 `analysis_name`
    literals in the file and becoming a hub that propagated every tree to every symbol
    referencing it. AST gives 1,223 chars and drops max-trees-per-symbol from 28 to 6.
    """
    out: dict[str, str] = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(src, node) or ""
            out[node.name] = out.get(node.name, "") + seg
    return out


def _support_bodies(repo_root: Path) -> dict[str, str]:
    """symbol -> concatenated source of every same-named def in the support layer."""
    bodies: dict[str, str] = {}
    for p in _iter_support_files(repo_root):
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name, seg in _defs_in(src).items():
            bodies[name] = bodies.get(name, "") + seg
    return bodies


def support_symbol_trees(repo_root: Path) -> dict[str, set[str]]:
    """Map each support-layer symbol to the analysis trees it leads to."""
    bodies = _support_bodies(repo_root)
    sym: dict[str, set[str]] = {}
    for name, body in bodies.items():
        found = set(_ANALYSIS_NAME_RE.findall(body))
        if found:
            sym[name] = found
    for _ in range(_SUPPORT_HOPS):
        for name, body in bodies.items():
            acc: set[str] = set()
            for other, trees in sym.items():
                if other != name and re.search(rf"\b{re.escape(other)}\b", body):
                    acc |= trees
            if acc:
                sym[name] = sym.get(name, set()) | acc
    return sym


#: pytest's runtime fixture-request API. A file containing this token is ANNOUNCING that
#: its resolved closure is incomplete, which is what makes the residual detectable rather
#: than silent.
_DYNAMIC_LOOKUP_MARKER = "getfixturevalue"


def _string_literals(src: str) -> set[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def file_trees(repo_root: Path, closures: dict[str, list[str]]) -> dict[str, set[str]]:
    """file -> trees, from pytest's REAL resolved fixture closure plus direct catalog calls.

    The closure half is structural: pytest resolves the full transitive fixture graph, so a
    tree reached through fixture -> fixture -> helper is named without this module guessing
    at the hops. The direct-call half covers the one shape a fixture closure cannot see --
    a test body calling a catalog function itself, measured at exactly ONE file in the
    corpus (`test_from_scratch_honesty.py`) -- and it scans a closed `retrieve_*` set rather
    than an open list of fixture names.
    """
    sym = support_symbol_trees(repo_root)
    catalog = {n for n in sym if n.startswith("retrieve_")}
    out: dict[str, set[str]] = {}
    for node_id, fixtures in closures.items():
        f = node_file(node_id)
        acc = out.setdefault(f, set())
        for name in fixtures:
            acc |= sym.get(name, set())
    for f in list(out):
        try:
            txt = (repo_root / f).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for c in catalog:
            if re.search(rf"\b{re.escape(c)}\b", txt):
                out[f] |= sym[c]
        # DYNAMIC FIXTURE LOOKUP. `request.getfixturevalue("name")` is resolved at CALL
        # time, so it appears in no fixture closure -- measured live:
        # test_experiments_from_doi.py:47 requests `rendered_synth_multi_sim` this way and
        # the name occurs nowhere else in the file, so both pytest's closure and the
        # support-layer scan miss it and the file lands in the cheap bucket while really
        # reaching `synth_multi_sim`.
        #
        # Scoped to files that SELF-IDENTIFY: `getfixturevalue` is a closed pytest API
        # token, not an open project vocabulary, and a runtime fixture request cannot be
        # made without emitting it. That is what makes this different from the name list
        # this derivation replaced.
        if _DYNAMIC_LOOKUP_MARKER not in txt:
            continue
        universe = dict(sym)
        local = _defs_in(txt)
        for name, seg in local.items():
            if name not in universe:
                universe[name] = set(_ANALYSIS_NAME_RE.findall(seg))
        resolved = [lit for lit in _string_literals(txt) if lit in universe]
        # FAIL LOUD when a dynamic lookup resolves to nothing at all. An unresolvable
        # request is a hole in the derivation, and a hole that returns an empty tree set is
        # indistinguishable from a file that genuinely touches no tree -- which is exactly
        # how the previous derivation under-connected 28 files while looking correct.
        if not resolved:
            raise PartitionDriftError(
                f"{f} calls {_DYNAMIC_LOOKUP_MARKER} but no string literal in it names a "
                "known fixture or helper, so the request cannot be resolved and the file's "
                "tree reach is unknown. Refusing to partition on an unresolvable lookup."
            )
        for lit in resolved:
            out[f] |= universe[lit]
    return out


def heavy_components(
    repo_root: Path, files: list[str], trees_by_file: dict[str, set[str]] | None = None
) -> list[list[str]]:
    """Group `files` into connected components joined by a shared ANALYSIS TREE.

    The join used to be a shared HEAVY FIXTURE LITERAL found by scanning each test file's
    text. That under-connected badly: measured 2026-08-23, 28 files reached three trees
    through a fixture hop no literal could see, against 22 the scan resolved -- the
    derivation was blind to more of the reach than it caught, and the residual raced.

    `trees_by_file` is derived from pytest's OWN resolved fixture closure (see
    `file_trees`), so the hops are resolved by the resolver rather than guessed at. It is
    REQUIRED; the legacy literal scan is not a fallback, because an empty closure and a
    missing one are the same bytes downstream and a silent fallback would under-connect
    exactly as before while looking like it worked.
    """
    if trees_by_file is None:
        raise PartitionDriftError(
            "heavy_components requires trees_by_file derived from the pytest fixture "
            "closure. Refusing to fall back to the literal scan: it under-connects on 28 "
            "known files and the failure surfaces as a flaky suite, not as an error."
        )
    parent: dict[str, str] = {f: f for f in files}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_tree: dict[str, list[str]] = {}
    for f in files:
        for tree in sorted(trees_by_file.get(f) or ()):
            by_tree.setdefault(tree, []).append(f)
    for members in by_tree.values():
        for other in members[1:]:
            union(members[0], other)

    heavy = {f for members in by_tree.values() for f in members}
    groups: dict[str, list[str]] = {}
    for f in sorted(heavy):
        groups.setdefault(find(f), []).append(f)
    return [sorted(v) for _, v in sorted(groups.items())]


#: Symbols whose presence in a test file mean the file touches a SHARED on-disk
#: analysis tree under the slug cache. Two chunks that both reach the same tree run
#: CONCURRENTLY and can mutate it under each other; within one chunk they are serial.
#: This is DETECTION, not scheduling: the partitioner reports the exposure and does
#: not silently repair it by merging, because merging hides a shared-mutable-state
#: defect behind a scheduling accident that the next partition change re-opens.
#:
#: ARM A of a two-arm check, and it has a HORIZON: this scan sees a literal symbol in
#: a test FILE. Measured 2026-08-23, a test reached `synth_multi_sim` through
#: fixture -> fixture -> helper-module, three hops past anything a file-level scan can
#: see, and the file carried zero matching symbols. Deepening the parse only moves the
#: horizon, so arm B (runtime observation, in run_suite.py) exists to be wrong in a
#: DIFFERENT way, and aggregate.py reports their disagreement.
SHARED_TREE_SYMBOL_RE = r"retrieve_(\w+?)_test_case|retrieve_(\w+?)_case"


def shared_tree_exposure(repo_root: Path, chunks: list[dict]) -> list[dict]:
    """Report trees reachable from MORE THAN ONE chunk.

    Returns one record per cross-chunk tree, naming the tree and the chunk ids that
    reach it. An empty list is the invariant holding -- as far as THIS arm can see.
    """
    by_tree: dict[str, set[int]] = {}
    for c in chunks:
        for f in c["files"]:
            try:
                text = (repo_root / f).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in re.finditer(SHARED_TREE_SYMBOL_RE, text):
                tree = m.group(1) or m.group(2)
                by_tree.setdefault(tree, set()).add(c["chunk_id"])
    return [{"tree": t, "chunks": sorted(ids)} for t, ids in sorted(by_tree.items()) if len(ids) > 1]


#: Imputed per-node cost for a heavy-component file the duration table does not cover, in
#: seconds. Measured from the cancelled run's two CHEAP chunks: 916 nodes in 139.0 s
#: (0.152 s/node) and 916 in 162.5 s (0.177 s/node). RE-MEASURED on the COMPLETED run
#: 20260904T145255Z_2853a077a3ee: 0.127 and 0.101 s/node. The constant is deliberately NOT
#: lowered to match -- 0.17 now OVER-estimates an unmeasured unit, which is the safe
#: direction (a unit costed too high is packed conservatively, never under-filled), and
#: re-fitting a behaviour-bearing constant to the newest run is how a cost model acquires
#: an unexamined dependency on one measurement. It is an ASSUMPTION and the manifest
#: records it as one. KNOWN WEAKNESS, stated because a silent one is worse: the imputed
#: population is the HEAVY component, so a cheap-tier rate is the optimistic prior. On the
#: 20260904T074157Z_cd4d8f6d9a91 measurement 584 of the 586 uncovered nodes are the
#: test_workflow_* unit tier and test_version_migration_golden.py, for which the rate is
#: right; the exception is test_synth_workflow_performance_generator_parity.py (6 nodes),
#: whose real cost is unmeasured and is under-counted here.
_IMPUTED_SECONDS_PER_NODE = 0.17


def node_split_candidates(
    files: list[str],
    by_file: dict[str, list[str]],
    closures: dict[str, list[str]],
    repo_root: Path,
) -> set[str]:
    """Files whose per-node parts would inherit a TRUTHFUL whole-file fixture set.

    The unit of partition is the FILE by default. A file joins this set only when both
    hold, and the second is the load-bearing one:

    1. Its text does not contain `getfixturevalue`. A dynamic fixture request is resolved
       at call time and cannot be attributed to a node, so a file that makes one cannot be
       split without guessing. `file_trees` already treats this token as a self-identifying
       incompleteness marker; this is the same token used for the same reason.
    2. For every node in the file, `_fixtures_used(file) <= set(closures[node])`.

    Why (2) is not optional. `_fixtures_used` is a WHOLE-FILE substring scan, so both parts
    of a split file inherit the union of every RECORDED_FIXTURES name the file mentions. If
    some node does not request one of them, that part declares an `expected_fixtures` entry
    it never sets up, `aggregate.py::classify_chunk` computes `dead = expected - ok`, and
    the chunk is VOID. That is a partition change making a green harder to earn for a reason
    that has nothing to do with the code under test, which is the exact outcome a faster
    suite must not buy. Under (2) no part can declare a fixture it does not request.

    Measured on run 20260904T074157Z_cd4d8f6d9a91: `tests/test_metadata_consolidation.py`
    satisfies both (84 lines, no dynamic lookup, names only `tritonswmm_cpu_compiled` of
    RECORDED_FIXTURES, and both its tests carry it via `usefixtures`).

    FIGURES CORRECTED against the first COMPLETED run, 20260904T145255Z_2853a077a3ee. The
    earlier ones came from a run CANCELLED at 48%, before the suite's most expensive file
    had run at all. Total call-time is 28,292.3 s over 3,021 node ids, of which chunk-00
    carried 28,046 s (99.1%) while the other six carried 246 s between them -- a seven-way
    array that delivered essentially no parallelism. The FILE-granularity floor is
    10,343.5 s (`tests/test_synth_timeseries_consolidation.py`, 7 nodes), not the 7,751 s
    the cancelled run suggested; the NODE-granularity floor is 5,178.3 s (that file's
    `test_toggle_on_consolidates_node_and_link_timeseries`). So the saving this gate buys
    is roughly 2.87 h at file granularity against 1.44 h at node granularity -- an order of
    magnitude larger than the ~22% the cancelled run implied.
    """
    out: set[str] = set()
    for f in files:
        try:
            txt = (repo_root / f).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _DYNAMIC_LOOKUP_MARKER in txt:
            continue
        declared = _fixtures_used(repo_root / f, RECORDED_FIXTURES)
        nodes = by_file.get(f) or []
        if len(nodes) < 2:
            continue
        if all(declared <= set(closures.get(n) or ()) for n in nodes):
            out.add(f)
    return out


def split_heavy_component(
    files: list[str],
    by_file: dict[str, list[str]],
    durations: dict[str, float],
    budget_s: float,
    node_split: set[str] | None = None,
) -> list[list[str]]:
    """Split ONE connected component into time-balanced parts, returning NODE ID lists.

    Returns node ids, not files, because `run_chunk` is ALREADY node-granular: it reads
    `chunks/NN.txt` line by line (run_suite.py:984) and passes them to pytest verbatim
    (:1036). Nothing downstream of the manifest has to change for the unit to be a node.

    The default unit is still the FILE -- a node-count packer is wrong by orders of
    magnitude here (7 nodes / 10,343.5 s vs 477 nodes / negligible), and a per-file cost is
    what the duration table measures. Files in `node_split` are the audited exception, and
    membership is decided by `node_split_candidates`, never by the caller.

    The number of parts is DERIVED from `budget_s`, then clamped so a part is never emptier
    than one unit. Deriving it is what stops a caller reaching for a K the measurements do
    not support: makespan saturates at the heaviest indivisible unit, so a K above the
    saturation point is chunk_count inflation with no wall-clock effect and one more
    isolated fixture build each. Measured on the completed run 20260904T145255Z_2853a077a3ee,
    that floor is 10,343.5 s at FILE granularity and 5,178.3 s at NODE granularity.
    """
    if budget_s <= 0:
        raise PartitionDriftError(f"heavy split budget must be > 0 s, got {budget_s}")
    node_split = node_split or set()
    units: dict[str, list[str]] = {}
    cost: dict[str, float] = {}
    for f in files:
        nodes = by_file[f]
        per_file = durations.get(f)
        if f in node_split:
            # Distribute the file's measured cost across its nodes only when the table
            # covers the node itself; otherwise fall back to the file's share. A node the
            # table does not name is imputed exactly as a file would be.
            for n in nodes:
                units[n] = [n]
                cost[n] = durations.get(
                    n,
                    (per_file / len(nodes)) if per_file is not None else _IMPUTED_SECONDS_PER_NODE,
                )
        else:
            units[f] = list(nodes)
            cost[f] = per_file if per_file is not None else len(nodes) * _IMPUTED_SECONDS_PER_NODE
    total = sum(cost.values())
    k = max(1, min(len(units), int(-(-total // budget_s))))
    if k == 1:
        return [sorted(n for u in units.values() for n in u)]
    bins: list[list[str]] = [[] for _ in range(k)]
    load = [0.0] * k
    for key in sorted(units, key=lambda p: (-cost[p], p)):
        i = load.index(min(load))
        bins[i].extend(units[key])
        load[i] += cost[key]
    return [sorted(b) for b in bins if b]


def build_manifest(
    *,
    repo_root: Path,
    node_ids: list[str],
    source_sha: str,
    run_id: str,
    closures: dict[str, list[str]],
    cheap_bins: int = 1,
    heavy_split_budget_s: float | None = None,
    durations: dict[str, float] | None = None,
    tree_isolation_per_chunk: bool = False,
) -> dict:
    if not node_ids:
        raise PartitionDriftError("collected universe is empty; refusing to partition")
    # FAIL LOUD ON A MISSING CLOSURE. An empty closure and an absent one are the same bytes
    # downstream, and this harness has now found that shape five times. A pytest upgrade
    # that renames `_fixtureinfo` must stop the run at plan time, not silently produce a
    # one-chunk partition that looks like a scheduling choice.
    if not closures:
        raise PartitionDriftError(
            "fixture closures are empty; the collect-time plugin did not run or "
            "`_fixtureinfo` is unavailable. Refusing to partition on no evidence."
        )
    missing = [n for n in node_ids if n not in closures]
    if missing:
        raise PartitionDriftError(
            f"{len(missing)} collected node id(s) have no fixture closure, e.g. "
            f"{missing[:3]}. A partial closure under-connects silently."
        )
    if cheap_bins < 1:
        raise PartitionDriftError(f"cheap_bins must be >= 1, got {cheap_bins}")
    # REFUSE, rather than split unsafely. A component is connected because its files reach
    # the SAME on-disk analysis tree, and rerun.sh:242 UNSETS HHEMT_TEST_RUNS_ROOT_OVERRIDE,
    # so two chunks of one component would write one tree concurrently. That failure does not
    # present as a partition bug; it presents as an intermittent red in an unrelated test.
    # The precondition is therefore encoded here rather than remembered.
    if heavy_split_budget_s is not None and not tree_isolation_per_chunk:
        raise PartitionDriftError(
            "heavy_split_budget_s was given without tree_isolation_per_chunk=True. Splitting "
            "a connected component while every chunk shares one analysis-tree root races that "
            "tree. Set the isolation flag (which makes run_chunk export a per-chunk "
            "HHEMT_TEST_RUNS_ROOT_OVERRIDE) or do not split."
        )
    # REFUSE on an empty duration table. Balancing this component on node count is wrong by
    # orders of magnitude (7 nodes / 10,343.5 s vs 477 nodes / negligible on the completed
    # run 20260904T145255Z_2853a077a3ee), and an absent table and an all-imputed one are the
    # same bytes downstream.
    if heavy_split_budget_s is not None and not durations:
        raise PartitionDriftError(
            "heavy_split_budget_s was given with no durations table. Node count is not a cost "
            "proxy for this corpus. Pass --durations-from {run_dir} naming a COMPLETED run."
        )

    files = sorted({node_file(n) for n in node_ids})
    components = heavy_components(repo_root, files, file_trees(repo_root, closures))
    heavy_files = {f for comp in components for f in comp}
    cheap_files = [f for f in files if f not in heavy_files]

    by_file: dict[str, list[str]] = {f: [] for f in files}
    for n in node_ids:
        by_file[node_file(n)].append(n)

    chunks: list[dict] = []
    for comp in components:
        if heavy_split_budget_s is None:
            chunks.append({"kind": "heavy", "files": comp})
            continue
        # Split BEFORE the per-chunk enrichment loop below, so each part derives its OWN
        # expected_fixtures from its own files rather than inheriting the parent's. A part
        # that inherited a fixture it does not use would report a lost precondition and
        # aggregate.py would VOID it -- which is why node_split_candidates gates on exactly
        # that condition rather than on the file being "small enough to split".
        splittable = node_split_candidates(comp, by_file, closures, repo_root)
        for part in split_heavy_component(comp, by_file, durations or {}, heavy_split_budget_s, node_split=splittable):
            # node_ids is set HERE because a part may be a strict subset of a file's nodes;
            # the enrichment loop below must not re-derive it from `files`.
            chunks.append(
                {
                    "kind": "heavy",
                    "files": sorted({node_file(n) for n in part}),
                    "node_ids": part,
                }
            )

    bins: list[list[str]] = [[] for _ in range(cheap_bins)]
    load = [0] * cheap_bins
    for f in sorted(cheap_files, key=lambda p: (-len(by_file[p]), p)):
        i = load.index(min(load))
        bins[i].append(f)
        load[i] += len(by_file[f])
    for b in bins:
        if b:
            chunks.append({"kind": "cheap", "files": sorted(b)})

    for i, c in enumerate(chunks):
        c["chunk_id"] = i
        # DO NOT re-derive node_ids when the splitter already set them. A node-split part is
        # a strict SUBSET of its file's nodes, so re-deriving from `files` would silently
        # re-merge the split -- every part would reclaim the whole file, _assert_covers would
        # then see each of those nodes in more than one chunk, and PartitionDriftError would
        # fire naming duplicates rather than the re-derivation that caused them.
        c.setdefault("node_ids", [n for f in c["files"] for n in by_file[f]])
        expected: set[str] = set()
        for f in c["files"]:
            expected |= _fixtures_used(repo_root / f, RECORDED_FIXTURES)
        c["expected_fixtures"] = sorted(expected)

    assigned = [n for c in chunks for n in c["node_ids"]]
    _assert_covers(collected=node_ids, assigned=assigned)

    exposure = shared_tree_exposure(repo_root, chunks)
    return {
        "shared_tree_exposure": exposure,
        "run_id": run_id,
        "source_sha": source_sha,
        "collected": sorted(node_ids),
        "chunk_count": len(chunks),
        # The INPUT, beside the consequence. `chunk_count` alone forced a reader to
        # reverse-engineer which `--cheap-bins` produced a run -- which is how a run at the
        # binder-suppressing default went unnoticed until the chunk count was compared
        # against a measured table. The three heavy-split inputs are recorded for the same
        # reason, and `tree_isolation_per_chunk` additionally CROSSES THE SEAM: run_chunk
        # reads it from here to decide whether to export a per-chunk runs-root override, so
        # the isolation a manifest was planned under is the isolation its chunks run under.
        #
        # DISCLOSURE FOR A LATER READER DIFFING TWO RUNS: the FIRST run with
        # tree_isolation_per_chunk=true is NOT red-set-comparable to any baseline before it.
        # Isolation gives each chunk a fresh analysis-tree root, so synthetic_multisim_completed
        # and synthetic_sensitivity_completed (tests/conftest.py:533, :596) gate on a
        # _marker_state that is never "complete" and REBUILD rather than adopt. Every test
        # downstream of them then runs against a tree built by that chunk instead of one
        # persisted in the slug cache. A test that passed partly because of residue another
        # test left behind changes outcome -- in either direction. A red appearing on that
        # first isolated run is therefore not evidence of a code regression until it has been
        # reproduced on a second isolated run.
        "cheap_bins": cheap_bins,
        "heavy_split_budget_s": heavy_split_budget_s,
        "heavy_split_imputed_seconds_per_node": (None if heavy_split_budget_s is None else _IMPUTED_SECONDS_PER_NODE),
        "heavy_split_durations_covered": (None if durations is None else sorted(durations)),
        "tree_isolation_per_chunk": tree_isolation_per_chunk,
        "chunks": chunks,
    }


def _assert_covers(*, collected: list[str], assigned: list[str]) -> None:
    cset, aset = set(collected), set(assigned)
    unassigned = sorted(cset - aset)
    extra = sorted(aset - cset)
    dupes = sorted({n for n in assigned if assigned.count(n) > 1}) if len(assigned) != len(aset) else []
    if unassigned or extra or dupes:
        raise PartitionDriftError(
            "chunk assignment does not cover the collected universe exactly.\n"
            f"  collected={len(cset)} assigned={len(aset)}\n"
            f"  in no chunk ({len(unassigned)}): {unassigned[:20]}\n"
            f"  not collected ({len(extra)}): {extra[:20]}\n"
            f"  in more than one chunk ({len(dupes)}): {dupes[:20]}"
        )


def write_manifest(manifest: dict, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "chunks").mkdir(exist_ok=True)
    for c in manifest["chunks"]:
        (run_dir / "chunks" / f"{c['chunk_id']:02d}.txt").write_text("\n".join(c["node_ids"]) + "\n", encoding="utf-8")
    p = run_dir / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p
