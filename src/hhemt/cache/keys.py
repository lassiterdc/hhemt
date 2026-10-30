"""WHAT names a cached tree: the identity of everything that can change its bytes.

A completion marker is only worth writing if it can be REFUSED, and what refuses it is a
disagreement about identity. NINE terms go into that identity, and they are the nine a
cached tree's bytes actually depend on:

  * ``builder_digest`` -- a SHA-256 over the bytes of the files that decide what the cached
    tree must contain. The list is supplied by the caller because it names files in the
    test tree, and this package may not import ``tests/``.
  * ``toolkit_src_digest`` -- a SHA-256 over every ``*.py`` file under the toolkit's own
    ``src/hhemt/``, read from the WORKING TREE, so a tree built by one state of the code and
    read from another is refused rather than silently reinterpreted.
  * ``tier0_key`` -- the key of the tier-0 artifact directory the tree was built from.
    COMPOSED rather than re-derived: a tier-0 change re-mints that directory, so carrying
    its key here invalidates every analysis tree built from it through one term.
  * ``solver_identity`` -- a SHA-256 over the module that declares the TRITON pin and
    remote. It moves on an edit to that module even when no caller overrides anything.
  * ``effective_pin`` -- a SHA-256 over the pin and software directory a CALLER actually
    provisioned, which the module default above cannot see.
  * ``configs_digest`` -- a SHA-256 over every report config the completed-tree fixtures
    hand to ``run()``.
  * ``environment_digest`` -- a SHA-256 over the SOLVER environment components, at the
    granularities :data:`ENVIRONMENT_COMPONENTS` states.
  * ``case_params_digest`` -- a SHA-256 over the case parameters a tree is built for
    (event counts, reporting timestep, generator params, system overlay).
  * ``payload_digest`` -- ``{rel_path: (size, sha256)}`` per named payload entry, so a
    payload that is PRESENT but no longer matches what the marker recorded is detectable
    rather than read as intact.

The two BYTE-level terms are deliberately over-inclusive, comment-only edits included. A
comment edit changes the digest and reads every cached marker as stale, and that
over-invalidation is accepted rather than narrowed: distinguishing a body from a comment
needs an AST normalization that is itself a drift surface, and the failure it would prevent
(a stale rebuild) costs one workflow run, while the failure it would introduce (a body edit
the key cannot see) costs a wrong result that reads as a real one.

``toolkit_src_digest`` used to be the repository's ``HEAD`` sha. ``HEAD`` cannot see an
uncommitted edit, so a builder editing ``src/`` mid-run left every marker reading CURRENT
while the code that built the tree had changed. Reading the working tree is over-inclusive
in the opposite direction on purpose -- during a build wave one builder's ``src/`` edit
invalidates every marker -- and that rebuild is the trade this term makes.

THE ENVIRONMENT TERM IS FILTERED, and the filter is a decision rather than an oversight.
:data:`ENVIRONMENT_COMPONENTS` carries one row per component, each with a granularity and a
ground, and rows whose ``tier`` is ``excluded`` are hashed by nothing. An unfiltered
environment digest is the over-sensitivity this key exists to avoid: a ``pyswmm`` patch bump
would invalidate every analysis tree, which is the same defect as a docs commit doing it.
The named gap that survives is stated where it binds -- patch-level changes to a real-input
component do not invalidate -- so a surprising one is a NAMED gap rather than a silent one.

NOT ESTABLISHED, two residuals, named rather than papered over:

  * ``toolkit_src_digest`` ranges over ``*.py`` only. ``src/hhemt/`` also ships non-Python
    files -- the report ``.j2`` templates and their captions, ``invalidating_fixes.yaml``,
    the SIF recipe defs and the vendored Tabulator assets -- and an edit to one of those
    does not move the digest. Widening the glob is a change to the key's scope rather than
    to its implementation, so it is recorded here instead of taken unilaterally.
  * ``toolkit_sha`` is still defined below and is no longer part of the composition. It
    stays because ``hhemt/cache/__init__.py`` re-exports it, and deleting a name that module
    imports would break ``import hhemt.cache`` for every consumer.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The tier-1 composition's key set, named once. A SET for assertion purposes: the shape
#: fixes WHICH terms compose the identity, never the order a dict literal is built in.
#: A term that is not supplied by its caller composes as ``""`` or an empty mapping, which
#: is a TRUTHFUL value -- absence is reported as absence so two callers that both lack a
#: term compare equal to each other rather than to a plausible-looking placeholder.
TIER1_TERMS = (
    "builder_digest",
    "toolkit_src_digest",
    "tier0_key",
    "solver_identity",
    "effective_pin",
    "configs_digest",
    "environment_digest",
    "case_params_digest",
    "payload_digest",
)


def builder_digest(sources: Iterable[Path]) -> str:
    """SHA-256 over the bytes of ``sources``, in the order supplied.

    Order matters and is not sorted here: the caller's sequence is the identity, so
    reordering it changes the digest. An unreadable source contributes the fixed byte string
    ``b"<unreadable>"`` instead of raising, because a marker written when a source cannot be
    read is still refusable later, whereas an exception here would take down fixture setup
    for a reason unrelated to the tree under test.
    """
    digest = hashlib.sha256()
    for src in sources:
        try:
            digest.update(src.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()


def file_identity(path: Path) -> str:
    """SHA-256 over ONE file's bytes, or ``""`` when it cannot be read.

    The ``solver_identity`` and ``configs_digest`` terms are both this measurement over a
    fixed file list, so they share one function rather than two spellings of the same read.

    Deliberately a SOURCE-DECLARATION digest and not the provisioned checkout's state: what
    moves it is an edit to the module that offers the TRITON pin and remote, which is the
    half a caller cannot override away. The pin a caller actually provisions is
    :func:`effective_pin_digest`'s subject, and keeping the two separate is what lets a
    reader tell "the default moved" from "this caller provisioned something else".
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def effective_pin_digest(branch_key: Any, software_directory: Any) -> str:
    """SHA-256 over the pin and software directory a CALLER provisioned.

    Two callers at one ``HEAD`` could otherwise provision different solvers under one key:
    the branch key is read from the caller's ``additional_system_configs`` and the software
    directory is overridable independently, while :func:`file_identity` over the declaration
    module moves only when that module is edited. ``None`` is hashed as ``None`` rather than
    as ``""`` so "no override declared" and "override declared as the empty string" are
    different identities -- the second is a caller bug, and collapsing it into the first
    would let the bug pass unrecorded.
    """
    return hashlib.sha256(
        json.dumps([branch_key, software_directory], sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def case_params_digest(
    *,
    n_events: Any,
    n_reporting_tsteps_per_sim: Any,
    triton_reporting_timestep_s: Any,
    params: Any,
    additional_system_configs: Mapping[str, Any] | None = None,
) -> str:
    """SHA-256 over the case parameters a tree is built for.

    ``additional_analysis_configs`` is deliberately ABSENT. A per-call analysis overlay is an
    ADDRESSING concern rather than a key term: the cache address must include its digest, or a
    tree cached under one overlay is reused under another. The system overlay stays a term for
    a narrower reason -- :func:`effective_pin_digest` extracts two of its members and the
    remainder reaches the written configs -- so the two overlays are treated differently on
    purpose and the difference is checkable by the arm that varies the analysis overlay and
    requires the key NOT to move.
    """
    return hashlib.sha256(
        json.dumps(
            {
                "n_events": n_events,
                "n_reporting_tsteps_per_sim": n_reporting_tsteps_per_sim,
                "TRITON_reporting_timestep_s": triton_reporting_timestep_s,
                "params": params,
                "additional_system_configs": dict(additional_system_configs or {}),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _directory_shape_digest(path: Path) -> tuple[int, str]:
    """``(total size, digest over sorted (rel_path, size) pairs)`` for a payload DIRECTORY.

    Deliberately a SHAPE digest rather than a content digest, and the reason is cost against
    the mechanism's own purpose: the real payloads are Zarr stores of many megabytes, and
    content-hashing one on every fixture construction would add, inside the package that
    exists to remove a cost, a cost larger than the rebuild it might prevent. Sizes catch
    truncation and deletion -- the damage a killed writer leaves -- and the residual is
    stated rather than hidden: a same-size content change is invisible to this measurement.
    """
    total = 0
    digest = hashlib.sha256()
    try:
        entries = sorted((child.relative_to(path).as_posix(), child) for child in path.rglob("*") if child.is_file())
    except OSError:
        return (0, "")
    for rel, child in entries:
        try:
            size = child.stat().st_size
        except OSError:
            continue
        total += size
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("utf-8"))
        digest.update(b"\0")
    return (total, digest.hexdigest())


def payload_digest(analysis_dir: Path, payload: Iterable[str]) -> dict[str, list[Any]]:
    """``{rel_path: [size, sha256]}`` for every payload entry that EXISTS.

    Absent entries are omitted rather than recorded as a placeholder, because their absence
    is already the payload check's own arm (:func:`hhemt.cache.markers.unmet_payload`); a
    placeholder here would make a missing file and an empty file share one digest, and the
    second is a real artifact that was written.

    A FILE contributes its byte count and the digest of its bytes. A DIRECTORY contributes
    its total file size and :func:`_directory_shape_digest`'s metadata digest, which is the
    inexpensive measurement for the Zarr stores that are this mechanism's real payloads.

    This is what separates "the tree was damaged after the marker was written" from "the
    marker is out of date". A truncated file still EXISTS, so an existence-only check reads
    it as intact; its digest does not match, and the marker state machine reports the tree as
    ``unsatisfied`` -- the arm whose warning text already says the tree may have been damaged
    after the marker was written.
    """
    out: dict[str, list[Any]] = {}
    for rel in payload:
        path = Path(analysis_dir) / rel
        if path.is_dir():
            size, digest = _directory_shape_digest(path)
            out[rel] = [size, digest]
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file():
            continue
        out[rel] = [stat.st_size, hashlib.sha256(path.read_bytes()).hexdigest()]
    return out


# --------------------------------------------------------------------------------------
# The environment digest, and the policy that filters it
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvironmentComponent:
    """One row of the environment policy: what it is, how coarse, and why it is admitted.

    ``tier`` names WHERE the component's effect lands rather than what kind of software it
    is. The rule is one measurement applied in two directions: a component whose change
    alters a SIMULATION OUTPUT is a solver component and attaches to tier 1; a component
    whose change alters a TIER-0 ARTIFACT is an artifact component and attaches to tier 0,
    reaching tier 1 only through the ``tier0_key`` composition. ``excluded`` rows are
    declared, not omitted -- a reader applying the admission rule to one of them finds a
    row stating why it is out, instead of an absence they must interpret.
    """

    name: str
    tier: str
    granularity: str
    ground: str


#: The policy. A component is a real input IFF changing it can change the bytes of a
#: produced artifact; the granularity is the COARSEST at which that stays true.
ENVIRONMENT_COMPONENTS: tuple[EnvironmentComponent, ...] = (
    EnvironmentComponent(
        "swmm-toolkit",
        "solver",
        "major.minor",
        "It IS the SWMM engine, so a version change alters simulation output.",
    ),
    EnvironmentComponent(
        "pyswmm",
        "solver",
        "major",
        "A binding layer does not change numerics; keying its patch is the over-sensitivity "
        "this table exists to exclude, and its major can change the call semantics.",
    ),
    EnvironmentComponent(
        "cxx-compiler",
        "solver",
        "compiler",
        "It produces the binary. Major.minor of the compiler's OWN version, taken as the "
        "LAST dotted-numeric token on its --version first line, because the first dotted run "
        "on that line is usually the distribution tag -- Ubuntu's '9.3.0-17ubuntu1~20.04' on a "
        "9.4.0 binary keys 9.3 under a first-run reading, which is the wrong toolchain.",
    ),
    EnvironmentComponent(
        "cmake",
        "solver",
        "major.minor",
        "It produces the build files. The one row where the rule is applied at 'can' rather "
        "than 'does': a minor often changes nothing for a given project, and admitting it "
        "beats leaving a generator-induced byte change silently unkeyed.",
    ),
    EnvironmentComponent(
        "kokkos-backend",
        "solver",
        "enum",
        "Host and device solvers are different code paths, so the enum value is the coarsest "
        "granularity at which the rule holds.",
    ),
    EnvironmentComponent(
        "gdal",
        "artifact",
        "major.minor",
        "It produces tier 0's rasters, where a driver or resampling change alters the bytes.",
    ),
    EnvironmentComponent(
        "proj",
        "artifact",
        "major.minor",
        "It produces tier 0's reprojected rasters and vectors, where a CRS-transform change alters the bytes.",
    ),
    EnvironmentComponent(
        "rasterio",
        "artifact",
        "major.minor",
        "It writes tier 0's GeoTIFFs, so its encoding defaults are produced bytes.",
    ),
    EnvironmentComponent(
        "rioxarray",
        "artifact",
        "major.minor",
        "It mediates the raster reads and writes that produce tier 0's rasters.",
    ),
    EnvironmentComponent(
        "geopandas",
        "artifact",
        "major.minor",
        "It writes tier 0's vector artifacts.",
    ),
    EnvironmentComponent(
        "shapely",
        "artifact",
        "major.minor",
        "It decides tier 0's geometry results, where a geometric-algorithm change alters the bytes.",
    ),
    EnvironmentComponent(
        "xarray",
        "artifact",
        "major.minor",
        "It serialises tier 0's NetCDF, whose encoding defaults are version-dependent within a minor.",
    ),
    EnvironmentComponent(
        "swmmio",
        "artifact",
        "major.minor",
        "It writes tier 0's three .inp files, so its writer formatting IS the produced bytes.",
    ),
    EnvironmentComponent(
        "numpy",
        "artifact",
        "major",
        "The generator draws no random numbers, so the only byte-affecting axis is dtype "
        "promotion and array serialisation, both declared stable within a major.",
    ),
    EnvironmentComponent(
        "pandas",
        "artifact",
        "major",
        "It carries tier 0's lookup and mapping CSVs; column ordering and float formatting "
        "are version-dependent at major and API-stable at minor.",
    ),
    EnvironmentComponent(
        "patch",
        "excluded",
        "none",
        "Patch components of every real-input row above. Semver patch is these "
        "distributions' declared no-behaviour-change axis; excluding it is a decision on the "
        "record, so a surprising patch-level change is a NAMED gap rather than a silent one.",
    ),
    EnvironmentComponent(
        "conda-env-name",
        "excluded",
        "none",
        "It changes identity and never produced bytes.",
    ),
    EnvironmentComponent(
        "python-patch",
        "excluded",
        "none",
        "The interpreter's patch level changes neither the solver nor the artifacts.",
    ),
    EnvironmentComponent(
        "hostname",
        "excluded",
        "none",
        "It changes identity and never produced bytes.",
    ),
    EnvironmentComponent(
        "cpu-count",
        "excluded",
        "none",
        "It changes scheduling, never produced bytes.",
    ),
    EnvironmentComponent(
        "locale",
        "excluded",
        "none",
        "It changes formatting of human-facing output and never produced bytes.",
    ),
    EnvironmentComponent(
        "gpu-model",
        "excluded",
        "none",
        "It already reaches the build through tier 2's path suffix, which is not a key; "
        "folding it here would duplicate a mechanism that exists.",
    ),
    EnvironmentComponent(
        "affine",
        "excluded",
        "none",
        "Pure affine algebra whose output is fully determined by its inputs, with no version-dependent numerics.",
    ),
)

_DOTTED = re.compile(r"\d+(?:\.\d+)*")


def _leading_major_minor(raw: str) -> str:
    """The first dotted-numeric run on ``raw``, truncated to major.minor."""
    match = _DOTTED.search(raw or "")
    if match is None:
        return ""
    parts = match.group(0).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


def _leading_major(raw: str) -> str:
    """The first dotted-numeric run on ``raw``, truncated to its major component."""
    match = _DOTTED.search(raw or "")
    return match.group(0).split(".")[0] if match is not None else ""


def _trailing_major_minor(raw: str) -> str:
    """The LAST dotted-numeric run on ``raw``, truncated to major.minor.

    The last run rather than the first, for the compiler's ``--version`` line specifically:
    see the ``cxx-compiler`` row's ground.
    """
    matches = _DOTTED.findall(raw or "")
    if not matches:
        return ""
    parts = matches[-1].split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


def _enum_value(raw: str) -> str:
    """A declared enum value, case- and whitespace-normalized."""
    return (raw or "").strip().lower()


_NORMALIZERS: dict[str, Callable[[str], str]] = {
    "major.minor": _leading_major_minor,
    "major": _leading_major,
    "compiler": _trailing_major_minor,
    "enum": _enum_value,
    "none": lambda _raw: "",
}


def _first_line_of(command: list[str]) -> str:
    """The first line of ``command``'s stdout, or ``""`` when it cannot be run."""
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        for line in completed.stdout.splitlines():
            if line.strip():
                return line.strip()
    return ""


def _package_version(distribution: str) -> str:
    """Installed version of ``distribution``, or ``""`` when it is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return ""


def _compiler_version_line() -> str:
    """The first ``--version`` line of the MPI C++ wrapper TRITON actually builds with.

    ``mpic++`` first because TRITON's ``machine.cmake`` FORCE-sets the compiler to it even
    under ``TRITON_IGNORE_MACHINE_FILES=ON``; the fallbacks keep the term populated on a box
    where the wrapper is absent, which is more useful than an empty term there.
    """
    for candidate in ("mpic++", "g++", "c++"):
        found = shutil.which(candidate)
        if found:
            return _first_line_of([found, "--version"])
    return ""


@functools.lru_cache(maxsize=1)
def environment_values() -> dict[str, str]:
    """Raw values for every PROBEABLE component, resolved once per process.

    Once per process rather than once per fixture: the digest is persisted in the marker, so
    re-probing four subprocess versions on each of hundreds of constructions would add a cost
    inside the package that exists to remove one. The excluded rows are not probed at all --
    they are hashed by nothing, so probing them would be work whose result no consumer reads.
    """
    distinct = sorted({component.name for component in ENVIRONMENT_COMPONENTS if component.tier != "excluded"})
    values: dict[str, str] = {}
    for name in distinct:
        if name in ("cxx-compiler",):
            values[name] = _compiler_version_line()
        elif name == "cmake":
            found = shutil.which("cmake")
            values[name] = _first_line_of([found, "--version"]) if found else ""
        elif name == "kokkos-backend":
            # Not probeable from the environment: the backend is a per-run CONFIG choice, so
            # the caller supplies it and an absent supply is a truthful empty term.
            values[name] = ""
        else:
            values[name] = _package_version(name)
    return values


def environment_digest(tier: str, *, values: Mapping[str, str] | None = None) -> str:
    """SHA-256 over the components of ``tier`` that the policy admits, at their granularity.

    ``values`` overrides the probed raw values for the components it names. That override is
    how the two sensitivity arms are runnable: one varies a real component's minor and
    requires the digest to move, the other varies a real component's PATCH digit and each
    excluded row and requires it not to. Without an override the arms could only assert
    something about the machine they happen to run on.

    An unknown ``tier`` digests nothing rather than raising: a caller that misspells a tier
    gets the digest of the empty string, which no other tier produces, so the mistake is
    visible in a marker rather than fatal to fixture setup.
    """
    base = dict(environment_values())
    if values:
        base.update(values)
    digest = hashlib.sha256()
    for component in ENVIRONMENT_COMPONENTS:
        if component.tier != tier:
            continue
        normalize = _NORMALIZERS[component.granularity]
        digest.update(component.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalize(base.get(component.name, "")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_stack_digest(*, values: Mapping[str, str] | None = None) -> str:
    """The artifact-stack half of the environment policy -- tier 0's env term.

    A thin, named entry point rather than a bare ``environment_digest("artifact")`` call at
    the tier-0 key site: the tier-0 module should not have to know the policy's tier names,
    and a reader of tier 0's key should see WHY that key carries an environment term at all.
    """
    return environment_digest("artifact", values=values)


# --------------------------------------------------------------------------------------
# The composition
# --------------------------------------------------------------------------------------


def toolkit_src_digest(src_root: Path) -> str:
    """SHA-256 over every ``*.py`` under ``src_root``, read from the WORKING TREE.

    WORKING TREE and not a git object, because that is the whole reason this term exists:
    ``git show HEAD:{path}`` would satisfy any description of "a digest of the toolkit
    sources" while restoring exactly the commit-blindness being removed -- an uncommitted
    edit would leave the digest unchanged and every marker reading current.

    Each file contributes its POSIX relative path, a NUL, its bytes, and a NUL, in sorted
    path order. The path is INSIDE the digest deliberately: renaming a module with identical
    bytes changes its import path, which is a real change to what the code builds, and a
    digest over bytes alone would call two different module layouts identical.

    The empty string, not the digest of no input, when ``src_root`` is not a directory --
    an installed wheel has no checkout ``src/``. That mirrors :func:`toolkit_sha`'s
    convention: absence is reported as absence, so every environment without a source tree
    compares equal to every other instead of to a plausible-looking empty digest.
    """
    if not src_root.is_dir():
        return ""
    entries = sorted((path.relative_to(src_root).as_posix(), path) for path in src_root.rglob("*.py"))
    digest = hashlib.sha256()
    for rel_path, path in entries:
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
        digest.update(b"\0")
    return digest.hexdigest()


def toolkit_sha(repo_root: Path) -> str:
    """``HEAD`` of the repository at ``repo_root``, or ``""`` when it cannot be read.

    NOT part of the key composition any more -- :func:`builder_provenance` stamps
    :func:`toolkit_src_digest` instead, because a commit identity cannot see an uncommitted
    edit. It is retained only because ``hhemt/cache/__init__.py`` re-exports it; a caller
    reaching for it as a cache term reintroduces the commit-blindness this module moved off.

    An empty string is a TRUTHFUL answer rather than a swallowed failure: a wheel installed
    outside any checkout has no commit, and every marker written in such an environment
    carries the same empty term, so they compare equal to each other. Raising instead would
    make the marker mechanism unusable from an installed package.
    """
    with contextlib.suppress(Exception):
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    return ""


def builder_provenance(
    sources: Iterable[Path],
    *,
    repo_root: Path,
    tier0_key: str = "",
    solver_identity: str = "",
    effective_pin: str = "",
    configs_digest: str = "",
    environment_digest: str = "",
    case_params_digest: str = "",
    payload_digest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The provenance block a marker records and a reader compares against.

    ``repo_root`` is explicit rather than derived from this file because the caller owns the
    tree whose sources are being stamped: a fixture run from one checkout against a toolkit
    installed from another must stamp the USING tree, and a derivation here would stamp the
    installed one. It locates the toolkit sources the same way the caller does, so a
    checkout root carries its own ``src/hhemt/``.

    The remaining seven terms are keyword-only and default to a truthful empty value, because
    each is derived from something only the CALLER holds: the tier-0 directory name, the
    pin a caller provisioned, the report configs it passes, the case parameters it builds
    for, and the payload it promises. A term this function re-derived itself would be a
    second spelling of a fact the caller already has, and the two would drift.

    Note the ORDER OF TRUST in ``payload_digest``: it belongs to the composition because a
    payload's bytes are part of what the tree IS, but :func:`hhemt.cache.markers.marker_state`
    compares it SEPARATELY from the rest of the block, so a damaged payload reports as
    ``unsatisfied`` (warn, then rebuild) rather than as ``stale`` (different builder, rebuild
    silently). The two diagnoses route to different human actions and one arm tests each.
    """
    return {
        "builder_digest": builder_digest(sources),
        "toolkit_src_digest": toolkit_src_digest(repo_root / "src" / "hhemt"),
        "tier0_key": tier0_key,
        "solver_identity": solver_identity,
        "effective_pin": effective_pin,
        "configs_digest": configs_digest,
        "environment_digest": environment_digest,
        "case_params_digest": case_params_digest,
        "payload_digest": dict(payload_digest or {}),
    }
