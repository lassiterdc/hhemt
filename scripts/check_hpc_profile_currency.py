#!/usr/bin/env python3
"""Fail when the example HPC profile on the docs site has fallen behind the config model.

THE PROBLEM. ``docs/how-to/hpc-profile-setup.md`` shows one complete ``hpc_system_config``
profile as a ``yaml`` block a reader copies verbatim. The block is hand-maintained and the
model it describes (``hhemt.config.hpc_system``) is not, so a field added to the model leaves
the page silently incomplete, and a field renamed or removed leaves the page with a key the
loader rejects. Neither event touches the page, so nothing else in the repository reports it.

WHAT IS CHECKED. The page must carry exactly one ``yaml`` block whose top level holds both
``hpc_name`` and ``partitions``; that block is the profile. Then:

1. Every field of the model appears in the profile, walking nested models: the fields of
   each partition entry are checked over the UNION of the declared partitions (a field shown
   on any one partition counts as shown), and the fields of the ``container`` block are
   checked inside that block. Any other mapping or list of specs is walked the same way
   as the partitions, at whatever depth the annotation declares -- an optional container,
   an optional element, or a mapping of lists of specs included. A field may be shown with its default value, ``null``
   included -- what is checked is that the reader can see the key, not that it is set.
2. Every key in the profile is a field of the model. This is the misnamed-field case: a
   rename shows up here as an unknown key, and also as an absent field unless another
   partition entry still carries the real name.
3. The profile loads through the model itself, so a value of the wrong type fails here
   rather than on the reader's cluster.

The field list is read from the model at run time and is never written down here, so a
field added to the model fails this check until the page shows it. Deliberately NOT a
hand-written expected-field list: that list is a second copy of the model and goes stale
exactly the way the page does, with this check still green.

WHAT A GREEN RUN DOES NOT SAY. It does not say the values on the page are right for the
cluster the page names; it says the page and the model agree on which keys exist and that
the values load.

Exit 0 = the profile is current, 1 = at least one finding (printed to stderr).
"""

from __future__ import annotations

import argparse
import re
import sys
import types
import typing
from dataclasses import dataclass
from pathlib import Path

import yaml

# Bind `import hhemt` to THIS checkout before the first package import. The editable
# install names one `src`, so a bare run from a second checkout would otherwise grade
# this checkout's page against the other checkout's model and report a gap that is not
# there, or miss one that is.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_src import bind_local_src  # noqa: E402

bind_local_src(Path(__file__).resolve().parent.parent / "src", {"hhemt"})

from pydantic import BaseModel, ValidationError  # noqa: E402

from hhemt.config.hpc_system import hpc_system_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE_REL = "docs/how-to/hpc-profile-setup.md"

# A fenced block opened with ```yaml at column 0. The closing fence is the next column-0
# run of three backticks, so an indented example inside an admonition is not picked up.
_YAML_FENCE_RE = re.compile(r"^```yaml[ \t]*\n(.*?)^```", re.M | re.S)

# The two keys that identify the profile block among other yaml blocks on the page. Both
# are required fields of the model, so a block lacking either cannot be a loadable profile.
_PROFILE_MARKER_KEYS = ("hpc_name", "partitions")


@dataclass(frozen=True)
class Finding:
    """One reason the page's profile is not current, located by its dotted path in the profile."""

    location: str  # dotted path inside the profile, or "<page>" for a whole-page finding
    code: str
    detail: str

    def render(self) -> str:
        """The one-line form printed on failure: location, code, then the explanation."""
        return f"{self.location}: {self.code}: {self.detail}"


def _nested_model(annotation: object) -> tuple[tuple[str, ...], type[BaseModel]] | None:
    """Classify a field annotation as holding a nested model, and through which containers.

    Returns ``(steps, M)``: ``M`` is the model the annotation holds, and ``steps`` lists the
    containers between the field and ``M``, outermost first -- ``"mapping"`` for a
    ``dict[str, ...]`` and ``"sequence"`` for a ``list[...]``. So ``M`` gives ``()``,
    ``dict[str, M]`` gives ``("mapping",)``, and ``dict[str, list[M]]`` gives
    ``("mapping", "sequence")``. Returns None for anything that holds no model (scalars,
    lists of scalars, plain ``dict``).

    ``X | None`` is classified as ``X`` wherever it appears -- on the field itself or on a
    container's element -- because a container's element is classified by this same
    function. So ``dict[str, M | None]`` and ``list[M | None]`` hold ``M`` exactly as
    ``dict[str, M]`` and ``list[M]`` do.

    Not handled, and returned as None: a Union of two or more non-None types, which has no
    single nested model to walk. No field of the live models takes that shape.
    """
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        # Unwrap `X | None` before classifying `X`. Deliberately NOT "any model among the
        # union's members": that rule found `ContainerSpec | None` but returned nothing for
        # `dict[str, M] | None` and `list[M] | None`, whose members are containers rather
        # than models, so the walk never descended and passed an incomplete page.
        non_none = [a for a in args if a is not type(None)]
        return _nested_model(non_none[0]) if len(non_none) == 1 else None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return ((), annotation)
    # A container's element is classified by recursing, deliberately NOT by testing whether
    # the element IS a model class: that test returned None for `dict[str, M | None]`,
    # `list[M | None]` and `dict[str, list[M]]`, each of which holds a model, so the walk
    # never descended and a page omitting a nested field passed.
    if origin is dict and len(args) == 2:
        step, element = "mapping", args[1]
    elif origin is list and len(args) == 1:
        step, element = "sequence", args[0]
    else:
        return None
    inner = _nested_model(element)
    if inner is None:
        return None
    steps, model = inner
    return ((step, *steps), model)


def _walk_location(model: type[BaseModel], loc: tuple) -> str:
    """Render a validation error location in the walk's own dotted form.

    The walk names a mapping key or a list index as ``*``, because a field shown on any
    entry counts as shown; a validation error names the entry. Translating one into the
    other is what lets ``check_profile`` tell whether the walk already reported an
    unknown key, rather than assuming it did.
    """
    parts: list[str] = []
    current: type[BaseModel] | None = model
    i = 0
    while i < len(loc):
        seg = loc[i]
        i += 1
        parts.append(str(seg))
        if current is None or seg not in current.model_fields:
            current = None
            continue
        nested = _nested_model(typing.get_type_hints(current).get(seg))
        if nested is None:
            current = None
            continue
        steps, current = nested
        # One mapping key or list index per container step, each rendered as `*`.
        for _step in steps:
            if i >= len(loc):
                break
            parts.append("*")
            i += 1
    return ".".join(parts)


def _walk(model: type[BaseModel], mappings: list[dict], prefix: str) -> tuple[list[Finding], int]:
    """Coverage findings for ``model`` over ``mappings``, plus the number of fields checked.

    ``mappings`` holds every profile mapping that instantiates ``model`` (one for the top
    level and the container block, one per entry of a mapping or list of specs such as the
    partitions); a field counts as shown when any of them carries it.
    """
    findings: list[Finding] = []
    # get_type_hints rather than FieldInfo.annotation: under `from __future__ import
    # annotations` the model module's forward reference to ContainerSpec stays a
    # ForwardRef on the FieldInfo, so the container block would read as a scalar field
    # and its contents would go unchecked.
    hints = typing.get_type_hints(model)
    checked = 0
    for name in model.model_fields:
        checked += 1
        where = f"{prefix}{name}"
        carriers = [m for m in mappings if name in m]
        if not carriers:
            findings.append(
                Finding(where, "field-absent", f"{model.__name__}.{name} is a model field the profile does not show")
            )
            continue
        nested = _nested_model(hints.get(name))
        if nested is None:
            continue
        steps, sub = nested
        # Descend exactly as many container levels as the ANNOTATION declares, never as the
        # values happen to be shaped: expanding every list-valued entry would also expand a
        # scalar list such as available_gpu_models and report a correct page stale. A value
        # of the wrong container type at a step contributes nothing, so its sub-fields read
        # absent rather than being guessed at.
        level = [m[name] for m in carriers]
        for step in steps:
            if step == "mapping":
                level = [v for value in level if isinstance(value, dict) for v in value.values()]
            else:
                level = [v for value in level if isinstance(value, list) for v in value]
        inner = [v for v in level if isinstance(v, dict)]
        sub_prefix = f"{where}." + "*." * len(steps)
        # A nested block set to null (or a container with no entries, or only null entries)
        # shows none of the nested model's fields, so it is walked with no mappings and every
        # field reports absent.
        sub_findings, sub_checked = _walk(sub, inner, sub_prefix)
        findings.extend(sub_findings)
        checked += sub_checked
    for mapping in mappings:
        for key in mapping:
            if key not in model.model_fields:
                findings.append(
                    Finding(f"{prefix}{key}", "key-not-in-model", f"{model.__name__} has no field named {key!r}")
                )
    return findings, checked


def extract_profile(page_text: str) -> tuple[dict | None, list[Finding]]:
    """Return the page's single profile block, or None with the finding that explains why not."""
    candidates: list[dict] = []
    for body in _YAML_FENCE_RE.findall(page_text):
        try:
            doc = yaml.safe_load(body)
        except yaml.YAMLError as exc:
            return None, [Finding("<page>", "yaml-parse-error", str(exc).replace("\n", " "))]
        if isinstance(doc, dict) and all(k in doc for k in _PROFILE_MARKER_KEYS):
            candidates.append(doc)
    if not candidates:
        return None, [
            Finding(
                "<page>",
                "no-profile-block",
                f"no column-0 ```yaml block with top-level {' and '.join(_PROFILE_MARKER_KEYS)} was found",
            )
        ]
    if len(candidates) > 1:
        return None, [
            Finding(
                "<page>",
                "ambiguous-profile-block",
                f"{len(candidates)} yaml blocks look like a profile; exactly one is expected",
            )
        ]
    return candidates[0], []


def check_profile(profile: dict, model: type[BaseModel] = hpc_system_config) -> tuple[list[Finding], int]:
    """All findings for one profile mapping against ``model``, plus the number of fields checked."""
    findings, checked = _walk(model, [profile], "")
    walked_unknown = {f.location for f in findings if f.code == "key-not-in-model"}
    try:
        model.model_validate(profile)
    except ValidationError as exc:
        for err in exc.errors():
            loc_parts = tuple(err.get("loc", ()))
            if err.get("type") == "extra_forbidden":
                # An unknown key the walk already reported is skipped, so a rename is not
                # printed twice. Only THAT key is skipped: an unknown key at a level the walk
                # did not reach is reported here, because skipping every extra_forbidden
                # error left such a key reported by neither path and the gate exited 0.
                where = _walk_location(model, loc_parts)
                if where in walked_unknown:
                    continue
                findings.append(
                    Finding(
                        where, "key-not-in-model", f"{loc_parts[-1]!r} is not a field here (reported by validation)"
                    )
                )
                continue
            loc = ".".join(str(part) for part in loc_parts) or "<profile>"
            findings.append(Finding(loc, "profile-invalid", err.get("msg", "validation failed")))
    return findings, checked


def check_page(page: Path, model: type[BaseModel] = hpc_system_config) -> tuple[list[Finding], int]:
    """All findings for the page at ``page``, plus the number of model fields checked."""
    try:
        text = page.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
        return [Finding("<page>", "unreadable", f"{page}: {exc}")], 0
    profile, findings = extract_profile(text)
    if profile is None:
        return findings, 0
    return check_profile(profile, model)


def main(argv: list[str] | None = None) -> int:
    """Check one page and return the process exit status: 0 when current, 1 on any finding."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root holding the docs tree")
    parser.add_argument("--page", type=Path, default=None, help=f"page to check (default: ROOT/{PAGE_REL})")
    args = parser.parse_args(argv)

    page = args.page if args.page is not None else args.root / PAGE_REL
    findings, checked = check_page(page)
    if not findings and checked == 0:
        # A walk that checked nothing and found nothing would print a clean pass over an
        # empty population; refuse it so a green run always means fields were compared.
        findings = [Finding("<model>", "zero-population", "the model walk checked zero fields")]
    if findings:
        print(f"HPC profile currency check FAILED for {page} ({checked} model fields checked):", file=sys.stderr)
        for f in findings:
            print(f"  {f.render()}", file=sys.stderr)
        return 1
    print(f"HPC profile currency check passed for {page}: {checked} model fields shown, 0 findings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
