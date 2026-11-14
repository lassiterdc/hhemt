"""Every configuration setting is read by some module, or an allowlist entry says it is not.

A setting that the pages document and no code reads does nothing for the user who sets
it. `report_html_mode` was such a setting until the report-format work wired it, and a
page described it as live. This test finds the class: it walks every field of the
configuration models under `src/hhemt/config/` and reports each field whose name no
module outside that package reads, where a read is an attribute access (`cfg.name`), a
string subscript (`cfg["name"]`), a `get` of the name (`cfg.get("name")`) or a
`getattr` with the name as a string constant. A field is also read when a method of its
model, or of a base class of that model other than `cfgBaseModel` (whose
`print_files_defined_in_yaml` dumps every model), formats a module-level template
constant with `**` applied to a `self.model_dump(...)` result and the field's name is a
placeholder of that template, read with `string.Formatter().parse`. Such a field is read
by the template, not by name. A subclass that formats the template reads the inherited
fields as well.

`unread_settings_allowlist.yaml` holds the fields the test finds today, each with the
field's path and one line saying what the source shows about it. The test fails on an
unread field with no entry and on an entry whose field is read or no longer exists, so
the list shrinks when a setting is wired and cannot grow without an edit that a
reviewer sees. No code in the repository writes the allowlist.

The comparison is by field name, not by owning model: a read of `.dpi` on any object
counts for every field named `dpi`. That makes the test miss a dead field that shares
its name with a live one, and it never reports a live field as dead. The test parses
source with `ast`; it imports no toolkit module and compiles or runs no solver.
"""

from __future__ import annotations

import ast
import string
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_ALLOWLIST = Path(__file__).resolve().parent / "unread_settings_allowlist.yaml"
_PACKAGE = Path("src") / "hhemt"
_CONFIG = _PACKAGE / "config"

# A configuration model reaches pydantic's `BaseModel` through `cfgBaseModel`; the class
# search below follows base-class names until no new model class appears.
_MODEL_ROOT = "BaseModel"


@dataclass(frozen=True)
class Setting:
    """One field of a configuration model: where it is declared and what it is called."""

    module: str
    model: str
    name: str

    @property
    def path(self) -> str:
        """The allowlist key: `src/hhemt/config/{module}::{model}.{name}`."""
        return f"{_CONFIG.as_posix()}/{self.module}::{self.model}.{self.name}"


def _python_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def config_settings(repo: Path) -> list[Setting]:
    """Every public field of every model class in the configuration package of `repo`.

    A field is an annotated assignment in the body of a class that reaches `BaseModel`
    through other classes of the package. Names that start with an underscore and
    `ClassVar` annotations are not settings.
    """
    classes: dict[tuple[str, str], tuple[ast.ClassDef, list[str]]] = {}
    for path in _python_files(repo / _CONFIG):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b).split("[")[0].split(".")[-1] for b in node.bases]
                classes[(path.name, node.name)] = (node, bases)
    models: set[str] = set()
    grew = True
    while grew:
        grew = False
        for (_, name), (_, bases) in classes.items():
            if name not in models and any(b == _MODEL_ROOT or b in models for b in bases):
                models.add(name)
                grew = True
    settings = []
    for (module, name), (node, _) in sorted(classes.items()):
        if name not in models:
            continue
        for stmt in node.body:
            if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                continue
            if stmt.target.id.startswith("_") or "ClassVar" in ast.unparse(stmt.annotation):
                continue
            settings.append(Setting(module, name, stmt.target.id))
    return settings


def names_read(repo: Path) -> set[str]:
    """Names that a module outside the configuration package reads.

    Counts `x.name`, `x["name"]`, `x.get("name")` and `getattr(x, "name")` in every
    `.py` file under `src/hhemt/` except `src/hhemt/config/`. The `getattr` form counts
    because it is attribute access with the name spelled as a string; `hasattr` and
    `setattr` do not. A string that appears in any other
    position (a keyword argument, a dict literal key, a template) is not a read here.
    """
    config_dir = repo / _CONFIG
    read: set[str] = set()
    for path in _python_files(repo / _PACKAGE):
        if config_dir in path.parents:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute):
                read.add(node.attr)
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                read.add(node.slice.value)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                read.add(node.args[0].value)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                read.add(node.args[1].value)
    return read


def _dump_call(node: ast.AST) -> ast.Call | None:
    """`node` when the node is a call `self.model_dump(...)`, else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "model_dump"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    ):
        return node
    return None


def _excluded(dump: ast.Call) -> set[str]:
    """String constants of an `exclude={...}` keyword of a dump call; empty when absent or not literal."""
    for kw in dump.keywords:
        if kw.arg == "exclude" and isinstance(kw.value, (ast.Set, ast.List, ast.Tuple)):
            return {e.value for e in kw.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    return set()


def template_fields(repo: Path) -> dict[tuple[str, str], set[str]]:
    """For each model class, the field names that a template formatted from its dump names.

    A method of the class formats a module-level string constant `T` as `T.format(**d)`
    where `d` is `self.model_dump(...)` written in place or a local name assigned from that
    call. The placeholders of `T` are read with `string.Formatter().parse`, less the names
    the dump excludes. The result is keyed by `(module file name, class name)`; the
    caller extends the result to subclasses, and a subclass inherits the method. A method of `cfgBaseModel`
    is skipped: its dump goes to `print`, not to a template.
    """
    found: dict[tuple[str, str], set[str]] = {}
    for path in _python_files(repo / _CONFIG):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        templates = {
            stmt.targets[0].id: stmt.value.value
            for stmt in tree.body
            if isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        }
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            if cls.name == "cfgBaseModel":
                continue
            for method in (n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
                dumps = {
                    t.id: d
                    for n in ast.walk(method)
                    if isinstance(n, ast.Assign) and (d := _dump_call(n.value)) is not None
                    for t in n.targets
                    if isinstance(t, ast.Name)
                }
                for call in (n for n in ast.walk(method) if isinstance(n, ast.Call)):
                    func = call.func
                    if not (
                        isinstance(func, ast.Attribute)
                        and func.attr == "format"
                        and isinstance(func.value, ast.Name)
                        and func.value.id in templates
                    ):
                        continue
                    for kw in call.keywords:
                        if kw.arg is not None:
                            continue
                        dump = _dump_call(kw.value) or (
                            dumps.get(kw.value.id) if isinstance(kw.value, ast.Name) else None
                        )
                        if dump is None:
                            continue
                        names = {
                            field.split(".")[0].split("[")[0]
                            for _, field, _, _ in string.Formatter().parse(templates[func.value.id])
                            if field
                        }
                        found.setdefault((path.name, cls.name), set()).update(names - _excluded(dump))
    return found


def names_read_by_template(repo: Path) -> dict[str, set[str]]:
    """`{declaring class name: field names a dump-fed template reads}`.

    A class inherits the methods of its bases, so the names a base class's template reads
    are read through every subclass, and a subclass's template reads the inherited
    fields. The result therefore maps each class, including a base class, to the names
    read by templates of the class itself and of every class that inherits from the class. Inheritance is
    followed by class name across the configuration package, as the model search does.
    """
    direct = template_fields(repo)
    bases: dict[str, list[str]] = {}
    for path in _python_files(repo / _CONFIG):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                bases[node.name] = [ast.unparse(b).split("[")[0].split(".")[-1] for b in node.bases]

    def lineage(name: str) -> set[str]:
        seen = {name}
        stack = [name]
        while stack:
            for b in bases.get(stack.pop(), []):
                if b not in seen:
                    seen.add(b)
                    stack.append(b)
        return seen

    own: dict[str, set[str]] = {}
    for (_, cls), names in direct.items():
        own.setdefault(cls, set()).update(names)
    result: dict[str, set[str]] = {}
    for cls in bases:
        # `cls` formats its own templates and those of its bases; its fields include every base's.
        names = set().union(*(own.get(a, set()) for a in lineage(cls)))
        for declaring in lineage(cls):
            result.setdefault(declaring, set()).update(names)
    return result


def unread_paths(settings: list[Setting], read: set[str], template_read: dict[str, set[str]] | None = None) -> set[str]:
    """The paths of the settings whose name is not in `read` and that no template reads."""
    template_read = template_read or {}
    return {s.path for s in settings if s.name not in read and s.name not in template_read.get(s.model, set())}


def load_allowlist(path: Path) -> dict[str, str]:
    """The allowlist as `{field path: reason}`; a malformed entry fails with its position."""
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))["entries"]
    allowed: dict[str, str] = {}
    for index, entry in enumerate(entries):
        assert set(entry) == {"field", "reason"}, f"entry {index} must carry exactly `field` and `reason`"
        field, reason = entry["field"], entry["reason"]
        assert isinstance(field, str) and field, f"entry {index} has no field path"
        assert isinstance(reason, str) and reason.strip() and "\n" not in reason.strip(), (
            f"entry {field} needs a one-line reason"
        )
        assert field not in allowed, f"entry {field} appears twice"
        allowed[field] = reason
    return allowed


def unlisted(
    settings: list[Setting], read: set[str], allowed: dict[str, str], template_read: dict[str, set[str]] | None = None
) -> list[str]:
    """Problems: an unread setting that has no allowlist entry."""
    return [
        f"{path} is read by no module outside src/hhemt/config/ and has no allowlist entry"
        for path in sorted(unread_paths(settings, read, template_read) - set(allowed))
    ]


def stale(
    settings: list[Setting], read: set[str], allowed: dict[str, str], template_read: dict[str, set[str]] | None = None
) -> list[str]:
    """Problems: an allowlist entry whose setting is read now or is no longer a field."""
    declared = {s.path for s in settings}
    unread = unread_paths(settings, read, template_read)
    problems = []
    for path in sorted(set(allowed) - unread):
        why = "is read by a module outside src/hhemt/config/" if path in declared else "is no longer a field"
        problems.append(f"allowlist entry {path} {why}; remove the entry")
    return problems


# --- the tree under test -----------------------------------------------------------------


@pytest.fixture(scope="module")
def tree():
    """Settings, names read, allowlist and template reads of this checkout."""
    return config_settings(_REPO), names_read(_REPO), load_allowlist(_ALLOWLIST), names_read_by_template(_REPO)


def test_every_unread_setting_is_on_the_allowlist(tree):
    """Property: no configuration field is unread and unlisted.

    Class: every field of every model class under `src/hhemt/config/`.
    Why required: a setting that no code reads misleads a user who sets it, and an
    entry on the allowlist is how a reviewer sees that the setting is still unwired.
    What kills it: adding a field that no module reads, or removing the read of a
    wired field (the tree before the report-format work fails here and names
    `report_html_mode`). A second correct implementation: a regular expression over
    the source text in place of `ast`, which gives the same unread set on a field name
    that appears as an attribute, a subscript or a `get`.
    """
    settings, read, allowed, template_read = tree
    assert unlisted(settings, read, allowed, template_read) == []


def test_no_allowlist_entry_is_stale(tree):
    """Property: every allowlist entry names a field that exists and is still unread.

    Class: every entry of `unread_settings_allowlist.yaml`.
    Why required: an entry for a field that a module now reads hides nothing and
    suggests the setting is still unwired; the list has to shrink as settings are wired.
    What kills it: wiring a listed field without deleting its entry, or renaming a field.
    A second correct implementation: derive the unread set from the compiled
    bytecode names instead of the syntax tree.
    """
    settings, read, allowed, template_read = tree
    assert stale(settings, read, allowed, template_read) == []


def test_allowlist_is_sorted_and_well_formed():
    """Property: entries are sorted by field path, each with a one-line reason.

    Class: the allowlist file. Why required: a sorted list is reviewed by diff and a
    reason that spans lines is not an entry's one line. What kills it: appending an
    entry out of order. A second correct implementation: sort on load and compare.
    """
    allowed = load_allowlist(_ALLOWLIST)
    assert list(allowed) == sorted(allowed)


# --- controls: each feeds the comparison an input it must refuse -------------------------


def _mini_repo(root: Path, outside_source: str, inside_source: str = "") -> Path:
    """A repository with one model of fields `alpha`, `beta`, `gamma`, `delta` and the given readers."""
    config = root / _CONFIG
    config.mkdir(parents=True)
    (config / "base.py").write_text("from pydantic import BaseModel\nclass cfgBaseModel(BaseModel):\n    pass\n")
    (config / "models.py").write_text(
        "from typing import ClassVar\n"
        "from hhemt.config.base import cfgBaseModel\n"
        "class Child(cfgBaseModel):\n"
        "    alpha: int = 1\n"
        "    beta: int = 2\n"
        "    gamma: int = 3\n"
        "    delta: int = 4\n"
        "    _private: int = 5\n"
        "    counter: ClassVar[int] = 6\n"
        "class NotAModel:\n"
        "    omega: int = 7\n" + inside_source
    )
    (root / _PACKAGE / "reader.py").write_text(outside_source)
    return root


def _check(tmp_path: Path, outside_source: str, inside_source: str = ""):
    repo = _mini_repo(tmp_path, outside_source, inside_source)
    return config_settings(repo), names_read(repo)


def test_control_settings_are_the_public_fields_of_model_classes_only(tmp_path):
    """Class: the field-finding step on a synthetic package.

    The private name, the `ClassVar` and the field of a class that is no model are
    absent from the result; killed by a walk that takes every annotated name.
    """
    settings, _ = _check(tmp_path, "")
    assert {s.name for s in settings} == {"alpha", "beta", "gamma", "delta"}


def test_control_each_read_form_counts_and_other_positions_do_not(tmp_path):
    """Class: attribute, subscript and `get` reads against a keyword argument and a literal.

    `alpha`, `beta` and `gamma` are read in the three counted forms and `delta` is
    named as a keyword argument and as a dict key literal, which are not reads. The
    result must flag exactly `delta`; killed by a matcher that counts any occurrence of
    the name, which would clear `delta`. A second tree reads `delta` by
    `getattr(c, "delta", None)` and names `alpha` as a keyword argument, `beta` in
    `hasattr` and `gamma` in `setattr`; the unread set there is `alpha`, `beta`, `gamma`.
    Killed by a walk that omits the `getattr` branch (it would flag `delta`) or one that
    counts every call's string argument (it would clear `beta` and `gamma`).
    """
    reader = (
        "def f(c):\n"
        "    a = c.alpha\n"
        '    b = c["beta"]\n'
        '    g = c.get("gamma")\n'
        "    h(delta=1)\n"
        '    return {"delta": 2}\n'
    )
    settings, read = _check(tmp_path, reader)
    assert {p.rsplit(".", 1)[1] for p in unread_paths(settings, read)} == {"delta"}

    # `getattr` with a string constant is attribute access and counts; the same name as a
    # keyword argument in the same module does not. `hasattr` and `setattr` count neither.
    getter = (
        "def f(c):\n"
        '    a = getattr(c, "delta", None)\n'
        "    h(alpha=1)\n"
        '    hasattr(c, "beta")\n'
        '    setattr(c, "gamma", 1)\n'
        "    return c\n"
    )
    settings, read = _check(tmp_path / "getattr", getter)
    assert {p.rsplit(".", 1)[1] for p in unread_paths(settings, read)} == {"alpha", "beta", "gamma"}


def test_control_a_read_inside_the_config_package_does_not_count(tmp_path):
    """Class: a field read only by a config-package method.

    `alpha` is read by `Child.double` inside `src/hhemt/config/` and by nothing
    outside it; the result must flag it. Killed by a walk that includes the config directory.
    """
    inside = "    def double(self):\n        return self.alpha * 2\n"
    settings, read = _check(tmp_path, "x = 1\n", inside)
    assert "alpha" not in read
    assert any(p.endswith("Child.alpha") for p in unread_paths(settings, read))


def test_control_an_unread_field_with_no_entry_is_reported_by_path(tmp_path):
    """Class: the comparison of unread fields with an empty allowlist.

    Must fire: the message names the field's path. Killed by a comparison that
    returns no problem for an unlisted field.
    """
    settings, read = _check(tmp_path, "def f(c):\n    return c.alpha, c.beta, c.gamma\n")
    problems = unlisted(settings, read, {})
    path = "src/hhemt/config/models.py::Child.delta"
    assert problems == [f"{path} is read by no module outside src/hhemt/config/ and has no allowlist entry"]


def test_control_a_listed_field_that_is_read_or_gone_is_reported_stale(tmp_path):
    """Class: the comparison of allowlist entries with the unread set.

    `alpha` is read, `omega` is not a field, `delta` is unread. Only the first two
    are stale. Killed by a comparison that checks the list against the declared fields alone.
    """
    settings, read = _check(tmp_path, "def f(c):\n    return c.alpha, c.beta, c.gamma\n")
    prefix = "src/hhemt/config/models.py::Child."
    allowed = {prefix + "alpha": "r", prefix + "delta": "r", prefix + "omega": "r"}
    assert stale(settings, read, allowed) == [
        f"allowlist entry {prefix}alpha is read by a module outside src/hhemt/config/; remove the entry",
        f"allowlist entry {prefix}omega is no longer a field; remove the entry",
    ]


def test_control_removing_the_report_html_mode_read_makes_the_real_tree_fail(tree):
    """Class: the real tree with the report-format wiring undone.

    Takes the real settings and allowlist, removes the one name `report_html_mode`
    from the set of names read, and expects the unlisted comparison to name it. This
    models the tree before the report-format work, so it fires without a checkout of
    that tree; killed by an `unlisted` that ignores the read set.
    """
    settings, read, allowed, template_read = tree
    problems = unlisted(settings, read - {"report_html_mode"}, allowed, template_read)
    assert any("InteractiveBackendConfig.report_html_mode" in p for p in problems)


# --- controls for the template rule --------------------------------------------------------

_TEMPLATE_MODULE = "T = 'a {alpha} b {beta}'\nU = 'no placeholders here'\n"


def _template_check(tmp_path: Path, method_body: str, extra_module_source: str = _TEMPLATE_MODULE):
    """Settings, names read and template reads of a synthetic package whose `Child.render` is `method_body`."""
    inside = extra_module_source + "def _unused():\n    pass\n"
    repo = _mini_repo(tmp_path, "x = 1\n", "")
    # the class body must sit after the module constants: rewrite models.py with both
    models = repo / _CONFIG / "models.py"
    text = models.read_text().replace("class Child(cfgBaseModel):\n", "class Child(cfgBaseModel):\n" + method_body, 1)
    models.write_text(inside + text)
    return config_settings(repo), names_read(repo), names_read_by_template(repo)


def _unread_names(settings, read, template_read):
    return {p.rsplit(".", 1)[1] for p in unread_paths(settings, read, template_read) if "Child." in p}


def test_control_a_dump_that_goes_to_print_clears_no_field(tmp_path):
    """Class: a method that dumps the model and prints the dump, as `print_files_defined_in_yaml` does.

    No template is formatted, so `alpha`, `beta`, `gamma` and `delta` all stay unread.
    Killed by a rule that clears every field of a model that calls `model_dump`.
    """
    body = "    def show(self):\n        print(self.model_dump())\n\n"
    settings, read, template_read = _template_check(tmp_path, body)
    assert _unread_names(settings, read, template_read) == {"alpha", "beta", "gamma", "delta"}


def test_control_a_template_formatted_from_the_dump_clears_exactly_its_placeholders(tmp_path):
    """Class: a method that formats a module-level template with `**self.model_dump()`.

    The template names `alpha` and `beta`, so those two are read and `gamma` and `delta` stay
    unread, whether the dump is written in place or assigned to a local name. Killed by a rule
    that clears every field of the formatting model and would clear `gamma` and `delta`.
    """
    in_place = "    def render(self):\n        return T.format(**self.model_dump())\n\n"
    settings, read, template_read = _template_check(tmp_path, in_place)
    assert _unread_names(settings, read, template_read) == {"gamma", "delta"}
    via_local = "    def render(self):\n        fields = self.model_dump()\n        return T.format(**fields)\n\n"
    settings, read, template_read = _template_check(tmp_path / "local", via_local)
    assert _unread_names(settings, read, template_read) == {"gamma", "delta"}


def test_control_a_placeholder_free_template_clears_nothing(tmp_path):
    """Class: a method that formats a template with no placeholder.

    Killed by a rule that clears the fields of any model whose method formats any template.
    """
    body = "    def render(self):\n        return U.format(**self.model_dump())\n\n"
    settings, read, template_read = _template_check(tmp_path, body)
    assert _unread_names(settings, read, template_read) == {"alpha", "beta", "gamma", "delta"}


def test_control_a_subclass_template_reads_the_fields_it_inherits(tmp_path):
    """Class: a base class's field named by a subclass method's template.

    Killed by a rule that checks only the declaring class's own methods.
    """
    repo = _mini_repo(tmp_path, "x = 1\n", "")
    models = repo / _CONFIG / "models.py"
    models.write_text(
        _TEMPLATE_MODULE
        + models.read_text()
        + "class Sub(Child):\n    def render(self):\n        return T.format(**self.model_dump())\n"
    )
    settings, read = config_settings(repo), names_read(repo)
    template_read = names_read_by_template(repo)
    assert _unread_names(settings, read, template_read) == {"gamma", "delta"}


@pytest.mark.parametrize(
    "body",
    [
        "    def render(self, other):\n        return T.format(**other.model_dump())\n\n",
        "    def render(self):\n        return T.format(x=self.model_dump())\n\n",
        "    def render(self):\n        values = {'alpha': 1, 'beta': 2}\n        return T.format(**values)\n\n",
    ],
    ids=["dump_of_another_object", "no_double_star", "plain_dict"],
)
def test_control_a_format_not_fed_by_the_models_own_dump_clears_no_field(tmp_path, body):
    """Class: a template formatted from something other than `**self.model_dump(...)`.

    The dump of another object, a keyword that is not `**`, and a plain dict all name `alpha`
    and `beta` without reading this model's fields. Killed by a rule that accepts any
    `model_dump` call, any keyword, or any `**` argument.
    """
    settings, read, template_read = _template_check(tmp_path, body)
    assert _unread_names(settings, read, template_read) == {"alpha", "beta", "gamma", "delta"}
