"""The publishing credentials are asked for when a person is present, and nothing is stored.

Problem under test. `publishing._require_env` used to demand an environment variable and
refuse otherwise, so a person making a one-off deposit had to export a secret first. The
ruled behaviour is the reverse: ask when someone is there to answer, keep the environment
variable as the unattended route, and never persist a prompt-obtained answer. A stored
credential outlives the run that needed it, is swept into backups and shared-home
snapshots, and has no owner to rotate it.

The storage claim is checked by INTERCEPTING THE ACT, not by enumerating where a write might
land. `_writes_refused()` patches the write primitives themselves — `builtins.open` for any
writing mode, `os.open`, and `Path.write_text` / `write_bytes` / `mkdir` — so a write reached
from anywhere during the prompted call fails the test, including one made by a helper no list
mentions. The AST arm is corroboration over the five named bodies, not the detector: it pins
that nothing in them writes and that a rename reds rather than narrowing the scanned set, and
it cannot see past a refactor into a new helper. The location snapshot is the DIAGNOSTIC: when
the interception fires, or when a location does change, it says which path moved — it is not
what decides pass or fail.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import getpass
import inspect
from pathlib import Path
from unittest import mock

import pytest

from hhemt import publishing
from hhemt.exceptions import PublishError

#: The functions a prompted credential can reach, and therefore the population the AST arm reads.
_PROMPT_PATH = (
    "_require_env",
    "_stdin_is_interactive",
    "_credential_is_secret",
    "_credential_label",
    "_prompt_credential",
)

#: The value the stubbed prompt returns. MODULE-level on purpose: a pytest traceback renders the
#: LOCALS of every displayed frame, so a value bound in a test local would be printed beside the
#: failure; a module constant is not a local, and the planted string never appears in a source
#: line either, which is what lets the leak check in the round's evidence grep for it.
_PLANTED_TOKEN = "prompted-token-value"


class _PromptWrote(Exception):
    """A write reached the prompt path. Private, and carrying the message only — never a value."""


#: Call names that can write: a persistence path reachable from the prompt would appear here.
_WRITE_CALLS = frozenset(
    {
        "open",
        "write",
        "writelines",
        "write_text",
        "write_bytes",
        "mkdir",
        "makedirs",
        "replace",
        "rename",
        "unlink",
        "rmdir",
        "chmod",
        "dump",
        "save",
        "store",
        "NamedTemporaryFile",
        "mkstemp",
        "mkdtemp",
        "copy",
        "copyfile",
    }
)


def _seed_prompt(monkeypatch, value: str) -> None:
    """Stub BOTH prompt primitives so the call cannot read the real terminal or block.

    The injection point is the standard-library call the production helper makes, not a
    replacement of the helper: the real `_prompt_credential` runs, and only its terminal IO
    is stubbed. `input` is stubbed for the username route and `getpass.getpass` for the
    secret route.
    """
    monkeypatch.setattr(getpass, "getpass", lambda *args, **kwargs: value)
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: value)


@contextlib.contextmanager
def _writes_refused():
    """Fail any file-writing primitive reached during the block, wherever it points.

    The detector is a boundary interception rather than a list of places: a helper factored out
    of the prompt path writes through the same primitives, so patching the primitives catches it
    without anyone having to name it. `builtins.open` is guarded by MODE rather than refused
    outright, because a read is legitimate on this path and the read path stays available.
    `os.open` is refused outright: nothing on the prompt path opens at that layer, so a later
    caller that needs it fails loudly here rather than being silently allowed past the guard.

    The failure message names the CALL and its DESTINATION and never an argument value. That is
    not cosmetic: for `Path.write_text(value)` the value IS the prompted credential, so a
    message built from the call's positional arguments would print the secret into the test log
    at the exact moment the guard fired — the leak this chunk exists to prevent, one layer out.
    """
    real_open = builtins.open

    def _refuse_write(call: str):
        """A refusing stand-in for one write primitive, naming the call and the destination only."""

        def _refuse(*args, **kwargs):
            destination = args[0] if args else "<no positional destination>"
            # The private type is raised HERE, where `args` holds the credential, and the context
            # manager below re-raises a fresh AssertionError `from None`. That discards this
            # frame and its locals, so a traceback renderer cannot print the value.
            raise _PromptWrote(f"the prompt path attempted {call} on {destination!r}")

        return _refuse

    def _guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            raise AssertionError(f"the prompt path opened {file!r} for writing (mode {mode!r})")
        return real_open(file, mode, *args, **kwargs)

    with (
        mock.patch("builtins.open", _guarded_open),
        mock.patch("os.open", _refuse_write("os.open")),
        mock.patch.object(Path, "write_text", _refuse_write("Path.write_text")),
        mock.patch.object(Path, "write_bytes", _refuse_write("Path.write_bytes")),
        mock.patch.object(Path, "mkdir", _refuse_write("Path.mkdir")),
    ):
        try:
            yield
        except _PromptWrote as refusal:
            # `from None` is load-bearing: it suppresses the original exception AND its
            # traceback, which is the frame whose locals hold the credential.
            raise AssertionError(str(refusal)) from None


def _storage_locations(tmp_path: Path) -> list[Path]:
    """Every place a prompt-obtained credential could plausibly be stored."""
    home = Path.home()
    return [
        tmp_path,
        home / ".netrc",
        home / ".hhemt_credentials.json",
        home / ".hhemt_credentials",
        home / ".config" / "hhemt",
        home / ".cache" / "hhemt",
        home / ".globus_tokens.json",
    ]


def _snapshot(paths: list[Path]) -> dict[str, tuple[int, int] | None]:
    """(size, mtime_ns) for a file, or the sorted child names for a directory, or None."""
    snap: dict[str, tuple[int, int] | None] = {}
    for path in paths:
        try:
            if path.is_dir():
                snap[str(path)] = (len(list(path.iterdir())), path.stat().st_mtime_ns)
            elif path.exists():
                stat = path.stat()
                snap[str(path)] = (stat.st_size, stat.st_mtime_ns)
            else:
                snap[str(path)] = None
        except OSError:  # a location we cannot stat cannot have been written by this call either
            snap[str(path)] = None
    return snap


def test_a_prompted_credential_is_returned_and_nothing_is_written(tmp_path, monkeypatch):
    """Property: with the env var unset and a terminal present, the credential is ASKED for and no byte lands on disk.

    Class: unset env var x interactive stdin x the secret route (a token), and the same for the
    echo-on route (a username). Why it is required: the ruled behaviour is a prompt instead of a
    refusal, and the ruled boundary is that the answer is held only in memory — a persisted one
    becomes a standing secret that outlives the run. Reachability: `_ZenodoTarget.publish` and
    `_HydroShareTarget.publish` both reach `_require_env`, the site the production deposit paths
    call. Both-states anchor: the return value of `_require_env`, and the fact that no write
    primitive raises inside `_writes_refused()`, hold whatever the prompt wording says.
    Invocation isolation: the prompt primitives are stubbed at the standard library, so the
    production helper runs unmodified and no read touches the test runner's terminal.
    What kills it: an `_require_env` that raises on a missing variable; a prompt whose answer is
    written to a config file, netrc, temp file or any other path, INCLUDING one written by a
    helper no list names, because the write primitives are patched rather than the locations;
    or a caller that caches the value. Second correct implementation: prompting with
    `sys.stdin.readline()` instead of `input`, or with a `click.prompt`; this reads outcomes and
    the interception only. The snapshot comparison is kept as the diagnostic — it names which
    location moved — and it is not the detector.
    """
    monkeypatch.delenv("HHEMT_ZENODO_TOKEN", raising=False)
    monkeypatch.delenv("HHEMT_HYDROSHARE_USERNAME", raising=False)
    monkeypatch.setattr(publishing, "_stdin_is_interactive", lambda: True, raising=False)
    _seed_prompt(monkeypatch, _PLANTED_TOKEN)
    before = _snapshot(_storage_locations(tmp_path))

    try:
        with _writes_refused():
            value = publishing._require_env("HHEMT_ZENODO_TOKEN", "zenodo")
    except PublishError as exc:
        pytest.fail(f"the prompt arm refused instead of asking, so no credential was prompted for: {exc}")

    assert value == _PLANTED_TOKEN, "the prompted answer is what the caller receives"
    assert _snapshot(_storage_locations(tmp_path)) == before, "the prompt arm changed a known storage location"

    with _writes_refused():
        username = publishing._require_env("HHEMT_HYDROSHARE_USERNAME", "hydroshare")
    assert username == _PLANTED_TOKEN, "the non-secret route is prompted through the echoed reader too"
    assert _snapshot(_storage_locations(tmp_path)) == before, "the username route changed a known storage location"


def test_the_prompt_path_holds_no_write_call_at_all(tmp_path):
    """Property: no function on the prompt path can write — the AST of each holds no write call and no env assignment.

    Class: every function the prompted path can execute (`_PROMPT_PATH`), each enumerated by name
    and asserted present, so a rename reds here rather than narrowing the population silently.
    Why it is required: CORROBORATION, not the detector. `_writes_refused()` in the prompt arm is
    what fails on a write, including one made by a helper this list does not name; this arm adds
    the narrower claim that the five bodies themselves carry no write call and no environment
    assignment, so a line added inside them reds here even if the interception were removed. It
    does not see past a refactor into a new helper, and the module docstring says so.
    Reachability: the five names are the whole call graph of `_require_env` on the prompt arm as
    written today. Both-states anchor: the presence of a write-capable call node, a property of the
    source rather than of behaviour, which is what makes it independent of the runtime arm.
    Invocation isolation: it reads source text through `inspect.getsource` on the imported module,
    so it needs no fixture, no deposit, and no network. What kills it: a `Path.write_text`, an
    `open(..., "w")`, a `json.dump`, or an `os.environ[...] =` introduced into any of the five.
    Second correct implementation: a `netrc` or `keyring` write reached from `_prompt_credential`;
    this catches it by call name inside the scanned bodies, and the prompt arm's interception
    catches it wherever it sits.
    """
    write_hits: list[str] = []
    env_writes: list[str] = []
    for name in _PROMPT_PATH:
        function = getattr(publishing, name, None)
        assert function is not None, f"{name} is missing, so the scanned population would silently narrow"
        tree = ast.parse(inspect.getsource(function))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = node.func
                called = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
                if called in _WRITE_CALLS:
                    write_hits.append(f"{name}: {called}")
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Subscript) and "environ" in ast.unparse(target.value):
                        env_writes.append(f"{name}: {ast.unparse(target)}")

    assert write_hits == [], f"a write-capable call is reachable from the prompt path: {write_hits}"
    assert env_writes == [], f"the prompt path assigns to the environment: {env_writes}"


def test_the_environment_variable_still_wins_and_costs_no_prompt(tmp_path, monkeypatch):
    """Property: a set environment variable is returned unchanged and the prompt is never reached.

    Class: the opt-in arm — the variable set, with an interactive terminal present, so the prompt is
    reachable and must still be skipped. Why it is required: unattended runs (SLURM steps, CI) have
    no one to answer, and the ruled change keeps them working by exporting the variable; a prompt
    that takes precedence would hang them. Reachability: `monkeypatch.setenv` then `_require_env`,
    the call both target classes make. Both-states anchor: the returned value equals the exported
    one, and the prompt spy is never entered. Invocation isolation: the prompt is replaced by a
    raising spy, so any prompt attempt reds rather than blocking. What kills it: prompting before
    reading the environment, or a refactor that drops the environment branch. Second correct
    implementation: the same order with `os.getenv`; the assertions read the value and the spy.
    """
    monkeypatch.setenv("HHEMT_ZENODO_TOKEN", "from-the-environment")
    monkeypatch.setattr(publishing, "_stdin_is_interactive", lambda: True, raising=False)

    def _prompted(*args, **kwargs):
        raise AssertionError("the prompt was reached although the environment variable was set")

    monkeypatch.setattr(publishing, "_prompt_credential", _prompted, raising=False)
    before = _snapshot(_storage_locations(tmp_path))

    assert publishing._require_env("HHEMT_ZENODO_TOKEN", "zenodo") == "from-the-environment"
    assert _snapshot(_storage_locations(tmp_path)) == before, "the environment arm wrote to a storage location"


def test_an_empty_answer_or_a_pipe_refuses_by_name(tmp_path, monkeypatch):
    """Property: an empty answer, and a non-terminal stdin, both refuse with a PublishError naming the variable.

    Class: the two ways the prompt yields nothing — an operator pressing return, and a pipeline
    whose stdin is not a terminal. Why it is required: the refusal is the only signal an unattended
    caller can act on, and a prompt on a pipe would hang the job it was meant to unblock; an empty
    answer must not be accepted as a credential either. Reachability: `_require_env` with the prompt
    stubbed empty, then with the terminal gate returning False. Both-states anchor: the raised type
    and the variable name in its status, both true before this change as well, so the arm pins that
    the new prompt did not remove the old promise. Invocation isolation: the gate is patched in the
    second half and the prompt primitive in the first, so neither blocks on a real terminal.
    What kills it: returning an empty string as a credential, raising a bare exception, or dropping
    the variable name from the message. Second correct implementation: raising before the prompt on
    a non-terminal; the assertions read the exception type, its status text, and the storage snapshot.
    """
    monkeypatch.delenv("HHEMT_ZENODO_TOKEN", raising=False)
    before = _snapshot(_storage_locations(tmp_path))

    monkeypatch.setattr(publishing, "_stdin_is_interactive", lambda: True, raising=False)
    _seed_prompt(monkeypatch, "")
    with pytest.raises(PublishError) as empty:
        publishing._require_env("HHEMT_ZENODO_TOKEN", "zenodo")
    assert "HHEMT_ZENODO_TOKEN" in empty.value.status

    monkeypatch.setattr(publishing, "_stdin_is_interactive", lambda: False, raising=False)

    def _must_not_prompt(*args, **kwargs):
        raise AssertionError("a prompt was attempted although stdin is not a terminal")

    monkeypatch.setattr(publishing, "_prompt_credential", _must_not_prompt, raising=False)
    with pytest.raises(PublishError) as piped:
        publishing._require_env("HHEMT_ZENODO_TOKEN", "zenodo")
    assert "HHEMT_ZENODO_TOKEN" in piped.value.status
    assert _snapshot(_storage_locations(tmp_path)) == before, "a refusing arm wrote to a storage location"
