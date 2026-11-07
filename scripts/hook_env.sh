#!/bin/sh
# Run a command in a SHARED uv environment that lives outside every tree.
#
# Usage:
#   scripts/hook_env.sh [--python VERSION] [--extra NAME]... [--print-env] [--] COMMAND [ARG]...
#
# WHAT PROBLEM THIS SOLVES. Every local pre-commit hook and several justfile recipes ran
# `uv run`, which creates and fills a `.venv/` inside whichever tree it runs in: about 1.6 GiB
# per tree on a laptop whose disk is nearly full, never removed. This wrapper keeps the
# dependencies in one environment per (uv.lock, Python version, extras) under a root held
# outside every tree, so no commit and no recipe creates an environment in a tree.
#
# WHY THE PROJECT IS NOT INSTALLED INTO THE SHARED ENVIRONMENT. An environment that installed
# hhemt itself would carry one editable pointer to one tree. Measured: a second tree's `uv run`
# against the same environment re-pointed it to the second tree, and uv held its environment
# lock only while syncing, so two trees committing at once could each grade their own files
# against the other's code. `--no-install-project` keeps the environment dependency-only.
#
# WHY THE TREE IS BOUND BY IMPORT PATH. The command runs with `PYTHONPATH={tree}/src` first,
# set per process, so each tree imports its own package and two trees cannot cross. A reader
# of installed metadata (`importlib.metadata.version("hhemt")`) finds no distribution and
# degrades rather than fails at the three sites known to read it.
#
# WHY THE DEFAULT EXTRAS ARE `test docs`. The `ty` hooks need `test`; the pytest and docs
# recipes need `docs`. One fixed default removes the class of failure that differs by extra,
# and lets hooks and recipes share one environment. The Python version and the extras are
# part of the key because a different interpreter or extra set is a different environment
# (`testall` runs 3.11 and 3.12 as two keys).
#
# WHY A STALE LOCK STILL FAILS. The sync runs `--locked`, so a `uv.lock` that disagrees with
# `pyproject.toml` is refused with uv's own message instead of being silently rewritten.
#
# REMOVAL IS DETERMINISTIC, NOT A CLEANUP THAT MUST RUN. Every run removes each key directory
# under the root that has gone unused for more than 14 days. Nothing per-commit exists to be
# left behind, so a killed commit leaves at most a part-written shared key, repaired by the
# next sync (step 4) or aged out (step 6).
#
# WHY EVERY COMMAND ALSO GETS ITS TREE'S DISTRIBUTION RECORD AND CONSOLE SCRIPT. Not installing
# the project leaves `importlib.metadata` with no `hhemt` distribution, and one reader of it
# RAISES: `provenance._default_code_repository` raises PackageNotFoundError by design, so
# `consolidate` fails under a bare wrapper on a fresh clone (measured on Rivanna: 12 failed and 1
# error through the wrapper against 47 passed under an editable install). Six other readers
# degrade silently, one of them (`_reprex_gate`, which reads `Author-email`) changing a verdict, and
# the `hhemt` console script is absent. A tree that happens to carry a gitignored
# `src/hhemt.egg-info` hides all of this, which is why it appeared only on a fresh clone. Step 4b
# therefore asks the project's own declared build backend (`setuptools.build_meta`, through its PEP
# 517 `prepare_metadata_for_build_wheel` hook, which writes nothing into the tree) for the real
# `.dist-info`, keeps it in an immutable content-keyed record inside the key directory, and step 7
# puts the record BEFORE `{tree}/src` on `PYTHONPATH` so a stale `src/*.egg-info` cannot win.
# Deliberately NOT a hand-written METADATA (a second implementation of what the backend computes,
# missing `Author-email` and `entry_points.txt`) and NOT an install of the project into the shared
# environment (the cross-tree re-pointing measured above). The backend reaches the environment as
# a dependency of a core dependency; if it is ever absent the record step refuses with exit 2.
#
# --print-env syncs the environment and prints its absolute path without running a command,
# which is how an editor is pointed at it (`{path}/bin/python`).
#
# Environment: HHEMT_HOOK_ENV_ROOT overrides the root (default
# ${XDG_CACHE_HOME:-$HOME/.cache}/hhemt/hook_envs). HHEMT_TY_ENV, if already set, is kept.
set -eu

# Step 1: the tree and the root.
tree=$(git rev-parse --show-toplevel)
root=${HHEMT_HOOK_ENV_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/hhemt/hook_envs}
# A relative root is made absolute against the CALLER's directory here, before any step changes
# directory. Step 4 syncs from inside the tree, so a relative root left as it is would build the
# environment under the tree while steps 5 and 6 look for it under the caller's directory.
case $root in
    /*) ;;
    *) root=$(pwd)/$root ;;
esac

# Step 2: the wrapper's own leading options, then the key.
python=3.12
extras=
print_env=0
while [ $# -gt 0 ]; do
    case $1 in
        --python)
            [ $# -ge 2 ] || { echo "hook_env.sh: --python needs a version" >&2; exit 2; }
            python=$2
            shift 2
            ;;
        --extra)
            [ $# -ge 2 ] || { echo "hook_env.sh: --extra needs a name" >&2; exit 2; }
            extras="$extras $2"
            shift 2
            ;;
        --print-env)
            print_env=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done
[ -n "$extras" ] || extras="test docs"

if [ ! -f "$tree/uv.lock" ]; then
    echo "hook_env.sh: no uv.lock at $tree; the shared environment is keyed by it" >&2
    exit 2
fi
extras_key=$(printf '%s\n' $extras | sort | paste -sd, -)
if command -v sha256sum >/dev/null 2>&1; then
    hasher="sha256sum"
else
    hasher="shasum -a 256"
fi
key=$({
    cat "$tree/uv.lock"
    printf '\0python=%s\0extras=%s' "$python" "$extras_key"
} | $hasher | cut -c1-16)

# Step 3: the environment path; no inherited virtualenv.
env=$root/$key
unset VIRTUAL_ENV
UV_PROJECT_ENVIRONMENT=$env
export UV_PROJECT_ENVIRONMENT

# Step 4: sync, repairing a stray directory. This is the only step that writes an environment,
# and it writes only under $root.
extra_flags=
for name in $extras; do
    extra_flags="$extra_flags --extra $name"
done
sync_env() {
    # shellcheck disable=SC2086
    (cd "$tree" && uv sync --locked --no-install-project --python "$python" $extra_flags --quiet)
}
status=0
sync_env || status=$?
if [ "$status" -ne 0 ] && [ ! -e "$env/pyvenv.cfg" ]; then
    # REPAIR-BEGIN
    # uv refuses a key directory that holds no Python environment (exit 2); one whose
    # pyvenv.cfg remains is repaired by uv itself. Removal runs only after this process's
    # own sync has failed, never before every sync.
    rm -rf -- "$env"
    status=0
    sync_env || status=$?
    # REPAIR-END
fi
[ "$status" -eq 0 ] || exit "$status"

# Step 4b: the tree's distribution record and console scripts, built by the project's own backend.
# Nothing is written to the tree: the backend's metadata hook writes only into its output directory.
# Backend check: the backend must be importable in the environment's own interpreter.
if ! "$env/bin/python" -c "import setuptools.build_meta" >/dev/null 2>&1; then
    echo "hook_env.sh: setuptools.build_meta (the project's build backend, from [build-system]) is not importable in $env; the distribution record cannot be produced without it" >&2
    exit 2
fi

# Record key: the record's key is a digest of pyproject.toml and of the readme and license files it names.
meta_files=$("$env/bin/python" -c '
import glob, sys, tomllib
tree = sys.argv[1]
with open(tree + "/pyproject.toml", "rb") as handle:
    project = tomllib.load(handle).get("project", {})
readme = project.get("readme")
names = []
if isinstance(readme, str):
    names.append(readme)
elif isinstance(readme, dict) and "file" in readme:
    names.append(readme["file"])
for pattern in project.get("license-files", []):
    names.extend(sorted(glob.glob(pattern, root_dir=tree, recursive=True)))
print("\n".join(names))
' "$tree")
meta_key=$({
    cat "$tree/pyproject.toml"
    printf '%s\n' "$meta_files" | while IFS= read -r meta_name; do
        [ -n "$meta_name" ] && [ -f "$tree/$meta_name" ] || continue
        printf '%s\0' "$meta_name"
        cat "$tree/$meta_name"
    done
} | $hasher | cut -c1-16)
record=$env/hhemt_meta/$meta_key

has_record() {
    for candidate in "$1"/*.dist-info/METADATA; do
        [ -e "$candidate" ] && return 0
    done
    return 1
}

# Record build: build the record once per meta key, publishing it by a plain POSIX rename (no GNU-only
# `mv` option exists on macOS). If two processes both pass the test, POSIX `mv` places the loser's
# directory INSIDE the winner's as $record/record; the winner's record stays intact at the top and
# the nested copy is never on PYTHONPATH or PATH.
if ! has_record "$record"; then
    tmp=$env/hhemt_meta/.tmp.$$
    mkdir -p "$tmp/out" "$tmp/record/bin"
    # The backend logs to stdout, so the name of the dist-info it returns is written to a file.
    (cd "$tree" && "$env/bin/python" -c "import setuptools.build_meta as b, sys; open(sys.argv[2], 'w').write(b.prepare_metadata_for_build_wheel(sys.argv[1]))" "$tmp/out" "$tmp/built" >/dev/null 2>"$tmp/log") || {
        cat "$tmp/log" >&2
        echo "hook_env.sh: the build backend failed to produce a distribution record for $tree" >&2
        exit 2
    }
    built=$(cat "$tmp/built")
    mv "$tmp/out/$built" "$tmp/record/"
    # One shim per console script, each importing that entry's module and calling its callable
    # as pip's console scripts do. Deliberately NOT `python -m {package}`: two entries of one
    # package (`toy` and `toy-admin`) would then run the same `__main__`.
    "$env/bin/python" -c '
import configparser, os, sys
record, dist_info = sys.argv[1], sys.argv[2]
parser = configparser.ConfigParser()
parser.optionxform = str
parser.read(os.path.join(record, dist_info, "entry_points.txt"))
if parser.has_section("console_scripts"):
    for name, target in parser.items("console_scripts"):
        module, _, attribute = target.split("[")[0].strip().partition(":")
        code = (
            "import functools, importlib, sys; "
            "f = functools.reduce(getattr, %r.split(\x27.\x27), importlib.import_module(%r)); "
            "sys.argv[0] = %r; sys.exit(f())"
        ) % (attribute.strip(), module.strip(), name)
        path = os.path.join(record, "bin", name)
        with open(path, "w") as handle:
            handle.write("#!/bin/sh\nexec python -c \x27" + code.replace("\x27", "\x22") + "\x27 \x22$@\x22\n")
        os.chmod(path, 0o755)
' "$tmp/record" "$built"
    [ -e "$record" ] || mv "$tmp/record" "$record"
    rm -rf -- "$tmp"
    if ! has_record "$record"; then
        echo "hook_env.sh: no distribution record at $record after the build" >&2
        exit 2
    fi
fi

# Mark the record used.
touch "$record/.last_used"

# Step 5: mark use.
touch "$env/.last_used"

# Step 6: remove key directories unused for more than 14 days. Names come from the
# directory listing and must be 16 hex digits; nothing outside $root is reachable.
for dir in "$root"/*/; do
    dir=${dir%/}
    [ -d "$dir" ] || continue
    [ -L "$dir" ] && continue
    name=${dir##*/}
    [ "${#name}" -eq 16 ] || continue
    case $name in *[!0-9a-f]*) continue ;; esac
    [ "$name" = "$key" ] && continue
    if [ -e "$dir/.last_used" ]; then
        age_path=$dir/.last_used
    elif [ -e "$dir/pyvenv.cfg" ]; then
        age_path=$dir/pyvenv.cfg
    else
        age_path=$dir
    fi
    if [ -n "$(find "$age_path" -prune -mtime +14)" ]; then
        rm -rf -- "$root/$name"
    fi
done

# Step 6, within the current key: age out every record other than this one (and any killed
# `.tmp.*` build) by the same three-tier rule.
for entry in "$env/hhemt_meta"/* "$env/hhemt_meta"/.tmp.*; do
    [ -d "$entry" ] || continue
    [ -L "$entry" ] && continue
    [ "${entry##*/}" = "$meta_key" ] && continue
    if [ -e "$entry/.last_used" ]; then
        age_path=$entry/.last_used
    elif has_record "$entry"; then
        age_path=$(ls -d "$entry"/*.dist-info/METADATA | head -n 1)
    else
        age_path=$entry
    fi
    if [ -n "$(find "$age_path" -prune -mtime +14)" ]; then
        rm -rf -- "$entry"
    fi
done

# Step 7: print the environment, or run the command bound to this tree.
if [ "$print_env" -eq 1 ]; then
    printf '%s\n' "$env"
    exit 0
fi
PYTHONPATH="$record:$tree/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
PATH="$record/bin:$PATH"
export PATH
: "${HHEMT_TY_ENV:=$env}"
export HHEMT_TY_ENV
exec uv run --no-sync "$@"
