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

# Step 7: print the environment, or run the command bound to this tree.
if [ "$print_env" -eq 1 ]; then
    printf '%s\n' "$env"
    exit 0
fi
PYTHONPATH="$tree/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
: "${HHEMT_TY_ENV:=$env}"
export HHEMT_TY_ENV
exec uv run --no-sync "$@"
