#!/usr/bin/env bash
# Run the test suite inside a container, for CI.
#
# Same reason as run_pytest.sh: the numeric stack sizes every thread pool at the
# core count it can see, so an unconstrained run saturates the machine and can
# freeze it. The caps below are what make the run survivable.
#
# The differences from run_pytest.sh are all about the CI environment:
#
#   * CPU image, not the GPU one. A CI runner has no GPU, and the GPU image
#     pulls multi-GB CUDA wheels it could never run.
#   * No --gpus flag, for the same reason.
#   * The repo is checked out fresh by Actions, so there is no staged data/
#     to shadow. data/ is not in git, and the tests that consume it gate on
#     the directory being absent, so an empty mount is what they expect.
#   * AO_PYTEST_ROOT is set so conftest.py's host guard recognises this as a
#     containerised run.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${AO_IMAGE:-ao-shaping-test:cpu}"
WORKDIR_IN_CONTAINER=/workspace
CPU_LIMIT="${AO_CPUS:-4}"
# 16g, not 8g: the simulator tests hold a ~10 GB working set (measured with
# `docker stats` on the wfless subtree, which plateaus at 10.2 GB and then
# releases it). At 8g the cgroup OOM killer fires mid-suite and takes the run
# down at ~25%. GitHub's standard runner has 16 GB, so the container limit
# cannot exceed that anyway.
MEM_LIMIT="${AO_MEMORY:-16g}"
PIDS_LIMIT="${AO_PIDS:-1024}"
SCRATCH="${AO_SCRATCH:-$HOME/ao-pytest-scratch}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required; the suite must not run on the host." >&2
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image $IMAGE not found. Build it with:" >&2
    echo "  docker build -f docker/Dockerfile.test.cpu -t $IMAGE ." >&2
    exit 1
fi

mkdir -p "$SCRATCH"

# data/ is staged, not empty. The repo's data/ holds the fixtures several
# tests read (the r50 channel CSV, the wfs_configs, the ccd_configs), and
# the R50 tests fail outright without them. The staging script copies
# everything except the multi-GB corpora, so the checkout's data/ stays
# untouched and the container gets a writable copy.
#
# It is rebuilt on every run: the container runs as root, so a previous run
# leaves root-owned files in there that the next run cannot overwrite.
SEED_DATA="${AO_TEST_DATA_SEED:-$HOME/ao-testdata}"
if [ "${AO_TEST_DATA_REFRESH:-1}" = "1" ]; then
    "$REPO/scripts/build/stage_test_data.sh" >/dev/null
    rm -rf "$SCRATCH/data" 2>/dev/null || true
    cp -r "$SEED_DATA" "$SCRATCH/data"
fi
# --tmpfs /workspace/logs 在这里行不通：/workspace 是 ro bind，docker 建
# tmpfs 挂载点要在底层 overlay 上 mkdir，ro 下必报 "read-only file system"
# (run 39697306909 的 exit 125)。data/ 早已用 scratch bind 绕开，logs 同理：
# 挂宿主机 scratch 目录，docker 不需要在 ro 底层建任何挂载点。
mkdir -p "$SCRATCH/data" "$SCRATCH/logs"

# .git is mounted separately, and git is told to trust it.
#
# Two separate problems, both silent:
#
# 1. The mount. .git is normally inside the repo mount, but
#    test_conventions.py and test_report_provenance.py shell out to git
#    ("git ls-files docs"), and in a linked worktree .git is a POINTER FILE
#    whose gitdir lives outside the tree -- so git finds no repository and the
#    tests fail with "git ls-files docs returned nothing". In a worktree git
#    needs BOTH the worktree's gitdir and the common dir it points at (the
#    object store), so both are mounted at the paths the pointers name.
# 2. The ownership. The container runs as root while the checkout belongs to
#    the invoking user, so git refuses the repository with "detected dubious
#    ownership" -- again silently, from the test's point of view. safe.directory
#    has to cover the checkout AND anything the pointer resolves to, hence "*".
#
# The mounts are read-only: the suite must not be able to write to the
# checkout's object store.
GIT_MOUNT=()
if [ -d "$REPO/.git" ]; then
    GIT_MOUNT=(-v "$REPO/.git:$WORKDIR_IN_CONTAINER/.git:ro")
elif [ -f "$REPO/.git" ]; then
    # A linked worktree: .git names a gitdir elsewhere, and that gitdir in turn
    # points at the COMMON dir (../..) which holds the object store. Git needs
    # both, so both are mounted at the paths the pointers name -- the pointer
    # inside the container must resolve to a real repository.
    WORKTREE_GITDIR="$(sed -nE 's/^gitdir: *//p' "$REPO/.git")"
    if [ -n "$WORKTREE_GITDIR" ] && [ -d "$WORKTREE_GITDIR" ]; then
        # The commondir file holds a bare path (e.g. "../.."), not a
        # "commondir: <path>" line, so it is read as-is and resolved
        # relative to the gitdir that contains it.
        COMMON_GITDIR="$(cat "$WORKTREE_GITDIR/commondir" 2>/dev/null || true)"
        if [ -n "$COMMON_GITDIR" ]; then
            case "$COMMON_GITDIR" in
                /*) ;;
                *) COMMON_GITDIR="$(cd "$WORKTREE_GITDIR/$COMMON_GITDIR" && pwd)" ;;
            esac
        else
            COMMON_GITDIR="$WORKTREE_GITDIR"
        fi
        GIT_MOUNT=(
            -v "$WORKTREE_GITDIR:$WORKTREE_GITDIR:ro"
            -v "$COMMON_GITDIR:$COMMON_GITDIR:ro"
        )
    fi
fi

# The CWD is the scratch directory, not the repo: tests legitimately write
# scratch files relative to it (e.g. savefig("cmp.png")), which the read-only
# repo mount forbids. The targets are therefore absolute paths into the
# container, because a relative one would resolve against the scratch dir.
# (pyproject's testpaths is relative, so it cannot be used as-is here.)
# Split the arguments: pytest's own options (-k, -m, -x, --collect-only, ...)
# are not paths, and passing a flag where a path is expected makes pytest
# collect nothing. Options are forwarded verbatim; whatever is left is a target
# path, and the default is the whole suite.
TARGETS=()
PASSTHROUGH=()
for arg in "$@"; do
    case "$arg" in
        -*) PASSTHROUGH+=("$arg") ;;
        /*) TARGETS+=("$arg") ;;
        *)  TARGETS+=("$WORKDIR_IN_CONTAINER/$arg") ;;
    esac
done
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=("$WORKDIR_IN_CONTAINER/tests")
fi

# The excludes from pyproject's addopts are RELATIVE paths ("tests/ao_shaping/
# drivers"), and pytest resolves them against the current working directory.
# The CWD here is the scratch dir, so they would silently match nothing and the
# hardware / ml / gui-hang areas would be collected -- the ml area alone is the
# 65 GB hwdataset corpus, and its multiproc + mmap has hung full-tree runs
# before. Re-state them as absolute paths so they still exclude.
#
# The same three areas are excluded, for the same reasons, as in pyproject:
#   (1) the only real Streamlit AppTest (needs web automation or it hangs),
#   (2) hardware drivers (SLM/CCD SDK waits have no timeout protection),
#   (3) ml (65 GB hwdataset heavy I/O + torch).
EXCLUDES=(
    "--ignore=$WORKDIR_IN_CONTAINER/tests/ao_shaping/gui/slm/test_zernike_abort_repro.py"
    "--ignore=$WORKDIR_IN_CONTAINER/tests/ao_shaping/drivers"
    "--ignore=$WORKDIR_IN_CONTAINER/tests/ao_shaping/ml"
)

exec docker run --rm \
    --name ao-pytest-ci \
    --cpus="$CPU_LIMIT" \
    --memory="$MEM_LIMIT" \
    --memory-swap="$MEM_LIMIT" \
    --pids-limit="$PIDS_LIMIT" \
    --shm-size=1g \
    -v "$REPO:$WORKDIR_IN_CONTAINER:ro" \
    "${GIT_MOUNT[@]}" \
    -v "$SCRATCH/logs:$WORKDIR_IN_CONTAINER/logs" \
    -v "$SCRATCH/data:$WORKDIR_IN_CONTAINER/data" \
    --tmpfs /tmp:size=4g,mode=1777 \
    -e AO_PYTEST_ROOT="$WORKDIR_IN_CONTAINER" \
    -e AO_CPU_THREADS="${AO_CPU_THREADS:-16}" \
    -e AO_VRAM_LIMIT_MB="${AO_VRAM_LIMIT_MB:-4096}" \
    -e PYTHONPATH="$WORKDIR_IN_CONTAINER/src:$WORKDIR_IN_CONTAINER:$WORKDIR_IN_CONTAINER/scripts" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e NUMBA_CACHE_DIR=/tmp/numba-cache \
    -e MPLBACKEND=Agg \
    -e PYTHONUNBUFFERED=1 \
    -e GIT_CONFIG_COUNT=2 \
    -e GIT_CONFIG_KEY_0=safe.directory \
    -e GIT_CONFIG_VALUE_0="$WORKDIR_IN_CONTAINER" \
    -e GIT_CONFIG_KEY_1=safe.directory \
    -e GIT_CONFIG_VALUE_1="*" \
    -w "$SCRATCH" \
    "$IMAGE" \
    python -m pytest \
        -p no:cacheprovider \
        --rootdir "$WORKDIR_IN_CONTAINER" \
        "${EXCLUDES[@]}" \
        "${PASSTHROUGH[@]}" \
        "${TARGETS[@]}"
