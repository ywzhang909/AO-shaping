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
    # 清理必须真的生效。容器以 root 运行，上一轮会在 scratch/data 里留下 root 属主
    # 的目录，普通用户的 rm -rf 删不掉；原先写成 `rm -rf … || true` 把失败吞掉，
    # 随后的 cp 就在**仍存在**的目标目录里嵌套出一份 data/ao-testdata/，夹具因此全部
    # "消失"，表现是十几个 r50 测试报 "CSV not found: …/1300-5-enriched.csv"，与被测
    # 代码毫无关系（2026-10-24 实测踩过）。删不掉就用一次性 root 容器删；再不行就
    # 大声失败，绝不带着被污染的 stage 继续跑。
    rm -rf "$SCRATCH/data" 2>/dev/null || true
    if [ -e "$SCRATCH/data" ]; then
        docker run --rm -v "$SCRATCH/data:/victim" --entrypoint sh "$IMAGE" \
            -c 'rm -rf /victim/* /victim/.[!.]*' \
            || { echo "FATAL: cannot clean $SCRATCH/data (root-owned leftovers?)" >&2; exit 1; }
    fi
    mkdir -p "$SCRATCH/data"
    # `src/.` -> `dst/`：把**内容**拷进目录，与 dst 是否存在无关（`cp -r src dst`
    # 在 dst 已存在时会嵌套成 dst/src，这正是上面那次假红的机制）。
    cp -a "$SEED_DATA/." "$SCRATCH/data/"
    seed_n=$(find "$SEED_DATA" -maxdepth 1 -mindepth 1 | wc -l)
    staged_n=$(find "$SCRATCH/data" -maxdepth 1 -mindepth 1 | wc -l)
    if [ "$staged_n" -ne "$seed_n" ] || [ "$staged_n" -eq 0 ]; then
        echo "FATAL: staged $staged_n entries into $SCRATCH/data but $SEED_DATA has" \
             "$seed_n -- fixtures would silently be missing inside the container." >&2
        exit 1
    fi
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
# Only one area is excluded here and in pyproject's addopts, and the two lists
# must stay in sync -- pinned by
# tests/ao_shaping/test_conventions.py::test_pytest_excludes_match_across_configs
# (they drifted apart once already: pyproject was narrowed while this list still
# carried the whole drivers tree, which silently kept the narrowed discovery off).
#   (1) ml (65 GB hwdataset heavy I/O + torch).
# The real Streamlit AppTest (tests/ao_shaping/gui/slm/test_zernike_abort_repro.py)
# is NOT excluded any more: it was excluded as "the only real AppTest, needs web
# automation or it hangs", but that is no longer true -- it is a browserless
# AppTest over the real GUI functions and it passes in ~2 s (3 passed, measured
# 2026-10-26). It had been reproducing a Zernike-selectbox abort that the current
# code no longer hits. It is the only test that renders the real GUI runtime
# end-to-end, so excluding it cost real coverage.
# drivers/ is fully INCLUDED now, per container measurement: ccd 204 passed /
# 25 skipped, sim 394 / 2 in 63.33 s, wfs 110 / 11, slm 158 / 127 in 7.95 s,
# adc 49 in 2.84 s, hardware 37 skipped in 0.06 s, dm 139 / 5 in 40.22 s.
# dm had looked disqualifying at 560.83 s, but that was an unenforced `hardware`
# mark (the skip hook lived only in the slm/ and hardware/ conftests) letting a
# fixture sweep 26 unreachable controllers -- 260 s of setup per test. Enforcing
# the mark in drivers/conftest.py fixed both the cost and 2 spurious failures.
# Excluding drivers/ wholesale is what let drivers/ccd rot to 6 failed + 1 error
# without CI seeing it (rot fixed in 29001826).
# Corpus / memory exemptions. One entry per reason, each with a comment, and never a
# whole top-level area: tests/ao_shaping/scripts/test_pytest_workflow_shards_cover_tree.py
# rejects a dangling entry, an entry without a reason, and any entry that would blank an
# entire area (the blanket ml ignore below is what let three authored-red ml tests sit
# invisible for a day -- TODO.md 8.9 items 102/103/104).
EXCLUDES=(
    # tests/ao_shaping/ml/gsnet_debug cannot join the `ml` shard, but not for the
    # reason recorded when this entry was first written. Measured, 4 files, one
    # container, per-file mode: at 6g the invocation exits 1 with only 3 summaries --
    # test_gsnet_offline.py is killed and prints nothing (no FAILED, no error). At 8g
    # and at 12g the same four files exit 0 with 4 summaries (22 / 78 / 80 passed + 7
    # skipped / 83). And at 6g that same file is green on its own (80 passed, 7
    # skipped) in its own container. So the ceiling does help -- the earlier note that
    # "raising the ceiling is not the fix" was borrowed from optimizer/wfless without
    # measuring it here, and is withdrawn.
    #
    # The entry stays only because the exemption mechanism is unconditional: EXCLUDES
    # is appended to every run_pytest_ci.sh invocation, so a dedicated 8g shard for
    # this area would collect zero items and report PASS while testing nothing -- the
    # exact failure this file's guards exist to prevent. Fixing that needs a
    # per-shard exemption switch plus a per-shard memory envelope (TODO.md 8.9 item
    # 107); the runner's own RAM is why 6g was chosen for the shards in the first
    # place, so the fix is a narrower envelope, not a global raise.
    "--ignore=$WORKDIR_IN_CONTAINER/tests/ao_shaping/ml/gsnet_debug"
)

# Interpreter and options are kept in separate arrays on purpose: the per-file mode
# re-invokes pytest *inside* the container, and if the interpreter words were part of
# the options array the inner call would become `python -m pytest python -m pytest …`
# (pytest then treats `python`, `-m`, `pytest` as targets -> "no tests collected").
PYTEST_INTERP=( python -m pytest )
PYTEST_OPTS=(
    -p no:cacheprovider
    --rootdir "$WORKDIR_IN_CONTAINER"
    "${EXCLUDES[@]}"
)

# AO_TEST_PER_FILE=1 gives every test file its own pytest process.
#
# Why the mode exists (measured 2026-10-24, not assumed): optimizer/wfless -- 1074
# items -- is SIGKILLed (exit 137) as a single process at BOTH 6g and 12g, yet every
# one of its 55 files passes alone inside 4g. The symptom is in-process accumulation
# across files, so raising the ceiling does not help and a hand-maintained grouping
# would silently stop covering any file someone adds later. The list is therefore
# produced by --collect-only, which already respects EXCLUDES.
#
# PASSTHROUGH flags are not forwarded in this mode (the per-file runs use -q).
if [ "${AO_TEST_PER_FILE:-0}" = "1" ]; then
    CONTAINER_CMD=(
        bash -c '
            set -u
            base=()
            while [ $# -gt 0 ] && [ "$1" != "--" ]; do base+=("$1"); shift; done
            [ $# -gt 0 ] && shift
            collected=$(python -m pytest "${base[@]}" "$@" --collect-only -q) || {
                printf "FATAL: --collect-only failed; refusing to run an unknown set.\n" >&2
                printf "%s\n" "$collected" >&2
                exit 1
            }
            files=$(printf "%s\n" "$collected" | sed -n "s/::.*//p" | sort -u | grep .)
            count=$(printf "%s\n" "$files" | grep -c . || true)
            if [ "$count" -eq 0 ]; then
                printf "FATAL: per-file mode collected 0 test files -- a shard that runs\n" >&2
                printf "nothing must never look green.\n" >&2
                exit 1
            fi
            printf "[ao-pytest-ci] per-file mode: %d test file(s), one process each\n" "$count"
            status=0
            for f in $files; do
                # --collect-only prints paths relative to --rootdir, but CWD here is the
                # scratch dir, so a relative path resolves to nothing. Re-anchor it.
                case "$f" in
                    /*) path="$f" ;;
                    *)  path="$AO_PYTEST_ROOT/$f" ;;
                esac
                python -m pytest "${base[@]}" -q "$path" || status=1
            done
            exit $status
        '
        _ "${PYTEST_OPTS[@]}" -- "${TARGETS[@]}"
    )
else
    CONTAINER_CMD=( "${PYTEST_INTERP[@]}" "${PYTEST_OPTS[@]}" "${PASSTHROUGH[@]}" "${TARGETS[@]}" )
fi

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
    "${CONTAINER_CMD[@]}"
