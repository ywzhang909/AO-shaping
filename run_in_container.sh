#!/usr/bin/env bash
# Run a command inside the AO-Shaping test container.
#
# `run_pytest.sh` is the sanctioned entry point for the **test suite**.  This
# script is its sibling for everything else that must not touch the host
# directly: the offline experiment drivers under `scripts/slm/`, a report
# generator, a one-off simulation.
#
# Why a container at all: the host has 384 cores.  Numeric libraries size their
# thread pools at the core count, and the vendored OOPAO hands
# `multiprocessing.cpu_count()` to pyfftw and joblib.  On the host that turns
# one simulation into a ~390-thread storm that starves the shell before it
# finishes.  The container's CFS quota (`--cpus`) plus the pinned thread-count
# environment variables keep the fan-out bounded.
#
# The repo is mounted READ-ONLY by default, exactly like `run_pytest.sh`: a
# command has no business writing into the checkout.  Pass `--rw` when the
# command genuinely must write into the tree (a report generator), and
# `--out-root DIR` to give the command a writable working directory (the
# container's CWD is a scratch dir, so relative paths would not resolve).
#
# Usage:
#   ./run_in_container.sh python scripts/slm/calibrate_gsnet_2f.py --skip-network
#   ./run_in_container.sh --out-root /workspace/report/x -- python -c 'print(1)'
#   ./run_in_container.sh --rw -- python scripts/slm/generate_foo_report.py
#
# Environment overrides (same names as run_pytest.sh):
#   AO_IMAGE   image name            (default ao-shaping-test:gpu)
#   AO_CPUS    CFS quota             (default 16)
#   AO_MEMORY  RAM ceiling           (default 32g)
#   AO_PIDS    pids limit            (default 2048)
#   AO_GPU     --gpus value          (default device=0)
#   AO_THREADS thread-pool size      (default = AO_CPUS)
#   AO_UID    run as this uid:gid    (default = the invoking user's)
#
# The image ships no CJK font, so a matplotlib figure with Chinese labels
# renders as tofu boxes.  The host has them; mounting them read-only costs
# nothing and keeps every render inside the container.  MPLCONFIGDIR points at
# the tmpfs so matplotlib's font cache is rebuilt per run instead of being
# served from a stale one.
#
# HOME must be writable: the image's default HOME is "/", and running as a
# non-root user (AO_UID) means CuPy's cache directory cannot be created there.
# /tmp is a 1777 tmpfs, so it is writable by anyone.
#
# The image's default user is root, and root-owned files left in the checkout
# are undeletable from the host.  AO_UID defaults to the invoking user so a
# `--rw` run leaves the tree owned by whoever can actually clean it up.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${AO_IMAGE:-ao-shaping-test:gpu}"
WORKDIR_IN_CONTAINER=/workspace
CPU_LIMIT="${AO_CPUS:-16}"
MEM_LIMIT="${AO_MEMORY:-32g}"
PIDS_LIMIT="${AO_PIDS:-2048}"
GPU="${AO_GPU:-device=0}"
THREADS="${AO_THREADS:-$CPU_LIMIT}"
AO_UID="${AO_UID:-$(id -u):$(id -g)}"
AO_SCRATCH="${AO_SCRATCH:-$HOME/ao-pytest-scratch}"
MOUNT_MODE=ro
OUT_ROOT="$AO_SCRATCH"
PASSTHROUGH=()

while [ $# -gt 0 ]; do
    case "$1" in
        --rw) MOUNT_MODE=rw; shift ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --) shift; PASSTHROUGH=("$@"); break ;;
        *) break ;;
    esac
done

if [ ${#PASSTHROUGH[@]} -eq 0 ]; then
    echo "usage: $0 [--rw] [--out-root DIR] -- <command> [args...]" >&2
    exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required; this must not run on the host." >&2
    exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image $IMAGE not found. Build it with:" >&2
    echo "  docker build -f docker/Dockerfile.test -t $IMAGE ." >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

# The thread-count env vars must be set BEFORE any wheel is imported, and they
# must agree with the CFS quota or the pools stay sized at the host core count.
exec docker run --rm \
    --name ao-container \
    --user "$AO_UID" \
    --cpus="$CPU_LIMIT" \
    --memory="$MEM_LIMIT" \
    --memory-swap="$MEM_LIMIT" \
    --pids-limit="$PIDS_LIMIT" \
    --gpus "\"$GPU\"" \
    --shm-size=1g \
    -v "$REPO:$WORKDIR_IN_CONTAINER:$MOUNT_MODE" \
    --tmpfs "$WORKDIR_IN_CONTAINER/logs:size=512m" \
    -v "$OUT_ROOT:/out" \
    -v /usr/share/fonts:/usr/share/fonts:ro \
    -v "$HOME/.local/share/fonts:$HOME/.local/share/fonts:ro" \
    --tmpfs /tmp:size=4g,mode=1777 \
    -e AO_CPU_THREADS="$THREADS" \
    -e AO_VRAM_LIMIT_MB="${AO_VRAM_LIMIT_MB:-4096}" \
    -e OMP_NUM_THREADS="$THREADS" \
    -e OPENBLAS_NUM_THREADS="$THREADS" \
    -e MKL_NUM_THREADS="$THREADS" \
    -e NUMEXPR_NUM_THREADS="$THREADS" \
    -e VECLIB_MAXIMUM_THREADS="$THREADS" \
    -e NUMBA_NUM_THREADS="$THREADS" \
    -e PYTHONPATH="$WORKDIR_IN_CONTAINER/src:$WORKDIR_IN_CONTAINER:$WORKDIR_IN_CONTAINER/scripts" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e NUMBA_CACHE_DIR=/tmp/numba-cache \
    -e HOME=/tmp \
    -e CUPY_CACHE_DIR=/tmp/cupy-cache \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    -e MPLBACKEND=Agg \
    -e PYTHONUNBUFFERED=1 \
    -e GIT_CONFIG_COUNT=1 \
    -e GIT_CONFIG_KEY_0=safe.directory \
    -e "GIT_CONFIG_VALUE_0=$WORKDIR_IN_CONTAINER" \
    -w "$OUT_ROOT" \
    "$IMAGE" \
    "${PASSTHROUGH[@]}"
