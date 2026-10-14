#!/usr/bin/env bash
# Run the AO-Shaping test suite in a resource-capped container.
#
# The host has 384 cores and two RTX 4090s. An unconstrained `pytest` there
# saturates both: numeric libraries size their thread pools at 384, and the
# vendored OOPAO hands `multiprocessing.cpu_count()` to pyfftw and joblib. The
# first symptom is not a failed test but the shell itself timing out.
#
# This wrapper is the sanctioned entry point. It enforces:
#   --cpus=16          CFS quota. Note this does NOT lower `nproc`.
#   --pids-limit=2048  the real backstop: threads count as pids in cgroup v2,
#                      so an explosion fails with EAGAIN instead of taking the
#                      host down.
#   --memory=32g       container RAM ceiling, swap disabled so the cgroup OOM
#                      killer fires here rather than thrashing the host.
#   --gpus device=0    one RTX 4090. VRAM is NOT capped by docker; the 4 GB
#                      ceiling is applied by conftest.py to the caching
#                      allocators, which is where the repo's tensors live.
#
# data/ is shadowed by a staged copy that deliberately omits data/debug.
# That directory holds 18 GB of pickles with individual files up to 3.42 GB,
# and the tests that consume it gate on the directory EXISTING -- they must see
# it absent so they skip, which an empty tmpfs cannot achieve because is_dir()
# would still be true. Everything else under data/ (the r50 wiring CSV, the
# calibration JSONs, ccd_configs) is copied in, since those tests read it.
# Refresh the staging area with scripts/stage_test_data.sh after new fixtures
# land in data/.
#
# Usage:
#   ./run_pytest.sh                          # whole suite
#   ./run_pytest.sh tests/ao_shaping/runners  # a subtree
#   ./run_pytest.sh -k "not slow" -- -x      # extra args after --
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${AO_IMAGE:-ao-shaping-test:gpu}"
WORKDIR_IN_CONTAINER=/workspace
CPU_LIMIT="${AO_CPUS:-16}"
MEM_LIMIT="${AO_MEMORY:-32g}"
PIDS_LIMIT="${AO_PIDS:-2048}"
GPU="${AO_GPU:-device=0}"
# CWD must be writable: tests legitimately write scratch files relative to
# it (e.g. savefig("cmp.png")), which the read-only repo mount forbids. The repo
# itself stays read-only; only this scratch directory is not.
AO_SCRATCH="${AO_SCRATCH:-$HOME/ao-pytest-scratch}"
mkdir -p "$AO_SCRATCH"

# data/ is copied fresh into the scratch area and mounted read-write (some tests
# write data/wfs_configs). It cannot be the host's staging directory directly:
# the container runs as root, so anything it leaves behind is undeletable from
# the host. Notably data/debug -- its mere EXISTENCE is what gates the real-data
# tests, so a polluted copy silently turns those skips into failures. Rebuilding
# from a pristine source on every run makes that impossible.
SEED_DATA="${AO_TEST_DATA_SEED:-$HOME/ao-testdata}"
AO_TEST_DATA="${AO_TEST_DATA:-$AO_SCRATCH/data}"
if [ "${AO_TEST_DATA_REFRESH:-1}" = "1" ]; then
    "$REPO/scripts/stage_test_data.sh" >/dev/null
    # A previous run may have left root-owned files here, which the host cannot
    # delete; docker can, so clean through it rather than failing the run.
    rm -rf "$AO_TEST_DATA" 2>/dev/null ||         docker run --rm -v "$AO_SCRATCH":/scratch alpine \
            sh -c "rm -rf /scratch/data" >/dev/null 2>&1
    cp -r "$SEED_DATA" "$AO_TEST_DATA"
fi
mkdir -p "$AO_TEST_DATA"

# Opt-in switches the suite itself reads (golden-file regeneration,
# fault-handler dumps). They must be forwarded explicitly, otherwise setting
# them on the host silently does nothing inside the container -- which looks
# exactly like the regeneration having failed.
AO_PASSTHROUGH_ENV=""
for var in AO_UPDATE_HELP_GOLDEN AO_CLI_CONTRACT_UPDATE AO_PROBE_HELP_UPDATE AO_FAULT_HANDLER AO_FAULT_SECS; do
    if [ -n "${!var:-}" ]; then
        AO_PASSTHROUGH_ENV="${AO_PASSTHROUGH_ENV:+$AO_PASSTHROUGH_ENV }${var}=${!var}"
    fi
done

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required; the suite must not run on the host." >&2
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image $IMAGE not found. Build it with:" >&2
    echo "  docker build -f docker/Dockerfile.test -t $IMAGE ." >&2
    exit 1
fi

# Anything after `--` is forwarded verbatim; everything before it is a target.
if [ "${1:-}" = "--" ]; then
    shift
    TARGETS=()
    PASSTHROUGH=("$@")
else
    TARGETS=()
    PASSTHROUGH=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do
        TARGETS+=("$1")
        shift
    done
    if [ "${1:-}" = "--" ]; then
        shift
        PASSTHROUGH=("$@")
    fi
fi
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=(tests)
fi

# The container's CWD is a scratch directory (tests write scratch files
# relative to it), so relative targets would not resolve. Anchor them to the
# in-container repo path instead.
ABS_TARGETS=()
for target in "${TARGETS[@]}"; do
    case "$target" in
        /*) ABS_TARGETS+=("$target") ;;
        -*) ABS_TARGETS+=("$target") ;;
        *)  ABS_TARGETS+=("$WORKDIR_IN_CONTAINER/$target") ;;
    esac
done

# Regenerating a help golden writes into the repo, which the read-only mount
# forbids. Opt in explicitly rather than mounting the tree read-write by default
# -- a plain test run has no business writing to the checkout.
MOUNT_MODE=ro
if [ -n "${AO_UPDATE_HELP_GOLDEN:-}${AO_CLI_CONTRACT_UPDATE:-}${AO_PROBE_HELP_UPDATE:-}" ]; then
    MOUNT_MODE=rw
fi

exec docker run --rm \
    --name ao-pytest \
    --cpus="$CPU_LIMIT" \
    --memory="$MEM_LIMIT" \
    --memory-swap="$MEM_LIMIT" \
    --pids-limit="$PIDS_LIMIT" \
    --gpus "\"$GPU\"" \
    --shm-size=1g \
    -v "$REPO:$WORKDIR_IN_CONTAINER:$MOUNT_MODE" \
    --tmpfs "$WORKDIR_IN_CONTAINER/logs:size=512m" \
    -v "$AO_TEST_DATA:$WORKDIR_IN_CONTAINER/data" \
    --tmpfs /tmp:size=4g,mode=1777 \
    -e AO_CPU_THREADS="${AO_CPU_THREADS:-16}" \
    -e AO_VRAM_LIMIT_MB="${AO_VRAM_LIMIT_MB:-4096}" \
    ${AO_PASSTHROUGH_ENV:+-e $AO_PASSTHROUGH_ENV} \
    -e PYTHONPATH="$WORKDIR_IN_CONTAINER/src:$WORKDIR_IN_CONTAINER:$WORKDIR_IN_CONTAINER/scripts" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e NUMBA_CACHE_DIR=/tmp/numba-cache \
    -e MPLBACKEND=Agg \
    -e PYTHONUNBUFFERED=1 \
    -w "$AO_SCRATCH" \
    -e AO_PYTEST_ROOT="$WORKDIR_IN_CONTAINER" \
    "$IMAGE" \
    python -m pytest \
        -p no:cacheprovider \
        --rootdir "$WORKDIR_IN_CONTAINER" \
        "${ABS_TARGETS[@]}" \
        ${PASSTHROUGH+"${PASSTHROUGH[@]}"}