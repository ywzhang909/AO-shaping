#!/usr/bin/env bash
# Stage the subset of data/ the test suite needs, excluding the huge corpora.
#
# Why this exists: data/ holds ~87 GB. The tests that consume data/debug gate on
# the directory EXISTING, so they must see it absent to skip -- shadowing it
# with an empty tmpfs does not work, because is_dir() stays true and the tests
# then run against nothing.
#
# Everything small IS copied, because tests both read and write here: for example
# gui/r50/test_r50_voltage_send.py writes data/wfs_configs/wfs. The staging area
# is scratch space mounted read-write, never a source of truth.
#
# Excluded (size, and no test needs them):
#   slm_dual_spot 62 GB  debug 18 GB  zernike_pred 5.8 GB
#   fouriergsnet_sim 1.3 GB  fouriergsnet_sim_replay 286 MB
#
# Run this after adding a fixture under data/ that a test depends on.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${AO_TEST_DATA:-$HOME/ao-testdata}"

EXCLUDE_RE='^(debug|slm_dual_spot|zernike_pred|fouriergsnet_sim|fouriergsnet_sim_replay)$'

rm -rf "$STAGE"
mkdir -p "$STAGE"

copied=0
skipped=0
for path in "$REPO"/data/*; do
    name="$(basename "$path")"
    if [[ "$name" =~ $EXCLUDE_RE ]]; then
        echo "  excluded $name ($(du -sh "$path" 2>/dev/null | cut -f1))"
        skipped=$((skipped + 1))
        continue
    fi
    cp -r "$path" "$STAGE/"
    copied=$((copied + 1))
done

for name in debug slm_dual_spot zernike_pred fouriergsnet_sim fouriergsnet_sim_replay; do
    if [ -e "$STAGE/$name" ]; then
        echo "ERROR: staging area contains $name; the bulk tests would run" >&2
        exit 1
    fi
done

echo "staged $STAGE ($(du -sh "$STAGE" | cut -f1)): $copied entries copied, $skipped excluded"