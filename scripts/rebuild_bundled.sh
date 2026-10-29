#!/usr/bin/env bash
# Rebuild the redistributable dictionary bundled in the wheel.
#
# Sources: packaged seed data (MIT) + khmerlbdict_*.tsv (MIT).
# The full research build (khPOS/ALT/RAC) is intentionally NOT used here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RAW_DIR="${RAW_DIR:-data/raw}"
BUILD_DIR="${BUILD_DIR:-data/build/permissive}"
INTERIM_DIR="${INTERIM_DIR:-data/interim/permissive}"
RESOURCE_DIC="src/kroch_chhmar/resources/system_khmer.dic"

if ! compgen -G "$RAW_DIR/khmerlbdict_*.tsv" > /dev/null; then
    echo "error: no khmerlbdict_*.tsv files found in $RAW_DIR" >&2
    echo "Fetch the MIT-licensed wordlists first:" >&2
    echo "  uv run kroch-chhmar fetch khmerlbdict" >&2
    exit 1
fi

echo "Preparing permissive dictionary in $BUILD_DIR ..."
uv run kroch-chhmar prepare \
    --profile permissive \
    --seed \
    --raw \
    --no-auto-holdout \
    --raw-dir "$RAW_DIR" \
    --build-dir "$BUILD_DIR" \
    --interim-dir "$INTERIM_DIR"

echo "Compiling binary dictionary ..."
uv run kroch-chhmar build --no-prepare --build-dir "$BUILD_DIR"

echo "Installing bundled dictionary -> $RESOURCE_DIC"
cp "$BUILD_DIR/system_khmer.dic" "$RESOURCE_DIC"

echo "Done. SHA-256:"
shasum -a 256 "$RESOURCE_DIC"
