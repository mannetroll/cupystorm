#!/usr/bin/env sh
# Copy the Qt-free solver modules from ../cupystorm into web/py/cupystorm
# so the browser app can fetch them and mount them in the Pyodide filesystem.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SRC="${SCRIPT_DIR}/../cupystorm"
DEST="${SCRIPT_DIR}/py/cupystorm"

mkdir -p "$DEST"
cp "${SRC}/turbo_simulator.py" "${SRC}/turbo_wrapper.py" "$DEST/"
: > "${DEST}/__init__.py"

echo "Synced solver sources into ${DEST}"
