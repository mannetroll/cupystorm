#!/usr/bin/env sh
# Deploy the web app to a web server, copying only the Pyodide files the app
# actually needs (core runtime + numpy/scipy wheels, ~30 MB instead of the
# full ~460 MB distribution).
#
#   ./deploy.sh user@host:/var/drtobbe/storm
set -eu

DEST="${1:?usage: deploy.sh [user@]host:/path/to/docroot}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

./sync-sources.sh

# --relative preserves the py/ and pyodide/ subdirectories on the server.
rsync -av --relative \
  index.html styles.css app.js web_sim.py \
  py/cupystorm/__init__.py \
  py/cupystorm/turbo_simulator.py \
  py/cupystorm/turbo_wrapper.py \
  pyodide/pyodide.js \
  pyodide/pyodide.mjs \
  pyodide/pyodide.asm.js \
  pyodide/pyodide.asm.wasm \
  pyodide/python_stdlib.zip \
  pyodide/pyodide-lock.json \
  pyodide/numpy-*.whl \
  pyodide/scipy-*.whl \
  "${DEST}/"
