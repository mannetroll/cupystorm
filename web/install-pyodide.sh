#!/usr/bin/env sh
set -eu

VERSION="${PYODIDE_VERSION:-0.29.4}"
ARCHIVE="pyodide-${VERSION}.tar.bz2"
URL="https://github.com/pyodide/pyodide/releases/download/${VERSION}/${ARCHIVE}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEST="${SCRIPT_DIR}/pyodide"
TMPDIR=$(mktemp -d)

cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT INT TERM

echo "Downloading Pyodide ${VERSION}..."
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --output "${TMPDIR}/${ARCHIVE}" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${TMPDIR}/${ARCHIVE}" "$URL"
else
  echo "curl or wget is required" >&2
  exit 1
fi

echo "Extracting..."
mkdir -p "${TMPDIR}/extract"
tar -xjf "${TMPDIR}/${ARCHIVE}" -C "${TMPDIR}/extract"

SOURCE_DIR=$(find "${TMPDIR}/extract" -type f -name pyodide.js -exec dirname {} \; | head -n 1)
if [ -z "$SOURCE_DIR" ]; then
  echo "pyodide.js was not found in ${ARCHIVE}" >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST"
cp -R "${SOURCE_DIR}/." "$DEST/"

echo "Installed Pyodide ${VERSION} into ${DEST}"
