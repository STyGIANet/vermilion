#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root and target directories relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLOS_DIR="${REPO_ROOT}/src/clos"
DC_DIR="${CLOS_DIR}/datacenter"

echo "[1/3] Cleaning datacenter (${DC_DIR})"
( cd "${DC_DIR}" && make clean )

echo "[2/3] Cleaning and building clos ("${CLOS_DIR}" )"
( cd "${CLOS_DIR}" && make clean && make -j )

echo "[3/3] Building datacenter ("${DC_DIR}" )"
( cd "${DC_DIR}" && make -j )

echo "Done."