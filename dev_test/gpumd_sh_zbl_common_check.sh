#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/.codex_tmp"
mkdir -p "$BUILD_DIR"

c++ -std=c++11 -O2 -I"$ROOT" \
  "$ROOT/codex_tests/gpumd_sh_zbl_common_check.cpp" \
  -o "$BUILD_DIR/gpumd_sh_zbl_common_check"
"$BUILD_DIR/gpumd_sh_zbl_common_check"
