#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /path/to/GPUMD-SUS2 /path/to/upstream-GPUMD" >&2
  exit 2
fi

overlay_root=$1
gpumd_root=$2

required_files=(
  "src/force/sus2_v11.cu"
  "src/force/sus2_v11.cuh"
  "src/force/force.cu"
  "src/model/read_xyz.cu"
  "tools/sus2_v11_codegen.py"
)

for rel in "${required_files[@]}"; do
  if [[ ! -f "${overlay_root}/${rel}" ]]; then
    echo "Missing overlay file: ${overlay_root}/${rel}" >&2
    exit 1
  fi
done

if [[ ! -d "${gpumd_root}/src/force" || ! -d "${gpumd_root}/src/model" ]]; then
  echo "The target does not look like a GPUMD source tree: ${gpumd_root}" >&2
  exit 1
fi

mkdir -p "${gpumd_root}/tools"

for rel in "${required_files[@]}"; do
  mkdir -p "${gpumd_root}/$(dirname "${rel}")"
  cp "${overlay_root}/${rel}" "${gpumd_root}/${rel}"
  echo "installed ${rel}"
done

chmod +x "${gpumd_root}/tools/sus2_v11_codegen.py"

cat <<EOF

SUS2 overlay installed into:
  ${gpumd_root}

Build example:
  cd ${gpumd_root}/src
  module purge
  module load gcc/12.2.0 cuda/12.4
  make -j2 gpumd CUDA_ARCH=-arch=sm_80
EOF

