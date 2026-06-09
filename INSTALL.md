# GPUMD-SUS2-SH Build Guide

This guide installs the SH-only SUS2 overlay into a GPUMD source tree, then
builds the `gpumd` executable.

The tested cluster build uses:

```text
gcc/12.2.0
cuda/12.4
A100 GPU target: sm_80
```

## 1. Prepare GPUMD

Start from either a clean upstream GPUMD checkout or a copied GPUMD-SUS2 runtime
tree. If the target tree already has the old moment-tensor SUS2 files, the
overlay script removes them so `sus2_v11.cu` is not compiled into this SH
build.

## 2. Apply The Overlay

```bash
/path/to/GPUMD-SUS2-SH/scripts/apply_overlay.sh \
  /path/to/GPUMD-SUS2-SH \
  /path/to/GPUMD-SUS2-SH-work
```

This copies:

```text
src/force/sus2_sh.cu
src/force/sus2_sh.cuh
src/force/sus2_zbl_common.cuh
src/force/force.cu
src/model/read_xyz.cu
```

and removes stale tensor-only files if present:

```text
src/force/sus2_v11.cu
src/force/sus2_v11.cuh
tools/sus2_v11_codegen.py
```

## 3. Build GPUMD

On the tested A100 cluster:

```bash
cd /path/to/GPUMD-SUS2-SH-work/src
module purge
module load gcc/12.2.0 cuda/12.4 cmake/3.25.2
make -j2 gpumd CUDA_ARCH=-arch=sm_80
```

The output binary is:

```text
/path/to/GPUMD-SUS2-SH-work/src/gpumd
```

## 4. Use A SUS2-SH Model

Use a SUS2-SH MTP model file whose first token is `MTP` and whose model section
contains:

```text
potential_tag = SUS2-SH
version = 1.1.0
```

In `run.in`, pass the model file followed by exactly `species_count` element
symbols:

```text
potential p.mtp Cu Zr
```

The first backend supports `RBChebyshev_sss`, `scaling_map = LK`,
`sh_l_max <= 4`, direct radial recurrence, and float moments by default.
If the model file contains `zbl_enabled = 1`, GPUMD-SUS2-SH automatically adds
the same NEP-style universal ZBL term used by the SUS2-SH trainer/LAMMPS
interface. Pair cutoffs are precomputed from `zbl_atomic_numbers`,
`zbl_inner`, `zbl_outer`, and optional `zbl_typewise_cutoff_factor` at model
load time.
