# GPUMD-SUS2 Build Guide

This guide installs the SUS2 v1.1 GPUMD overlay into an upstream GPUMD source tree, then builds the `gpumd` executable.

The tested cluster build uses:

```text
gcc/12.2.0
cuda/12.4
A100 GPU target: sm_80
```

## 1. Prepare Upstream GPUMD

Clone GPUMD into a working directory:

```bash
git clone https://github.com/brucefan1983/GPUMD.git GPUMD-SUS2-work
cd GPUMD-SUS2-work
```

The current overlay was developed against the upstream GPUMD tree used in the SUS2 porting work. If you want the closest tested baseline, start from a clean GPUMD checkout before applying the overlay.

## 2. Apply The SUS2 Overlay

From the `GPUMD-SUS2` overlay repository:

```bash
/path/to/GPUMD-SUS2/scripts/apply_overlay.sh /path/to/GPUMD-SUS2 /path/to/GPUMD-SUS2-work
```

This copies:

```text
src/force/sus2_v11.cu
src/force/sus2_v11.cuh
src/force/force.cu
src/model/read_xyz.cu
tools/sus2_v11_codegen.py
```

Manual copy is also possible:

```bash
cp /path/to/GPUMD-SUS2/src/force/sus2_v11.cu  /path/to/GPUMD-SUS2-work/src/force/
cp /path/to/GPUMD-SUS2/src/force/sus2_v11.cuh /path/to/GPUMD-SUS2-work/src/force/
cp /path/to/GPUMD-SUS2/src/force/force.cu     /path/to/GPUMD-SUS2-work/src/force/
cp /path/to/GPUMD-SUS2/src/model/read_xyz.cu  /path/to/GPUMD-SUS2-work/src/model/
mkdir -p /path/to/GPUMD-SUS2-work/tools
cp /path/to/GPUMD-SUS2/tools/sus2_v11_codegen.py /path/to/GPUMD-SUS2-work/tools/
chmod +x /path/to/GPUMD-SUS2-work/tools/sus2_v11_codegen.py
```

## 3. Build GPUMD

On the tested A100 cluster:

```bash
cd /path/to/GPUMD-SUS2-work/src
module purge
module load gcc/12.2.0 cuda/12.4
make -j2 gpumd CUDA_ARCH=-arch=sm_80
```

Use a small `-j` value on login nodes. For production rebuilds, prefer compiling on a GPU/build node according to local cluster policy.

The output binary is:

```text
/path/to/GPUMD-SUS2-work/src/gpumd
```

## 4. Use A SUS2 Model

Use a SUS2 MTP model file whose first token is `MTP` and whose model section has `version = 1.1.0`.

In `run.in`, pass the model file followed by exactly `species_count` element symbols:

```text
potential p.mtp H C N I Pb
velocity 200 seed 9174
ensemble npt_mttk temp 200 200 aniso 0.0001 0.0001 tperiod 100 pperiod 1000
time_step 0.5
dump_thermo 100
run 2000
```

The element symbols are needed because SUS2 MTP files do not store the GPUMD element list directly.

## 5. Lookup Table Controls

By default, all supported SUS2 radial basis types use a GPU lookup table with:

```text
dr = 1.0e-4 A
lut_span = ceil(max_dist / dr) + 2
```

This matches the LAMMPS table-spacing convention used in the SUS2 interface work.

You can override the spacing in `run.in`:

```text
potential p.mtp H C N I Pb sus2_lut_dr=0.0001
```

or through environment variables:

```bash
export SUS2_GPUMD_LUT_DR=0.0001
export SUS2_GPUMD_LUT_SPAN=65002
```

`sus2_lut_span` and `SUS2_GPUMD_LUT_SPAN` override the number of table entries directly. In normal use, prefer setting `sus2_lut_dr`.

## 6. Optional Float Gradient Workspace

The default SUS2 reverse-mode moment-gradient workspace is double precision. To reduce memory traffic and memory footprint, enable the experimental float-gradient workspace:

```text
potential p.mtp H C N I Pb sus2_grad_float=1
```

or:

```bash
export SUS2_GPUMD_GRAD_FLOAT=1
```

Only the reverse-gradient workspace is changed to float; forward moments remain double. On the tested l3k3 98k MA case, this improved speed from `4.357e6` to `5.195e6 atom-step/s` and reduced GPUMD process GPU memory from about `4124 MiB` to `3366 MiB`.

## 7. Optional NEP-Like Float Moment Path

For a more aggressive NEP-inspired mixed-precision test, enable:

```text
potential p.mtp H C N I Pb sus2_float=1
```

or:

```bash
export SUS2_GPUMD_FLOAT=1
```

This implies float reverse gradients and additionally stores forward moments, fitted scalar coefficients used by the kernel, and local moment/force arithmetic in float. GPUMD positions and final potential/force/virial arrays remain double.

Static first-frame 98k MA l3k3 check versus default double:

```text
energy_diff = 0.25911 eV total = 0.0026 meV/atom
force_MAE = 1.55e-5 eV/A
force_RMSE = 2.54e-5 eV/A
force_max_abs = 2.84e-4 eV/A
```

NPT 2000-step 98k MA l3k3 performance on one A100:

```text
double:      4.381e6 atom-step/s, 4124 MiB GPUMD process peak
grad_float:  5.090e6 atom-step/s, 3366 MiB GPUMD process peak
sus2_float:  6.488e6 atom-step/s, 2608 MiB GPUMD process peak
```

The `sus2_float=1` path is experimental and should be accepted only after first-frame energy/force/virial checks for the target model.

## 8. Supported Radial Basis Types

The current GPUMD-SUS2 v1.1 overlay supports:

```text
RBJacobi_sss
RBJacobi_sss_lmp
RBJacobi_sss_noweight
RBJacobi_sss_noweight_lmp
RBChebyshev_sss
RBChebyshev_sss_lmp
RBLaguerre_log1p
RBLaguerre_log1p_lmp
RBLaguerre_log1p_noenv
RBLaguerre_log1p_noenv_lmp
RBLaguerre_log1p_pos
RBLaguerre_log1p_pos_lmp
```

The `_lmp` suffix is accepted for compatibility. In GPUMD-SUS2, supported radial types default to lookup-table evaluation regardless of the suffix.

## 8. Product-Graph Codegen Cache

The code generator can produce a model-topology-specific CUDA core and cache it for reuse:

```bash
cd /path/to/GPUMD-SUS2-work
tools/sus2_v11_codegen.py /path/to/current.mtp \
  --out-dir codex_codegen/current \
  --cache-dir codegen_cache/sus2_v11 \
  --chunk-size 512 \
  --compile \
  --arch sm_80
```

The first compile of a large l4k3 topology on the tested A100 environment takes about `160 s`. The tool prints progress while `nvcc` and `ptxas` are working. If the same product topology is reused later, the cached object is reused and startup is typically sub-second.

The cache key depends on the SUS2 product topology:

```text
L
scaling_map
radial_funcs_count
alpha_index_basic
alpha_index_times
alpha_moment_mapping
active moment count
```

The key intentionally ignores element names, fitted coefficients, radial coefficients, scaling coefficients, and the radial basis type. Those quantities affect radial values or fitted weights, not the product graph itself.

## 9. Validation Snapshot

The current formal SUS2-GPUMD binary on the development cluster is:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/src/gpumd
```

Representative validation results:

```text
Chebyshev_sss load smoke: GPUMD_RC=0, LUT=65002, dr=0.0001 A
1.18M atom MA/Jacobi NPT2000: 4.12257e6 atom-step/s
l4k3 codegen cache miss: 158.16 s
l4k3 codegen cache hit: 0.19 s
```

For development notes and known limitations, see:

```text
docs/sus2-gpumd-v1.1-porting-notes.md
```
