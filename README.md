# GPUMD-SUS2

Experimental SUS2 v1.1 inference support for GPUMD.

This repository is an overlay on top of upstream GPUMD, not a full GPUMD fork. It contains the core files needed to add the `MTP`/SUS2 potential backend, the SUS2 topology code-generation utility, and build notes used on the A100 cluster.

## Current Scope

- SUS2 model format: `version = 1.1.0`
- GPUMD potential token: model files beginning with `MTP`
- Default radial evaluation: GPU lookup table for all supported radial basis types
- Default LUT spacing: `dr = 1.0e-4 A`, matching the LAMMPS table convention
- Runtime LUT controls: `sus2_lut_dr=...`, `sus2_lut_span=...`, `SUS2_GPUMD_LUT_DR`, `SUS2_GPUMD_LUT_SPAN`
- Optional memory-saving reverse-gradient workspace: `sus2_grad_float=1` or `SUS2_GPUMD_GRAD_FLOAT=1`
- Optimized tensor path: automatic for standard `lLkK` `alpha_index_basic` layouts up to `l4k4`
- Experimental product-graph controls: `sus2_fused_graph=...`, `sus2_local_product_graph=...`
- Supported radial basis types:
  - `RBJacobi_sss`, `RBJacobi_sss_lmp`
  - `RBJacobi_sss_noweight`, `RBJacobi_sss_noweight_lmp`
  - `RBChebyshev_sss`, `RBChebyshev_sss_lmp`
  - `RBLaguerre_log1p`, `RBLaguerre_log1p_lmp`
  - `RBLaguerre_log1p_noenv`, `RBLaguerre_log1p_noenv_lmp`
  - `RBLaguerre_log1p_pos`, `RBLaguerre_log1p_pos_lmp`

## Core Files

```text
src/force/sus2_v11.cu
src/force/sus2_v11.cuh
src/force/force.cu
src/model/read_xyz.cu
tools/sus2_v11_codegen.py
```

`sus2_v11.cu/.cuh` implements the SUS2 backend. `force.cu` wires GPUMD potential dispatch to SUS2 when the potential file starts with `MTP`. `read_xyz.cu` lets GPUMD read element symbols for SUS2 models from `run.in`. `sus2_v11_codegen.py` generates and caches model-topology-specific CUDA product-graph cores.

## Quick Install

See [INSTALL.md](INSTALL.md).

## Example GPUMD `run.in`

```text
potential p.mtp H C N I Pb
velocity 200 seed 9174
ensemble npt_mttk temp 200 200 aniso 0.0001 0.0001 tperiod 100 pperiod 1000
time_step 0.5
dump_thermo 100
run 2000
```

To override the table spacing:

```text
potential p.mtp H C N I Pb sus2_lut_dr=0.0001
```

or:

```bash
export SUS2_GPUMD_LUT_DR=0.0001
```

To test the optional float moment-gradient workspace:

```text
potential p.mtp H C N I Pb sus2_grad_float=1
```

or:

```bash
export SUS2_GPUMD_GRAD_FLOAT=1
```

This keeps the SUS2 forward moment values in double precision but stores the reverse-mode moment-gradient workspace in float. The default remains double.

To test the more aggressive NEP-like float path:

```text
potential p.mtp H C N I Pb sus2_float=1
```

or:

```bash
export SUS2_GPUMD_FLOAT=1
```

This stores SUS2 moments and reverse gradients in float, evaluates local moment arithmetic in float, and still writes GPUMD energy/force/virial outputs to the existing double arrays.

To disable or probe the current safe fast paths:

```text
potential p.mtp Cu Zr sus2_float=1 sus2_l3k3_force_grad_cache=0
potential p.mtp Cu Zr sus2_float=1 sus2_tensor_force_grad_cache=0
potential p.mtp Cu Zr sus2_float=1 sus2_fused_graph=0
potential p.mtp Cu Zr sus2_float=1 sus2_local_product_graph=1
```

`sus2_local_product_graph=1` is kept as an experimental model-topology path. It preserves the same product DAG and reverse-mode chain rule, but on the tested Cu-Zr l3k3 model it was slower because the local per-thread graph workspace increased register/local-memory pressure.

## Codegen Cache

The topology code generator uses a persistent cache:

```bash
tools/sus2_v11_codegen.py model.mtp \
  --out-dir codex_codegen/model \
  --cache-dir codegen_cache/sus2_v11 \
  --chunk-size 512 \
  --compile \
  --arch sm_80
```

On the tested l4k3 model, the first cache miss took about `158-160 s`; a later cache hit took about `0.2 s`.

The cache key is based on the product topology, not element identities or fitted coefficients. It includes `L`, `scaling_map`, `radial_funcs_count`, `alpha_index_basic`, compressed `alpha_index_times`, compressed `alpha_moment_mapping`, and active moment count.

## Validation Snapshot

The experimental A100 build used:

```text
gcc/12.2.0
cuda/12.4
CUDA_ARCH=sm_80
```

Representative results:

```text
Chebyshev_sss load smoke: GPUMD_RC=0, LUT=65002, dr=0.0001 A
1.18M atom MA/Jacobi NPT2000: 4.12257e6 atom-step/s
l4k3 codegen cache miss: 158.16 s
l4k3 codegen cache hit: 0.19 s
l3k3 98k grad-float test: 4.357e6 -> 5.195e6 atom-step/s, GPU process memory 4124 -> 3366 MiB
Cu-Zr l3k3 1.024M sus2_float opt pass: 1.24453e7 -> 1.69611e7 atom-step/s, GPU process memory about 9.6 GiB
MA l4k3 98k tensor cache test: 1.50229e6 -> 1.75504e6 atom-step/s
```

More detailed notes are in [docs/sus2-gpumd-v1.1-porting-notes.md](docs/sus2-gpumd-v1.1-porting-notes.md).
