# GPUMD-SUS2-SH

Experimental SUS2-SH inference support for GPUMD.

This repository is the spherical-harmonic line of the GPUMD SUS2 interface. It
is split from the moment-tensor `GPUMD-SUS2` project so the SH model reader,
real-spherical-harmonic basic moments, CG contraction path, and later
matrix-style kernels can evolve without destabilizing the older tensor backend.

Initial design and implementation notes are tracked in
[docs/gpumd-sus2-sh-interface-plan.md](docs/gpumd-sus2-sh-interface-plan.md).

## Status

This initial commit preserves the mature moment-tensor backend as the starting
point and documents the SUS2-SH design. The next implementation step is a new
`SUS2_SH` backend selected by `potential_tag = SUS2-SH`.

## Inherited Tensor Backend

- SUS2 model format: `version = 1.1.0`
- GPUMD potential token: model files beginning with `MTP`
- Default radial evaluation: direct basis recurrence for supported radial basis types
- LUT fallback spacing: `dr = 1.0e-4 A`, matching the LAMMPS table convention
- Runtime LUT controls: `sus2_lut_dr=...`, `sus2_lut_span=...`, `SUS2_GPUMD_LUT_DR`, `SUS2_GPUMD_LUT_SPAN`
- Default precision path: NEP-like float moments/gradients/local arithmetic; use `sus2_float=0` or `SUS2_GPUMD_FLOAT=0` for double moments
- Default force path: self-force buffer enabled; use `sus2_force_self_buffer=0` or `SUS2_GPUMD_FORCE_SELF_BUFFER=0` to disable it
- Optimized tensor path: automatic for standard `lLkK` `alpha_index_basic` layouts up to `l4k4`, using programmatic rank-block tensor contractions
- Product-graph controls: graph-specific grouped product path is automatic for safe supported models; use `sus2_graph_specific=0` or `SUS2_GPUMD_GRAPH_SPECIFIC_PRODUCT=0` to force the mature product graph
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

## Cluster Usage

For the maintained server paths, A100 runtime modules, GPUMD `run.in` syntax,
SUS2 precision modes, radial lookup-table controls, and LSF job template, see
[docs/sus2-gpumd-usage.md](docs/sus2-gpumd-usage.md).

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

To force radial lookup tables instead of direct basis recurrence:

```text
potential p.mtp Cu Zr sus2_radial_direct=0
```

or:

```bash
export SUS2_GPUMD_RADIAL_DIRECT=0
```

The default is direct recurrence for supported radial families with
`radial_basis_size <= 16`; larger radial bases fall back to LUT mode unless
direct mode is requested explicitly. LUT mode remains available for debugging
and parity checks.

To force double moment values and gradients instead of the default NEP-like
float path:

```text
potential p.mtp H C N I Pb sus2_float=0
```

or:

```bash
export SUS2_GPUMD_FLOAT=0
```

To keep double forward moments but store only the reverse-gradient workspace in
float:

```text
potential p.mtp H C N I Pb sus2_float=0 sus2_grad_float=1
```

or:

```bash
export SUS2_GPUMD_FLOAT=0
export SUS2_GPUMD_GRAD_FLOAT=1
```

The default `sus2_float=1` stores SUS2 moments and reverse gradients in float,
evaluates local moment arithmetic in float, and still writes GPUMD
energy/force/virial outputs to the existing double arrays.

To disable or probe the current safe fast paths:

```text
potential p.mtp Cu Zr sus2_float=1 sus2_l3k3_force_grad_cache=0
potential p.mtp Cu Zr sus2_float=1 sus2_tensor_force_grad_cache=0
potential p.mtp Cu Zr sus2_float=1 sus2_fused_graph=0
potential p.mtp Cu Zr sus2_float=1 sus2_graph_specific=0
potential p.mtp Cu Zr sus2_float=1 sus2_force_self_buffer=0
potential p.mtp Cu Zr sus2_float=1 sus2_local_product_graph=1
```

`sus2_graph_specific=0` keeps product-assign but forces the mature grouped product graph. The default graph-specific path is enabled only when the model has a supported tensor-basic layout, assignable product DAG, unique scalar moment mapping, and `uint16`-packable product rules. `sus2_local_product_graph=1` is kept as an experimental model-topology path. It preserves the same product DAG and reverse-mode chain rule, but on the tested Cu-Zr l3k3 model it was slower because the local per-thread graph workspace increased register/local-memory pressure.

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

To identify whether a model can reuse an already preserved product graph:

```bash
tools/sus2_v11_codegen.py model.mtp \
  --cache-dir codegen_cache/sus2_v11 \
  --query-cache
```

The cache also maintains `registry.json`; use `--list-cache` to inspect saved graph entries.

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
MA l4k3 98k tensor L/K-specialized cache test: 1.50229e6 -> 2.28901e6 atom-step/s
```

More detailed notes are in [docs/sus2-gpumd-v1.1-porting-notes.md](docs/sus2-gpumd-v1.1-porting-notes.md).
