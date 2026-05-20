# GPUMD-SUS2-SH

Experimental SUS2-SH inference support for GPUMD.

This repository is the spherical-harmonic line of the GPUMD SUS2 interface. It
is split from the moment-tensor `GPUMD-SUS2` project so the SH model reader,
real-spherical-harmonic basic moments, CG contraction path, and later
matrix-style kernels can evolve without destabilizing the older tensor backend.
The active build is SH-only: moment-tensor SUS2 models remain supported by the
separate `GPUMD-SUS2` repository.

## Project Relationship

| Repository | Role |
| --- | --- |
| `SUS2-SH` | Generates and trains SUS2-SH `.mtp` models on the CPU/reference path. |
| `SUS2-SH-GPU` | Trains the same SUS2-SH model format with CUDA objective/gradient paths. |
| `GPUMD-SUS2-SH` | This repository: GPUMD runtime backend for trained SUS2-SH models. |
| `GPUMD-SUS2` | Separate GPUMD backend for SUS2 v1.1 moment-tensor models. |
| `PySUS2SH` | Python/ASE/phonon workflow package for SUS2-SH models. |

Use this repository for MD production in GPUMD after training a SUS2-SH model.
Do not use it for moment-tensor SUS2-MLIP models; that path belongs to
`GPUMD-SUS2`.

Initial design and implementation notes are tracked in
[docs/gpumd-sus2-sh-interface-plan.md](docs/gpumd-sus2-sh-interface-plan.md).

## Status

The first generic `SUS2_SH` backend is implemented. GPUMD still recognizes
model files beginning with `MTP`; if the file also contains
`potential_tag = SUS2-SH`, dispatch switches to the SH backend. Other SUS2
`MTP` models are rejected by this repository so the old tensor backend and its
large graph-specific code are not compiled into the SH line.

The first SH path prioritizes correctness:

- real spherical harmonics are evaluated directly for `sh_l_max <= 4`;
- `alpha_index_basic = {(k,l,m), ...}` is read as SH basic moments;
- saved `sh_products = {left,right,target,coeff}` are executed in topological
  order and reversed for forces;
- the standard SUS2-SH real-CG graph is reconstructed on load and validated
  against the saved flat products;
- `RBChebyshev_sss`, `scaling_map = LK`, float moments, and direct radial
  recurrence are the default supported path.
- basic metadata is packed as `(mu,yidx)` for the current low-risk fast path;
  experimental tensor-product execution is available for profiling, but is not
  the default production path yet.

## Runtime Defaults

- SUS2-SH model format: `version = 1.1.0`, `potential_tag = SUS2-SH`
- GPUMD potential token: model files beginning with `MTP`
- Supported first radial path: `RBChebyshev_sss`
- Supported scaling map: `LK`
- Supported angular cutoff: `sh_l_max <= 4`
- Default radial evaluation: direct basis recurrence
- Default precision path: NEP-like float moments/gradients/local arithmetic
- Default force path: self-force buffer enabled

## Core Files

```text
src/force/sus2_sh.cu
src/force/sus2_sh.cuh
src/force/force.cu
src/model/read_xyz.cu
```

`sus2_sh.cu/.cuh` implements the first SH backend. `force.cu` wires GPUMD
potential dispatch to SUS2-SH when the potential file starts with `MTP` and
contains `potential_tag = SUS2-SH`. `read_xyz.cu` lets GPUMD read element
symbols for SUS2-SH models from `run.in`.

## Quick Install

See [INSTALL.md](INSTALL.md).

## Example GPUMD `run.in`

```text
potential p.mtp Cu Zr
velocity 200 seed 9174
ensemble npt_mttk temp 200 200 aniso 0.0001 0.0001 tperiod 100 pperiod 1000
time_step 0.5
dump_thermo 100
run 2000
```

To force double moment values and gradients instead of the default NEP-like
float path:

```text
potential p.mtp Cu Zr sus2_float=0
```

or:

```bash
export SUS2_GPUMD_FLOAT=0
```

To disable the self-force buffer:

```text
potential p.mtp Cu Zr sus2_sh_force_self_buffer=0
```

or:

```bash
export SUS2_GPUMD_FORCE_SELF_BUFFER=0
```

The default `sus2_float=1` stores SH moments and reverse gradients in float,
evaluates local moment arithmetic in float, and still writes GPUMD
energy/force/virial outputs to the existing double arrays.
For models with at most 64 SH basic components, the force kernel now caches the
center basic gradients by default; set `sus2_sh_force_grad_cache=0` to disable
that profiling path.
The default product path uses the compact serial row program; set
`sus2_sh_compact_serial_product=0` to fall back to the older flat product loop.

The experimental tensor-product path can be enabled with:

```text
potential p.mtp Cu Zr sus2_sh_tensor_product_parallel=1
```

It reconstructs the standardized CG rows/layers from the model and uses a
source-adjoint reverse pass. The current default tensor grid cap is 8192 CTAs;
override it for profiling with `sus2_sh_tensor_grid_cap=<n>`.

## Validation Snapshot

The experimental A100 build uses:

```text
gcc/12.2.0
cuda/12.4
CUDA_ARCH=sm_80
```

Current status:

```text
Cu-Zr l3k3 1.024M first SH backend: 4.65405e6 atom-step/s
Cu-Zr l3k3 1.024M packed-basic metadata path: 4.77022e6 atom-step/s
Cu-Zr l3k3 1.024M default flat path after basis/cache pass: 5.14507e6 atom-step/s
Cu-Zr l3k3 1.024M compact serial product path: 5.23633e6 atom-step/s
Cu-Zr l3k3 1.024M tensor row-adjoint path: correct, 4.92559e6 atom-step/s
```
