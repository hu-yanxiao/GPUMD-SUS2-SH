# SUS2-GPUMD Usage Guide

This note records the operational GPUMD-SUS2 v1.1 workflow used on the
`phy-weigw` cluster. It is meant to be the quick reference for future SUS2
GPUMD runs, rebuilds, and three-way synchronization checks.

## Three-Way Sync

Authoritative local clean worktree:

```bash
/Users/hu-yanxiao/Projects/SUS2MLIP/.codex_tmp/gpumd-sus2-clean-wt
```

GitHub private repository:

```bash
https://github.com/hu-yanxiao/GPUMD-SUS2
```

Server runtime/build tree:

```bash
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex
```

Server binary:

```bash
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/src/gpumd
```

Synchronization rule:

1. Make persistent source changes in the local clean worktree.
2. Commit and push the local change to `hu-yanxiao/GPUMD-SUS2`.
3. Mirror the same source files to the server runtime/build tree.
4. Rebuild `src/gpumd` on the server when source changes affect the binary.
5. Verify local, GitHub, and server agree before treating the update as complete.

The server tree is a build/runtime mirror, not the GitHub authority. It may
contain benchmark directories, build logs, codegen caches, and backup files that
are intentionally not committed.

## Server Environment

Runtime modules for A100 jobs:

```bash
module purge
module load gcc/12.2.0 cuda/12.4
```

Build modules:

```bash
module purge
module load gcc/12.2.0 cuda/12.4 cmake/3.25.2
```

Preferred A100 build command with low login-node CPU pressure:

```bash
cd /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/src
make -j2 gpumd CUDA_ARCH=-arch=sm_80 > ../build_gpumd_sm80.log 2>&1
```

Use `sm_80` for A100. Keep generated benchmark outputs and codegen caches out
of GitHub unless they are promoted intentionally.

## Required Run Files

A normal GPUMD-SUS2 run directory needs:

```text
run.in
model.xyz
p.mtp
```

`p.mtp` is a SUS2 model file whose first token selects the SUS2 backend through
the GPUMD `MTP` potential path. `model.xyz` is the GPUMD structure file. The
element symbols after the model filename in `run.in` define the element mapping
used by GPUMD for the SUS2 model.

## Basic `run.in`

Example for a BAs run:

```text
potential p.mtp B As
velocity 300 seed 9174
ensemble npt_mttk temp 300 300 aniso 0.0001 0.0001 tperiod 50 pperiod 500
time_step 1
dump_thermo 100
run 2000
```

The pressure value `0.0001` is in GPUMD pressure units and corresponds to the
1 bar style setting used in the matching LAMMPS tests.

## SUS2 Modes

Default double mode:

```text
potential p.mtp B As
```

Float reverse-gradient workspace:

```text
potential p.mtp B As sus2_grad_float=1
```

NEP-like SUS2 float path:

```text
potential p.mtp B As sus2_float=1
```

Equivalent environment variables:

```bash
export SUS2_GPUMD_GRAD_FLOAT=1
export SUS2_GPUMD_FLOAT=1
```

`sus2_grad_float=1` keeps forward moment values in double precision but stores
the reverse-gradient workspace in float. `sus2_float=1` stores SUS2 moments and
reverse gradients in float and writes the final GPUMD energy, force, and virial
outputs to the existing double arrays. For precision-sensitive checks, compare
the first frame before relying on long MD trajectories because later frames can
diverge through normal chaotic MD dynamics.

## Radial Lookup Tables

The GPUMD-SUS2 backend uses radial lookup tables by default.

Default spacing:

```text
dr = 1.0e-4 A
```

This matches the current LAMMPS SUS2 table convention.

Override from `run.in`:

```text
potential p.mtp B As sus2_lut_dr=0.0001
```

Override from the environment:

```bash
export SUS2_GPUMD_LUT_DR=0.0001
export SUS2_GPUMD_LUT_SPAN=...
```

Experimental direct Chebyshev recurrence:

```text
potential p.mtp Cu Zr sus2_radial_direct=1
```

or:

```bash
export SUS2_GPUMD_RADIAL_DIRECT=1
```

This bypasses the radial lookup table and evaluates the
`RBChebyshev_sss[_lmp]` radial values and derivatives by the same recurrence
used to build the LUT. It is opt-in and currently rejects Jacobi/Laguerre
families so the default behavior and broad radial-basis support stay table
based.

## LSF Job Template

Submit from the intended run directory so GPUMD sees the local `run.in`,
`model.xyz`, and `p.mtp`.

```bash
#!/bin/bash
#BSUB -J gpumd_sus2
#BSUB -q gpu-phy-zhangwq
#BSUB -n 1
#BSUB -gpu "num=1/host"
#BSUB -o job.%J.out
#BSUB -e job.%J.err

set -e
module purge >/dev/null 2>&1 || true
module load gcc/12.2.0 cuda/12.4

cd "$LS_SUBCWD"
gpumd_bin=/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/src/gpumd

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
stat --printf="%y %n\n" "$gpumd_bin" > binary.info

rm -f gpumd.log wall.time
start_epoch=$(date +%s)
"$gpumd_bin" > gpumd.log 2>&1
rc=$?
end_epoch=$(date +%s)
echo wall_seconds=$((end_epoch-start_epoch)) > wall.time
tail -80 gpumd.log
exit $rc
```

Submit:

```bash
bsub < run.lsf
```

On `gpu-phy-zhangwq`, `CUDA_VISIBLE_DEVICES` can be unset even when the
scheduler has assigned an A100. Do not fail the script only because this
variable is unset.

## Outputs To Check

Main GPUMD log:

```bash
tail -80 gpumd.log
```

Speed line:

```bash
grep -E "Time used for this run|Speed of this run" gpumd.log
```

Thermodynamic output:

```bash
tail thermo.out
```

If a benchmark wrapper records GPU memory, it may also create `vram.csv`; that
file is produced by the wrapper, not by GPUMD itself.

## Supported SUS2 v1.1 Scope

Current supported model format:

```text
version = 1.1.0
```

Supported radial basis types include:

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

The optimized tensor path is automatic for standard `lLkK`
`alpha_index_basic` layouts up to `l4k4`.

## Product-Graph Codegen Cache

The topology code generator can cache model-topology-specific product graph
cores. The cache key depends on product topology, not element identities or
fitted coefficients.

Query or build a cached core:

```bash
cd /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex
tools/sus2_v11_codegen.py model.mtp \
  --cache-dir codegen_cache/sus2_v11 \
  --query-cache
```

On a cache miss, compilation can take about `160 s`. On a cache hit, lookup is
typically sub-second.

## Recent Benchmark Locations

BAs GPU benchmark:

```bash
/work/phy-weigw/hyx/bas/bench_gpu_lammps_gpumd_npt2000_20260428
```

The GPUMD logs are stored by scale and mode, for example:

```bash
/work/phy-weigw/hyx/bas/bench_gpu_lammps_gpumd_npt2000_20260428/n884736/gpumd/sus2_float/gpumd.log
```

List all GPUMD logs:

```bash
find /work/phy-weigw/hyx/bas/bench_gpu_lammps_gpumd_npt2000_20260428 \
  -path "*/gpumd.log" | sort
```
