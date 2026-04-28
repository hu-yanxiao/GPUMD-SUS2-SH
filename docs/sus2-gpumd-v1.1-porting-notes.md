# SUS2 v1.1 GPUMD Porting Notes

Date: 2026-04-26 to 2026-04-27

## Goal

Start from SUS2 v1.1 inference first, keep the tabulated radial-basis advantage, and evaluate/implement a GPUMD force backend without touching the existing SUS2 developer baseline.

## Current Three-Way Sync State

Local clean GPUMD-SUS2 worktree:

```bash
/Users/hu-yanxiao/Projects/SUS2MLIP/.codex_tmp/gpumd-sus2-clean-wt
```

GitHub private repository:

```bash
https://github.com/hu-yanxiao/GPUMD-SUS2
```

Current GitHub branch:

```bash
main
```

Server runtime/build tree:

```bash
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex
```

Server GPUMD binary:

```bash
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/src/gpumd
```

Upstream GPUMD reference:

```bash
https://github.com/brucefan1983/GPUMD.git
```

Initial upstream commit used as the base:

```bash
ff9b0dd Merge pull request #1466 from duanzaixu/gitpr
```

Important synchronization rule:

1. Treat the local clean worktree as the authority for GitHub commits and pushes.
2. Treat the server tree as the build/runtime mirror used for A100 tests and production benchmark runs.
3. For a persistent GPUMD-SUS2 code update, commit it locally, push it to `hu-yanxiao/GPUMD-SUS2`, sync the same source files to the server tree, rebuild `src/gpumd`, and verify file hashes across local/GitHub/server.
4. Do not consider a GPUMD-SUS2 update complete until the local clean worktree, GitHub `main`, and the server runtime tree agree on the relevant source files.
5. Do not upload benchmark outputs, `codex_bench/`, `codex_smoke/`, build logs, or codegen caches unless they are intentionally promoted to documentation or test fixtures.

Server git note:

- The server has `origin` set to `https://github.com/hu-yanxiao/GPUMD-SUS2.git` and `upstream` set to `https://github.com/brucefan1983/GPUMD.git`.
- The GitHub repository is private, and the server currently has no GitHub credentials. In practice, push/pull should be done from the local clean worktree; server updates should be mirrored over SSH/rsync.

## Toolchain

Preferred A100 build modules:

```bash
module purge
module load gcc/12.2.0 cuda/12.4 cmake/3.25.2
```

Observed versions:

```bash
gcc 12.2.0
nvcc 12.4.131
cmake 3.25.2
```

Clean GPUMD build command, using low login-node CPU pressure:

```bash
cd /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/src
make -j2 gpumd CUDA_ARCH=-arch=sm_80 > ../build_gpumd_sm80.log 2>&1
```

This produced a working `src/gpumd` binary for A100 `sm_80`.

## Runtime Smoke Test

Smoke directory:

```bash
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_smoke/gpumd_static
```

Robust LSF script:

```bash
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_smoke/gpumd_static/smoke_a100_safe.lsf
```

Successful job:

```bash
3616901
```

Queue and node:

```bash
gpu-phy-zhangwq
b05u08g
NVIDIA A100-SXM4-80GB
```

The official GPUMD NEP smoke case completed with `GPUMD_RC=0`.

Important script lesson: avoid `set -u` for LSF GPU scripts unless every scheduler variable is guarded. On `gpu-phy-zhangwq`, `CUDA_VISIBLE_DEVICES` can be unset even when a GPU is allocated, and GPUMD still sees the assigned A100 through the scheduler environment. Use `${CUDA_VISIBLE_DEVICES:-unset}` in diagnostics.

## Existing Reference Prototype

There is an older dirty reference tree:

```bash
/work/phy-weigw/apps/gpumd
```

Do not overwrite it. It contains an early `SUS2_MTP` prototype and user/generated test artifacts. It is useful as a reference only.

The prototype already adds `MTP` parsing in GPUMD and has files such as:

```bash
src/force/sus2_mtp.cu
src/force/sus2_mtp.cuh
src/force/sus2_mtp_generated.inc
```

Current limitations of that prototype:

- It targets `version=1.1.0` but is not a general v1.1 reader.
- It is effectively single-species.
- It assumes fixed `l2k2` dimensions and `RBChebyshev_sss`.
- It hardcodes moment/radial sizes instead of reading arbitrary SUS2 model dimensions.

## SUS2 v1.1 Implementation Status

Implemented experimental SUS2 v1.1 backend files in the clean GPUMD worktree:

```bash
src/force/sus2_v11.cu
src/force/sus2_v11.cuh
src/force/force.cu
src/model/read_xyz.cu
```

Current supported target:

- `version = 1.1.0`
- `radial_basis_type = RBJacobi_sss_lmp`
- Dynamic model dimensions read from the `.mtp` file.
- Radial basis values and derivatives are pretabulated in a GPU LUT.
- GPUMD `run.in` uses the model symbols after the potential file, for example:

```bash
potential p1.1.mtp H C N I Pb
```

GPUMD itself still reads the first token `MTP` from the SUS2 model file to select this backend.

## MA/Jacobi v1.1 Correctness Case

The first real v1.1 validation case uses the previous MA/Jacobi benchmark:

```bash
/work/phy-weigw/hyx/ma/l3k3/jacobi/benchmark_lmp/p1.1.mtp
/work/phy-weigw/hyx/ma/l3k3/jacobi/benchmark_lmp/data.in
```

Parity test directory:

```bash
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_smoke/ma_v11_parity
```

Reference method:

- GPUMD: `time_step 0`, `run 1`, `dump_force 1`.
- LAMMPS: CPU `sus2mtp`, `run 0`, `dump custom ... fx fy fz`.
- Comparison script: `compare_ma_gpumd_lammps.py`.

Clean build and parity result after the multi-image neighbor fix:

```text
atoms = 192
energy_gpumd_eV = -8.2216971022999996e+02
energy_lammps_eV = -8.2216971000000001e+02
energy_diff_eV = -2.2999995508143911e-07
energy_diff_meV_per_atom = -1.1979164327158287e-06
force_mae_eV_A = 1.1169451700354228e-07
force_rmse_eV_A = 1.8504171572672147e-07
force_max_abs_eV_A = 8.0230331420128032e-07
```

Important lesson: the MA cell has a short z dimension, and the SUS2 cutoff is slightly larger than half the box thickness. A simple minimum-image neighbor list gave nonzero but wrong results:

```text
energy_diff_meV_per_atom ~= 1.36
force_mae_eV_A ~= 6.3e-3
```

The correct GPUMD-SUS2 neighbor path must store each neighbor edge's actual periodic-image displacement `dx, dy, dz`, not only the neighbor atom id. Otherwise multiple periodic images of the same atom collapse onto the minimum image and one image contribution is lost.

## MA/Jacobi v1.1 Virial/Stress Parity

Stress parity directory:

```bash
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_smoke/ma_v11_virial_parity
```

Reference method:

- GPUMD: `time_step 0`, `run 1`, `dump_thermo 1`, zero velocities in `model.xyz`.
- LAMMPS: CPU `sus2mtp`, `run 0`, `compute pressure NULL virial`.
- Comparison script: `compare_ma_gpumd_lammps_virial.py`.

Clean parity result after the hybrid neighbor-list implementation:

```text
energy_diff_eV = -2.2999995508143911e-07
stress_order = xx yy zz xy xz yz
stress_gpumd_GPa = 1.4690688041 1.4783693494 1.9406601283 0.001504167978 -0.023581090475 -0.020751830916
stress_lammps_GPa = 1.4690687 1.4783692 1.9406597 0.0015041466 -0.023581077 -0.020751841
stress_mae_GPa = 1.2112283333565120e-07
stress_rmse_GPa = 1.9032355850115235e-07
stress_max_abs_GPa = 4.2829999991056411e-07
```

The stress agreement is at numerical-noise level, so the sign/order convention for virial stress is consistent with the LAMMPS SUS2 reference for this case.

## Atomic Virial Orientation

Update date: 2026-04-29

The SUS2-GPUMD atomic virial non-diagonal terms are aligned with the native NEP
ordered-pair convention:

```text
W_xy += -r_x * g_y
W_xz += -r_x * g_z
W_yx += -r_y * g_x
...
```

where `g = dE_i / dr_ij` for the center-site energy contribution. This keeps
the 9-component storage order unchanged:

```text
xx yy zz xy xz yz yx zx zy
```

The change only fixes the orientation of the unsymmetrized per-atom virial
tensor so that SUS2 follows NEP's internal convention. Energies, forces, and the
physical symmetric stress are unchanged. The MA/Jacobi static virial parity was
rechecked after this update:

```text
energy_diff_eV = 0.0
stress_mae_GPa = 5.07e-07
stress_rmse_GPa = 6.50e-07
stress_max_abs_GPa = 1.24e-06
```

## 98k Atom NPT Performance Smoke

Benchmark directory:

```bash
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/ma_98k_npt
```

Source data:

```bash
/work/phy-weigw/hyx/ma/l3k3/jacobi/benchmark_lmp/bench_npt2000_latest_iface_100k/data_98304.in
```

GPUMD NPT settings were translated from the LAMMPS benchmark:

- `time_step 0.5` fs, matching LAMMPS `timestep 0.0005` ps.
- `velocity 200 seed 9174`.
- `ensemble npt_mttk temp 200 200 aniso 0.0001 0.0001 tperiod 100 pperiod 1000`.
- `dump_thermo 100`, `run 2000`.

Short 10-step smoke:

```text
job = 3617026
atoms = 98304
steps = 10
speed = 58398.4 atom-step/s
```

Pre-optimization full 2000-step run:

```text
job = 3617029
atoms = 98304
steps = 2000
run_seconds = 3202.01
speed = 61401.4 atom-step/s
```

This run completed normally with finite thermo output, but it exposed a severe performance issue that was later traced to the wrong neighbor-list path.

## Hybrid Neighbor Strategy

The current implementation uses two neighbor paths:

- Before choosing a path, explicitly initialize `box.thickness_x/y/z = volume / area_{x/y/z}`. This matches the NEP logic and avoids reading uninitialized thickness values.
- If any periodic box thickness is smaller than `2 * cutoff`, use the verified multi-image O(N^2) path and store each edge's actual periodic-image displacement `dx, dy, dz`, so duplicate periodic images are not lost.
- Otherwise use GPUMD's cached large-box neighbor path: `Neighbor::find_neighbor_global(rc, ...)` followed by `Neighbor::find_local_neighbor_from_global(rc, ...)`. Large-box SUS2 kernels store neighbor ids only and recompute the minimum-image displacement inside the kernels.

This preserves the small-cell correctness case while making the 98k atom NPT smoke feasible.

## First GPUMD-SUS2 Optimization Pass

Date: 2026-04-27

Implemented low-risk GPU-path optimizations while keeping the SUS2 mathematical expression unchanged:

- Large-box path no longer stores per-edge `dx, dy, dz`; it stores neighbor ids only and recomputes the minimum-image displacement inside the kernels. Small-box/multi-image path still stores the exact image displacement to preserve duplicate-image correctness.
- Radial LUT values and derivatives are stored as `float` on device and converted back to `double` in the arithmetic path. The model expression and moment arithmetic remain double.
- Large-box force accumulation uses a direct no-atomic center-thread path. For each edge `i -> j`, the thread computes the same chain-rule objects as the old implementation but accumulates `D_i(r_ij) - D_j(r_ji)` into atom `i`, avoiding force atomics. The small-box path still uses the original cached-displacement atomic route.

Correctness checks:

```text
small-box 192-atom parity:
energy_diff_meV_per_atom = -3.1250024790097086e-07
force_mae_eV_A = 1.8212621905130897e-07
force_max_abs_eV_A = 1.0163574218902127e-06
stress_mae_GPa = 1.2125895000285095e-07
stress_max_abs_GPa = 2.4909999996047816e-07
```

```text
large-box 192-atom force parity, box doubled without replicating atoms:
energy_diff_meV_per_atom = 1.4635416434316539e-05
force_mae_eV_A = 3.4390296550870349e-07
force_max_abs_eV_A = 5.1308822630602435e-06
```

Performance checks on the 98,304-atom NPT case:

```text
previous 10-step smoke speed = 58398.4 atom-step/s
first f12-gather trial speed = 58176.5 atom-step/s
direct no-atomic 10-step speed = 58300.3 atom-step/s
direct no-atomic 100-step speed = 60983.3 atom-step/s
previous 2000-step full speed = 61401.4 atom-step/s
```

Conclusion from this intermediate pass: removing large-box force atomics alone did not improve performance because the job was still taking the wrong neighbor path. The direct no-atomic force kernel also computes both center derivatives inside each atom thread, which doubles the expensive SUS2 edge-derivative work relative to a directed-edge atomic accumulation.

## Large-Box Neighbor and Force Bottleneck Fix

Date: 2026-04-27

Profiling before the fix showed that the 98k atom NPT run was dominated by neighbor construction:

```text
SUS2_PROFILE calls=100 avg_ms:
neighbor = 1568.08 ms
neighbor_global = 0
neighbor_local = 0
measured_total = 1605 ms
speed ~= 6.12e4 atom-step/s
```

The zero `neighbor_global/local` timers were the clue: the 98k large-box run was not entering the GPUMD cached neighbor path. Root cause: `box.thickness_x/y/z` were used before being initialized, so they effectively looked like zero and the code always selected the small-box multi-image fallback.

Fixes:

- Initialize `box.thickness_x/y/z` explicitly before the small-box/large-box decision.
- Keep exact cached displacements only for the small-box multi-image path.
- Use GPUMD's global neighbor cache plus local cutoff filter for large boxes.
- Change the default large-box force path back to directed-edge atomic accumulation. This computes each directed SUS2 derivative once. The older no-atomic pairwise path is retained only as an experiment behind `SUS2_GPUMD_PAIRWISE_NO_ATOMIC_FORCE=1`.

Final correctness checks after the neighbor and force fixes:

```text
small-box 192-atom parity:
energy_diff_meV_per_atom = -3.1250024790097086e-07
force_mae_eV_A = 1.8550532338051470e-07
force_max_abs_eV_A = 1.0163574218902127e-06
```

```text
large-box 192-atom force parity, box doubled without replicating atoms:
energy_diff_meV_per_atom = 1.4635416434316539e-05
force_mae_eV_A = 3.4235210590389136e-07
force_max_abs_eV_A = 5.1308822630602435e-06
```

```text
stress parity:
stress_mae_GPa = 1.2125895000285095e-07
stress_max_abs_GPa = 2.4909999996047816e-07
```

98k atom NPT profiling after the thickness fix but before restoring directed-edge atomic force:

```text
SUS2_PROFILE calls=30 avg_ms:
neighbor = 0.189057 ms
neighbor_global = 0.030497 ms
neighbor_local = 0.153407 ms
force = 24.818870 ms
measured_total = 50.941961 ms
speed = 1.90776e6 atom-step/s
```

98k atom NPT profiling after restoring directed-edge atomic force:

```text
SUS2_PROFILE calls=30 avg_ms:
neighbor = 0.187716 ms
neighbor_global = 0.029334 ms
neighbor_local = 0.153527 ms
force = 11.521814 ms
measured_total = 37.619717 ms
speed = 2.56215e6 atom-step/s
```

Final 2000-step 98k atom NPT performance run:

```text
job = 3617310
atoms = 98304
steps = 2000
run_seconds = 75.7826
wall_seconds = 83
speed = 2.59437e6 atom-step/s
```

Net result on this case: `6.14014e4 -> 2.59437e6 atom-step/s`, about `42.3x` faster than the pre-fix GPUMD-SUS2 run. This is also above the earlier single-A100 LAMMPS Kokkos SUS2 reference for the same 98k-scale MA benchmark, which was about `1.99e6 atom-step/s`.

## L3K3 Fast Path and Product-Rule Table Optimization

Date: 2026-04-27

Implemented a second low-risk optimization pass for the remaining non-neighbor bottlenecks:

- Added an `l3k3` `alpha_index_basic` fast path. When the model has the standard 12-radial, rank-0/1/2/3 grouped layout, GPUMD-SUS2 directly evaluates the 20 Cartesian monomials per `k` group instead of interpreting `(mu,a,b,c)` tuples for every edge.
- Added a matching `l3k3` force-derivative fast path. It directly expands the rank-0/1/2/3 geometric derivatives while keeping the same radial values and chain rule as the generic implementation.
- Packed `alpha_index_times` into a `uint16` constant-memory table when all moment ids and multipliers fit. For the MA/Jacobi model, `alpha_index_times_count = 5230`, moment ids are `0..2023`, and multipliers are at most `6`, so this path is valid. This reduces repeated product-rule index loads in forward/backward moment propagation.
- Replaced large zero-fill kernels/thrust fills with `gpuMemset` for `moment_vals`, `moment_grads`, `force_tmp`, and `virial_tmp`.

All fast paths are guarded. If a model does not match the `l3k3` layout or cannot pack product rules into `uint16`, the implementation falls back to the generic v1.1 path.

Correctness checks after this pass:

```text
small-box 192-atom parity:
energy_diff_meV_per_atom = -3.1250024790097086e-07
force_mae_eV_A = 1.8290885714458053e-07
force_max_abs_eV_A = 1.2547760009917752e-06
```

```text
large-box 192-atom force parity, box doubled without replicating atoms:
energy_diff_meV_per_atom = 1.4635416434316539e-05
force_mae_eV_A = 3.4216217523721834e-07
force_max_abs_eV_A = 5.1308822630602435e-06
```

```text
stress parity:
stress_mae_GPa = 1.2125895000285095e-07
stress_max_abs_GPa = 2.4909999996047816e-07
```

98k atom NPT profiling after this pass:

```text
SUS2_PROFILE calls=30 avg_ms:
neighbor = 0.184859 ms
zero = 1.702167 ms
basic = 5.461010 ms
forward = 4.449905 ms
energy_grad = 1.433769 ms
backward = 9.711979 ms
force = 3.799738 ms
accumulate = 0.021658 ms
measured_total = 26.765085 ms
```

Final 2000-step 98k atom NPT performance run after this pass:

```text
job = 3617375
atoms = 98304
steps = 2000
run_seconds = 54.0395
wall_seconds = 61
speed = 3.63823e6 atom-step/s
```

Incremental result over the previous fixed-neighbor/force version: `2.59437e6 -> 3.63823e6 atom-step/s`, about `1.40x` faster. Net result over the original correctness-first GPUMD-SUS2 implementation: `6.14014e4 -> 3.63823e6 atom-step/s`, about `59.3x` faster.

Million-scale single-A100 run using a `2x2x3` replication of the 98k MA/Jacobi cell:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/ma_1m_npt/full_2000_1179648
job = 3617387
atoms = 1179648
steps = 2000
run_seconds = 678.141
wall_seconds = 689
speed = 3.47907e6 atom-step/s
```

This completed normally on one A100 with the same NPT settings as the 98k case. The 1.18M atom speed is close to the 98k speed, so the optimized GPUMD-SUS2 v1.1 path shows good large-system throughput on this benchmark.

## L3K3 Basic-Moment Local Accumulation

Date: 2026-04-27

Implemented a third optimization pass for the `basic` kernel. The earlier `l3k3` fast path still updated the 60 basic moments in global memory for every neighbor edge. The new path accumulates all 60 basic moments in thread-local storage for each center atom and writes each basic moment to global memory once after the neighbor loop.

This keeps the same mathematical expression and the same radial table values, but removes most repeated global read-modify-write traffic from the basic-moment construction.

Correctness checks after this pass:

```text
small-box 192-atom parity:
energy_diff_meV_per_atom = -3.1250024790097086e-07
force_mae_eV_A = 1.8117595912034688e-07
force_max_abs_eV_A = 1.2547760009917752e-06
```

```text
large-box 192-atom force parity, box doubled without replicating atoms:
energy_diff_meV_per_atom = 1.4635416434316539e-05
force_mae_eV_A = 3.4258052259474612e-07
force_max_abs_eV_A = 5.1308822630602435e-06
```

```text
stress parity:
stress_mae_GPa = 1.2125895000285095e-07
stress_max_abs_GPa = 2.4909999996047816e-07
```

98k atom NPT profiling after this pass:

```text
SUS2_PROFILE calls=30 avg_ms:
neighbor = 0.183532 ms
zero = 1.702933 ms
basic = 1.139293 ms
forward = 4.448650 ms
energy_grad = 1.431435 ms
backward = 9.708696 ms
force = 3.795585 ms
accumulate = 0.021382 ms
measured_total = 22.431506 ms
```

The important change is:

```text
basic ~= 5.46 ms -> 1.14 ms
```

Final 2000-step 98k atom NPT performance run after this pass:

```text
job = 3617431
atoms = 98304
steps = 2000
run_seconds = 45.6732
wall_seconds = 53
speed = 4.30466e6 atom-step/s
```

Final 2000-step 1.18M atom NPT performance run after this pass:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/ma_1m_npt/full_2000_1179648_l3k3_basic_accum
job = 3617432
atoms = 1179648
steps = 2000
run_seconds = 579.121
wall_seconds = 590
speed = 4.07393e6 atom-step/s
```

Incremental results over the previous pass:

```text
98k: 3.63823e6 -> 4.30466e6 atom-step/s, about 1.18x faster
1.18M: 3.47907e6 -> 4.07393e6 atom-step/s, about 1.17x faster
```

Net result over the original correctness-first GPUMD-SUS2 implementation:

```text
98k: 6.14014e4 -> 4.30466e6 atom-step/s, about 70.1x faster
```

The remaining bottleneck is now dominated by reverse product-rule propagation:

```text
backward ~= 9.7 ms
forward ~= 4.45 ms
force ~= 3.8 ms
zero ~= 1.7 ms
basic ~= 1.1 ms
```

Splitting constant-memory `alpha_index_times` into dedicated forward/backward kernels was tested and did not materially improve beyond the existing constant table path. The bottleneck is therefore mostly the global-memory traffic and dependency structure of the moment DAG rather than product-rule branch overhead.

Further meaningful gains likely require deeper moment-DAG work, such as changing the per-atom forward/backward propagation strategy or reducing global `N * alpha_moments_count` memory traffic. Those changes are more invasive than the current safe fast paths.

## Next Implementation Direction

The single-GPU correctness and 98k-scale performance path is now validated for the MA/Jacobi v1.1 model. Next work should focus on:

1. Expand v1.1 reader coverage beyond `RBJacobi_sss_lmp` if needed.
2. Optimize the moment calculation path further: reduce full `N * alpha_moments_count` global memory traffic, fuse kernels where practical, and avoid repeated scans over all basic moments for every neighbor.
3. Add direct performance comparisons against LAMMPS for the same MA model and more sizes.
4. Only after single-GPU performance is acceptable, consider GPUMD multi-GPU scaling.

## Laguerre l4k3 Reader And Codegen Probe

Date: 2026-04-27

Reference model:

```bash
/work/phy-weigw/hyx/ma/laguerre-l4k3/current.mtp
```

The GPUMD-SUS2 v1.1 reader was expanded beyond the original Jacobi-only path:

- Added `RBLaguerre_log1p`, `RBLaguerre_log1p_lmp`, `RBLaguerre_log1p_noenv`, `RBLaguerre_log1p_noenv_lmp`, `RBLaguerre_log1p_pos`, and `RBLaguerre_log1p_pos_lmp` host LUT generation.
- Added `RBJacobi_sss`, `RBJacobi_sss_lmp`, `RBJacobi_sss_noweight`, and `RBJacobi_sss_noweight_lmp` as accepted Jacobi v1.1 inference types.
- Added `RBChebyshev_sss` and `RBChebyshev_sss_lmp` host LUT generation.
- The interface still does not cover every historical SUS2 radial basis type, such as Shapeev, old Chebyshev variants, Bessel, or Taylor.

LUT control:

- Default now matches the LAMMPS table convention: `dr = 1.0e-4 A`, implemented as `lut_span = ceil(cutoff / 1.0e-4)`.
- Runtime controls now exist via environment variables `SUS2_GPUMD_LUT_SPAN` and `SUS2_GPUMD_LUT_DR`.
- `run.in` potential-line controls also work after the required model species symbols, for example:

```text
potential /work/phy-weigw/hyx/ma/laguerre-l4k3/current.mtp H C N I Pb sus2_lut_span=2000
potential /work/phy-weigw/hyx/ma/laguerre-l4k3/current.mtp H C N I Pb sus2_lut_dr=0.00325
```

The GPUMD xyz reader now takes exactly `species_count` symbols from the SUS2 `potential` line and ignores later SUS2 options, so options are not mistaken for element names.

Smoke tests after rebuilding `src/gpumd` with `gcc/12.2.0 cuda/12.4 sm_80`:

```text
Laguerre l4k3 load smoke:
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_smoke/laguerre_l4k3_load
job = 3617553
GPUMD_RC = 0
radial_type = RBLaguerre_log1p
species = 5
radial = 15
basics = 105
moments = 4065
scalars = 1767
LUT = 2002
dr = 0.00325 A
```

```text
Jacobi l3k3 compatibility smoke:
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_smoke/ma_v11_parity
job = 3617554
GPUMD_RC = 0
radial_type = RBJacobi_sss_lmp
LUT = 200002
dr = 3.25e-05 A
l3k3 fast path = enabled
constant uint16 alpha_index_times = enabled
```

After switching the default LUT spacing to `1.0e-4 A`, Chebyshev and large MA/Jacobi smoke tests were repeated:

```text
Chebyshev_sss load smoke:
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_smoke/chebyshev_sss_load
job = 3617704
GPUMD_RC = 0
radial_type = RBChebyshev_sss
species = 2
radial = 12
basics = 60
moments = 514
scalars = 349
LUT = 65002
dr = 0.0001 A
```

```text
1.18M atom MA/Jacobi NPT2000, new default LUT spacing:
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/ma_1m_npt/full_2000_1179648_lut1e4
job = 3617705
atoms = 1179648
steps = 2000
LUT = 65002
dr = 0.0001 A
run_seconds = 572.288
wall_seconds = 579
speed = 4.12257e6 atom-step/s
```

Previous same-case reference with the old dense LUT was:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/ma_1m_npt/full_2000_1179648_l3k3_basic_accum
job = 3617432
LUT = 200002
run_seconds = 579.121
wall_seconds = 590
speed = 4.07393e6 atom-step/s
```

The new LAMMPS-style default table density is therefore slightly faster in this 1.18M atom run:

```text
speed change = 4.07393e6 -> 4.12257e6 atom-step/s, about +1.19%
run-time change = 579.121 s -> 572.288 s, about -1.18%
```

Product-graph code generation probe:

```text
tool = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/tools/sus2_v11_codegen.py
model = /work/phy-weigw/hyx/ma/laguerre-l4k3/current.mtp
hash = d42a91c26f5253168e83e30294f3238e458b0e265febd3923b644500a783b67a
L = 4
k_count = 3
alpha_basic_count = 105
alpha_basic layout = l4k3 matched
alpha_times_count = 18245
active DAG compression = no inactive product moments
```

Generated full product-graph CUDA core result:

```text
out_dir = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_codegen/laguerre_l4k3_chunk512
chunk_size = 512 product rules per noinline device chunk
source_size = 5.40 MB
object_size = 8.16 MB
compile_seconds = 160.62
```

The codegen tool now prints activity progress during `nvcc/ptxas`, controlled by `--progress-interval`. A repeated l4k3 chunk512 compile with visible progress produced:

```text
out_dir = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_codegen/laguerre_l4k3_chunk512_progress
progress_interval = 20 s
progress_log = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_codegen/laguerre_l4k3_chunk512_progress.log
compile_seconds = 160.17
object_size = 8.16 MB
```

Example progress lines:

```text
[sus2-codegen] nvcc/ptxas working | [#...................] elapsed=20.0s
[sus2-codegen] nvcc/ptxas working / [##..................] elapsed=40.0s
[sus2-codegen] nvcc/ptxas working - [###.................] elapsed=60.1s
```

The codegen tool now also has a persistent topology cache. Default cache location from the GPUMD-SUS2 work root is:

```bash
codegen_cache/sus2_v11
```

It can be overridden with:

```bash
SUS2_CODEGEN_CACHE_DIR=/path/to/cache
tools/sus2_v11_codegen.py ... --cache-dir /path/to/cache
```

Cache key policy:

- Includes: `version`, `L`, `scaling_map`, `radial_funcs_count`, `alpha_index_basic`, compressed `alpha_index_times`, compressed `alpha_moment_mapping`, and compressed active moment count.
- Excludes: `species_count`, element names, radial coefficients, scaling coefficients, shift/species/moment coefficients, and `radial_basis_type`.
- This matches the current generated core scope, which is the product/moment topology core rather than radial evaluation.

Cache miss/hit test on the Laguerre l4k3 model:

```text
model = /work/phy-weigw/hyx/ma/laguerre-l4k3/current.mtp
cache_key = 4c5cba1e8d377067c527bf0df43c381ae32866adcad1208b2d802a6079ff0c44
cache_dir = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codegen_cache/sus2_v11/4c5cba1e8d377067c527bf0df43c381ae32866adcad1208b2d802a6079ff0c44
miss_compile_seconds = 158.16
hit_elapsed_seconds = 0.19
object_size = 8.16 MB
```

Cache entry files:

```text
generated.cu
generated.o
metadata.json
build.log
```

A synthetic topology-only probe changed `species_count` from 5 to 7 and changed `radial_basis_type` from `RBLaguerre_log1p` to `RBChebyshev_sss` while keeping the same alpha topology. It hit the same cache key and completed in about `0.17 s`, confirming that element/radial-type metadata is not part of this product-graph cache key.

An actual Cu-Zr l4k3 model did not hit the Laguerre l4k3 cache because its final scalar mapping/product graph is smaller:

```text
Cu-Zr l4k3 alpha_basic_count = 105
Cu-Zr l4k3 alpha_times_count = 3172
Cu-Zr l4k3 alpha_scalar_moments = 535
cache_key = 68c932d007dcb70c46d0875eb82e573887780b26a2c630e9cd4d2ed7900001bc
```

This is expected: same l4k3 basic basis does not guarantee the same final scalar graph.

Conclusion: automatic topology recognition and model-specific CUDA generation are feasible, but a complete l4k3 `alpha_index_times` graph is too large for startup-time JIT if fully baked into one cubin. Treat this as an AOT/cache path, or specialize only the cheaper and clearly profitable parts first, especially `alpha_index_basic`/force derivative kernels. Keep `alpha_index_times` on the constant-table path until a lower-compile-cost graph strategy is designed.

Current element-pair LUT status:

- The implementation still builds LUTs for all `species_count * species_count` model pairs.
- Skipping unused element pairs is mathematically safe for a concrete simulation, but GPUMD currently constructs the potential before the atom type vector is passed into `SUS2_V11`, so it does not yet know which pairs are unused at construction time.
- The right next design is lazy LUT construction on first `compute()` after seeing the actual type vector, or an explicit active-pair mask/cache keyed by the species present in `model.xyz`.

## Optional Float Moment-Gradient Workspace

Implemented an experimental switch that keeps forward moments in double precision but stores the reverse-mode moment-gradient workspace in float:

```text
potential p.mtp H C N I Pb sus2_grad_float=1
```

or:

```bash
export SUS2_GPUMD_GRAD_FLOAT=1
```

The default remains double:

```text
SUS2 v1.1 GPUMD moment-gradient workspace: double.
```

The float mode prints:

```text
SUS2 v1.1 GPUMD moment-gradient workspace: float.
```

Test directory:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/ma_l3k3_98k_gradfloat_20260427
```

MA l3k3, 98304 atoms, one A100, NPT 2000 steps, `time_step 0.5 fs`:

```text
double workspace:
  run_seconds = 45.1244
  wall_seconds = 49
  speed = 4.35702e6 atom-step/s
  GPUMD process GPU memory peak = 4124 MiB

float gradient workspace:
  run_seconds = 37.8451
  wall_seconds = 42
  speed = 5.19508e6 atom-step/s
  GPUMD process GPU memory peak = 3366 MiB
```

Observed change:

```text
speedup = 1.192x
run-time reduction = 16.1%
GPU process memory reduction = 758 MiB
```

Static 98k consistency check:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/ma_l3k3_98k_gradfloat_static_20260427
energy_diff_eV = 0.0
force_mae_eV_A = 1.3093445e-5
force_rmse_eV_A = 2.0695372e-5
force_max_abs_eV_A = 1.0588951e-4
```

For l4k3 million-atom runs, the expected memory saving is much larger because the saved half of `moment_grads` scales as:

```text
alpha_moments_count * N * 4 bytes
```

For `alpha_moments_count=4065` and `N=995328`, this is about `15.5 GiB` saved compared with double gradients, while preserving double forward moments.

## Optional NEP-Like Float Moment Path

Implemented a second experimental switch that follows NEP's mixed-precision strategy more closely:

```text
potential p.mtp H C N I Pb sus2_float=1
```

or:

```bash
export SUS2_GPUMD_FLOAT=1
```

This mode implies float reverse gradients and additionally stores forward moments, fitted scalar coefficients used inside the device kernel, and local moment/force arithmetic in float. GPUMD positions and final potential/force/virial arrays remain double.

The first-frame check is the hard correctness gate because later MD trajectories diverge chaotically. Static 98k MA l3k3 result versus default double:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/ma_l3k3_98k_nepfloat_static_20260427
energy_diff_eV = 0.25911
energy_diff_meV_per_atom = 0.00264
force_mae_eV_A = 1.5548e-5
force_rmse_eV_A = 2.5351e-5
force_max_abs_eV_A = 2.8420e-4
```

NPT 2000-step 98k MA l3k3 performance on one A100 after rebuilding the same binary:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/ma_l3k3_98k_nepfloat_npt2000_20260427

double:
  run_seconds = 44.8793
  speed = 4.38082e6 atom-step/s
  GPUMD process GPU memory peak = 4124 MiB

grad_float:
  run_seconds = 38.6276
  speed = 5.08983e6 atom-step/s
  GPUMD process GPU memory peak = 3366 MiB
  speedup_vs_double = 1.162x

sus2_float:
  run_seconds = 30.3050
  speed = 6.48763e6 atom-step/s
  GPUMD process GPU memory peak = 2608 MiB
  speedup_vs_double = 1.481x
```

The NEP-like float path should remain opt-in until each target model passes a first-frame energy/force/virial comparison against the default double path.

## Cu-Zr l3k3 NEP-Like Optimization Pass

Date: 2026-04-28

Target system and reference paths:

```text
model = /work/phy-weigw/hyx/cu-zr/7.5/lmp/bench_npt2000_dt1fs_latest_iface/p.mtp
atoms = 1,024,000
ensemble = npt_mttk
steps = 2000
time_step = 1 fs
GPUMD-SUS2 path = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex
```

The optimization kept the SUS2 v1.1 mathematical expression unchanged. The accepted changes are storage/order optimizations of the same forward product DAG and reverse-mode chain rule:

```text
1. constant-memory uint16 scalar moment mapping
2. constant-memory float shift/species/moment coefficients in sus2_float mode
3. fused site-energy-gradient plus product-backward path
4. l3k3 center basic-gradient cache in the force kernel
5. l3k3 tensor-polynomial aggregation for rank 0/1/2/3 force derivatives
```

Static first-frame old/new check against the pre-pass binary:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_smoke/cuzr_sus2_opt1_old_new_static_20260428
atoms = 2304
force_mae = 3.92e-7 eV/A
force_rmse = 5.27e-7 eV/A
force_max = 2.58e-6 eV/A
thermo_max = 4.05e-5
```

Short 200-step 1.024M profile after the accepted pass:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/cuzr_sus2_opt1_profile_1m_200step_20260428
speed = 1.67943e7 atom-step/s
representative avg_ms:
  neighbor = 3.72-5.86
  zero = 2.25
  basic = 9.95
  product graph = 20.10
  force = 20.48
```

Full 2000-step result:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/cuzr_l3k3_1m_sus2_opt_final_npt2000_20260428/sus2_float
run_seconds = 120.747
wall_seconds = 126
speed = 1.69611e7 atom-step/s
GPU process memory = about 9636 MiB
```

Reference comparisons on the same Cu-Zr 1.024M setup:

```text
previous GPUMD-SUS2 sus2_float = 1.24453e7 atom-step/s
optimized GPUMD-SUS2 sus2_float = 1.69611e7 atom-step/s
speedup_vs_previous = 1.36x

native GPUMD NEP = 1.86685e7 atom-step/s
optimized_sus2_vs_nep = 90.9%

LAMMPS single A100 SUS2 baseline = 2.10322e6 atom-step/s
optimized_gpumd_sus2_vs_lammps = 8.06x
```

Rejected or non-default experiments from this pass:

```text
fused full product graph:
  mathematically correct
  speed essentially unchanged
  kept because it reduces launch count but does not reduce global-memory traffic

local per-atom product graph:
  mathematically correct
  zero step improved from about 2.25 ms to 0.04 ms
  product graph worsened from about 20.1 ms to 23.9 ms
  speed dropped to 1.6303e7 atom-step/s
  kept only behind sus2_local_product_graph=1

kBlockSize = 256:
  mathematically correct
  force worsened from about 20.5 ms to 23.3 ms
  speed dropped to 1.60575e7 atom-step/s
  reverted to kBlockSize = 128
```

Interpretation:

The remaining gap to native NEP is no longer basic-moment construction. It is mainly product-DAG traffic and force contraction. A future model-specific fast path should not simply move the whole product DAG into a large per-thread local array; that spills and becomes slower. The better direction is an AOT or cached generated kernel that recognizes the concrete multiplication graph and emits a register-aware packed product/reverse plan, ideally writing only the basic gradients needed by the force kernel without exceeding register pressure.

## General Tensor Basic-Gradient Cache Up To l4k4

Date: 2026-04-28

The first cached-gradient implementation was hand-written for the exact `l3k3` layout:

```text
L = 3
K = 3
radial_funcs_count = 12
alpha_index_basic_count = 60
```

This pass generalized the same math to standard tensor layouts up to `l4k4` without changing the SUS2 expression:

```text
L <= 4
K <= 4
radial_funcs_count = K * (L + 1)
basic_per_group = C(L + 3, 3)
alpha_index_basic_count = K * basic_per_group
```

The detector verifies that `alpha_index_basic` is ordered by group, then rank, then Cartesian exponents:

```text
for group = 0..K-1:
  for rank = 0..L:
    mu = group * (L + 1) + rank
    for a = rank..0:
      for b = rank-a..0:
        c = rank-a-b
        alpha_index_basic includes (mu, a, b, c)
```

The mathematical contraction remains:

```text
B_{i,p} = sum_j phi_p(r_ij)
g_{i,p} = dE_i / dB_{i,p}
dE_i/dr_ij = sum_p g_{i,p} * d phi_p(r_ij)/dr_ij
```

The first version of this optimization only loaded `g_{i,p}` once per center atom and reused it across all neighbors. Exact `l3k3` used the earlier hand-aggregated rank-0/1/2/3 implementation, while other standard tensor layouts used a generic cached tensor derivative path.

Runtime controls:

```text
sus2_tensor_force_grad_cache=0/1
SUS2_GPUMD_TENSOR_FORCE_GRAD_CACHE=0/1
```

The old `sus2_l3k3_force_grad_cache` and `SUS2_GPUMD_L3K3_FORCE_GRAD_CACHE` names remain accepted aliases.

Correctness checks used the same binary with tensor cache on/off:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_smoke/tensor_l4k4_cache_parity_20260428

l3k3 Cu-Zr:
  force_mae = 3.9344e-7 eV/A
  force_rmse = 5.2721e-7 eV/A
  force_max = 2.5928e-6 eV/A
  thermo_max = 4.1244e-5

l4k3 MA/Laguerre:
  force_mae = 7.4612e-7 eV/A
  force_rmse = 1.3103e-6 eV/A
  force_max = 9.9093e-6 eV/A
  thermo_max = 6.4100e-5

l4k4 drug/Chebyshev:
  force_mae = 8.0854e-7 eV/A
  force_rmse = 1.2315e-6 eV/A
  force_max = 5.7220e-6 eV/A
  thermo_max = 5.0864e-7
```

Performance check on a 98,304-atom `l4k3` MA/Laguerre case, 100 NPT steps:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/tensor_l4k3_98k_cache_profile_20260428

cache off:
  speed = 1.50229e6 atom-step/s
  force ~= 18.8 ms

cache on:
  speed = 1.75504e6 atom-step/s
  force ~= 9.1 ms

speedup = 1.17x
```

Interpretation:

This pass established the correct and guarded general path up to the current expected maximum `l4k4`, but the first generic `l4` implementation still spent too much time in rank/exponent loops.

## Programmatic Rank-Block Tensor Path Up To l4k4

Date: 2026-04-28

The standard `lLkK` layouts are regular enough that they do not need separate handwritten kernels for every `L,K` combination. For each `k` group, the basic moments are ordered as rank blocks:

```text
rank 0: 1
rank 1: x, y, z
rank 2: xx, xy, xz, yy, yz, zz
rank 3: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
rank 4: xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz, yyyy, yyyz, yyzz, yzzz, zzzz
```

The implementation now detects the same standard `alpha_index_basic` layout and evaluates both directions with fixed rank blocks:

```text
basic accumulation:
  B_i += s_rank(r_ij) * CartesianMonomial_rank(dx, dy, dz)

force contraction:
  dE_i/dr_ij =
    radial_derivative_part * P_rank(dx, dy, dz)
    + radial_value_part * grad_xyz P_rank(dx, dy, dz)
```

Here `P_rank` is the gradient-weighted polynomial for one rank block. This keeps the SUS2 mathematical expression unchanged; it only replaces the inner `(rank,a,b,c)` interpreter loops with a compact programmatic expansion shared by `l2k*`, `l3k*`, and `l4k*` layouts. The exact old `l3k3` path remains enabled for the known fastest Cu-Zr/MA case, but the non-exact tensor path is now also "hand-like" rather than fully generic.

Correctness checks after rank-block contraction, using cache on/off parity:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_smoke/tensor_l4k4_cache_parity_20260428

l3k3 Cu-Zr:
  force_mae = 3.9370e-7 eV/A
  force_rmse = 5.2757e-7 eV/A
  force_max = 2.5481e-6 eV/A
  thermo_max = 4.2898e-5

l4k3 MA/Laguerre:
  force_mae = 1.0801e-6 eV/A
  force_rmse = 1.8777e-6 eV/A
  force_max = 1.5453e-5 eV/A
  thermo_max = 4.6116e-5

l4k4 drug/Chebyshev:
  force_mae = 1.1563e-6 eV/A
  force_rmse = 1.6118e-6 eV/A
  force_max = 5.8413e-6 eV/A
  thermo_max = 5.4511e-7
```

Performance check on the same 98,304-atom `l4k3` MA/Laguerre case, 100 NPT steps:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/tensor_l4k3_98k_cache_profile_20260428/on_rankblock_20260428

old cache off:
  speed = 1.50229e6 atom-step/s
  basic ~= 6.92 ms
  force ~= 18.76 ms

old cache on:
  speed = 1.75504e6 atom-step/s
  basic ~= 6.91 ms
  force ~= 9.07 ms

rank-block cache on:
  speed = 2.05907e6 atom-step/s
  basic ~= 4.08 ms
  force ~= 3.86 ms

speedup vs old cache on = 1.17x
speedup vs cache off = 1.37x
```

Takeaway: the useful pattern is not to maintain many handwritten `l2k3`, `l3k4`, `l4k4` branches. The robust version is to detect the regular tensor layout, then use one rank-block implementation covering the supported `L <= 4`, `K <= 4` family.

## L/K-Specialized Tensor Kernels

Date: 2026-04-28

The next pass pushed the same idea one step closer to handwritten code. Instead of one dynamic rank-block kernel with runtime `tensor_l`, `tensor_k`, `tensor_basic_per_group`, and `MaxBasic`, the code now dispatches standard layouts to compile-time specializations:

```text
gpu_compute_basic_moments_tensor_accum_static<RealT, L, K>
gpu_compute_forces_tensor_cached_grads_static<GradT, RealT, L, K>
```

The supported specialized grid is:

```text
L = 1..4
K = 1..4
basic_per_group = C(L + 3, 3)
basic_count = K * basic_per_group
```

Unsupported or nonstandard layouts still fall back to the dynamic tensor implementation. The exact `l3k3` path remains enabled by default, but forcing it off now routes `l3k3` through the `L=3,K=3` specialization.

Correctness checks after L/K specialization, using cache on/off parity:

```text
directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_smoke/tensor_l4k4_cache_parity_20260428

l3k3 Cu-Zr:
  force_mae = 3.9340e-7 eV/A
  force_rmse = 5.2679e-7 eV/A
  force_max = 2.5481e-6 eV/A
  thermo_max = 1.7242e-5

l4k3 MA/Laguerre:
  force_mae = 9.6303e-7 eV/A
  force_rmse = 1.7336e-6 eV/A
  force_max = 1.7881e-5 eV/A
  thermo_max = 5.5635e-5

l4k4 drug/Chebyshev:
  force_mae = 9.3807e-7 eV/A
  force_rmse = 1.3336e-6 eV/A
  force_max = 6.3181e-6 eV/A
  thermo_max = 3.1296e-7
```

Performance updates:

```text
l4k3 MA/Laguerre, 98,304 atoms, 100 NPT steps:
  directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/tensor_l4k3_98k_cache_profile_20260428/on_lk_static_20260428
  speed = 2.28901e6 atom-step/s
  basic ~= 0.98 ms
  force ~= 2.10 ms

Previous rank-block cache:
  speed = 2.05907e6 atom-step/s
  basic ~= 4.08 ms
  force ~= 3.86 ms

Original dynamic cache-off baseline:
  speed = 1.50229e6 atom-step/s
  basic ~= 6.92 ms
  force ~= 18.76 ms
```

The `l3k3` forced-static check shows that the general `L=3,K=3` specialization has effectively caught the old exact path:

```text
exact l3k3:
  speed = 1.67767e7 atom-step/s
  basic ~= 9.97 ms
  force ~= 20.50 ms

forced L=3,K=3 static path:
  directory = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/cuzr_l3k3_generic_vs_exact_1m_200step_20260428/l3_static_lk_20260428
  speed = 1.67617e7 atom-step/s
  basic ~= 9.97 ms
  force ~= 20.60 ms
```

Takeaway: the model reader still discovers the layout automatically, but the runtime now lands on a compile-time `L,K` specialized path. This keeps the maintainability of automatic detection while recovering handwritten-path performance for the regular tensor families currently used in SUS2-SL.

## Product-Graph Fingerprint And Registry Prototype

Date: 2026-04-28

The product graph must be preserved independently of fitted coefficients so later applications can check whether a model has an existing specialized graph core. The codegen tool now has three explicit cache/registry modes:

```bash
tools/sus2_v11_codegen.py model.mtp --fingerprint-only
tools/sus2_v11_codegen.py model.mtp --cache-dir codegen_cache/sus2_v11 --query-cache
tools/sus2_v11_codegen.py model.mtp --cache-dir codegen_cache/sus2_v11 --list-cache
```

The fingerprint key is the SHA256 of a canonical product-topology payload:

```text
included:
  version
  L
  scaling_map
  radial_funcs_count
  alpha_index_basic
  compressed alpha_index_times
  compressed alpha_moment_mapping
  compressed active moment count

excluded:
  species_count
  element names
  radial_basis_type
  radial/scaling/shift/species/moment fitted coefficients
```

This means the graph core can be reused when the topology is the same even if the chemical elements, radial type, or fitted numeric coefficients differ. It will not be reused when the final scalar graph is different, even if both models are called `l3k3` or `l4k3`.

`l3k3` Cu-Zr prototype:

```text
model = /work/phy-weigw/hyx/cu-zr/7.5/lmp/p.mtp
cache_dir = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codegen_cache/sus2_v11_l3k3_probe_20260428
cache_key = 993c0ffd3b8d1d62e97d9a1b714fba487b0a0d153ca0e7a8f46f9a250bb167f0
layout = l3k3 matched
compressed alpha_times_count = 1291
compressed alpha_moments_count = 514
alpha_scalar_moments = 349
```

First compile preserved the graph:

```text
out_dir = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_codegen/cuzr_l3k3_graph_cache_probe_20260428_first
chunk_size = 512
cache_hit = false
compile_seconds = 10.51
object_bytes = 488544
```

Second compile of the same model hit the preserved graph:

```text
out_dir = /work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_codegen/cuzr_l3k3_graph_cache_probe_20260428_second
cache_hit = true
compile_seconds = 0.0
usable_compiled_core = true
```

Reuse/miss checks:

```text
Synthetic Cu-Zr copy with changed species_count and radial_basis_type:
  same cache_key as original = true
  usable_compiled_core = true

MA/Jacobi l3k3 model:
  model = /work/phy-weigw/hyx/ma/l3k3/jacobi/benchmark_lmp/p1.1.mtp
  layout = l3k3 matched
  cache_key = a1e9a1d3bd1cff4c55dce0e1cb0d2b47789a189389ae27ef68af7502f3167a5c
  same as Cu-Zr key = false
  usable_compiled_core = false
```

The cache directory now keeps a `registry.json` index. For the Cu-Zr probe it contains one entry with the graph key, dimensions, compressed counts, object size, and original compile seconds. This is the piece future GPUMD runtime or build tooling should query before deciding whether to compile a new product graph core.

Next technical step: connect a preserved graph core to runtime execution. The safest path is AOT/cached build integration first: detect the model fingerprint, locate `registry.json`, link or load the matching generated object/cubin if available, otherwise fall back to the current constant-table product graph. Runtime CUDA-driver module loading is possible, but should be added only after the AOT cache path is stable.
