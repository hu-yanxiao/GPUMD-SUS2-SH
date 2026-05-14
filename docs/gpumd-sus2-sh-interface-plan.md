# GPUMD-SUS2-SH Interface Plan

Date: 2026-05-13

## Scope

This project is the independent GPUMD interface for SUS2-SH models. It starts
from the mature moment-tensor `GPUMD-SUS2` implementation, but the angular
representation is replaced by real spherical harmonics and Clebsch-Gordan
coupling products saved by SUS2-SH.

Current target test model:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-work-codex/l3_3333/p.mtp
```

Current target million-atom Cu-Zr test structure:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/cuzr_l3k3_1m_nepfloat_npt2000_20260427/model.xyz
```

The target model is a Cu-Zr metallic-glass SUS2-SH model with:

```text
potential_tag = SUS2-SH
sh_l_max = 3
sh_k_max = 3
sh_body_order = 5
sh_body_l_max = {3, 3, 3, 3}
radial_basis_type = RBChebyshev_sss
max_dist = 7.5
radial_basis_size = 10
radial_funcs_count = 12
alpha_index_basic_count = 48
alpha_moments_count = 1349
sh_product_count = 5475
alpha_scalar_moments = 572
```

## Old Moment-Tensor Interface Bottleneck

The old GPUMD-SUS2 moment-tensor path is already quite optimized for the
`l3k3` Cu-Zr million-atom benchmark. The best mature configuration uses direct
radial recurrence, float moments, product-assign, graph-specific product
groups, and a self-force buffer.

Representative old benchmark records:

```text
CuZr l3k3 1M, old compact tensor model:
  Time used for this run = 11.84 s / 200 steps
  Speed = 1.73e7 atom*step/s
  profile: basic ~9.6 ms, product forward/backward ~18.7 ms, force ~21.3 ms

CuZr l3k3 1M, newer larger tensor graph:
  Time used for this run = 23.92 s / 200 steps
  Speed = 8.56e6 atom*step/s
  profile: basic ~10.0 ms, product forward/backward ~78.4 ms, force ~22.5 ms
```

The main weakness is not radial evaluation or neighbor traversal. It is the
per-atom product DAG. The tensor graph stores a long list of graph-specific
scalar/tensor multiplication rules. Even after grouping and assign-forward,
each atom still performs many small dependent products in a mostly serial
per-atom loop. Previous attempts to parallelize one atom with one CUDA block
were slower because block scheduling, synchronization, shared-memory traffic,
and shared atomic pressure outweighed the extra intra-atom parallelism.

This is a structural mismatch: moment tensors expose products as graph-specific
component rules, while the GPU wants regular blocks of the same small operation
over many atoms.

## SH Advantage

SUS2-SH basic moments are:

```text
B_i[k,l,m] = sum_j R_{k,l}(r_ij) * Y_lm(rhat_ij)
```

For fixed `l`, the angular channel is a dense irrep vector of size `2*l+1`.
For `l <= 4`, the largest vector length is only 9. A product is a standard
Clebsch-Gordan contraction:

```text
C[L,M] += sum_{m1,m2} CG(l1,m1,l2,m2;L,M) * A[l1,m1] * B[l2,m2]
```

This gives two important engineering advantages:

1. The product graph can be grouped into small dense/sparse CG blocks instead
   of arbitrary tensor-component rules.
2. The same `(l1,l2,L)` CG matrix is reused across many `k` combinations and
   across all atoms.

The goal is therefore not to use cuBLAS for one huge GEMM. The operation is
pointwise bilinear per atom, so the efficient GPU form is custom kernels that
map threads over `(atom, coupling block, output M)` and use compact constant or
read-only CG tables.

## Implementation Strategy

### Phase 1: Correct Generic SUS2-SH Backend

Add a new `SUS2_SH` potential class, selected when the model file starts with
`MTP` and contains:

```text
potential_tag = SUS2-SH
```

Do not compile the old tensor `SUS2_V11` class in this independent SH build.
Moment-tensor SUS2 models remain in the separate `GPUMD-SUS2` repository. This
keeps the SH compile path short and avoids carrying tensor product-codegen
optimizations that cannot be reused by the SH `sh_products`/CG path.

Required parser fields:

```text
species_count
scaling
scaling_map
radial_basis_type
min_dist / max_dist
radial_basis_size
radial_funcs_count
shift_coeffs
scal_coeffs
radial_coeffs
alpha_index_basic = {(k,l,m), ...}
sh_product_count
sh_products = {(src0, src1, dst, coeff), ...}
alpha_moment_mapping
moment_coeffs
```

The first correctness path should be deliberately simple:

- evaluate real spherical harmonic basics `B[k,l,m]`;
- run the saved `sh_products` in topological order using assign/accumulate
  semantics matching SUS2-SH;
- initialize scalar gradients from `alpha_moment_mapping` and `moment_coeffs`;
- run reverse-mode over `sh_products`;
- compute forces and virials by differentiating the real SH basics and radial
  basis with respect to each neighbor displacement.

This path is expected to be slower than the final target, but it fixes parsing,
physics, units, force signs, virial convention, periodic image handling, and
model compatibility first.

First implementation status:

- `src/force/sus2_sh.cu/.cuh` implements this generic path;
- `src/force/force.cu` dispatches `MTP` plus `potential_tag = SUS2-SH` to
  `SUS2_SH`;
- non-SH `MTP` files are rejected instead of falling back to the old tensor
  backend;
- the default SH runtime mode is the tested SUS2 setting: float moments and
  direct radial recurrence;
- the initial supported subset is `RBChebyshev_sss`, `scaling_map = LK`, and
  `sh_l_max <= 4`.

### Phase 2: Real-SH Basic Fast Path

Use real spherical harmonics directly. Do not evaluate complex harmonics and
transform them. For `l <= 4`, implement or generate real polynomial
expressions for both `Y_lm(x/r,y/r,z/r)` and their Cartesian derivatives.

The force kernel needs:

```text
d/dx [R(r) * Y_lm(rhat)]
  = R'(r) * (x/r) * Y_lm + R(r) * dY_lm/dx
```

and similarly for `y,z`. This must be checked against the SUS2-SH CPU
implementation before large tests.

### Phase 3: CG Block Product Kernels

Analyze `sh_products` on load and pack it into coupling blocks:

```text
src0_base, src1_base, dst_base
l0, l1, L
dim0 = 2*l0 + 1
dim1 = 2*l1 + 1
dim_dst = 2*L + 1
cg_coeff_offset
```

Forward kernel shape:

```text
one or more threads per (atom, coupling block, M)
C[M] = sum_{m1,m2} CG[M,m1,m2] * A[m1] * B[m2]
```

This replaces thousands of flat product rows with regular small CG contractions
across all atoms. Since `N` is about 1,024,000 for the Cu-Zr test, mapping work
over atoms gives high occupancy without the one-block-per-atom overhead that
hurt the tensor experiment.

Backward kernel shape:

```text
dA[m1] += sum_{M,m2} CG[M,m1,m2] * dC[M] * B[m2]
dB[m2] += sum_{M,m1} CG[M,m1,m2] * dC[M] * A[m1]
```

The first optimized version may use global atomics to basic/intermediate
gradients for simplicity. The preferred version should build reverse adjacency
by source component and assign each source gradient once per reverse layer, so
the backward contraction is also a matrix-style reduction without global atomic
contention.

### Phase 4: Layered Execution and Memory Control

The SH standard trees have natural product layers. The loader should recover
layers from the `sh_products` topological order and execute one packed kernel
per layer. This gives deterministic forward/backward order and avoids races in
assign-forward moments.

The initial memory layout should stay structure-of-arrays:

```text
moments[moment_id * N + atom]
grads[moment_id * N + atom]
```

For the target SH model, float moments and gradients require about:

```text
1349 moments * 1,024,000 atoms * 4 bytes * 2 arrays ~= 10.5 GB
```

This is acceptable on A100 80 GB and smaller than the newer old tensor graph
with 2024 moments. Later work can shrink memory by keeping only live layers,
but correctness and comparable performance should come first.

## Correctness Tests

1. Static parser smoke on the target `p.mtp`.
2. Single-structure parity against `SUS2-SH` CPU `calc-efs` for energy and force.
3. Virial/stress parity using GPUMD `time_step 0`, `run 1`, and `dump_thermo`.
4. Small Cu-Zr dynamic smoke.
5. Million-atom Cu-Zr benchmark using the existing 1,024,000 atom `model.xyz`.

The multi-image neighbor displacement fix from the old GPUMD-SUS2 project must
be preserved. GPUMD-SUS2-SH must store and use the actual periodic image
displacement per neighbor edge, not only the neighbor atom id.

## Benchmark Baseline

Initial large-system comparison should reuse the old run style:

```text
potential p.mtp Cu Zr sus2_float=1 sus2_radial_direct=1
velocity 200 seed 9174
ensemble npt_mttk temp 200 200 aniso 0.0001 0.0001 tperiod 50 pperiod 500
time_step 1
dump_thermo 50
run 200
```

The benchmark directory should be under the new server project, for example:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-work-codex/codex_bench/cuzr_sh_l3k3_1m_profile200_YYYYMMDD
```

Do not mix new SH benchmark outputs into the old `GPUMD-SUS2-v1.1-work-codex`
tree except as read-only references.

## Optimization Notes

First 1,024,000 atom Cu-Zr SH benchmark:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_first_profile200_20260513
steps = 200
time = 44.0047 s
speed = 4.65405e6 atom*step/s
```

Profile on the same model shows the generic flat `sh_products` path is not the
final SH direction:

```text
neighbor   ~= 7.2 ms/step
memset     ~= 5.9 ms/step
basic      ~= 43.3 ms/step
product    ~= 71.3 ms/step
force      ~= 89.8 ms/step
accumulate ~= 0.15 ms/step
```

The first accepted low-risk optimization packs basic metadata from `(mu,l,m)` to
`(mu,yidx)`, where `yidx = l*l + (m+l)`. This reduces repeated integer loads and
index arithmetic in both basic and force kernels without changing any floating
point operation order.

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_packedbasic_200_20260513
steps = 200
time = 42.933 s
speed = 4.77022e6 atom*step/s
```

The force basic-gradient cache was retested after the SH basic kernel changes
and is now enabled by default when `alpha_index_basic_count <= 64`. It can still
be disabled with `sus2_sh_force_grad_cache=0`. The cache keeps each center
atom's basic adjoints in a small local array while traversing its neighbors;
this preserves the force expression but avoids repeated global gradient loads.

The next material optimization should not continue tuning the flat product DAG.
SUS2-SH's intended engineering advantage is a standardized block/layer
execution model:

```text
basic block:   base, k, mu, l, count = 2*l + 1
CG block:      layer, left_block, right_block, target_base, l1, l2, L
CG terms:      M_offset, m1_offset, m2_offset, coeff
scalar map:    scalar_index -> block/component or moment id
```

For long-term robustness, SUS2-SH model generation should write explicit
`sh_cg_blocks`, `sh_cg_terms`, and `sh_cg_layers` metadata, together with the
real-SH and CG phase convention/version. GPUMD can then validate and execute the
standard representation directly instead of inferring it from a flat product
list.

### 2026-05-13 Tensor-Product Tests

The GPUMD loader now reconstructs the same standard real-CG graph used by
SUS2-SH model generation and validates it against the saved `sh_products`:

```text
tensor_blocks = 701
cg_blocks     = 689
cg_terms      = 5787
cg_rows       = 1301
cg_row_terms  = 5787
cg_back_rows  = 783
cg_back_terms = 11574
layers        = 2
```

The first true tensor-product experiment uses component rows for the forward
CG contraction and source-component adjoint rows for the reverse pass. This is
mathematically correct on the 1,024,000 atom Cu-Zr test: the one-step total
energy agrees with the flat path to displayed precision, with stress-level
differences around `1e-5` from floating-point ordering.

Performance on A100:

```text
flat packed-basic path:
  product ~= 71-73 ms/step
  speed   ~= 4.79e6 atom*step/s

tensor row-adjoint, original 65535 CTA cap:
  product ~= 87.8 ms/step
  speed   ~= 4.58e6 atom*step/s

tensor row-adjoint, 8192 CTA cap:
  product ~= 84.4 ms/step
  speed   ~= 4.65e6 atom*step/s
```

A 2D `(row, atom-tile)` launch that removed the flattened `task / N` and
`task % N` indexing was tested and rejected. It remained correct, but product
time rose to about `141 ms/step` because each row received too few CTAs and the
kernel turned into long per-thread atom loops. That experiment was reverted.

Current conclusion: the tensor-product direction is right, but global
component-row kernels are not the final GPU mapping. The next implementation
should use layer-synchronous CG block/source-block tiles:

```text
forward:  (layer, CG block, atom tile, output M)
backward: (layer, source block, atom tile, source m)
```

This keeps the standardized SH/CG tensor-product representation while avoiding
both per-atom serial DAG execution and overly fragmented component-row kernels.

### 2026-05-13 Basis/Force Cache Pass

Accepted low-risk changes:

- `eval_real_sh()` clears only the active `(lmax+1)^2` values for the no-derivative
  basic-moment path, while the force derivative path keeps the original fixed
  `kMaxSHComponents` clearing.
- `gpu_sh_compute_basic` dispatches to 64/128/basic-capacity specializations so
  the common l3k3 case uses a 64-entry local basic array instead of the 256-entry
  fallback.
- `sus2_sh_force_grad_cache` defaults on for `alpha_index_basic_count <= 64`.
- The block-forward tensor experiment was removed after profiling showed it was
  slower than the row-adjoint path.

Verified on the 1,024,000 atom Cu-Zr l3k3 model on A100:

```text
default flat path:
  case    = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_default_flat_profile200_20260513
  basic   ~= 35.8 ms/step
  product ~= 71.5 ms/step
  force   ~= 77.7-78.0 ms/step
  speed   = 5.14507e6 atom*step/s

tensor row-adjoint path:
  case    = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_default_tensor_profile200_20260513
  basic   ~= 35.8 ms/step
  product ~= 83.8 ms/step
  force   ~= 77.1-77.6 ms/step
  speed   = 4.92559e6 atom*step/s
```

Current conclusion: basis/basic is no longer the immediate bottleneck for l3k3;
the default flat product path is still faster than the current tensor row-adjoint
implementation. The next meaningful tensor-product optimization should target
block/source-block tiled contractions or a compact standard layer program, not
the rejected one-thread-per-block forward kernel.

### 2026-05-13 Compact Serial Row Program

Accepted change: the default non-tensor product path now uses the compact
target-row/source-row program inside the existing per-atom fused kernel. This
keeps one product kernel launch and the same layer-ordered chain rule, but each
forward target moment and each backward source gradient is accumulated in a
register and written once. Because all tensor targets are assigned explicitly,
the product value buffer no longer needs the full `alpha_moments_count * N`
memset. The old flat product loop remains available with
`sus2_sh_compact_serial_product=0`.

Verified on the 1,024,000 atom Cu-Zr l3k3 model on A100:

```text
case    = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_compact_serial_profile200_20260513
basic   ~= 35.7-35.8 ms/step
product ~= 71.4-71.5 ms/step
force   ~= 77.0-77.2 ms/step
memset  ~= 3.0 ms/step
speed   = 5.23633e6 atom*step/s
```

Thermo comparison against the previous flat path over the 200-step profile run
showed maximum total-energy difference `3.1e-3 eV` for 1,024,000 atoms, i.e.
about `3e-9 eV/atom`, consistent with float summation order differences.

Follow-up basis probe from the Cartesian-polynomial observation: a dedicated
value-only `eval_real_sh` path was tested so the basic-moment kernel would not
share the derivative helper. This was not retained. On the same l3k3 A100
profile, `basic` stayed at about `35.7-36.0 ms/step` and speed was
`5.23573e6 atom*step/s`, indistinguishable from the compact-serial run. This
confirms that the current SH evaluation is already polynomial-based and that the
remaining basic-stage cost is dominated by neighbor traversal, radial
tanh/Chebyshev evaluation, and basic-channel accumulation rather than the
Cartesian harmonic value formulas.

### 2026-05-13 Block/Topology Tile Probe

Tested a topology-aware CG-block/atom-tile forward kernel grouped by layer and
`(l1,l2)->L` topology, while keeping the existing source-row backward path for
chain-rule safety. The code was reverted after profiling because it increased
the product stage substantially:

```text
case    = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_tensor_blocktile_profile200_20260513
basic   ~= 35.8 ms/step
product ~= 120.3 ms/step
force   ~= 77.2 ms/step
speed   = 4.18693e6 atom*step/s
```

Conclusion: merely batching one CG block over atom tiles is too coarse and
loads too much shared/register state for l3k3. The useful form of topology
classification is more likely to be specialized compact component-row kernels
or layer programs per `(l1,l2,L)` type, not a generic block tile that serializes
all terms for each atom thread.

### 2026-05-13 NEP/Moment Comparison and Rejected Force Probes

The current working explanation for the speed gap against the old moment
interface is:

- The old moment path is fast because it has mature low-level fast paths:
  Cartesian tensor basic/force kernels, graph-specific packed metadata,
  assign-forward product groups, selective gradient initialization, float
  intermediates, direct radial tables, and force self-buffering.
- Official GPUMD NEP uses the spherical-harmonic descriptor form in the
  documentation, but the CUDA implementation evaluates Cartesian/solid-harmonic
  polynomials and contracts them into fixed invariants. NEP avoids generic CG
  tensor products; it stores directed-edge partial forces and later combines
  reverse edges.
- SUS2-SH has a different requirement: it must preserve the trained standard
  SH/CG scalar list and tensor-product path. Therefore NEP's polynomial basis
  and mixed precision are transferable, but NEP's fixed invariant contraction
  cannot replace the SUS2-SH CG graph without changing the model definition.

Rejected probes on the 1,024,000 atom Cu-Zr l3k3 model:

```text
directed-edge force reduce:
  idea    = NEP-style f12 storage plus reverse-edge reduce, avoiding force atomics
  case    = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_force_edge_reduce_profile200_20260513
  force   ~= 115 ms/step
  speed   = 4.32113e6 atom*step/s
  result  = rejected; reverse-edge lookup and extra edge-force traffic outweighed atomic removal

radial value-only basic:
  idea    = basic stage calls a value-only direct radial evaluator instead of the value/derivative helper
  same-GPU A/B speed = 5.23653e6 -> 5.23833e6 atom*step/s
  result  = not retained; mathematically safe but too small to justify extra code complexity

grouped (k,l) basic/force:
  idea    = regroup force chain rule by mu=(k,l), forming sum_m adj*Y_lm and sum_m adj*dY_lm
  case    = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_grouped_basic_force_profile200_20260513
  basic   ~= 47 ms/step
  force   ~= 103 ms/step
  speed   = 4.34924e6 atom*step/s
  result  = rejected; extra local state/control flow increased register pressure
```

Accepted tiny cleanup: `force_self_tmp` and `virial_tmp` are no longer memset
before the force kernel because the force kernel overwrites all per-atom entries.
The same l3k3 profile is unchanged within noise:

```text
case    = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_nomemset_single_profile200_20260513
speed   = 5.23439e6 atom*step/s
```

Accepted metadata cleanup: the compact serial forward row table is now also
packed into CUDA constant memory when it fits the 64 KiB constant-memory budget.
This borrows the old moment interface idea of keeping small, read-only graph
metadata off the regular global-memory path. The backward source-row table stays
in global memory because it is too large for this budget. The feature is enabled
by default for the float compact path and can be disabled with
`sus2_sh_const_forward=0` or `SUS2_SH_GPUMD_CONST_FORWARD=0`.

Same-GPU A/B on `a05u22g` for the 1,024,000 atom Cu-Zr l3k3 model:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_ab_const_forward_20260513

const forward off:
  product ~= 75.0-75.2 ms/step
  speed   = 5.03222e6 atom*step/s

const forward on:
  product ~= 74.0-74.2 ms/step
  speed   = 5.06498e6 atom*step/s
```

The maximum thermo difference over the A/B run was `3.5e-3` in the printed
columns, consistent with the already accepted float summation-order scale for
this million-atom test. This is a small gain, but it is low-risk and keeps the
path closer to the old optimized moment metadata layout.

Rejected full direct-polynomial expansion probe: expanding all scalar outputs
for the l3k3 5body model into weighted products of basic `B(k,l,m)` channels
created `11536` merged monomials with derivative occurrence work around
`45028`. The degree distribution was dominated by 4-factor terms. This is not
clearly cheaper than the current compact forward/backward row walk unless it is
recast into a block/quadratic pair-tensor form, so no direct scalar polynomial
kernel was kept.

### 2026-05-13 Static SH Basic and Terminal Scalar Fusion

Accepted change: the GPUMD basic-stage kernel now has a guarded static path for
the SUS2-SH generator's full `(l,k,m)` layout with `l <= 4`, `k <= 6`, and
`radial_basis_size = 10`. The guard checks the actual `alpha_index_basic` order
from the model: active `q=(k,l)` tensors are generated in the training order
`l` descending, `k` descending, and `m=-l..l`. If the model does not match that
layout, GPUMD falls back to the generic SH path. The static path can be disabled
with `sus2_sh_static_basic=0` or `SUS2_SH_GPUMD_STATIC_BASIC=0`.

Same-GPU A/B on `c04u01g`, 1,024,000 atom Cu-Zr l3k3:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_ab_static_basic_20260513

static basic off:
  basic ~= 35.7 ms/step
  speed = 5.25315e6 atom*step/s

static basic on:
  basic ~= 7.1 ms/step
  speed = 6.16893e6 atom*step/s
```

The maximum printed thermo difference over the run was `2.1e-3`, consistent
with float summation-order changes. This confirms that the previous SH basic
bottleneck was mostly the generic per-basic accumulation and repeated
value-form bookkeeping, not the radial recurrence itself.

Accepted change: the compact product kernel now fuses terminal scalar rows.
For scalar targets that are not used as sources by any later tensor product, the
kernel accumulates site energy and the direct chain-rule contributions to the
left/right source moments inside the forward row. The corresponding terminal
targets are skipped in scalar seeding, and their back-row terms are removed
from the reverse table. This keeps the trained scalar definition unchanged but
avoids materializing and then immediately differentiating terminal scalar nodes.
The feature is enabled by default for the compact product path and can be
disabled with `sus2_sh_terminal_scalar_fusion=0` or
`SUS2_SH_GPUMD_TERMINAL_SCALAR_FUSION=0`.

Same-GPU A/B on `a05u22g`, with static basic and constant forward metadata both
enabled:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_ab_terminal_scalar_20260513

terminal fusion off:
  product ~= 76.6-77.1 ms/step
  speed   = 5.82452e6 atom*step/s

terminal fusion on:
  terminal_scalars = 545
  removed_back_terms = 4470
  product ~= 65.7-65.8 ms/step
  speed   = 6.19759e6 atom*step/s
```

The maximum printed thermo difference over the terminal-fusion A/B run was
`1.03e-2` on the million-atom system, i.e. about `1e-8` per atom in the energy
scale. This is larger than the pure static-basic A/B because scalar accumulation
order changes more substantially, but still in the expected float-order range
for this system size.

Next high-confidence direction: product and force are now the dominant costs.
The next product step should group the remaining non-terminal rows by
`(l1,l2,L)` topology or by pair-tensor block so the current per-atom serial row
walk can be replaced by a smaller set of structured contractions. The next
force step should mirror the static-basic value path with an equally structured
derivative path, while keeping the current cached-gradient force kernel as the
correctness reference.

Accepted change: the force kernel now has a static standard-layout path for
full SUS2-SH `(l,k,m)` models with `l <= 4`, `k <= 6`, and
`radial_basis_size = 10`. It keeps the same chain rule,
`sum_m g_{klm} (R'_{kl} Y_lm rhat + R_{kl} dY_lm/dr)`, but contracts each
fixed `(k,l)` group before applying the radial derivative. This removes the
generic per-basic metadata loop in the force stage. The path is guarded by the
same full-layout check as static basic and can be disabled with
`sus2_sh_static_force=0` or `SUS2_SH_GPUMD_STATIC_FORCE=0`.

Same-GPU A/B on `a05u22g`, 1,024,000 atom Cu-Zr l3k3, with static basic,
constant forward metadata, and terminal scalar fusion enabled:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_ab_static_force_20260513

static force off:
  product ~= 65.4-65.7 ms/step
  force   ~= 79.5-80.1 ms/step
  speed   = 6.21011e6 atom*step/s

static force on:
  product ~= 65.5-65.6 ms/step
  force   ~= 18.8-19.0 ms/step
  speed   = 9.86625e6 atom*step/s
```

The maximum printed thermo-column difference was `4.6e-3` in total energy for
the million-atom system, consistent with changed float contraction order. This
recovers the old moment interface's force cost level while preserving the
standard SH basis and CG graph.

Accepted product change: the compact product path can split the final source-row
reverse pass into a row-by-atom parallel kernel. Forward row evaluation,
terminal scalar energy/gradient fusion, and scalar gradient seeding remain in
the compact per-atom kernel; only the pruned back-row table is run in parallel.
This is enabled by default when the validated graph has at least `4096`
back-row terms and can be disabled with `sus2_sh_parallel_back_rows=0` or
`SUS2_SH_GPUMD_PARALLEL_BACK_ROWS=0`.

Same-GPU A/B on `a05u22g`, with static force enabled:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_l3k3_1m_ab_parallel_back_20260513

parallel back rows off:
  product ~= 65.1-65.3 ms/step
  force   ~= 18.4-18.8 ms/step
  speed   = 9.95839e6 atom*step/s

parallel back rows on:
  product ~= 62.3-62.4 ms/step
  force   ~= 18.9-19.0 ms/step
  speed   = 1.01514e7 atom*step/s
```

The maximum printed thermo-column difference was `2.9e-3` in total energy,
again within the float-order scale. Product remains the dominant gap versus the
old moment backend, so the next serious direction is still a more compact
standard layer program for the layer-1 pair tensor contractions and terminal
scalar contractions, not more force work.

Accepted product changes after the two-model optimization pass:

- Active scalar seeding now stores only scalar moments that still need a
  separate seed after terminal scalar fusion. This removes the loop over the
  full scalar list in the compact, flat, cg-block, and tensor-parallel product
  paths.
- The back-row metadata can be packed into `uint32` row and term words. This is
  enabled only when the model also uses the parallel back-row path; on smaller
  graphs such as l3322 the old in-kernel back-row loop is faster.
- The compact terminal-scalar path now uses an explicit terminal-row predicate,
  which keeps the same chain rule but makes later row-scalar experiments
  possible.
- `sus2_sh_row_scalar_fusion=1` is available as an experimental option, but is
  not the default because same-binary off/on timing showed no reliable extra
  speed beyond the terminal-row restructuring.

Same-GPU A/B on `c04u01g`, 1,024,000 atom Cu-Zr, 200 NPT steps:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_rowscalar_ab_20260513

l3333, default row scalar off:
  terminal_scalars = 545
  active_scalar_seeds = 27
  product ~= 58.33-58.36 ms/step
  speed   = 1.09579e7 atom*step/s

l3333, row scalar on:
  active_scalar_seeds = 3
  product ~= 58.33 ms/step
  speed   = 1.09553e7 atom*step/s

l3322, default row scalar off:
  terminal_scalars = 240
  active_scalar_seeds = 21
  product ~= 14.98-14.99 ms/step
  speed   = 2.12200e7 atom*step/s

l3322, row scalar on:
  active_scalar_seeds = 3
  product ~= 14.98-14.99 ms/step
  speed   = 2.12581e7 atom*step/s
```

The row-scalar off/on `run 1` A/B showed only float-order differences:
total-energy changes were about `1.1e-1` eV for l3333 and `4.4e-2` eV for
l3322 over the million-atom box, i.e. about `1e-7` eV/atom or less. Because the
timing gain was not distinct from noise, the row-scalar option stays off by
default.

Follow-up math check, same code, l3333 model, first 4096 atoms, `time_step 0`,
`dump_force 1`, `run 1`:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_rowscalar_mathcheck_20260513

float, row scalar off/on:
  total energy diff = 7.7e-5 eV over 4096 atoms
  force max abs diff = 7.674e-6
  force rms diff     = 7.886e-7

double, row scalar off/on:
  total energy diff = 0 at printed precision
  force max abs diff = 1.192e-7
  force rms diff     = 9.331e-9

float off versus double off:
  force max abs diff = 4.295e-5
  force rms diff     = 1.874e-6
```

This separates trajectory divergence from the algebra. Row-scalar fusion changes
where scalar site-energy contributions and gradient seeds are accumulated, but
does not change the tensor-product expression in exact arithmetic. The double
off/on result is essentially identical at output precision, and the float off/on
force difference is smaller than the normal float-versus-double force difference.
The larger differences seen after hundreds of NPT steps are therefore consistent
with finite-precision summation order and chaotic trajectory divergence, not a
missing CG contraction or chain-rule term. The option remains experimental and
off by default because it also has no clear speed benefit.

The next product strategy should exploit the fact that SH coupling topology is
independent of `k`: CG term patterns depend on `(l1,l2,L,m)` while `k` only
selects radial channels and tensor instances. A useful implementation must
batch multiple `k` instances of the same `(l1,l2,L)` topology so coefficient
loads and selected left/right component values can be reused. Merely storing a
single copy of the CG coefficient table is unlikely to be enough, because the
current constant-memory row table is already cheap; the target is reduced
global moment traffic and fewer repeated per-row loops. The safest first
experiment is a limited terminal-scalar or layer-1 pair-tensor microkernel for
the hottest `(l1,l2,L)` groups, with the compact row program retained as the
correctness fallback.

2026-05-14 product optimization pass:

- `sus2_sh_const_back=1` was tested by moving packed back-row metadata into the
  constant buffer while disabling forward constant metadata. It was slower on
  l3333 (`product ~=57.9 ms` versus `56.5 ms`), so it remains an experimental
  off-by-default option.
- A terminal-scalar dot fast path is now used for rows whose terms are exactly
  contiguous dot products with one uniform CG coefficient. In the l3333 model
  all 545 terminal scalar rows match this pattern (`2235` terms); in l3322 all
  240 terminal rows match (`738` terms).
- Duplicate back terms with the same `(target, other)` inside a source row are
  merged by summing coefficients. This removes `204/6480` back terms in l3333
  and `90/1356` in l3322.
- A coarse product basic-moment register cache was tested and rejected. Although
  it removes repeated basic-moment global loads, caching 48 basics per thread
  increases register pressure enough to slow l3333 (`product ~=56.8 ms` to
  `58.0 ms`). Any future basic reuse should be topology-local, not a full basic
  cache.

Four-way l3333 A/B on the 1,024,000 atom Cu-Zr box, 200 NPT steps:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_tdot_merge_matrix_20260514

base:
  cg_back_terms = 6480
  product = 56.980 ms
  speed   = 1.11158e7 atom*step/s

terminal-dot only:
  product = 56.800 ms
  speed   = 1.11377e7 atom*step/s

merge-back only:
  cg_back_terms = 6276
  product = 56.374 ms
  speed   = 1.12032e7 atom*step/s

terminal-dot + merge-back:
  product = 56.204 ms
  speed   = 1.12130e7 atom*step/s
```

The l3322 default check with the same final code gave:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_final_l3322_check_20260514

cg_back_terms = 1266
product = 14.764 ms
speed   = 2.13467e7 atom*step/s
```

Correctness check, l3333 first 4096 atoms, `time_step 0`, `run 1`, comparing
base (`terminal_dot=0`, `merge_back_duplicates=0`) against the final default
path:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_tdot_merge_mathcheck_20260514

total-energy diff = 1.87e-4 eV over 4096 atoms
force max abs diff = 5.96e-7
force rms diff     = 1.26e-7
```

This is within float summation-order scale. The next high-potential direction
is not full per-thread basic caching, but a generated topology-local product
program for the hottest layer-1 `(l1,l2,L)` groups, where a small set of basic
components can be reused without inflating register pressure across the whole
kernel.

2026-05-14 follow-up optimization pass:

- Accepted change: selective gradient zeroing is now enabled for the compact
  product path. Instead of clearing every `alpha_moments_count * N` gradient
  entry, the loader marks only moments that can receive a gradient seed or a
  reverse-mode contribution. The full memset path remains available with
  `sus2_sh_selective_grad_zero=0` or `SUS2_SH_GPUMD_SELECTIVE_GRAD_ZERO=0`.
- l3333 marks `804/1349` moments for gradient clearing. In the million-atom,
  200-step A100 check, `memset` dropped from about `2.96 ms` to about
  `2.59-2.63 ms` and speed improved from `1.12077e7` to `1.12467e7`
  atom-step/s. A rebuild after reverting the failed direct-back experiment
  reproduced `product ~=56.21 ms` and `speed = 1.12568e7`.
- l3322 marks `285/525` moments. The current binary gives `memset ~=0.93 ms`,
  `product ~=14.77 ms`, and `speed = 2.1461e7` atom-step/s on the same
  million-atom, 200-step A100 check.
- Correctness check for l3333 first 4096 atoms, `time_step 0`, `run 1`,
  comparing selective zero off/on:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_selective_grad_zero_mathcheck_20260514

force max abs diff = 1.192092896e-7
force rms diff     = 1.4153e-8
```

Rejected experiments from this pass:

- Layer-1 block-forward microkernel: force differences were only float
  summation-order scale (`max ~=4.8e-7`, `rms ~=6.9e-8`), but the million-atom
  run slowed from `1.12605e7` to `1.08194e7` atom-step/s. The likely cause is
  register/local-array pressure, so the code was reverted.
- Terminal direct-back expansion: replacing terminal-row reverse propagation
  with nested direct propagation preserved energy and kept force differences
  near float scale (`max ~=6.1e-6`, `rms ~=1.1e-6`), but it slowed product from
  about `61 ms` to about `123 ms`. The nested per-terminal CG expansion
  duplicated too much work, and the runtime branch also slowed the off path.
  The implementation was removed completely; no `terminal_direct` code remains.

CUDA SH / tensor-product references checked for future work:

- [`sphericart`](https://github.com/lab-cosmo/sphericart): useful reference for
  real/solid spherical harmonics and derivatives with CUDA C++ APIs. The most
  relevant idea is generating low-order SH values and derivatives together
  using fixed polynomial/recurrence structure.
- [`OpenEquivariance`](https://github.com/vbharadwaj-bk/OpenEquivariance) and
  [`cuEquivariance`](https://github.com/NVIDIA/cuEquivariance): useful design
  references for sparse equivariant tensor-product kernels, segmented tensor
  products, and generated contraction schedules.
- [`e3nn.c`](https://github.com/teddykoker/e3nn.c): useful small C reference
  for spherical harmonics and tensor products, especially the distinction
  between generic sparse CG traversal and generated straight-line formulas.

Current product conclusion: product remains the largest cost, and the next
high-confidence direction is a guarded product-v2 path rather than more tuning
inside the current compact row loop. The old compact path must stay as the
default fallback while product-v2 is tested. The product-v2 candidate should
group rows by `(l_left, l_right, L, layer, term pattern)` and exploit that the
CG topology is independent of `k`: the same component contraction repeats for
many radial-channel pairs. The implementation should avoid large per-thread
arrays and avoid recursive terminal expansion. A generated or prepacked
topology-local contraction plan, with coefficients in constant memory and only
small per-group reusable component windows in registers, is the most plausible
route to a larger product reduction.

2026-05-14 `k <= 6` update:

- The GPUMD backend now treats `l <= 4, k <= 6` as the supported static-layout
  design range. Static basic and static force dispatch include `K=5` and `K=6`.
  The maximum static force-gradient cache is extended to 256 basic channels,
  which covers the largest intended `l=4,k=6` layout (`150` basics).
- The forward-row constant-memory table limit was raised from `16000` to
  `16384` `uint32` words, matching the 64 KiB CUDA constant-memory boundary.
  This lets the `l4k5_4422` model use `const forward rows: on`; its packed
  forward table needs just over the previous artificial limit.
- Added benchmark models to the regular profile matrix:

```text
l4k4_4422 = /work/phy-weigw/20260321_Test/SUS2-SH-work-codex/l4k4_4422/p.mtp
l4k5_4422 = /work/phy-weigw/20260321_Test/SUS2-SH-work-codex/l4k4_4422/k5/p.mtp
```

Current 1,024,000 atom Cu-Zr 200-step A100 profile matrix, using the average of
the last three 50-step profile windows:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_k6_const16384_profile200_20260514

model       speed(atom*step/s)  neighbor  memset  basic   product  force   accumulate  total
l3333       1.06872e7           5.220     2.886   7.787   57.444   18.969  0.148       92.455
l3322       2.07820e7           5.001     0.973   7.402   14.970   17.648  0.147       46.140
l4k4_4422   1.15609e7           5.452     1.644   14.979  32.324   31.184  0.147       85.731
l4k5_4422   7.95493e6           5.166     2.418   16.017  60.597   41.474  0.150       125.821
```

The constant-table boundary change is useful but small. For `l4k5_4422`, product
time moved from about `60.78 ms` to about `60.60 ms`. The main optimization
target remains product-v2: share forward-row CG term patterns across repeated
`k`/block instances while keeping the current compact path as fallback.

2026-05-14 A100 `sm_80` product-pattern update:

- The server build must target A100 with `CUDA_ARCH=-arch=sm_80`. A previous
  build command accidentally used the upstream makefile default `sm_60`; the
  benchmark below is from a clean `sm_80` rebuild.
- Added guarded forward product pattern rows. The row instances keep their own
  `target` and base moment ids, while repeated relative `(left_offset,
  right_offset, coeff)` term patterns are stored once. This changes metadata
  packing and readout only; the compact tensor-product graph and reverse-mode
  back rows are unchanged.
- The option is controlled by `sus2_sh_product_pattern_rows=` or
  `SUS2_SH_GPUMD_PRODUCT_PATTERN_ROWS`. Default heuristic: enable it when
  `cg_row_terms >= 2500`; this keeps the smaller `l3322` model on the older
  const-forward row path, where pattern decode was slightly slower.
- Pattern packing statistics for the current models:

```text
model       rows  row_terms  patterns  pattern_terms  old_u32  pattern_u32
l3333       1301  5475       165       811            14853    4554
l3322       477   1416       48        180            4263     1410
l4k4_4422   968   2996       48        180            8896     2392
l4k5_4422   1730  5410       48        180            16010    3916
```

`sm_80` 1,024,000 atom Cu-Zr 200-step A/B profile matrix. Times are averages
over the last three 50-step profile windows:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_sm80_pattern_ab_profile200_20260514

model       mode     speed(atom*step/s)  neighbor  memset  basic   product  force   accumulate  total
l3333       off      1.05156e7           5.156     2.836   7.714   59.507   18.727  0.147       94.087
l3333       pattern  1.07244e7           5.128     2.808   7.623   57.844   18.578  0.148       92.128
l3322       off      2.14351e7           4.857     0.931   7.110   14.891   16.880  0.147       44.816
l3322       pattern  2.13570e7           4.848     0.934   7.107   15.043   16.870  0.148       44.949
l4k4_4422   off      1.15164e7           5.443     1.645   14.953  32.663   31.131  0.148       85.984
l4k4_4422   pattern  1.16588e7           5.446     1.645   14.967  31.555   31.160  0.147       84.921
l4k5_4422   off      7.90063e6           5.170     2.418   16.026  61.435   41.518  0.150       126.718
l4k5_4422   pattern  7.99383e6           5.171     2.418   16.032  59.849   41.510  0.149       125.130
```

Current default behavior after the heuristic is therefore:

```text
model       default path  speed(atom*step/s)  product  total
l3333       pattern       1.07244e7           57.844   92.128
l3322       const rows    2.14351e7           14.891   44.816
l4k4_4422   pattern       1.16588e7           31.555   84.921
l4k5_4422   pattern       7.99383e6           59.849   125.130
```

Correctness check: first 4096 Cu-Zr atoms, `l4k5_4422`, same LSF job and same
GPU, `run 1`, comparing pattern off/on with dumped forces:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_sm80_pattern_correctness_20260514/l4k5

force max abs diff = 4.613399506e-5
```

2026-05-14 terminal-dot-group update:

- A large fraction of product rows are terminal scalar dot products, and many
  rows share the same left-vector base. The new grouped path packs rows by
  `(layer, left0, dot_count)`, loads the left vector once, loops over all right
  entries in that group, and writes the left gradients once per group. This
  keeps the same scalar value and reverse-mode chain rule as the terminal-dot
  row path; only the local accumulation order changes.
- The option is controlled by `sus2_sh_terminal_dot_groups=` or
  `SUS2_SH_GPUMD_TERMINAL_DOT_GROUPS`. Default heuristic: enable it when
  `cg_row_terms >= 2500`. The smaller `l3322` graph is deliberately left on
  the previous path because grouped dots were slower there.
- Terminal dot grouping statistics:

```text
model       terminal_dot_rows  dot_terms  groups  entries
l3333       545                2235       103     545
l3322       240                738        57      240
l4k4_4422   560                1790       99      560
l4k5_4422   1105               3525       148     1105
```

Sequential same-GPU A/B check on A100 `sm_80`, 1,024,000 Cu-Zr atoms, 200
steps. Times are averages over the last three 50-step profile windows:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_tdot_group_seq_profile200_20260514
case_l3322 = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_tdot_group_l3322_seq_20260514

model       mode  speed(atom*step/s)  neighbor  memset  basic   product  force   accumulate  total
l3333       off   1.11647e7           4.852     2.591   7.101   56.916   16.854  0.147       88.461
l3333       on    1.17553e7           4.885     2.599   7.115   52.263   16.906  0.146       83.915
l3322       off   2.14221e7           4.861     0.931   7.112   14.894   16.879  0.147       44.824
l3322       on    2.10571e7           4.886     0.931   7.109   15.653   16.876  0.147       45.602
l4k4_4422   off   1.16285e7           5.471     1.644   14.977  31.793   31.175  0.147       85.207
l4k4_4422   on    1.19949e7           5.464     1.644   14.979  29.111   31.173  0.147       82.518
l4k5_4422   off   7.99510e6           5.197     2.417   16.031  59.852   41.502  0.149       125.149
l4k5_4422   on    8.43227e6           5.157     2.417   16.029  53.275   41.494  0.150       118.523
```

Current default behavior after the grouped-dot heuristic:

```text
case_default = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_default_tdotgrp_profile200_20260514

model       default path             speed(atom*step/s)  neighbor  memset  basic   product  force   accumulate  total
l3333       pattern + dot groups     1.17703e7           4.855     2.591   7.095   52.179   16.856  0.147       83.722
l3322       const rows               2.14010e7           4.881     0.931   7.112   14.864   16.886  0.147       44.821
l4k4_4422   pattern + dot groups     1.19878e7           5.445     1.644   14.982  29.101   31.182  0.147       82.502
l4k5_4422   pattern + dot groups     8.42709e6           5.197     2.417   16.034  53.285   41.532  0.150       118.615
```

Correctness check: first 4096 Cu-Zr atoms, `l4k5_4422`, comparing grouped dots
off/on with dumped forces:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_tdot_group_correctness_20260514/l4k5

force max abs diff = 1.221001148e-4
thermo max abs diff = 1.409000000e-3
```

This is larger than the pattern-row packing repeatability check because grouped
dots intentionally change the local float accumulation order for terminal dot
rows. The mathematical contraction and reverse chain are unchanged.

An off-vs-off repeat under the same conditions produced the same maximum force
difference, so this is the current float GPUMD non-bitwise repeatability scale,
not a pattern-row math regression:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_sm80_determinism_20260514/l4k5_off_off

force max abs diff = 4.613399506e-5
```

2026-05-14 terminal-dot row-list update:

- When terminal dot groups are enabled, the old compact row loop still scanned
  every terminal dot row only to immediately skip it and run the grouped path
  later. The new path builds a per-layer non-dot row list and makes the compact
  product loop visit only those rows. Terminal dot groups are still evaluated in
  the same grouped code as before.
- The option is controlled by `sus2_sh_terminal_dot_row_list=` or
  `SUS2_SH_GPUMD_TERMINAL_DOT_ROW_LIST`. The default is on, but it only has an
  effect when terminal dot groups are on.
- This is a metadata/control-flow reduction, not a mathematical change. It does
  not alter CG coefficients, scalar coefficients, terminal dot grouping, or the
  reverse-mode chain rule.

Correctness check: first 4096 Cu-Zr atoms, `l4k5_4422`, comparing terminal dot
groups with row-list off/on:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_nondot_rowlist_correctness_20260514

force max abs diff = 1.203268766e-6
thermo max abs diff = 3.600000159e-11
```

Sequential same-GPU A/B check on A100 `sm_80`, 1,024,000 Cu-Zr atoms, 200
steps. Times are averages over the last three 50-step profile windows:

```text
case_l3333_l4k5 = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_nondot_rowlist_ab_20260514
case_l4k4       = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_nondot_rowlist_l4k4_20260514

model       row_list  speed(atom*step/s)  nondot_rows  product  total
l3333       off       1.17492e7           0            52.380   83.917
l3333       on        1.18210e7           756          51.852   83.416
l4k4_4422   off       1.19504e7           0            29.260   82.797
l4k4_4422   on        1.20354e7           408          28.757   82.185
l4k5_4422   off       8.40697e6           0            53.573   118.897
l4k5_4422   on        8.46227e6           625          52.658   118.088
```

Current default behavior after the row-list update:

```text
model       default path                        expected product effect
l3333       pattern + dot groups + row-list     about -0.53 ms product
l3322       const rows                          unchanged; dot groups stay off
l4k4_4422   pattern + dot groups + row-list     about -0.50 ms product
l4k5_4422   pattern + dot groups + row-list     about -0.92 ms product
```

2026-05-14 forward-only compact product specialization:

- The compact product kernel now has compile-time `DoBackward=0/1`
  specializations. With `parallel_back_rows=on`, the main compact kernel uses
  the forward-only instance and the existing packed parallel back kernels remain
  responsible for reverse propagation.
- This does not change the computation graph. It only lets NVCC eliminate the
  unused serial-backward branch from the compact kernel used by the default
  large-model path.

Correctness check: first 4096 Cu-Zr atoms, `l4k5_4422`, comparing the row-list
version before and after the forward-only specialization:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_forwardonly_correctness_20260514

force max abs diff = 2.384185791e-7
thermo max abs diff = 1.599999430e-13
```

A100 `sm_80`, 1,024,000 Cu-Zr atoms, 200 steps. Times are averages over the
last three 50-step profile windows:

```text
case = /work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/codex_bench/cuzr_sh_forwardonly_profile200_20260514

model       previous row-list product  forward-only product  speed(atom*step/s)
l3333       51.852 ms                  51.803 ms             1.18313e7
l4k5_4422   52.658 ms                  52.481 ms             8.47799e6
```

The gain is small. It is worth keeping as a clean kernel specialization, but it
also confirms that substantial product improvement requires changing the tensor
contraction path rather than only trimming inactive branches.

Next memory-friendly product directions:

- Do not transpose `moments[moment * N + atom]` to atom-major globally. The
  current SoA layout is already coalesced for warp-contiguous atoms in product,
  basic, and force kernels; a global transpose would likely move cost rather
  than remove it.
- The next high-confidence direction is a feature-gated tensor-block forward
  path for non-terminal CG blocks: pack blocks by layer/topology, keep
  terminal dot groups as-is, load the small left/right component vectors
  (`l <= 4`, at most 9 components) into per-thread registers, and write all
  target components for the block once. Backward should initially remain on the
  current packed row path to avoid atomics and preserve the validated chain
  rule.
- Avoid large shared-memory tiles and atom-major scratch buffers until a small
  block-forward prototype proves that component-vector reuse beats the extra
  register/control-flow pressure. Previous right-cache grouping reduced loads
  on paper but slowed product, so each structural optimization must be guarded
  by an option and A/B tested before becoming default.
