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

The tested force basic-gradient cache did not improve performance and is kept
disabled by default. It can still be enabled with `sus2_sh_force_grad_cache=1`
for experiments when `alpha_index_basic_count <= 64`.

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
