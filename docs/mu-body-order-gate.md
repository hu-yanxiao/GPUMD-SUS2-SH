# SUS2-SH Mu-Body Gate GPUMD Notes

Branch: `codex/mu-body-order-gate`

Server build tree:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-mu-body-order-gate-build-codex
```

Old gate reference binary, kept untouched:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex/src/gpumd
sha256 d86b7b7315510a269dfeae2fe1c1fa8198ce8edf77457c8a69228c5d2778e1cf
```

New test binary:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-mu-body-order-gate-build-codex/src/gpumd
sha256 6b809fe9c0e1a8afdc568a42bdf45741e6894fe650a21d0ebd004069d7e2456b
```

## Supported Gate Modes

The branch supports the SUS2-SH mu gate model files with:

```text
two_layer_gate_mode = mu-body-linear-combo
two_layer_gate_mode = mu-scalar-full
two_layer_gate_site_mode = neighbor
```

Legacy gate files are rejected in this branch to avoid silent fallback. Double-site gate files are also rejected for now; the benchmarked GPUMD path is the single neighbor gate path.

For `mu-body-linear-combo`, scalar body signals are shared:

```text
b_{a,p} = sum_{q: body_order(q)=p+2} w_q s_q(a)
h_{a,mu} = sum_p c_{mu,p} b_{a,p}
```

For `mu-scalar-full`, every mu channel has an independent scalar combination:

```text
h_{a,mu} = sum_q W_{mu,q} s_q(a)
```

The GPUMD neighbor gate is:

```text
G_{ij,mu} = 1 + A tanh(a_{z_j} h_{j,mu})
```

and the gated moment is:

```text
M_{i,mu m} = sum_{j in N(i)} t_{z_i} t_{z_j} G_{ij,mu}
             R_mu(r_ij) Y_lm(rhat_ij)
```

The additive coefficient `a` is species-indexed only, matching the current SUS2-SH model format.

## Optimization Record

The first working version computed `mu-scalar-full` gate values serially per atom over all `mu` and all gate scalars. This left `l4k4` full mode too slow at 10k atoms.

The retained optimization is exact: for `mu-scalar-full`, gate value contraction is parallelized over `(atom, mu)`. Each thread still sums scalar weights in the same `q` order for its own `h_{atom,mu}`. This keeps the mathematical result unchanged while exposing more CUDA parallelism.

Test history is stored under:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-mu-body-order-gate-build-codex/codex_mu_gate_history
/work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-mu-body-order-gate-build-codex/codex_mu_gate_bench_20260619_v1b
```

## Final Benchmark

Benchmark directory:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-mu-body-order-gate-build-codex/codex_mu_gate_bench_20260619_v1b/matrix_v2_full
```

Settings:

```text
sus2_float=1
sus2_radial_direct=1
sus2_sh_factor_pruning=q-total
```

Systems:

```text
10k  atoms: 10368 B atoms
100k atoms: 105456 B atoms
```

All 24 GPUMD runs completed with status `0`. Ratios below are `new_total_ms / main_old_gate_total_ms`; all are within the target `<= 1.2`.

```text
lk    size  mode                    ratio_total
l2k2  10k   mu_body_linear_combo    1.064267
l2k2  10k   mu_scalar_full          1.066838
l2k2  100k  mu_body_linear_combo    1.080020
l2k2  100k  mu_scalar_full          1.081033
l2k3  10k   mu_body_linear_combo    1.067311
l2k3  10k   mu_scalar_full          1.080201
l2k3  100k  mu_body_linear_combo    1.138488
l2k3  100k  mu_scalar_full          1.150960
l3k3  10k   mu_body_linear_combo    1.032787
l3k3  10k   mu_scalar_full          1.064680
l3k3  100k  mu_body_linear_combo    1.106215
l3k3  100k  mu_scalar_full          1.148464
l4k4  10k   mu_body_linear_combo    0.915722
l4k4  10k   mu_scalar_full          1.023429
l4k4  100k  mu_body_linear_combo    0.918645
l4k4  100k  mu_scalar_full          1.106543
```

Maximum ratio:

```text
1.150960 at l2k3 100k mu_scalar_full
```
