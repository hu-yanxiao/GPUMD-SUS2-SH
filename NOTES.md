# GPUMD-SUS2 Notes

This repository stores the SUS2 v1.1 GPUMD overlay, not a full GPUMD source fork.

## Development Reference

Formal development tree:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex
```

Formal binary:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/src/gpumd
```

Local working copy used to prepare this overlay:

```text
/Users/hu-yanxiao/Projects/SUS2MLIP/.codex_tmp/gpumd-sus2-v1.1-work-codex
```

## Important Design Choices

- SUS2 support is detected from potential files whose first token is `MTP`.
- The current implementation targets SUS2 model format `version = 1.1.0`.
- The GPUMD `potential` line provides the element symbols after the model filename.
- Supported radial basis types use lookup tables by default with `dr = 1.0e-4 A`.
- `sus2_grad_float=1` or `SUS2_GPUMD_GRAD_FLOAT=1` switches the reverse-mode moment-gradient workspace from double to float; default remains double.
- `sus2_float=1` or `SUS2_GPUMD_FLOAT=1` switches to an experimental NEP-like float path for SUS2 moments, reverse gradients, fitted scalar coefficients in the device kernel, and local arithmetic while retaining double GPUMD output arrays.
- Unused element-pair tables are not built unless the model and run-time element mapping require them.
- Product-graph codegen is cached by topology, so models with the same multiplication graph can reuse compiled artifacts even if element identities or coefficients differ.

## Grad-Float Probe

Test case:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex/codex_bench/ma_l3k3_98k_gradfloat_20260427
```

Result on one A100, l3k3 MA, 98304 atoms, NPT 2000 steps, `time_step 0.5 fs`:

```text
double workspace:     4.35702e6 atom-step/s, 4124 MiB GPUMD process GPU memory
float grad workspace: 5.19508e6 atom-step/s, 3366 MiB GPUMD process GPU memory
```

Static 98k force comparison against default double:

```text
energy_diff_eV = 0.0
force_mae_eV_A = 1.31e-5
force_rmse_eV_A = 2.07e-5
force_max_abs_eV_A = 1.06e-4
```

## Current Limitation

The model-topology code generator and cache are available as a tool. The main GPUMD runtime still uses the generic SUS2 backend path by default; dynamic loading of generated objects into the production `gpumd` executable is a separate follow-up step.
