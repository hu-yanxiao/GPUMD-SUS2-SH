# GPUMD-SUS2-SH Notes

This repository stores the SH-only SUS2 GPUMD overlay, not a full GPUMD source
fork.

## Development Reference

Current server overlay path:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-work-codex
```

Current server build path:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-SH-build-codex
```

The old moment-tensor GPUMD-SUS2 tree remains a read-only implementation
reference:

```text
/work/phy-weigw/20260321_Test/GPUMD-SUS2-v1.1-work-codex
```

## Important Design Choices

- SUS2-SH support is detected from potential files whose first token is `MTP`
  and whose body contains `potential_tag = SUS2-SH`.
- Non-SH `MTP` files are rejected in this project; the tensor backend stays in
  the separate `GPUMD-SUS2` repository.
- The first implementation targets SUS2 model format `version = 1.1.0`.
- The GPUMD `potential` line provides the element symbols after the model
  filename.
- `RBChebyshev_sss`, `scaling_map = LK`, `sh_l_max <= 4`, direct radial
  recurrence, and float moments are the first supported/default path.
- The old tensor product-graph codegen and graph-specific kernels are not used
  by SH. SH optimization should instead pack `sh_products` into CG coupling
  blocks and layers.
