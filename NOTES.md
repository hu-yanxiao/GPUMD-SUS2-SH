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
- Unused element-pair tables are not built unless the model and run-time element mapping require them.
- Product-graph codegen is cached by topology, so models with the same multiplication graph can reuse compiled artifacts even if element identities or coefficients differ.

## Current Limitation

The model-topology code generator and cache are available as a tool. The main GPUMD runtime still uses the generic SUS2 backend path by default; dynamic loading of generated objects into the production `gpumd` executable is a separate follow-up step.

