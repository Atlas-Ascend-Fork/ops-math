# Globalavgpoolpass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T16:56:50.046Z pushedAt=2026-10-13T01:47:29.733Z -->

## Fusion Pattern

This fusion replaces the `GlobalAveragePool` operator in the graph with the `ReduceMean` operator, which computes the mean along the spatial dimensions (all dimensions from the second one onward), as shown in the following figure.

![](../../../docs/en/figures/GlobalavgpoolFusionPass.png)

After the replacement, the attributes of `ReduceMean` are set to `keep_dims=true` and `noop_with_empty_axes=true`.

## Constraints

- Version constraint: The fusion takes effect when the GE compiler version is not lower than 9.1.0.
- Input dimension constraint: The input `x` of `GlobalAveragePool` must be 3D, 4D, or 5D. Inputs of other dimensions are not fused.
- Data types and data formats follow the constraints of the underlying `GlobalAveragePool`/`ReduceMean` operators.

## Applicable Products

This fusion rule is platform-independent. For the specific supported models, see the Applicable Products of the underlying `GlobalAveragePool` and `ReduceMean` operators.
