# ReduceMeanWithCastFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T17:09:24.202Z pushedAt=2026-10-13T01:47:29.768Z -->

## Fusion Pattern

This fusion decomposes the `ReduceMeanWithCast` operator in the graph into `Cast` + `ReduceMean`. When `ReduceMeanWithCast` carries a valid `dtype` attribute (not `DT_UNDEFINED`), a `Cast` node is inserted first to convert the input to the target data type, and then `ReduceMean` is connected. When `dtype` is `DT_UNDEFINED`, no `Cast` is inserted, and the operator is directly replaced with `ReduceMean`.

**Use case 1: dtype is valid (Cast is inserted)**

![](../../../docs/en/figures/ReduceMeanWithCastFusionPass_1.png)

**Use case 2: dtype is DT_UNDEFINED (Cast is not inserted)**

![](../../../docs/en/figures/ReduceMeanWithCastFusionPass_2.png)

After the replacement, the `keep_dims` and `noop_with_empty_axes` attributes of `ReduceMean` are inherited from the original `ReduceMeanWithCast` node, and `dst_type` of `Cast` takes the value of the `dtype` attribute of the original node.

## Constraints

- Version constraint: The fusion takes effect when the GE compiler version is 9.1.0 or later.
- Attribute constraints:
  - `keep_dims`: Boolean. Default value: `false`. It is inherited as-is by `ReduceMean`.
  - `noop_with_empty_axes`: Boolean, defaults to `true`, inherited as-is by `ReduceMean`.
  - `dtype`: target data type, defaults to `DT_UNDEFINED`. If the value is `DT_UNDEFINED`, no `Cast` is inserted; otherwise, `Cast` is inserted and the value is used as `dst_type`.
- Input shape and data type are not validated. Relevant constraints follow those of the underlying `Cast` and `ReduceMean` operators.

## Applicable Products

This fusion rule is platform-independent. The supported models depend on the applicable products of the underlying `Cast` and `ReduceMean` operators.
