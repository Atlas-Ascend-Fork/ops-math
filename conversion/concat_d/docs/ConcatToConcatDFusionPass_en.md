# ConcatToConcatDFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-11T12:38:25.692Z pushedAt=2026-11-16T06:28:26.330Z nl=2 -->

## Description

Replaces the Concat, ConcatV2, and ConcatV2D operators in the graph with the ConcatD operator. The source of `concat_dim` varies by operator type:

- Concat operator: `concat_dim` is read from the 0th input (a constant).
- ConcatV2 operator: `concat_dim` is read from the last input (a constant).
- ConcatV2D operator: `concat_dim` is read from the `concat_dim` attribute.

The graph structures before and after fusion are as follows:

![](../../../docs/en/figures/ConcatToConcatDFusionPass.png)

After replacement, the dtype, shape, and format of the ConcatD output stay the same as those of the original operator output.

## Constraints

- The graph contains Concat/ConcatV2/ConcatV2D operators.
- The `concat_dim` input of the Concat/ConcatV2 operator must be a constant, and its data type must be int32 or int64. The ConcatV2D operator must have the `concat_dim` attribute.

## Applicable Products

<!-- npu="950" id2 -->
Ascend 950PR&950DT products
<!-- end id2 -->
