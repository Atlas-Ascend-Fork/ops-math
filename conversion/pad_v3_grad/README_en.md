# PadV3Grad

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |

## Function Description

- Operator function:

  Invokes different underlying padgrad operators (pad_v3_grad_replicate/pad_v3_grad_replication/pad_v4_grad/reflection_pad3d_grad) according to different conditions.

## Invocation Description

  | Operator Name | Corresponding README                                                        |Invocation Condition     |
  |--------------|------------------------------------------------------------------------|-------------------------|
  | circular_pad_grad | [circular_pad_grad](../circular_pad_grad/README_en.md) |When the input mode is "circular", the operator circular_pad_grad is invoked.|
  | pad_v4_grad | [pad_v4_grad](../pad_v4_grad/README_en.md) |When the number of dimensions of the input tensor is less than or equal to 4 and the input mode is "reflect", the operator pad_v4_grad is invoked.|
  | pad_v3_grad_replicate | [pad_v3_grad_replicate](../pad_v3_grad_replicate/README_en.md) |When the number of dimensions of the input tensor is less than or equal to 4 and the input mode is "edge", the operator pad_v3_grad_replicate is invoked.|
  | pad_v3_grad_replication | [pad_v3_grad_replication](../pad_v3_grad_replication/README_en.md) |When the number of dimensions of the input tensor is equal to 5 and the input mode is "edge", the operator pad_v3_grad_replication is invoked.|
  | reflection_pad3d_grad | [reflection_pad3d_grad](../reflection_pad3d_grad/README_en.md) |When the number of dimensions of the input tensor is equal to 5 and the input mode is "reflect", the operator reflection_pad3d_grad is invoked.|
