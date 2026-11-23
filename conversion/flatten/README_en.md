# Flatten

English | [简体中文](./README.md)

## Product Support Status

| Product                                           | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    √     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |

## Function Description

- **Operator function**: flattens the input Tensor into a 2D Tensor based on the given axis.
- **Computation formula**:
  - If the shape of the input x is (d₀, d₁, ..., dₙ), the shape of the output y is (d₀×d₁×...×dₐₓᵢₛ₋₁, dₐₓᵢₛ×dₐₓᵢₛ₊₁×...×dₙ)
  - If axis is 0, the shape of the output y is (1, d₀×d₁×...×dₙ)
  - If axis is rank(x), the shape of the output y is (d₀×d₁×...×dₙ, 1)

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor. | ALL | ND |
| axis | Optional attribute | Reference axis for the flatten computation. The value range is [-rank(x), rank(x)]. Defaults to 1. | INT | - |
| y | Output | Flattened 2D tensor. The data type is the same as that of `x`. | Same as `x` | ND |

## Constraint Description

- `axis` must be in the range [-rank(x), rank(x)] (both ends inclusive). Inference fails if it is out of range.
- If the input has unknown rank (-2), the output shape is also -2.
- If a dimension involved in the product is unknown (-1), the corresponding output dimension is -1.
- If the input is 1-D and its only dimension is unknown (-1), the output keeps rank 1 (-1).
- The aclnn API currently supports only axis = 1 and inputs of 2D to 8D.
- Supported data types are defined by the IR in [`flatten_proto.h`](op_graph/flatten_proto.h).

## Invocation Description

| Invocation Method  | Sample Code                                                | Description                                                         |
| --------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| Graph mode | [test_geir_flatten](./examples/test_geir_flatten.cpp) | Invokes the Flatten operator through the [operator IR](./op_graph/flatten_proto.h) graph construction. |
| aclnn API | [test_aclnn_flatten](./examples/test_aclnn_flatten.cpp) | Invokes the Flatten operator through the [aclnnFlatten](docs/aclnnFlatten_en.md) API. |
