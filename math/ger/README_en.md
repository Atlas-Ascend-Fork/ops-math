# Ger

English | [简体中文](./README.md)

## Product Support Status

| Product                                         | Supported |
| :------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>      |    √    |
| <term>Atlas A3 products</term>                |    √    |
| <term>Atlas A2 products</term>                |    √    |
| <term>Atlas 200I/500 A2 inference products</term>       |    ×    |
| <term>Atlas inference products</term>               |    ×    |
| <term>Atlas training products</term>               |    √    |

## Function Description

- Operator function: computes the outer product of the one-dimensional vectors x1 and x2 to obtain a two-dimensional matrix.
- Computation formula:
  $$
  y = x_1^T \times x_2
  $$

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description                                       | Data Type               | Data Format |
| :--- | :--- | :----------------------------------------- | :--------------------- | :------- |
| x1   | Input | Input vector x1 in the formula, the first vector of the outer product computation. | BFLOAT16, FLOAT16, FLOAT | ND       |
| x2   | Input | Input vector x2 in the formula, the second vector of the outer product computation. | BFLOAT16, FLOAT16, FLOAT | ND       |
| y    | Output | Output tensor y in the formula, the result of the outer product computation. Its shape is {x1.shape[0], x2.shape[0]}. | BFLOAT16, FLOAT16, FLOAT | ND       |

- Atlas A3 products, Atlas A2 products, and Atlas training products: The BFLOAT16 data type is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method   | Invocation Sample                                  | Description                                             |
| :--- | :--- | :--- |
| aclnn API  | [test_aclnn_ger](./examples/test_aclnn_ger.cpp) | Invokes the Ger operator through the [aclnnGer](./docs/aclnnGer_en.md) API. |
| GE graph mode   | -                                        | Invokes the Ger operator by building a graph through the [operator IR](./op_graph/ger_proto.h). |
