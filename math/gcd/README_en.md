# Gcd

English | [简体中文](./README.md)

## Product Support Status

| Product                                         | Supported |
| :------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>      |    √    |
| <term>Atlas A3 products</term>                |    √    |
| <term>Atlas A2 products</term>                 |    √    |
| <term>Atlas 200I/500 A2 inference products</term>        |    ×    |
| <term>Atlas inference products</term>                |    √    |
| <term>Atlas training products</term>                |    √    |

## Function Description

- Operator function: computes the greatest common divisor of two input tensors element-wise. The inputs must be of integer types.
- Computation formula:
  $$
  y = \gcd(x_1, x_2)
  $$

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description                                       | Data Type                           | Data Format |
| :--- | :--- | :----------------------------------------- | :--------------------------------- | :------- |
| x1   | Input | Input tensor x1 in the formula. Its shape satisfies the broadcast relationship with x2. | UINT8, INT8, INT16, INT32, INT64  | ND       |
| x2   | Input | Input tensor x2 in the formula. Its shape satisfies the broadcast relationship with x1. | UINT8, INT8, INT16, INT32, INT64  | ND       |
| y    | Output | Output tensor y in the formula. Its shape is the shape of x1 and x2 after broadcasting. | UINT8, INT8, INT16, INT32, INT64 | ND       |

- Atlas A3 products, Atlas A2 products, and Atlas inference products: The UINT8 and INT8 data types are not supported.
- Atlas training products: Only the INT32 data type is supported.

## Constraint Description

None

## Invocation Description

| Invocation Method   | Invocation Sample                                  | Description                                             |
| :--- | :--- | :--- |
| aclnn API  | [test_aclnn_gcd](./examples/test_aclnn_gcd.cpp) | Invokes the Gcd operator through the [aclnnGcd](./docs/aclnnGcd_en.md) API. |
| GE graph mode   | -                                        | Invokes the Gcd operator by building a graph through the [operator IR](./op_graph/gcd_proto.h). |
