# Add

English | [简体中文](./README.md)

## Product Support Status

| Product                                    | Supported |
| :-------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> |    √    |
| <term>Atlas A3 products</term>           |    √    |
| <term>Atlas A2 products</term>           |    √    |
| <term>Atlas 200I/500 A2 inference products</term>  |    ×    |
| <term>Atlas inference products</term>          |    ×    |
| <term>Atlas training products</term>          |    √    |
| <term>Kirin X90 processor series products</term>    |    √    |
| <term>Kirin 9030 processor series products</term>   |    √    |

## Function Description

- Operator function: adds the inputs.
- Computation formula:
  $$
  y = x_1 + x_2
  $$

## Parameter Description

<table style="table-layout: fixed; width: 880px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 320px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>The input tensor x_1 in the formula</td>
      <td>BOOL, INT8, INT16, INT32, INT64, UINT8, FLOAT64, FLOAT16, BFLOAT16, FLOAT32, COMPLEX128, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The input tensor x2 in the formula</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output tensor y in the formula</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The STRING data type is not supported. Registering DT\_STRING in the operator IR (add\_proto.h) is only for compatibility with the TensorFlow Add operator. Although the graph-mode (GEIR) invocation path can pass type inference, no corresponding kernel implementation exists, and execution fails at runtime.
- COMPLEX32 is supported only by the kernel implementation of Ascend 950. Neither the aclnnAdd / aclnnInplaceAdd APIs nor the graph-mode path supports it. Use COMPLEX64 / COMPLEX128 instead.
- Kirin X90/Kirin 9030 processor series products support the INT8, UINT8, INT32, FLOAT16, and FLOAT32 data types

## Invocation Description

| Invocation Method   | Invocation Sample                                       | Description                                                                                 |
| ---------- | ---------------------------------------------- | ------------------------------------------------------------------------------------ |
| aclnn invocation  | [test\_aclnn\_add](./examples/test_aclnn_add.cpp) | Invokes the Add operator through the [aclnnAdd and aclnnInplaceAdd](./docs/aclnnAdd&aclnnInplaceAdd_en.md) APIs |
| Graph-mode invocation | [test\_geir\_add](./examples/test_geir_add.cpp)   | Invokes the Add operator by building a graph through the [operator IR](./op_graph/add_proto.h)                                 |
