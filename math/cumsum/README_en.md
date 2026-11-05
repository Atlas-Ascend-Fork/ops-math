# Cumsum

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Operator function: accumulates the elements of the input tensor self in sequence along the specified dimension dim and saves the result to the output tensor out.
- Computation formula: $x\_{i}$ is an element in the input tensor self from the perspective of dimension dim (the subscripts of other dimensions remain unchanged, and only the subscript of the dim dimension increases in sequence), $y\_{i}$ is the element at the corresponding position in the output tensor out, then:

$$
y_{i} = x_{1} + x_{2} + x_{3} + ...... + x_{i}
$$

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>x</td>
      <td>Input</td>
      <td>Input Tensor.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64<br>
      FLOAT16, FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Input</td>
      <td>The dimension along which elements are accumulated in sequence.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exclusive</td>
      <td>Attribute</td>
      <td>The default value is false, indicating an inclusive cumsum, where the first output element is the same as the first input element;<br>
      true indicates an exclusive cumsum, where the first output element is 0 and subsequent elements are the prefix sums of the input.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reverse</td>
      <td>Attribute</td>
      <td>The default value is false, indicating that the cumulative sum is computed from the beginning to the end of the tensor (forward computation);<br>
      true indicates that the cumulative sum is computed from the end to the beginning of the tensor (backward computation).</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output Tensor after accumulation along the specified dimension.</td>
      <td>Same as input Tensor x</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- None.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation | [test_aclnn_cumsum_v2](./examples/test_aclnn_cumsum_v2.cpp) | Invokes the cumsum operator through the [aclnnCumsumV2](./docs/aclnnCumsumV2_en.md) API.    |
| Graph-mode invocation | [test_geir_cumsum](./examples/test_geir_cumsum.cpp)   | Invokes the cumsum operator by building a graph through the [operator IR](./op_graph/cumsum_proto.h). |
