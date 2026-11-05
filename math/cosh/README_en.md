# Cosh

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>          |      √   |
| <term>Atlas A3 products</term>     |    √    |
| <term>Atlas A2 products</term> |    √    |
| <term>Atlas 200I/500 A2 inference products</term>             |    √    |
| <term>Atlas inference products</term>                       |     √    |
| <term>Atlas training products</term>                       |     √    |

## Function Description

- Interface function: performs the hyperbolic cosine operation on each element of the input matrix and outputs the result.

- Computation formula:

$$
out_{i}=cosh(input_{i})
$$

## Parameter Description

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>self</td>
      <td>Input</td>
      <td>Input parameter for the cosh computation, which is input_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the cosh computation, which is out_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference products</term> and <term>Atlas training products</term>: the BFLOAT16 data type is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Sample Code | Description                                                           |
|--------------|--------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_cosh](examples/test_aclnn_cosh.cpp) | Invokes the Cosh operator through the [aclnnCosh](./docs/aclnnCosh&aclnnInplaceCosh_en.md) API. |
| GE graph mode | [test_geir_cosh](examples/test_geir_cosh.cpp) | Invokes the Cosh operator by building a graph through the [operator IR](op_graph/cosh_proto.h).             |
