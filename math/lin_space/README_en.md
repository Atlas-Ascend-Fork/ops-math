# LinSpace

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |
| <term>Kirin X90 processor series products</term>                      | √ |
| <term>Kirin 9030 processor series products</term>                       | √ |

## Function Description

- Operator function: generates a sequence of evenly spaced values. Creates a one-dimensional vector of size num, whose values are linearly and uniformly distributed from start to stop (inclusive).

- Computation formula:

$$
output = (start, start + \frac{stop - start}{num - 1},...,start + (num - 2) * \frac{stop - start}{num -1}, stop)
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
      <td>start</td>
      <td>Input</td>
      <td>Input tensor start in the formula, which must be a one-dimensional tensor.</td>
      <td>INT8, UINT8, INT16, INT32, FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>stop</td>
      <td>Input</td>
      <td>Input tensor stop in the formula, which must be a one-dimensional tensor.</td>
      <td>INT8, UINT8, INT16, INT32, FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num</td>
      <td>Input</td>
      <td>Input tensor num in the formula, which must be a one-dimensional tensor.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Output tensor output in the formula.</td>
      <td>INT8, UINT8, INT16, INT32, FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products, Atlas training products, and Atlas inference products: BFLOAT16 and INT16 are not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_lin_space](./examples/test_aclnn_lin_space.cpp) | Invokes the LinSpace operator through the [aclnnLinspace](./docs/aclnnLinspace_en.md) API. |
| Graph-mode invocation | [test_geir_lin_space](./examples/test_geir_lin_space.cpp)   | Invokes the LinSpace operator by building a graph through the [operator IR](./op_graph/lin_space_proto.h). |
