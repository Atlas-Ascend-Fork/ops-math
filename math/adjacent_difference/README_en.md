# AdjacentDifference

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term>      |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: compares the differences between adjacent elements of the input tensor. If two adjacent elements are the same, 0 is returned; otherwise, 1 is returned.

- Computation formula:

$$
out_i = {(x_i == x_j ?\ 0 : 1)}.to(y_dtype) \ j = i - 1
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
      <td>x</td>
      <td>Input</td>
      <td>Input parameter for the AdjacentDifference computation, which is x in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_dtype</td>
      <td>Input</td>
      <td>An attribute that specifies the output type, which is y_dtype in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the AdjacentDifference computation, which is out in the formula.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.

## Constraint Description

- None.

## Invocation Description

| Invocation Method | Sample Code |   Description    |
| :-------- | :---------- | :------------------------ |
|  Graph mode  |[test_geir_adjacent_difference](examples/test_geir_adjacent_difference.cpp) | Invokes the AdjacentDifference operator by building a graph through the [operator IR](./op_graph/adjacent_difference_proto.h). |
