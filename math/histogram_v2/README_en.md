# HistogramV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: computes the histogram of a tensor.
- Computation formula: uses min and max as the upper and lower bounds of the statistics, divides the space between min and max into bins intervals of equal width, and counts the number of elements in tensor x in each interval. If min and max are equal, the minimum and maximum values of all elements in the tensor are used as the upper and lower bounds of the statistics. Elements smaller than min and greater than max are not counted.

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
      <td>Input tensor x in the formula.</td>
      <td>FLOAT, FLOAT16, INT64, INT32, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>Input</td>
      <td>Input tensor min in the formula, containing only 1 element.</td>
      <td>FLOAT, FLOAT16, INT64, INT32, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max</td>
      <td>Input</td>
      <td>Input tensor max in the formula, containing only 1 element.</td>
      <td>FLOAT, FLOAT16, INT64, INT32, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, which is the histogram computation result.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bins</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates the number of divided intervals.</li><li>The default value is 100.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_histc](./examples/test_aclnn_histc.cpp) | Invokes the HistogramV2 operator through the [aclnnHistc](./docs/aclnnHistc_en.md) API. |
| Graph-mode invocation | [test_geir_histogram_v2](./examples/test_geir_histogram_v2.cpp)   | Invokes the HistogramV2 operator by building a graph through the [operator IR](./op_graph/histogram_v2_proto.h). |
