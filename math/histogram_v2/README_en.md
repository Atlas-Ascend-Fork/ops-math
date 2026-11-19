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

> Note: The BFLOAT16 data type is supported only on Ascend 950PR&950DT products (arch35/regbase).

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
      <td>FLOAT, FLOAT16, BFLOAT16, INT64, INT32, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>Input</td>
      <td>Input tensor min in the formula, containing only 1 element.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT64, INT32, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max</td>
      <td>Input</td>
      <td>Input tensor max in the formula, containing only 1 element.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT64, INT32, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, which is the histogram computation result.</td>
      <td>FLOAT32, INT32 (derived from input x on Ascend 950PR&950DT products; selected by y_dtype on other products)</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bins</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates the number of divided intervals.</li><li>The default value is 100.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y_dtype</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates the output data type: INT32 or FLOAT (float32).</li><li>On Ascend 950PR&950DT products, the output type is derived from input x (floating-point → FLOAT32, integer → INT32), so this attribute does not need to be set; if set explicitly, it must match the derived type, otherwise an error is reported.</li><li>On other products, this attribute selects the output type; the default value is INT32.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- `bins` must be in the range (0, 2147483647]: greater than 0 and no greater than INT32_MAX (the kernel bin index is int32).
- `min` must not be greater than `max`. When `min == max`, the statistical bounds are recomputed from the minimum/maximum element; if they are still equal, they are expanded to the adjacent representable values.
- `min`/`max` must be finite; only both being +inf or both being -inf is allowed.
- On Ascend 950PR&950DT products (arch35), `min`/`max` are interpreted as int64 for integer inputs; when a scalar is outside the value range of the input data type, the input and the scalar are widened together before computation. Other products keep the existing semantics.
- Output type: on Ascend 950PR&950DT products it is derived from input x (floating-point → FLOAT32, integer → INT32) and `y_dtype` does not need to be set; on other products it is selected by `y_dtype` (default INT32).
- Floating-point output uses atomic accumulation by default (non-deterministic); determinism can be enabled through `aclrtSetSysParamOpt`. Integer/INT32 output is deterministic.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_histc](./examples/test_aclnn_histc.cpp) | Invokes the HistogramV2 operator through the [aclnnHistc](./docs/aclnnHistc_en.md) API. |
| Graph-mode invocation | [test_geir_histogram_v2](./examples/test_geir_histogram_v2.cpp)   | Invokes the HistogramV2 operator by building a graph through the [operator IR](./op_graph/histogram_v2_proto.h). |
