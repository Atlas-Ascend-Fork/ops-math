# ClipByValue

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

## Function Description

- Operator function: limits all elements of the input to the range [clipValueMin,clipValueMax]. If an element is greater than clipValueMax, it is limited to clipValueMax. If an element is less than clipValueMin, it is limited to clipValueMin. Otherwise, it equals the element itself.
- Computation formula:

$$
{y}_{i} = max(min({{x}_{i}},{max\_value}_{i}),{min\_value}_{i})
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
      <td>Input tensor.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, <br>
      FLOAT16, FLOAT, DOUBLE, BOOL, BFLOAT16, COMPLEX32, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>clip_value_min</td>
      <td>Input</td>
      <td>Minimum value that limits the element range.</td>
      <td>Same as the input tensor x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>clip_value_max</td>
      <td>Input</td>
      <td>Maximum value that limits the element range.</td>
      <td>Same as the input tensor x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor after clipping the input tensor x.</td>
      <td>Same as the input tensor x</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_clip_by_value](./examples/test_geir_clip_by_value.cpp)   | Invokes the clip_by_value operator by building a graph through the [operator IR](./op_graph/clip_by_value_proto.h). |
