# Div

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Interface function: performs the division computation

- Computation formula:

  $$
  out_i = \frac{input_i}{other_i}
  $$

- Exception: when a Complex computation is involved and the modulus of the denominator is 0, the general complex number computation formula is still used without special handling. This behavior differs from that of CPU/GPU.

## Parameter Description

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>x1</td>
      <td>Input</td>
      <td>input in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, UINT8, INT8, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>other in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, UINT8, INT8, COMPLEX32, COMPLEX64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>out in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, UINT8, INT8, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_div](examples/test_aclnn_div.cpp) | Invokes the Div operator through the [AclnnDiv](docs/aclnnDiv&aclnnInplaceDiv_en.md) API. |
| Graph-mode invocation | [test_geir_div](examples/test_geir_div.cpp) | Invokes the Div operator by building a graph through the [operator IR](op_graph/div_proto.h). |
