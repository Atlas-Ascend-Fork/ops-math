# Real

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: returns the real part of each element in the input tensor.
- Computation formula:

  $$
  output_i = \mathrm{Re}(input_i)
  $$

  where:
  - When $input_i = a_i + b_i \cdot j$ is a complex number, $output_i = a_i$;
  - When $input_i$ is a real number, $output_i = input_i$ (pass-through).

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 280px">
<col style="width: 360px">
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
    <td>input</td>
    <td>Input</td>
    <td>Input tensor for real part extraction, which is $input_i$ in the formula.</td>
    <td>FLOAT, FLOAT16, COMPLEX64, COMPLEX32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>Tout</td>
    <td>Optional Attribute</td>
    <td><ul><li>Expected output dtype enumeration value.</li><li>The default value is DT_FLOAT. The actual output type is deduced from the input dtype.</li></ul></td>
    <td>INT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>output</td>
    <td>Output</td>
    <td>Output tensor after real part extraction, which is $output_i$ in the formula. The shape is the same as that of input.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

Correspondence between input and output dtypes:

| input | output |
| :--- | :--- |
| FLOAT | FLOAT |
| FLOAT16 | FLOAT16 |
| COMPLEX64 | FLOAT |
| COMPLEX32 | FLOAT16 |

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_real](./examples/test_aclnn_real.cpp) | Invokes the Real operator through the [aclnnReal](./docs/aclnnReal_en.md) API. |
| Graph-mode invocation | [test_geir_real](./examples/test_geir_real.cpp) | Invokes the Real operator by building a graph through the [operator IR](./op_graph/real_proto.h). |
