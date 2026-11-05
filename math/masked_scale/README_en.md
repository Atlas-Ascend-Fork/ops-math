# MaskedScale

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: performs elementwise masked scale computation.
- Computation formula:

  $$
  y = x \times mask \times scale
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
      <td>Input tensor, which is x in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>Mask tensor, which is mask in the formula. The shape is the same as that of x.</td>
      <td>INT8, UINT8, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Attribute</td>
      <td>Scale factor, which is scale in the formula.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, which is y in the formula. The data type and shape are the same as those of x.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_masked_scale](./examples/arch35/test_aclnn_masked_scale.cpp) | Invokes the MaskedScale operator through the [aclnnMaskedScale](./docs/aclnnMaskedScale_en.md) API. |
