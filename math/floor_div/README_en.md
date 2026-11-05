# FloorDiv

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

- Operator function: performs the division computation and rounds the result down.
- Computation formula:

  $$
  out_i = floor(\frac{self_i}{other_i})
  $$

- **Parameter Description:**

  <table style="table-layout: fixed; width: 1496px"><colgroup>
  <col style="width: 149px">
  <col style="width: 120px">
  <col style="width: 205px">
  <col style="width: 305px">
  <col style="width: 317px">
  <col style="width: 121px">
  <col style="width: 134px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input self in the formula.</td>
      <td>
        <ul>
          <li>The data type of self and the data type of other must satisfy the data type deduction rules (for details, see [Deduction Relationship](../../docs/en/context/deduction_relationship.md)).</li>
          <li>The shape of self must satisfy the [broadcast relationship](../../docs/en/context/broadcast_relationship.md) with other.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
      <td>No more than 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input other in the formula.</td>
      <td>
        <ul>
          <li>The data type of other and the data type of self must satisfy the data type deduction rules (for details, see [Deduction Relationship](../../docs/en/context/deduction_relationship.md)).</li>
          <li>The shape of other must satisfy the [broadcast relationship](../../docs/en/context/broadcast_relationship.md) with self.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
      <td>No more than 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output out in the formula.</td>
      <td>
        <ul>
          <li>The data type must be a data type to which the deduced data types of self and other can be converted (for details, see [Conversion Relationship](../../docs/en/context/conversion_relationship.md)).</li>
          <li>The shape must be the shape of self and other after broadcasting.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
      <td>No more than 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the operator executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas training products</term>: The BFLOAT16 data type is not supported.

## Invocation Description

| Invocation Method | Invocation Sample                                             | Description                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_floor_divide](./examples/test_aclnn_floor_divide.cpp) | Invokes the FloorDiv operator through the [aclnnFloorDivide and aclnnInplaceFloorDivide](./docs/aclnnFloorDivide&aclnnInplaceFloorDivide_en.md) APIs  |
