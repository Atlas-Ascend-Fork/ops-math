# TensorEqual

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

- API function: computes whether two tensors have the same size and elements, and returns a value of the Bool type.
- Computation expression:

  $$
  out = (self == other)  ?  True : False
  $$

## Parameter Description

  <table style="table-layout: fixed; width: 1549px"><colgroup>
  <col style="width: 168px">
  <col style="width: 136px">
  <col style="width: 258px">
  <col style="width: 271px">
  <col style="width: 311px">
  <col style="width: 116px">
  <col style="width: 142px">
  <col style="width: 147px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Instructions</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>The first input.</td>
      <td>The data types of self and other satisfy the data type deduction rule (see <a href="../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</td>
      <td>FLOAT16, FLOAT, INT32, INT8, UINT8, BOOL, DOUBLE, INT64, INT16, UINT16, UINT32, UINT64, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>The second input.</td>
      <td>The data types of other and self satisfy the data type deduction rule (see <a href="../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</td>
      <td>FLOAT16, FLOAT, INT32, INT8, UINT8, BOOL, DOUBLE, INT64, INT16, UINT16, UINT32, UINT64, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The output. The output is a one-dimensional tensor of the BOOL type that contains one element.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- <term>Atlas inference products</term> and <term>Atlas training products</term>: The supported data types are FLOAT16, FLOAT, and DOUBLE.

## Constraint Description

None

## Invocation Description

| Invocation Method | Sample Code                                                | Description                                                             |
|--------------|-----------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_equal.cpp](examples/test_aclnn_equal.cpp) | Invokes the TensorEqual operator through the [aclnnEqual](docs/aclnnEqual_en.md) API. |
