# LeftShift

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: performs the bitwise left shift operation on the input tensors x and y element-wise.

- Computation formula:

$$
z_i = x_i << y_i
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
      <td>The input x is a k-dimensional tensor and is the input parameter for the left_shift computation, which is x_i in the formula.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Input</td>
      <td>Input parameter for the left_shift computation, which is y_i in the formula.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>Output</td>
      <td>Output parameter for the left_shift computation, which is z_i in the formula.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

If shiftBits is negative or is not less than the bit width of the x data type, the output is 0, which is consistent with torch.bitwise_left_shift of PyTorch.

## Invocation Description

| Invocation Method | Invocation Sample                                                          | Description                                                          |
|--------|---------------------------------------------------------------|-------------------------------------------------------------|
| aclnn invocation | [test_aclnn_left_shift](./examples/test_aclnn_left_shift.cpp) | Invokes the left_shift operator through the [aclnnLeftShift](./docs/aclnnLeftShift_en.md) API.|
| aclnn invocation | [test_aclnn_left_shifts](./examples/test_aclnn_left_shifts.cpp) | Invokes the left_shift operator through the [aclnnLeftShifts](./docs/aclnnLeftShifts_en.md) API (shiftBits is a scalar).|
| Graph-mode invocation | [test_geir_left_shift](./examples/test_geir_left_shift.cpp) | Invokes the LeftShift operator by building a graph through the [operator IR](./op_graph/left_shift_proto.h). |
