# Atan2

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: performs the element-wise arctangent computation on input tensors self and other. self indicates the y-coordinate, and other indicates the x-coordinate. The computation result is the angle between the point (other, self) and the positive direction of the x-axis (in radians), with a value range of [-π, π]. Broadcasting is supported.

- Computation formula:

  $$
  out_{i}=atan2(self_{i}, other_{i})
  $$

  where $self_{i}$ indicates the $i$-th element of the input tensor self, and $other_{i}$ indicates the $i$-th element of the input tensor other. The computation result is determined by the quadrant in which (other, self) is located:

  $$
  out_{i}=\begin{cases}
  atan(self_{i}/other_{i})+\pi & \text{if } other_{i}<0, self_{i}\ge 0 \\
  atan(self_{i}/other_{i})-\pi & \text{if } other_{i}<0, self_{i}<0 \\
  atan(self_{i}/other_{i}) & \text{if } other_{i}>0 \\
  \pi/2 & \text{if } other_{i}=0, self_{i}>0 \\
  -\pi/2 & \text{if } other_{i}=0, self_{i}<0 \\
  nan & \text{if } other_{i}=0, self_{i}=0
  \end{cases}
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
      <td>self</td>
      <td>Input</td>
      <td>Input parameter for the atan2 computation, which is self_i in the formula and indicates the y-coordinate. The shape must satisfy the broadcast relationship with other.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, BOOL, FLOAT, FLOAT16, DOUBLE, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input parameter for the atan2 computation, which is other_i in the formula and indicates the x-coordinate. The shape must satisfy the broadcast relationship with self.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, BOOL, FLOAT, FLOAT16, DOUBLE, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the atan2 computation, which is out_i in the formula. The shape is the shape of self and other after broadcasting.</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- The inputs self and other support the INT8, INT16, INT32, INT64, UINT8, and BOOL types. The API automatically converts them to floating-point types before the computation.
- DOUBLE is computed through the AICPU path.
- Atlas inference products and Atlas training products: BFLOAT16 is not supported.

## Constraint Description

- The shapes of self and other must satisfy the broadcast relationship, and the shape of out is the shape of self and other after broadcasting.
- The dimensions of input and output tensors do not exceed 8, and non-contiguous Tensors are supported.
- Deterministic computation: aclnnAtan2&aclnnInplaceAtan2 uses a deterministic implementation by default.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_atan2](./examples/test_aclnn_atan2.cpp) | Invokes the Atan2 operator through the [aclnnAtan2 and aclnnInplaceAtan2](./docs/aclnnAtan2&aclnnInplaceAtan2_en.md) API. |
