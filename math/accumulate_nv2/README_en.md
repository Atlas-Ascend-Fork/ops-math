# AccumulateNV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    √     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |

## Function Description

- Operator function: returns the sum of each input tensor in the input tensors list by adding them in sequence. AccumulateNv2 performs the same computation as AddN, but does not wait for all inputs to be ready before starting the summation, which saves memory when inputs become ready at different times.

- Computation formula:

$$y = \sum_{i=0}^{N-1} x_i$$

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
      <td>The list of input tensors to be summed, which is x_i in the formula. In graph-mode direct connection, the shapes of all inputs must be exactly the same (the operator itself does not support broadcast). When the operator is invoked through the aclnnSum API, the shapes of tensors can be different, and the API layer completes the alignment.</td>
      <td>FLOAT16, FLOAT, INT8, INT32, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The summation result tensor, which is y in the formula.</td>
      <td>FLOAT16, FLOAT, INT8, INT32, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>N</td>
      <td>Attribute</td>
      <td>The size of the input tensor list. N must be greater than or equal to 1 and must be consistent with the actual number of input tensors. If they are inconsistent, an error is reported during graph compilation.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- When the operator is invoked through the aclnnSum API, the shapes of tensors in the input tensors list must satisfy the broadcast relationship, and the implementation layer completes the alignment. In graph-mode (GE/GEIR) direct connection, the shapes of all inputs must be exactly the same as each other and as the output y (the operator itself does not support broadcast). If the shapes are inconsistent, an error is reported during graph compilation.
- The data types of the input tensors and the output must be the same, and the data types of tensors in the tensors list must be consistent. If they are inconsistent, an error is reported during graph compilation.
- The number of input dimensions cannot exceed 8: the first-stage validation of the aclnnSum API intercepts the violation (ACLNN_ERR_PARAM_INVALID 161002), and in graph-mode direct connection, the tiling stage intercepts it.
- In graph-mode (GE/GEIR) direct connection, the number of inputs must be within the range [1, 40] (intercepted at the tiling stage, and an error is reported when the limit is exceeded). The aclnnSum API internally groups and accumulates tensors in groups of 16, and is not subject to this limit.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_sum](./examples/test_aclnn_sum.cpp) | Invokes the AccumulateNV2 operator through the [aclnnSum](./docs/aclnnSum_en.md) API. |
| Graph-mode (GEIR) | [accumulate_nv2_proto.h](./op_graph/accumulate_nv2_proto.h) | Invoked in a GE graph through the [AccumulateNV2 operator IR definition](./op_graph/accumulate_nv2_proto.h) (the shapes of all inputs must be exactly the same). |
