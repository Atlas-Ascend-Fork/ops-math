# Lerp

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: performs linear interpolation between the start and end tensors based on the given weight, and returns the interpolated tensor.

- Computation formula:

$$
\text { out }_i=\text { start }_i+\text { weight }_i \times\left(\text { end }_i-\text { start }_i\right)
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
      <td>start</td>
      <td>Input</td>
      <td>Input parameter for the lerp computation, which is start_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>end</td>
      <td>Input</td>
      <td>Input parameter for the lerp computation, which is end_i in the formula. The dtype is the same as start.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Input parameter for the lerp computation, which is weight_i in the formula. The dtype is the same as start.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the lerp computation, which is out_i in the formula. The dtype is the same as start.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

<term>Ascend 950PR&950DT products</term>: When start is float16 or bfloat16, weight also supports float32.

## Invocation Description

| Invocation Method   | Sample Code                                       | Description                                                                                 |
| ---------- | ---------------------------------------------- | ------------------------------------------------------------------------------------ |
| aclnn API  | [test\_aclnn\_lerp](./examples/test_aclnn_lerp.cpp) <br> [test\_aclnn\_inplace\_lerp](./examples/test_aclnn_inplace_lerp.cpp) | Invokes the Lerp operator through the [aclnnLerp&aclnnInplaceLerp](./docs/aclnnLerp&aclnnInplaceLerp_en.md) APIs. |
| aclnn API  | [test\_aclnn\_lerps](./examples/test_aclnn_lerps.cpp) <br> [test\_aclnn\_inplace\_lerps](./examples/test_aclnn_inplace_lerps.cpp) | Invokes the lerp operator through the [aclnnLerps&aclnnInplaceLerps](./docs/aclnnLerps&aclnnInplaceLerps_en.md) APIs |
| GE graph mode | [test\_geir\_lerp](./examples/test_geir_lerp.cpp)   | Invokes the Lerp operator by building a graph through the [operator IR](./op_graph/lerp_proto.h). For compilation and verification, see [Compile and Run Samples](../../docs/en/context/compile_and_run_sample.md). |
