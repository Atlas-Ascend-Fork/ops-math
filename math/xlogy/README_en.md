# Xlogy

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

- Operator function: computes x1 * log(x2) element-wise and supports broadcasting.

- Computation formula:

$$out_i = \begin{cases} 0 & \text{if } x1_i = 0 \\ x1_i \cdot \ln(x2_i) & \text{otherwise} \end{cases}$$

When x2 is NaN, the output propagates NaN.

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
      <td>x1</td>
      <td>Input</td>
      <td>The first input of xlogy, which is x1_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The second input of xlogy, which is x2_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output of xlogy, which is out_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The inputs x1 and x2 must have the same data type.
- The maximum number of dimensions is 8.

## Invocation Description

| Invocation Method | Sample Code | Description |
|--------------|------|------|
| aclnn API | [test_aclnn_x_log_y_tensor](./examples/test_aclnn_x_log_y_tensor.cpp) | Invokes the Xlogy operator through the [aclnnXLogYTensor](./docs/aclnnXLogYTensor&aclnnInplaceXLogYTensor_en.md) and [aclnnInplaceXLogYTensor](./docs/aclnnXLogYTensor&aclnnInplaceXLogYTensor_en.md) APIs. |
| aclnn API | [test_aclnn_x_log_y_scalar_self](./examples/test_aclnn_x_log_y_scalar_self.cpp) | Invokes the Xlogy operator through the [aclnnXLogYScalarSelf](./docs/aclnnXLogYScalarSelf_en.md) API. |
| aclnn API | [test_aclnn_x_log_y_scalar_other](./examples/test_aclnn_x_log_y_scalar.cpp) | Invokes the Xlogy operator through the [aclnnXLogYScalarOther](./docs/aclnnXLogYScalarOther&aclnnInplaceXLogYScalarOther_en.md) and [aclnnInplaceXLogYScalarOther](./docs/aclnnXLogYScalarOther&aclnnInplaceXLogYScalarOther_en.md) APIs. |
| GE graph mode | [test_geir_xlogy](./examples/test_geir_xlogy.cpp) | Invokes the Xlogy operator by building a graph through the [operator IR](./op_graph/xlogy_proto.h). |
