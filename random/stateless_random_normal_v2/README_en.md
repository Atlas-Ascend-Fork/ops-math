# stateless_random_normal_v2

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term>    |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: returns a random number tensor, where the random numbers are drawn from independent normal distributions.

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
      <td>shape</td>
      <td>Input</td>
      <td>Shape of the input tensor.</td>
      <td>INT64, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>Key used by the counter-based random number generation algorithm.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counter</td>
      <td>Input</td>
      <td>Initial counter value used by the counter-based random number generation algorithm.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alg</td>
      <td>Input</td>
      <td>Algorithm used to generate random numbers.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Outputs random values with the specified shape.</td>
      <td>FLOAT, BF16, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Sample Code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_normal_float_float](./examples/test_aclnn_normal_float_float.cpp) | Invokes the stateless_random_normal_v2 operator through the [aclnnNormalFloatFloat](./docs/aclnnNormalFloatFloat_en.md) API. |
| aclnn invocation | [test_aclnn_normal_float_tensor](./examples/test_aclnn_normal_float_tensor.cpp) | Invokes the stateless_random_normal_v2 operator through the [aclnnNormalFloatTensor](./docs/aclnnNormalFloatTensor_en.md) API. |
| aclnn invocation | [test_aclnn_normal_tensor_float](./examples/test_aclnn_normal_tensor_float.cpp) | Invokes the stateless_random_normal_v2 operator through the [aclnnNormalTensorFloat](./docs/aclnnNormalTensorFloat_en.md) API. |
| aclnn invocation | [test_aclnn_normal_tensor_tensor](./examples/test_aclnn_normal_tensor_tensor.cpp) | Invokes the stateless_random_normal_v2 operator through the [aclnnNormalTensorTensor](./docs/aclnnNormalTensorTensor_en.md) API. |
