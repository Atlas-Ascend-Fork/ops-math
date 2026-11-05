# StatelessRandomNormalV3

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

- Operator function: generates a random number tensor that follows the normal distribution based on the specified mean and standard deviation (stdev). It uses the counter-based Philox random number generation algorithm and supports mean and stdev in the form of a scalar or a tensor.

- Computation formula:

  $$
  result = StatelessRandomNormalV2 + Mul(std) + Add(mean)
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
      <td>shape</td>
      <td>Input</td>
      <td>Shape of the output tensor.</td>
      <td>INT64, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>Key used by the counter-based random number generation algorithm.</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counter</td>
      <td>Input</td>
      <td>Initial counter value used by the counter-based random number generation algorithm.</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Mean of the normal distribution, which can be a scalar or a tensor with the same shape as the output.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>stdev</td>
      <td>Input</td>
      <td>Standard deviation of the normal distribution, which can be a scalar or a tensor with the same shape as the output.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Outputs random values that follow the normal distribution with the specified shape.</td>
      <td>FLOAT, BF16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Optional attribute that specifies the data type of the output.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Sample Code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_normal_float_float](../stateless_random_normal_v2/examples/test_aclnn_normal_float_float.cpp) | Invokes the stateless_random_normal_v3 operator through the [aclnnNormalFloatFloat](../stateless_random_normal_v2/docs/aclnnNormalFloatFloat_en.md) API. |
| aclnn invocation | [test_aclnn_normal_float_tensor](../stateless_random_normal_v2/examples/test_aclnn_normal_float_tensor.cpp) | Invokes the stateless_random_normal_v3 operator through the [aclnnNormalFloatTensor](../stateless_random_normal_v2/docs/aclnnNormalFloatTensor_en.md) API. |
| aclnn invocation | [test_aclnn_normal_tensor_float](../stateless_random_normal_v2/examples/test_aclnn_normal_tensor_float.cpp) | Invokes the stateless_random_normal_v3 operator through the [aclnnNormalTensorFloat](../stateless_random_normal_v2/docs/aclnnNormalTensorFloat_en.md) API. |
| aclnn invocation | [test_aclnn_normal_tensor_tensor](../stateless_random_normal_v2/examples/test_aclnn_normal_tensor_tensor.cpp) | Invokes the stateless_random_normal_v3 operator through the [aclnnNormalTensorTensor](../stateless_random_normal_v2/docs/aclnnNormalTensorTensor_en.md) API. |
