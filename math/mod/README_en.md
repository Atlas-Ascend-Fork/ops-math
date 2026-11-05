# Mod

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                      |    √     |
| <term>Atlas A3 products</term>      |    √     |
| <term>Atlas A2 products</term>       |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: returns the remainder of self divided by other.

- Computation formula:

  After the tensor self and the scalar or tensor other are broadcast into tensors of the same shape, Mod can be expressed by the following mathematical formula:

  $$
  out_{i} = self_{i} - trunc \left(\frac{self_{i}}{other_{i}}\right) * other_{i}
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
      <td>Input parameter for the mod computation, which is self_i in the formula.</td>
      <td>DOUBLE, BFLOAT16, FLOAT16, FLOAT32, INT32, INT64, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input parameter for the mod computation, which is other_i in the formula.</td>
      <td>DOUBLE, BFLOAT16, FLOAT16, FLOAT32, INT32, INT64, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the mod computation, which is out_i in the formula.</td>
      <td>DOUBLE, BFLOAT16, FLOAT16, FLOAT32, INT32, INT64, INT8, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The data types must satisfy the data type inference rules. The inferred data type must be within the range of supported data types.
- The shapes of self and other must satisfy the [broadcast rules](../../docs/en/context/broadcast_relationship.md#broadcast-rules).
- Data with more than 8 dimensions is not supported.

## Invocation Description

| Invocation Method | Sample Code  | Description  |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_fmod_scalar](examples/test_aclnn_fmod_scalar.cpp) | Invokes the Mod operator through the [aclnnFmodScalar](docs/aclnnFmodScalar&aclnnInplaceFmodScalar_en.md) API. |
| aclnn API | [test_aclnn_fmod_tensor](examples/test_aclnn_fmod_tensor.cpp) | Invokes the Mod operator through the [aclnnFmodTensor](docs/aclnnFmodTensor&aclnnInplaceFmodTensor_en.md) API. |
