# BitwiseXor

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

- Operator function: computes the bitwise XOR of each element in the input tensor self and the element at the corresponding position in the input tensor other. The inputs self and other must be integer or Boolean types. For Boolean types, the logical XOR is computed.
- Computation formula:

  $$
  \text{out}_i =
  \text{self}_i \, \bigoplus\, \text{other}_i
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
      <td>Input self in the formula.</td>
      <td>INT16, UINT16, INT32, INT64, INT8, UINT8, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input other in the formula.</td>
      <td>INT16, UINT16, INT32, INT64, INT8, UINT8, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output out in the formula.</td>
      <td>INT16, UINT16, INT32, INT64, INT8, UINT8, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - Atlas inference products and Atlas training products: the data types INT8, UINT8, UINT32, INT64, and UINT64 are not supported.
  - Atlas A2 products and Atlas A3 products: INT8, UINT8, UINT32, and UINT64 are not supported.

## Constraint Description

- None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                                                         |
| ---------------- | --------------------------- |----------------------------------------------------------------------------|
| aclnn API  | [test_aclnn_bitwise_xor_scalar](examples/test_aclnn_bitwise_xor_scalar.cpp) | Invokes the BitwiseXor operator through the [aclnnBitwiseXorScalar](docs/aclnnBitwiseXorScalar&aclnnInplaceBitwiseXorScalar_en.md) API. |
| aclnn API  | [test_aclnn_bitwise_xor_tensor](examples/test_aclnn_bitwise_xor_tensor.cpp) | Invokes the BitwiseXor operator through the [aclnnBitwiseXorTensor](docs/aclnnBitwiseXorTensor&aclnnInplaceBitwiseXorTensor_en.md) API. |
| aclnn API  | [test_aclnn_inplace_bitwise_xor_scalar](examples/test_aclnn_inplace_bitwise_xor_scalar.cpp) | Invokes the BitwiseXor operator through the [aclnnInplaceBitwiseXorScalar](docs/aclnnBitwiseXorScalar&aclnnInplaceBitwiseXorScalar_en.md) API. |
| aclnn API  | [test_aclnn_inplace_bitwise_xor_tensor](examples/test_aclnn_inplace_bitwise_xor_tensor.cpp) | Invokes the BitwiseXor operator through the [aclnnInplaceBitwiseXorTensor](docs/aclnnBitwiseXorTensor&aclnnInplaceBitwiseXorTensor_en.md) API. |
