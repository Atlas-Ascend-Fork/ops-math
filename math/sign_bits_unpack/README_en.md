# SignBitsUnpack

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

- Operator function: unpacks the 1-bit Adam sign bits of the UINT8 type into a tensor of the FLOAT32 or FLOAT16 type.

- Computation formula:

Assume that the input `self` is a 1D UINT8 tensor with the length of N. Each byte contains 8 sign bits in LSB-first order. The unpacking rule for the $i$-th bit (from the least significant to the most significant) in the $j$-th byte $b_j$ ($i = 0, 1, \dots, 7$) is as follows:

$$
out_{j \cdot 8 + i} = \begin{cases} +1.0, & \text{bit}_i(b_j) = 1 \\ -1.0, & \text{bit}_i(b_j) = 0 \end{cases}
$$

The output tensor has $N \times 8$ elements in total and is reshaped to 2D. Its shape is:

$$
out.shape = (size,\ \frac{N \times 8}{size})
$$

Here, `size` is the size of the first dimension of the output specified by the input parameter.

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
      <td>Input parameter for the SignBitsUnpack computation. It is a 1D tensor, and each UINT8 element contains 8 sign bits. Empty tensors are supported.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input attribute</td>
      <td>Indicates dimension processing, which is the first dimension of the output tensor during reshape.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Input attribute</td>
      <td>Indicates the data type of the quantized output tensor.</td>
      <td>FLOAT16, FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the SignBitsUnpack computation. It is a 2D tensor, and its data type is determined by the dtype parameter.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- `self` must be a 1-dimensional tensor, and `out` must be a 2-dimensional tensor.
- `size` must be greater than 0, and (the number of elements in `self` × 8) must be integer-divisible by `size`.
- The first dimension of `out` must be equal to `size`.
- The `dtype` parameter supports only FLOAT16 and FLOAT, and the data type of `out` must be the same as that specified by the `dtype` parameter. UINT8 is the data type of `self`, and INT64 is the API type of `size`. Neither of them is an optional value of the `dtype` parameter.
- Only the ND data format is supported.
- The empty tensor scenario is supported.
- Deterministic computation: aclnnSignBitsUnpack defaults to a deterministic implementation.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_sign_bits_unpack](./examples/test_aclnn_sign_bits_unpack.cpp) | Invokes the SignBitsUnpack operator through the [aclnnSignBitsUnpack](./docs/aclnnSignBitsUnpack_en.md) API. |
| Graph-mode invocation | [test_geir_sign_bits_unpack](./examples/test_geir_sign_bits_unpack.cpp)   | Invokes the SignBitsUnpack operator by building a graph through the [operator IR](./op_graph/sign_bits_unpack_proto.h). |
