# SignBitsPack

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: extracts the sign bits of floating-point numbers and packs them into uint8 bytes. The sign bits of every 8 floating-point numbers (0 indicates non-negative, and 1 indicates negative) are packed into 1 uint8 byte, where the sign bit of the 1st floating-point number corresponds to bit7 and that of the 8th corresponds to bit0. It is mainly used in the 1-bit Adam optimizer scenario to compress and store sign information.
- Computation formula:

$$
b_i = \begin{cases} 0 & x_i \geq 0 \\ 1 & x_i < 0 \end{cases}
$$

$$
y_j = \sum_{k=0}^{7} b_{8j+k} \cdot 2^{7-k}
$$

$$
\text{out} = \mathrm{reshape}\!\left(\{y_j\},\; (\text{size},\; L\,/\,\text{size})\right)
$$

Here, $L = \lceil N / 8 \rceil$, and $N$ is the number of input elements. When $N$ is not a multiple of 8, the tail is padded with $-1.0$ (the sign bit is 1).

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
      <td>Floating-point tensor to be packed, 1D. The sign bit of each element is extracted. Non-contiguous tensors are automatically converted to contiguous.</td>
      <td>FLOAT16/FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Required attribute</td>
      <td><ul><li>Size of the first dimension of the output.</li><li>The value must be a positive integer (size &ge; 1).</li><li>ceil(N/8) % size == 0 must be satisfied.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Sign bit packing result, 2D. Its shape is [size, ceil(N/8)/size], and its dtype is always uint8.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- Only the ND data format is supported.
- The input x must be 1D (rank=1). Scalars or multi-dimensional tensors are not supported.
- The type combination is fixed to FLOAT16→UINT8 or FLOAT→UINT8, and other dtype combinations are not supported.
- The attribute size must be a positive integer (size &ge; 1), and ceil(N/8) must be integer-divisible by size.
- Empty tensors (N=0) are supported, and an empty output with the shape [size, 0] is returned directly.
- Non-contiguous tensors are supported, and the framework automatically converts them to contiguous before execution.
- Both +0 and -0 are considered non-negative (the sign bit is 0). NaN is not supported, and its behavior is undefined.
- Deterministic implementation by default. The same input always produces the same output.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Single-operator invocation | [test_aclnn_sign_bits_pack](./examples/test_aclnn_sign_bits_pack.cpp) | Invokes the SignBitsPack operator through the aclnn API. |
| Graph-mode invocation | [test_geir_sign_bits_pack](./examples/test_geir_sign_bits_pack.cpp) | Invokes the SignBitsPack operator by building a graph through GE-IR. |
