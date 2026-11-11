# Eltwise

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: the Eltwise operator performs element-wise operations on 1 to 32 input tensors of the same shape and the same dtype, and supports three computation modes.

- Computation formula:

  - **mode=0 (PRODUCT)**: element-wise product

    $$out_i = x^{(0)}_i \times x^{(1)}_i \times \cdots \times x^{(n-1)}_i$$

  - **mode=1 (SUM)**: element-wise weighted sum

    $$out_i = c_0 \cdot x^{(0)}_i + c_1 \cdot x^{(1)}_i + \cdots + c_{n-1} \cdot x^{(n-1)}_i$$

    Here $c_k$ is the weighting coefficient (coeff) of the $k$-th input, and the default value is 1.0.

  - **mode=2 (MAX)**: element-wise maximum

    $$out_i = \max(x^{(0)}_i, x^{(1)}_i, \ldots, x^{(n-1)}_i)$$

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>The list of input tensors, which contains 1 to 32 tensors. All tensors must have the same shape and the same dtype. Empty tensors (0 elements) are supported.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>N</td>
    <td>Attribute (required)</td>
    <td><ul><li>The number of input tensors.</li><li>This attribute must be set when building a graph in GE graph mode, and the value must be the same as the actual number of input tensors.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>mode</td>
    <td>Attribute (optional)</td>
    <td><ul><li>The computation mode, and the default value is 1.</li><li>0=PRODUCT (element-wise product), 1=SUM (element-wise weighted sum), 2=MAX (element-wise maximum).</li></ul></td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>coeff</td>
    <td>Attribute (optional)</td>
    <td><ul><li>The array of weighting coefficients, which is valid only when mode=1.</li><li>The length must be the same as the number of input tensors.</li><li>If it is not provided, all coefficients default to 1.0.</li><li>This parameter is ignored when mode=0 and mode=2 are used.</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>The output tensor. Its shape is the same as that of the input tensors, and its dtype is the same as that of the input tensors. An empty output tensor is returned for empty input (0 elements).</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraint Description

- All input tensors must have the same shape and the same dtype, and broadcasting is not supported.
- The number of input tensors is in the range [1, 32].
- The output dtype and shape must be exactly the same as those of the input.
- Only the ND data format is supported, and the shape dimension range is 0 to 8.
- The input and output are configured with automatic contiguity (AutoContiguous), and non-contiguous Tensors are converted to contiguous Tensors by the framework before entering the Kernel.
- mode supports only three values: 0 (PRODUCT), 1 (SUM), and 2 (MAX).
- When mode=1, if coeff is provided, its length must equal the number of input tensors; if it is not provided, the coefficients default to 1.0.
- Empty Tensor (0 elements): it is accepted normally, an empty out is returned, and no computation is performed.
- Deterministic computation: yes (element-wise operations do not involve Reduce, and the output result is deterministic).

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode  | [test_geir_eltwise.cpp](examples/test_geir_eltwise.cpp) | Invokes the Eltwise operator in graph mode. Port names (x/y) and attribute definitions (N/mode/coeff) are defined in [eltwise_proto.h](op_graph/eltwise_proto.h). |
