# Asinh

English | [简体中文](./README.md)

> The inverse hyperbolic sine (Inverse Hyperbolic Sine) operator, corresponding to PyTorch `torch.asinh` / `torch.asinh_`.
> Online aclnn API documentation: [docs/aclnnAsinh&aclnnInplaceAsinh_en.md](docs/aclnnAsinh&aclnnInplaceAsinh_en.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>          |      √   |
| <term>Atlas A3 products</term>     |    √    |
| <term>Atlas A2 products</term> |    √    |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×    |
| <term>Atlas inference products</term>                       |     √    |
| <term>Atlas training products</term>                       |     √    |

## Function Description

- Interface function: performs the inverse hyperbolic sine operation on each element in the input Tensor and outputs the result.

- Computation formula:

$$
y_{i}=ln(x_{i} + \sqrt{x_{i}^2 + 1})
$$

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 240px">
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
    <td>Input x in the formula, whose dtype is the same as that of y. The shape is arbitrary (1 to 8 dimensions). Empty Tensors are supported, and non-contiguous Tensors are supported.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Output y in the formula. The shape is exactly the same as that of x. It is specific to the non-in-place API.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>selfRef</td>
    <td>Input/Output</td>
    <td>Specific to the in-place API. It is a tensor that serves as both input and output, and the computation result is written back in place.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

- <term>Ascend 950PR&950DT products</term>, <term>Atlas A2 products</term>, and <term>Atlas A3 products</term>: support BFLOAT16.
- <term>Atlas training products</term> and <term>Atlas inference products</term>: do not support BFLOAT16.
- Input types additionally supported by the aclnn interface: INT8, INT16, INT32, INT64, UINT8, and BOOL are converted to FLOAT32 for computation; DOUBLE, COMPLEX64, and COMPLEX128 are computed through the AICPU path.

## Constraint Description

None

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn invocation (non-in-place)</td>
    <td><a href="./examples/test_aclnn_asinh.cpp">examples/test_aclnn_asinh.cpp</a></td>
    <td rowspan="2">Refer to build.sh in the root directory of the ops-math repository to complete operator compilation and operator package installation, and then verify with the examples (on a physical NPU).</td>
  </tr>
  <tr>
    <td>Graph-mode (GE IR) invocation</td>
    <td><a href="./examples/test_geir_asinh.cpp">examples/test_geir_asinh.cpp</a></td>
  </tr>
</tbody></table>
