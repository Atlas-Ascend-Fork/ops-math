# Acosh

English | [简体中文](./README.md)

> Inverse hyperbolic cosine operator, which corresponds to PyTorch `torch.acosh` / `torch.acosh_`.
> Online aclnn API documentation: [docs/aclnnAcosh&aclnnInplaceAcosh_en.md](docs/aclnnAcosh&aclnnInplaceAcosh_en.md)

## Product Support Status

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function Description

- API function: computes the inverse hyperbolic cosine of each element in the input Tensor and outputs the result.

- Computation formula:

  $$
  out=cosh^{-1}(self)
  $$

## Parameter Description

<table style="table-layout: fixed; width: 1000px"><colgroup>
<col style="width: 50px">
<col style="width: 70px">
<col style="width: 200px">
<col style="width: 100px">
<col style="width: 50px">
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
    <td>The input x in the formula, whose dtype is the same as that of y and whose shape is arbitrary (0 to 8 dimensions). Empty tensors are supported. A non-contiguous tensor input is automatically converted to a contiguous tensor by the framework before execution.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>The output y in the formula, whose shape is exactly the same as that of x. A non-contiguous tensor output is automatically converted to a contiguous tensor by the framework before execution.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>selfRef</td>
    <td>Input/Output</td>
    <td>A parameter specific to the aclnn in-place API (corresponding to the input x and output y in the operator definition). It is a tensor that serves as both input and output, and the computation result is written back in place. A non-contiguous tensor is automatically converted to a contiguous tensor by the framework before execution.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>


## Constraint Description

- The value range of the real input is [1, +∞). When x is less than 1, NaN is output.
- The number of shape dimensions ∈ [0, 8], and the shape of y is exactly the same as that of x.

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
    <td><a href="./examples/test_aclnn_acosh.cpp">examples/test_aclnn_acosh.cpp</a></td>
    <td rowspan="2">Refer to build.sh in the root directory of the ops-math repository to complete operator compilation and operator package installation, and then use the examples for verification (on a real NPU).</td>
  </tr>
  <tr>
    <td>aclnn invocation (in-place)</td>
    <td><a href="./examples/test_aclnn_inplace_acosh.cpp">examples/test_aclnn_inplace_acosh.cpp</a></td>
  </tr>
</tbody></table>
