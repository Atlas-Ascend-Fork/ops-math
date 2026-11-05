# Log

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |     ×    |

## Function Description

- Operator function: computes the logarithm of the elements in the input tensor x element-wise, and saves the result to the output tensor y. It supports input tensors of multiple types and supports the scaling (shift/scale) operation on the input tensor x.
- Computation formula:

    $$
    y = log_{base}(shift + scale * x)
    $$

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>Input tensor, which is x in the formula.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, FLOAT16, FLOAT, DOUBLE <br>
      BOOL, BFLOAT16, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>base</td>
      <td>Optional attribute</td>
      <td><ul><li>Base of the logarithm.</li><li>The default value is -1.0 (indicating the natural logarithm base e).</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Optional attribute</td>
      <td><ul><li>Scaling factor of the input tensor x.</li><li>The default value is 1.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>Optional attribute</td>
      <td><ul><li>Offset of the input tensor x.</li><li>The default value is 0.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor after the logarithm computation, which is y in the formula.</td>
      <td>If the input type is an integer type (INT type), the output type is FLOAT;<br>
      in other cases, the output type is the same as that of the input tensor x</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- The parameter base must be greater than 0. The default value -1 indicates the base e.
- If the input value is in the specific range (0,0.01] or [0.95,1.05), the output precision may be unstable
- Integer inputs are automatically converted to the FLOAT type for output

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation | [test_aclnn_log](./examples/test_aclnn_log.cpp) <br> [test_aclnn_inplace_log](./examples/test_aclnn_inplace_log.cpp) | Invokes the Log operator through the [aclnnLog or aclnnInplaceLog](./docs/aclnnLog&aclnnInplaceLog_en.md) API.    |
| aclnn invocation | [test_aclnn_log2](./examples/test_aclnn_log2.cpp) <br> [test_aclnn_inplace_log2](./examples/test_aclnn_inplace_log2.cpp) | Invokes the Log operator through the [aclnnLog2 or aclnnInplaceLog2](./docs/aclnnLog2&aclnnInplaceLog2_en.md) API.    |
| aclnn invocation | [test_aclnn_log10](./examples/test_aclnn_log10.cpp) <br> [test_aclnn_inplace_log10](./examples/test_aclnn_inplace_log10.cpp) | Invokes the Log operator through the [aclnnLog10 or aclnnInplaceLog10](./docs/aclnnLog10&aclnnInplaceLog10_en.md) API.    |
| Graph-mode invocation | [test_geir_log](./examples/test_geir_log.cpp)   | Invokes the Log operator by building a graph through the [operator IR](./op_graph/log_proto.h). |
