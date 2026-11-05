# ApproximateEqual

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: determines element-wise whether two input tensors are approximately equal.
- Computation formula:

  $$
  y_i = |x1_i - x2_i| < tolerance
  $$

  where tolerance is a non-negative floating-point number, and the default value is 1e-5. The output is a tensor of the BOOL type, whose element values are true (approximately equal) or false (not approximately equal).

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
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
    <td>x1</td>
    <td>Input</td>
    <td>Input x1 in the formula.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>Input</td>
    <td>Input x2 in the formula. Its data type and shape must be the same as those of x1.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>tolerance</td>
    <td>Optional attribute</td>
    <td><ul><li>Threshold for determining approximate equality.</li><li>The default value is 1e-5.</li><li>The value must be a non-negative finite floating-point number.</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>The element-wise comparison result. The shape is the same as that of x1.</td>
    <td>BOOL</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraint Description

- The data types of x1 and x2 must be the same. Implicit type conversion is not supported.
- The shapes of x1 and x2 must be strictly equal. Broadcasting is not supported.
- tolerance must be a non-negative finite floating-point number (NaN or Inf is not allowed).

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/test_geir_approximate_equal.cpp">test_geir_approximate_equal</a></td>
    <td>Invokes the ApproximateEqual operator by building a graph through the <a href="./op_graph/approximate_equal_proto.h">operator IR</a>.</td>
  </tr>
</tbody></table>
