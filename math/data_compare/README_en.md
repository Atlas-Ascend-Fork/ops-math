# DataCompare

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

- Operator function: compares the two input tensors `x1` and `x2`, which have the same shape and dtype, element by element, counts the total number of elements whose differences exceed the tolerance range (All Reduce), and outputs a FLOAT scalar indicating the total number of mismatched elements.
- Computation formula:

  Element-wise mismatch judgment:

  $$
  diff_i = |x1_i - x2_i|
  $$

  $$
  threshold_i = atol + rtol \times |x2_i|
  $$

  $$
  mismatch_i = \begin{cases} 1.0, & diff_i > threshold_i \\ 0.0, & diff_i \leq threshold_i \end{cases}
  $$

  Reduction output (total number of mismatched elements):

  $$
  output = \sum_{i} mismatch_i \quad (\text{dtype: FLOAT})
  $$

- Output `0`: all elements are within the tolerance range, and the two tensors are considered matched.
- Output `> 0`: elements with differences exceeding the tolerance exist. A larger value indicates a greater difference.

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
    <td>The first input tensor, which corresponds to x1 in the formula. Its shape and dtype must be exactly the same as those of x2, and broadcasting is not supported. Tensors of 0 to 8 dimensions and empty Tensors are supported.</td>
    <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>Input</td>
    <td>The second input tensor, which corresponds to x2 in the formula. Its shape and dtype must be exactly the same as those of x1, and broadcasting is not supported. Tensors of 0 to 8 dimensions and empty Tensors are supported.</td>
    <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>num</td>
    <td>Output</td>
    <td>The total number of mismatched elements, which corresponds to output in the formula. Fixed as a FLOAT scalar (0D tensor).</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>atol</td>
    <td>Attribute</td>
    <td><ul><li>Absolute tolerance, which corresponds to atol in the formula.</li><li>Must be >= 0.</li><li>The default value is 1e-5.</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>rtol</td>
    <td>Attribute</td>
    <td><ul><li>Relative tolerance, which corresponds to rtol in the formula.</li><li>Must be >= 0.</li><li>The default value is 1e-3.</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraint Description

- The shapes of x1 and x2 must be exactly the same, and broadcasting is not supported.
- The dtypes of x1 and x2 must be exactly the same, and mixed dtypes are not supported.
- Only the ND format is supported for x1, x2, and the output num; non-ND format inputs are rejected by validation.
- For input types other than FLOAT (FLOAT16/BFLOAT16/INT8/UINT8/INT32), the operator first casts them to FLOAT (fp32) internally and then performs the entire comparison chain; both the comparison threshold and the difference are computed in fp32 precision.
- The value range of atol/rtol is $[0, 1\times10^{38}]$, and out-of-range inputs are rejected by validation.
- The reduction method is All Reduce (reduction over all axes), and users cannot specify the axis.
- Dynamic shape/dynamic rank inputs are supported: graphs may be built with shapes containing unknown dims (-1); at execution time the shape must have been materialized, the operator performs the comparison based on the total number of elements after materialization, the output is always a scalar, and the semantics are identical to those of static shapes.
- The output is fixed as a FLOAT scalar.
- Deterministic implementation: the same input is guaranteed to produce the same output.
- When the total number of input elements exceeds $2^{24}$ (about 16M), the count may be inaccurate due to the limitation of the exact representation range of FLOAT.

## Invocation Description

| Invocation Method | Sample Code | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| GE graph mode | [test_geir_data_compare](./examples/arch35/test_geir_data_compare.cpp)   | Invokes the DataCompare operator by building a graph through the [operator IR](./op_graph/data_compare_proto.h). |
