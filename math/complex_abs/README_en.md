# ComplexAbs

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    ×     |
| <term>Atlas training products</term>                       |    ×     |

## Function Description

- Operator function: computes the absolute value (magnitude) of each element in the input complex tensor.

- Computation formula:

  For the complex input $x_i = a_i + b_i \cdot j$, the magnitude is output:

$$
y_{i}=|x_{i}|=\sqrt{a_{i}^{2}+b_{i}^{2}}
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
      <td>x</td>
      <td>Input</td>
      <td>Input parameter for the complex absolute value computation, which is the input tensor x in the formula.</td>
      <td>COMPLEX32, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the complex absolute value computation, which is the output tensor y in the formula. The output data type is determined by the input data type: FLOAT is output when the input is COMPLEX64, FLOAT16 is output when the input is COMPLEX32, and DOUBLE is output when the input is COMPLEX128.</td>
      <td>FLOAT, FLOAT16, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>Tout</td>
      <td>Attribute</td>
      <td>Output data type. This attribute is retained for compatibility with the TensorFlow ComplexAbs operator and is currently not used. The output data type is deduced from the input data type.</td>
      <td>Type, and the default value is DT_FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- The input supports only the COMPLEX32, COMPLEX64, and COMPLEX128 types, and other data types are not supported.
- The output data type is determined by the input data type and cannot be explicitly specified: COMPLEX64 corresponds to the FLOAT output, COMPLEX32 corresponds to the FLOAT16 output, and COMPLEX128 corresponds to the DOUBLE output.
- The shapes of the input and output are the same. Dynamic shape (DynamicShape) and dynamic dimension (DynamicRank) are supported.
- Empty tensors (with 0 in the shape) are not supported.
- The data format supports ND. Non-contiguous tensors are supported, and the number of dimensions is not greater than 8.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|--------|------|
| Graph-mode invocation | [test_geir_complex_abs](./examples/test_geir_complex_abs.cpp) | Invokes the ComplexAbs operator by building a graph through the [operator IR](./op_graph/complex_abs_proto.h). |
