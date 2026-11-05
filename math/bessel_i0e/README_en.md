# BesselI0e

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |

## Function Description

- Operator function: computes the value of the exponentially scaled modified Bessel function of the first kind of order zero for each element in the input tensor.

- Computation formula:

$$y = e^{-|x|} \cdot I_0(x)$$

where $I_0(x)$ is the modified Bessel function of the first kind of order zero.

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
      <td>Input parameter for the BesselI0e computation, which is x in the formula.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the BesselI0e computation, which is y in the formula.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraint Description

- The data type and shape of the input must be consistent with those of the output.
- Dynamic shapes and dynamic ranks are supported (the shape dimensions can be determined at runtime, and the rank supports 0 to 8 dimensions).
- Empty tensors are supported (when the number of elements is 0, the output is an empty tensor).
- The input and output support only the ND data format (non-ND formats are explicitly rejected by Tiling on the host side).
- Only the GE graph-mode invocation is supported, and no aclnn operator API is provided.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_bessel_i0e](./examples/test_geir_bessel_i0e.cpp) | Invokes the BesselI0e operator in graph mode. |

## Third-Party Framework Compatibility

- This operator corresponds to the `tf.math.bessel_i0e` operator of TensorFlow.
