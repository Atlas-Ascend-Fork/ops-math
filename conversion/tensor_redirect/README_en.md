# TensorRedirect

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

- Operator function: passes the values of the input tensor `x` to the output tensor `output_x`. The output has the same data type and shape as the input. This operator is used in graph optimization or reference redirection scenarios, and performs no type conversion, broadcasting, or data layout transformation.

- Computation formula:

$$
(\mathrm{output\_x})_i=x_i
$$

where $i$ traverses all elements of $x$.

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
      <td>Input tensor whose values need to be passed.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_x</td>
      <td>Output</td>
      <td>Tensor for the pass-through result. The data type, shape, and data format are all the same as those of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- No in-place invocation API is provided.
- <term>Ascend 950PR&950DT products</term>:
  - All data types listed in the Parameter Description are supported.
  - Both static shape and dynamic shape support input ND→output ND. Dynamic shape supports fixed rank and dynamic rank.
  - The value range of rank is [0,8]. A rank of 0 indicates a scalar. Empty tensors are supported.
  - In a graph inference shape, `-1` indicates an unknown dimension, and `{-2}` indicates an unknown rank. During actual execution, each dimension of the input tensor must be a specific non-negative integer.
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>:
  - BFLOAT16 is not supported, and the other data types are the same as those in the Parameter Description.
  - Static shape supports input ND→output ND. Dynamic shape is not supported.
  - The value range of rank is [1,8]. Empty tensors are not supported.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_tensor_redirect](./examples/test_geir_tensor_redirect.cpp) | Invokes the TensorRedirect operator by building a graph through the [operator IR](./op_graph/tensor_redirect_proto.h). |

This operator does not provide an ACLNN API with the same name, and supports only the GE graph-mode invocation in the preceding table.
