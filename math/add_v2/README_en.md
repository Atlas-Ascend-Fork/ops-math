# AddV2

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

- Operator function: performs element-wise addition on input tensors `x1` and `x2`, supports broadcasting, and is compatible with TensorFlow AddV2 semantics.

- Version note: V2 in the name corresponds to the TensorFlow AddV2 graph node and does not indicate any change to the addition or broadcasting rules. Within the data types supported by both AddV2 and Add, the two operators have the same mathematical semantics. The data types and invocation paths of AddV2 are subject to this document.

- Computation formula:

$$
y_i=x_{1,i}+x_{2,i}
$$

where $i$ iterates over all elements of the broadcast result, and $x_{1,i}$ and $x_{2,i}$ denote the elements at the corresponding positions after broadcasting.

- Broadcasting example:

```text
x1 shape (3, 4), x2 shape (1, 4) -> y shape (3, 4)
x1 shape (3, 1), x2 shape (1, 4) -> y shape (3, 4)
```

During broadcasting, dimensions are aligned from the last dimension forward. For each pair of corresponding dimensions, the lengths must be equal, or the dimension length of at least one input must be 1.

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
      <td>x1</td>
      <td>Input</td>
      <td>The first input tensor of the addition, which corresponds to x1 in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, INT16, UINT8, INT8, INT64, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The second input tensor of the addition, which corresponds to x2 in the formula. The data type is the same as that of x1.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, INT16, UINT8, INT8, INT64, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output tensor of the addition, which corresponds to y in the formula. The data type is the same as that of x1, and the shape is the broadcast result of x1 and x2.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, INT16, UINT8, INT8, INT64, COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The data types of inputs and outputs are subject to the Parameter Description. DOUBLE and COMPLEX128 are not supported.
- Both static shape and dynamic shape support input ND → output ND. Dynamic shape supports a fixed rank.
- The value range of rank is [0,8]. A rank of 0 indicates a scalar.
- <term>Ascend 950PR&950DT products</term>:
  - Supports all data types listed in the Parameter Description.
  - Supports dynamic rank.
  - Supports empty tensors. The shape of output y must still be the broadcast result of the two inputs.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - Supports BFLOAT16, FLOAT16, FLOAT, INT32, and INT64.
  - Does not support dynamic rank.
  - For empty tensors, only the broadcast scenarios where the two inputs have the same shape, or one input is a single-element tensor, are supported.
- <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>:
  - Supports FLOAT16, FLOAT, INT32, and INT64.
  - Does not support dynamic rank.
  - For empty tensors, only the broadcast scenarios where the two inputs have the same shape, or one input is a single-element tensor, are supported.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_add_v2](./examples/test_geir_add_v2.cpp) | Invokes the AddV2 operator by building a graph through the [operator IR](./op_graph/add_v2_proto.h). |

This operator does not provide an ACLNN API with the same name. Only the GE graph-mode invocation in the preceding table is supported.
