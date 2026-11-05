# NotEqual

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

* Operator function: compares whether the two input tensors are not equal element by element
* Computation formula:

$$
\text{y}_i =
\begin{cases}
True,&\text{if }x_{i, 1} \ne x_{i,2} \\
False,&\text{otherwise}
\end{cases}
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
      <td>x1</td>
      <td>Input</td>
      <td>Input x1 in the formula.</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input x2 in the formula.</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>y in the formula.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn API | [test_aclnn_ne_tensor](./examples/test_aclnn_ne_tensor.cpp) | Invokes the NotEqual operator through the [aclnnNeTensor](./docs/aclnnNeTensor&aclnnInplaceNeTensor_en.md) API    |
| GE graph mode | [test_geir_not_equal](./examples/test_geir_not_equal.cpp)   | Invokes the NotEqual operator by building a graph through the [operator IR](./op_graph/not_equal_proto.h) |
