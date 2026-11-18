# AdjacentDifference

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term>      |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: compares the differences between adjacent elements of the input tensor. If two adjacent elements are the same, 0 is returned; otherwise, 1 is returned.

- Computation formula:

$$
out_i = {(x_i == x_j ?\ 0 : 1)}.to(y_dtype) \ j = i - 1
$$

- First element: $out_0 = 0$. The first element has no preceding adjacent element to compare with, so it always outputs 0.

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
      <td>Input parameter for the AdjacentDifference computation, which is x in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_dtype</td>
      <td>Attribute</td>
      <td>An attribute that specifies the data type of the output tensor y, which is y_dtype in the formula. Optional; if not set, it defaults to INT32 (DT_INT32(3)). Valid values are INT32 (DT_INT32(3)) or INT64 (DT_INT64(9)).</td>
      <td>INT32, INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the AdjacentDifference computation, which is out in the formula. It has the same shape as the input x, and its data type is determined by the y_dtype attribute (INT32 or INT64) and must be consistent with y_dtype.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The input x and the output y have the same shape; only the ND format is supported; the input rank range is [0, 8]. Dynamic shape and dynamic rank inputs are supported.
- Empty tensors (with zero-length dimensions, i.e., zero elements) are supported: the output is an empty tensor of the same shape. Rank-0 scalar inputs (shape ()) are supported: they are processed as a single element and output the scalar 0.
- Non-contiguous tensors are supported as input; see [Non-contiguous Tensor](../../docs/en/context/non_contiguous_tensor.md). The output y is allocated as contiguous storage.
- The input x is read-only and will not be modified; the output is written to an independently allocated y, and in-place modification is not supported. The storage regions of the input and output should not overlap; behavior is undefined if they overlap.
- The data type of the output y must be consistent with the y_dtype attribute (when y is INT32, y_dtype must be DT_INT32(3); when y is INT64, y_dtype must be DT_INT64(9)); the y_dtype value only allows DT_INT32(3)/DT_INT64(9).
- Floating-point inputs (FLOAT, FLOAT16, BFLOAT16) compare adjacent elements by numeric equality: NaN is not equal to any value (including NaN itself), and the corresponding output is 1; +Inf equals +Inf and -Inf equals -Inf, and the corresponding output is 0.
- Exception scenarios (validated and rejected with an error at the graph compilation stage in graph mode; error messages follow the GE framework logs, and this operator does not provide separate public error codes):
  - The data type of y is inconsistent with y_dtype;
  - y_dtype is not DT_INT32(3)/DT_INT64(9);
  - The data type of the input x is not in the supported list.

## Invocation Description

| Invocation Method | Sample Code |   Description    |
| :-------- | :---------- | :------------------------ |
|  Graph mode  |[test_geir_adjacent_difference](examples/test_geir_adjacent_difference.cpp) | Invokes the AdjacentDifference operator by building a graph through the [operator IR](./op_graph/adjacent_difference_proto.h). |
