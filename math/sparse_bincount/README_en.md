# SparseBincount

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: counts the occurrences or weighted accumulated sum of each value (bin index) in a sparse tensor. Both the 1D and multi-dimensional modes are supported, and the binary_output mode is supported.

- Computation formula:

$$
\text{1D: } output[v] += w_i \quad (\text{if } v < size)
$$

$$
\text{Multi-dim: } output[batch][v] += w_i \quad (\text{if } v < size \text{ and } batch < dense\_shape[0])
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
      <td>indices</td>
      <td>Input</td>
      <td>Indices of the sparse tensor, a 2D tensor with shape (N, R).</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>Input</td>
      <td>Values of the sparse tensor (bin index), a 1D tensor with shape (N,).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dense_shape</td>
      <td>Input</td>
      <td>Dense shape of the sparse tensor, a 1D tensor with shape (R,).</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input</td>
      <td>Number of bins (scalar tensor) with shape (1,).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>Input</td>
      <td>Weights, a 1D tensor with shape (N,) or (0,). If empty, each occurrence is counted as 1.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>binary_output</td>
      <td>Optional attribute</td>
      <td>Whether to output binary values. The default value is false. If True, bins that have appeared are set to 1. If False, the count or weighted accumulation is output.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Statistics result. In 1D mode, the shape is [size]. In multi-dimensional mode, the shape is [dense_shape[0], size].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- indices is a 2D tensor with shape (N, R); values, dense_shape, size, and weights are all 1D tensors
- The number of rows of indices (N) must be equal to the number of elements of values (N)
- The number of columns of indices (R) must be equal to the number of elements of dense_shape (R), and the number of elements of dense_shape (R) must be at least 1 (dense_shape is not empty)
- indices[:, j] must be in the range [0, dense_shape[j])
- values[i] must be in the range [0, size)
- If weights is not empty, the number of elements must be equal to the number of elements of values (N)
- size is a non-negative scalar with shape (1,)
- If weights is empty (the number of elements is 0), each occurrence is counted as 1.0
- If binary_output = True, bins that have appeared are set to 1.0 and weights are ignored

## Invocation Description

| Invocation Method | Sample Code | Description |
|--------------|--------|------|
| GE graph mode | [test_geir_sparse_bincount](./examples/test_geir_sparse_bincount.cpp) | Invokes the SparseBincount operator by building a graph through the [operator IR](./op_graph/sparse_bincount_proto.h). |
