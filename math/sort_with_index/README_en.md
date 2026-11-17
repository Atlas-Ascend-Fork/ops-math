# SortWithIndex

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> |√ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: Sorts the input tensor x along the specified dimension in ascending or descending order and reorders the input indices index accordingly, producing y and sorted_index.

- Formulas:

$$p = \operatorname{argsort}(x, \text{axis}, \text{descending}, \text{stable})$$

$$y = \operatorname{gather}(x, \text{axis}, p)$$

$$\text{sorted\_index} = \operatorname{gather}(\text{index}, \text{axis}, p)$$

Here, p denotes the positions of the sorted elements in the input tensor x.

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
      <td>Input tensor to be sorted.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Input</td>
      <td>Input index tensor to be reordered with x. Its shape must match x. Index values are not used for sorting comparisons.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>Dimension along which to sort. Defaults to -1 (the last dimension).</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>descending</td>
      <td>Attribute</td>
      <td>Whether to sort in descending order. true indicates descending order and false indicates ascending order. Defaults to false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stable</td>
      <td>Attribute</td>
      <td>Whether to perform stable sorting. true indicates stable sorting and false indicates unstable sorting. Defaults to false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Sorted result of x, with the same shape, data type, and data format as x.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted_index</td>
      <td>Output</td>
      <td>Input indices index reordered according to the sorting order of x, with the same shape as x.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: x and y support only FLOAT16 and BFLOAT16; index and sorted_index support only INT32.

## Constraint Description

- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
  - Sorting is supported only along the last dimension. axis must be -1 or the input rank minus 1.
  - Scalars and empty tensors are not supported. All input dimensions must have positive lengths.
  - The last dimension must not exceed 2147483647. The product of the remaining dimensions must not exceed 4294967295.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_sort_with_index](./examples/test_geir_sort_with_index.cpp) | Invokes the SortWithIndex operator by building a graph through the [operator IR](../less/op_graph/less_proto.h). |
