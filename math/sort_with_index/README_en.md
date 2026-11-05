# SortWithIndex

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                    |    √      |
| <term>Atlas A3 products</term>    |    ×      |
| <term>Atlas A2 products</term>    |    ×      |
| <term>Atlas 200I/500 A2 inference products</term>                    |    ×      |
| <term>Atlas inference products</term>                            |   ×       |
| <term>Atlas training products</term>                            |   √       |

## Function Description

- Operator function: sorts the input tensor by element value, and sorts the index values along with the corresponding element values.

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
      <td>Input tensor for element value sorting.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Input</td>
      <td>Input tensor for element value sorting.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Input</td>
      <td>Dimension used as the sorting criterion.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>descending</td>
      <td>Input</td>
      <td>Controls the sorting order. True indicates descending order, and False indicates ascending order.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stable</td>
      <td>Input</td>
      <td>Whether to perform stable sorting. True indicates stable sorting, and False indicates unstable sorting.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output result after elements are sorted by value.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted_index</td>
      <td>Output</td>
      <td>Output result after the index values are sorted along with the element values.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The shapes of input x and index must be the same.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_sort_with_index](./examples/test_geir_sort_with_index.cpp) | Invokes the SortWithIndex operator by building a graph through the [operator IR](../less/op_graph/less_proto.h). |
