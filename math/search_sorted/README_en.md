# SearchSorted

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: searches a sorted tensor sorted_sequence for the positions where the given tensor values should be inserted. It returns a tensor of the same size as values, where each element indicates the position where the given value should be inserted in the original tensor.

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
      <td>sorted_sequence</td>
      <td>Input</td>
      <td> Input tensor. The values along its last dimension are sorted in ascending order.</td>
      <td>FLOAT16, FLOAT, INT8, INT16, INT32, INT64, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>Input</td>
      <td>Tensor to be inserted. Its data type must be the same as that of the input sorted_sequence. Except for the last dimension, its other dimensions must be the same as those of sorted_sequence.</td>
      <td>FLOAT16, FLOAT, INT8, INT16, INT32, INT64, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorter</td>
      <td>Optional input</td>
      <td>Optional input tensor. Its shape is the same as that of the unsorted sorted_sequence, and it contains the index sequence that sorts sorted_sequence in ascending order along the innermost dimension.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Optional attribute</td>
      <td>
          • Specifies the data type of the output tensor. Only INT64 and INT32 are supported.<br>
          • The default value is INT64.
      </td>
      <td>TYPE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>right</td>
      <td>Optional attribute</td>
      <td>
          • Optional Boolean value. false indicates that when identical values exist in the sequence and the insertion position is not unique, the insertion position is aligned to the left. true indicates that the insertion position is aligned to the right in this case.<br>
          • The default value is false.
      </td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor. Its shape is the same as that of the input values.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                              | Description                                                                     |
|--------|-------------------------------------------------------------------|------------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_search_sorted](./examples/test_aclnn_search_sorted.cpp) | Invokes the SearchSorted operator through the [aclnnSearchSorted](./docs/aclnnSearchSorted_en.md) API. |
| aclnn invocation | [test_aclnn_search_sorteds](./examples/test_aclnn_search_sorteds.cpp) | Invokes the SearchSorted operator through the [aclnnSearchSorteds](./docs/aclnnSearchSorteds_en.md) API. |
| Graph-mode invocation | [test_geir_search_sorted](./examples/test_geir_search_sorted.cpp)   | Invokes the SearchSorted operator by building a graph through the [operator IR](./op_graph/search_sorted_proto.h). |
