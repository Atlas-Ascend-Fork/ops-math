# Sort

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

- Operator function: sorts the elements in the input tensor along the specified dimension in ascending or descending order, and returns the corresponding sorted values and index values.

- Computation formula:

$$y1 = \operatorname{sort}(x, \text{axis}, \text{descending}, \text{stable})$$

$$y2 = \operatorname{argsort}(x, \text{axis}, \text{descending}, \text{stable})$$

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
      <td>FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Denotes the sorted values of the tensor along the specified dimension. It has the same type and format as x.</td>
      <td>FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Denotes the index of each element in the original tensor after sorting.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>Dimension along which the elements are sorted. The default value is -1 (the last dimension).</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>descending</td>
      <td>Attribute</td>
      <td>Whether to sort in descending order. true indicates descending order, and false indicates ascending order. The default value is false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stable</td>
      <td>Attribute</td>
      <td>Whether to perform stable sorting. true indicates stable sorting, and false indicates unstable sorting. The default value is false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y2_dtype</td>
      <td>Attribute</td>
      <td>Data type of y2. The default value is DT_INT32.</td>
      <td>Type</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_sort](./examples/test_aclnn_sort.cpp) | Invokes the Sort operator through the [aclnnSort](./docs/aclnnSort_en.md) API. |
| aclnn invocation | [test_aclnn_argsort](./examples/test_aclnn_argsort.cpp) | Invokes the Sort operator through the [aclnnArgsort](./docs/aclnnArgsort_en.md) API. |
