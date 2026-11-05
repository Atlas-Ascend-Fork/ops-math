# SplitV

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √      |
| <term>Atlas A2 products</term>    |    √      |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

## Function Description

- Operator function: splits a tensor into num_split smaller tensors along the dimension split_dim according to size_splits.

## Parameter Description

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 180px">
  <col style="width: 213px">
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
      <td>x</td>
      <td>Input</td>
      <td>List of tensors to be split.</td>
      <td>FLOAT16, FLOAT32, DOUBLE, INT64, INT32, UINT8, UINT16, UINT32, UINT64, INT8, INT16, BOOL, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size_splits</td>
      <td>Input</td>
      <td>Specifies a list that contains the size of each output tensor along the split dimension.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>split_dim</td>
      <td>Input</td>
      <td>Dimension along which the tensor is split.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_split</td>
      <td>Attribute</td>
      <td>Number of tensors to be split.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output result.</td>
      <td>FLOAT16, FLOAT32, DOUBLE, INT64, INT32, UINT8, UINT16, UINT32, UINT64, INT8, INT16, BOOL, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

* Atlas training products, Atlas inference products, Atlas 200I/500 A2 inference products, and Atlas 200/300/500 inference products: BFLOAT16 is not supported.

## Constraint Description

* Each element in size_splits is greater than or equal to 1.
* The length of size_splits equals the value of num_split.
* The sum of the elements in size_splits equals the size of the dimension split_dim.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_split_tensor](examples/test_aclnn_split_tensor.cpp) | Invokes the SplitV operator through the [aclnnSplitTensor](docs/aclnnSplitTensor_en.md) API. |
| aclnn API | [test_aclnn_split_with_size](examples/test_aclnn_split_with_size.cpp) | Invokes the SplitV operator through the [aclnnSplitWithSize](docs/aclnnSplitWithSize_en.md) API. |
