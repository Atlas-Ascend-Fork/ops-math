# CoalesceSparse

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    ×     |
| <term>Atlas training products</term>                       |    ×     |

## Function Description

- Operator function: accumulates and sums the values of the same coordinate points, thereby reducing the memory size of the Coo_Tensor.

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
      <td>unique_len</td>
      <td>Input</td>
      <td>Number of indexes after deduplication.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>unique_indices</td>
      <td>Input</td>
      <td>Index array after deduplication.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Index array.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>Input</td>
      <td>Element value corresponding to each coordinate.</td>
      <td>INT32, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>new_indices</td>
      <td>Output</td>
      <td>Merged index array.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>new_values</td>
      <td>Output</td>
      <td>Merged element values.</td>
      <td>INT32, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The operator can be called only from the PyTorch side. To use the aclnn API, you need to manually construct the unique_indices input parameter and ensure that the new_values tensor has been set to all zeros before the kernel is invoked.

- The indices value after re-indexing cannot exceed the upper limit of int32.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn invocation | [test_aclnn_coalesce_sparse](./examples/test_aclnn_coalesce_sparse.cpp)   | Invokes the CoalesceSparse operator through the aclnnCoalesceSparse API. |
