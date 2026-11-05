# DynamicPartition

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

API function: dynamically partitions the input data into multiple tensors based on partition indices. Example:
Suppose the input Tensor is [1, 6, 3, 8, 2, 9]. The one-dimensional tensor is divided into two groups by condition (for example, element values ≤ 5 and > 5). The values are as follows:

```text
>>> x = tf.constant([1, 6, 3, 8, 2, 9])
>>> partitions = tf.cast(data ≤ 5, tf.int32)
>>> tf.dynamic_partition(x, partitions, num_partitions=2)
[array([1, 3, 2]), array([6, 8, 9])]
```

Partition operations on multi-dimensional tensors are supported, for example, splitting a matrix by row or column:

```text
>>> x = tf.reshape(tf.range(12), [3, 4])
// Partition by row
>>> tf.dynamic_partition(matrix, [0, 1, 0], num_partitions=2)
array([[0, 1, 2, 3],
       [8, 9, 10, 11]])
```

## Parameter Description

  <table style="table-layout: fixed; width: 1528px"><colgroup>
  <col style="width: 132px">
  <col style="width: 120px">
  <col style="width: 256px">
  <col style="width: 253px">
  <col style="width: 333px">
  <col style="width: 126px">
  <col style="width: 160px">
  <col style="width: 145px">
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
      <td>Original input tensor to be partitioned</td>
      <td>FLOAT, DOUBLE, FLOAT16, COMPLEX64, UINT8, INT8, INT16, INT32, INT64, UINT16, UINT32, UINT64, BOOL, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>partitions</td>
      <td>Input</td>
      <td>Partition index tensor, which determines the partition number that each element belongs to</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_partitions</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the total number of partitions</li><li>The default value is 1.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor after partitioning</td>
      <td>FLOAT, DOUBLE, FLOAT16, COMPLEX64, UINT8, INT8, INT16, INT32, INT64, UINT16, UINT32, UINT64, BOOL, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Example

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_dynamic_partition](./examples/test_geir_dynamic_partition.cpp) | Invokes the DynamicPartition operator by building a graph through the [operator IR](./op_graph/dynamic_partition_proto.h). |
