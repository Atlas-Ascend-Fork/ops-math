# SwitchN

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

SwitchN is a basic control flow operator. It forwards the input tensor `data` to the corresponding output port based on the integer tensor input `pred_value`. `output` is a dynamic output, in which only one output receives `data`, and its shape and data type are the same as those of the input `data`. This operator usually works with the [Merge](../merge/README_en.md) operator to implement multi-way branch control.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 996px"><colgroup>
  <col style="width: 102px">
  <col style="width: 168px">
  <col style="width: 203px">
  <col style="width: 403px">
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
      <td>data</td>
      <td>Input</td>
      <td>The tensor to be forwarded.</td>
      <td>float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pred_value</td>
      <td>Input</td>
      <td>An int64 tensor that determines the output port to which the data is forwarded.</td>
      <td>DT_INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Dynamic output</td>
      <td>Output tensors, one of which receives data. The data type is the same as that of data.</td>
      <td>float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- The data type of the input `data` must be one of `float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool`.
- The data type of the input `pred_value` must be `DT_INT64`.

## Invocation Description

The SwitchN operator is a basic control flow operator. It is usually used as a multi-way branch selection node in a graph, and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_switch_n](./examples/test_geir_switch_n.cpp) | Invokes the SwitchN operator by building a graph through the [operator IR](./op_graph/switch_n_proto.h). |
