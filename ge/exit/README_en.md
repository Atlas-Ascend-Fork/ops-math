# Exit

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

Exits the current frame to its parent frame. The Exit operator works with the Enter operator to create loops in a graph. It passes the input tensor x to the parent frame to make the tensor available there. The output tensor y has the same shape and data type as the input tensor x.

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
      <td>x</td>
      <td>Input</td>
      <td>Input tensor, which needs to be passed to the parent frame.</td>
      <td>float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Tensor that has the same shape and data type as the input tensor x.</td>
      <td>float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- The data type and shape of the output tensor y must be consistent with those of the input tensor x.

## Invocation Description

The Exit operator is a control flow operator. It usually works with the Enter operator to create loops in a graph and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_exit](./examples/test_geir_exit.cpp) | Invokes the Exit operator by building a graph through the [operator IR](./op_graph/exit_proto.h) |
