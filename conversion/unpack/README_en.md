# Unpack

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                      |    √    |
| <term>Atlas A3 products</term>      |    √    |
| <term>Atlas A2 products</term>      |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √    |
| <term>Atlas inference products</term>                              |    √    |
| <term>Atlas training products</term>                              |    √    |

## Function Description

- Operator function: splits a tensor along a dimension into multiple sub-tensors.

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
      <td>Input tensor</td>
      <td>INT64, INT32, UINT32, FLOAT, FLOAT16, INT8, UINT8, BFLOAT16, INT16, UINT16, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num</td>
      <td>Input attribute</td>
      <td>Number of splits</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Input attribute</td>
      <td>Dimension along which the tensor is split</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Size of the input tensor in the specified dimension</td>
      <td>INT64, INT32, UINT32, FLOAT, FLOAT16, INT8, UINT8, BFLOAT16, INT16, UINT16, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_unpack](./examples/test_geir_unpack.cpp)   | Invokes the Unpack operator by building a graph through the [operator IR](./op_graph/unpack_proto.h). |
