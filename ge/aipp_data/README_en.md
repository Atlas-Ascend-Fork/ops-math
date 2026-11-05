# AippData

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

Used in dynamic AIPP (AI Pre-Processing) scenarios. When aipp-mode in the AIPP configuration file is set to dynamic, the framework automatically adds an AippData node at the end of the graph. The AippData operator directly passes the input tensor data to the output tensor out, and identifies the AIPP sequence number through the index attribute.

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
      <td>Input tensor. In dynamic AIPP scenarios, this is usually uint8 data in NCHW or NHWC layout.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Attribute</td>
      <td>AIPP sequence number.</td>
      <td>Int, with the default value of 0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Tensor that has the same shape and data type as the input tensor data.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- The shape and data type of the output tensor out must be the same as those of the input tensor data.

## Invocation Description

The AippData operator is automatically inserted by the framework in dynamic AIPP scenarios, and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_aipp_data](./examples/test_geir_aipp_data.cpp) | Invokes the AippData operator by building a graph through the [operator IR](./op_graph/aipp_data_proto.h). |
