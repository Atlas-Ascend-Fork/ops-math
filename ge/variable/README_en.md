# Variable

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

Creates a variable tensor. The Variable operator assigns a value through the input x, and records the value of the variable tensor internally. The output y is the created variable tensor. The InferShape function returns success but does not explicitly set the output description, which is inferred by the framework by default (the output follows the input).

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
      <td>The input tensor used to assign a value to the variable tensor. The caller does not need to pass the value of the variable tensor.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Attribute</td>
      <td>The index of the input tensor.</td>
      <td>Int, with the default value of 0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Attribute</td>
      <td>Used to pass and record the value of the variable tensor.</td>
      <td>Tensor, with the default value of Tensor()</td>
      <td>-</td>
    </tr>
    <tr>
      <td>container</td>
      <td>Attribute</td>
      <td>The container of the variable tensor.</td>
      <td>String, with the default value of ""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shared_name</td>
      <td>Attribute</td>
      <td>The shared name of the variable tensor.</td>
      <td>String, with the default value of ""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The created variable tensor.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

None

## Invocation Description

The Variable operator is a basic graph engine operator. It is usually used as a variable node in a graph, and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_variable](./examples/test_geir_variable.cpp) | Invokes the Variable operator by building a graph through the [operator IR](./op_graph/variable_proto.h). |
