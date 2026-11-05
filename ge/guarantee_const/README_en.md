# GuaranteeConst

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

Guarantees to the runtime that the input tensor is a constant. This operator is a pass-through operator: the output tensor y is the input tensor x, without changing any shape, data type, or value. It only passes the semantic information that "the input is a constant" in the graph for graph optimization decisions. The shape and data type of the output tensor y are consistent with those of the input tensor x.

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
      <td>Tensor that needs to be guaranteed as a constant.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, which is exactly the same as the input x (shape, data type, and value).</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- This operator is an experimental feature (EXPERIMENTAL) and must not be used in production environments.
- This operator is a TensorFlow-compatible pass-through operator. It is eliminated in the graph optimization phase (the output is directly connected to the input) and does not generate actual computation tasks at runtime.

## Invocation Description

The GuaranteeConst operator is a basic operator of the graph engine. It is used to mark constant semantics in a graph and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_guarantee_const](./examples/test_geir_guarantee_const.cpp) | Invokes the GuaranteeConst operator by building a graph through the [operator IR](./op_graph/guarantee_const_proto.h) |
