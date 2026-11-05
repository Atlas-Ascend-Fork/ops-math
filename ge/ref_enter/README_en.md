# RefEnter

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

Creates or finds a child frame, and makes the input tensor x, which can be overwritten by references, available in the child frame. The RefEnter operator works with the Exit operator to create loops in a graph, and identifies a frame through the unique frame_name attribute. When is_constant is true, the output tensor y is constant in the child frame; otherwise, it may be changed in the child frame. Unlike the Enter operator, the input of RefEnter can be overwritten by ref-type operators and serves as a variable.

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
      <td>Input tensor. A tensor that can be overwritten by references and needs to be available in the child frame.</td>
      <td>float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>frame_name</td>
      <td>Attribute</td>
      <td>Name of the child frame. A frame is identified by this unique name.</td>
      <td>String, required</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_constant</td>
      <td>Attribute</td>
      <td>If true, the output is constant in the child frame; otherwise, it may be changed in the child frame.</td>
      <td>Bool, required</td>
      <td>-</td>
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

- frame_name and is_constant are required attributes.
- The data type and shape of the output tensor y must be the same as those of the input tensor x.

## Invocation Description

The RefEnter operator is a control flow operator. It usually works with the Exit operator to create loops in a graph, and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_ref_enter](./examples/test_geir_ref_enter.cpp) | Invokes the RefEnter operator by building a graph through the [operator IR](./op_graph/ref_enter_proto.h). |
