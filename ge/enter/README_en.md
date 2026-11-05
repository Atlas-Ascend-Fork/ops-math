# Enter

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

Creates or finds a child frame and makes the input tensor x available in the child frame. The Enter operator works with the Exit operator to create loops in a graph. Frames are identified by the unique frame_name attribute. When is_constant is true, the output tensor y is constant within the child frame; otherwise, it may be changed within the child frame. The output tensor y has the same shape and data type as the input tensor x.

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
      <td>Input tensor, which needs to be available in the child frame.</td>
      <td>float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>frame_name</td>
      <td>Attribute</td>
      <td>Name of the child frame. Frames are identified by this unique name.</td>
      <td>String, required</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_constant</td>
      <td>Attribute</td>
      <td>If true, the output is constant within the child frame; otherwise, it may be changed within the child frame.</td>
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
- The data type and shape of the output tensor y must be consistent with those of the input tensor x.

## Invocation Description

The Enter operator is a control flow operator. It usually works with the Exit operator to create loops in a graph and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_enter](./examples/test_geir_enter.cpp) | Invokes the Enter operator by building a graph through the [operator IR](./op_graph/enter_proto.h) |
