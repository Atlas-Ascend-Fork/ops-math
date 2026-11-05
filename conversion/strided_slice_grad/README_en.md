# StridedSliceGrad

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                      |    √    |
| <term>Atlas A3 products</term>      |    √    |
| <term>Atlas A2 products</term>      |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                              |    √    |
| <term>Atlas training products</term>                              |    √    |

## Function Description

- Operator function: maps the gradient of a sub-tensor back to the corresponding position in the original tensor.

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
      <td>shape</td>
      <td>Input</td>
      <td>Input tensor</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>begin</td>
      <td>Input</td>
      <td>Start position</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>end</td>
      <td>Input</td>
      <td>End position</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Input</td>
      <td>Step</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>Gradient</td>
      <td>FLOAT16, BFLOAT16, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>begin_mask</td>
      <td>Input attribute</td>
      <td>Input format</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>end_mask</td>
      <td>Input attribute</td>
      <td>Input format</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ellipsis_mask</td>
      <td>Input attribute</td>
      <td>Input format</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>new_axis_mask</td>
      <td>Input attribute</td>
      <td>Input format</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shrink_axis_mask</td>
      <td>Input attribute</td>
      <td>Input format</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Padded gradient tensor</td>
      <td>FLOAT16, BFLOAT16, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_strided_slice_grad](./examples/test_geir_strided_slice_grad.cpp)   | Invokes the StridedSliceGrad operator by building a graph through the [operator IR](./op_graph/strided_slice_grad_proto.h). |
| aclnn API | [test_aclnn_strided_slice_grad](./examples/arch35/test_aclnn_strided_slice_grad.cpp) | Invokes the StridedSliceGrad operator through the [aclnnStridedSliceGrad](./docs/aclnnStridedSliceGrad_en.md) API. |
