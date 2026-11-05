# Reshape

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: reinterprets the input tensor to the target shape without changing the underlying data layout.

## Parameter Description

<table style="table-layout: fixed; width: 1400px"><colgroup>
 <col style="width: 140px">
 <col style="width: 180px">
 <col style="width: 360px">
 <col style="width: 500px">
 <col style="width: 120px">
 </colgroup>
 <thead>
  <tr>
   <th>Parameter Name</th>
   <th>Input/Output/Attribute</th>
   <th>Description</th>
   <th>Data Type</th>
   <th>Data Format</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>x</td>
   <td>Input</td>
   <td>Input tensor to be reshaped.</td>
    <td>BOOL, FLOAT16, FLOAT, INT4, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, DOUBLE, COMPLEX64</td>
   <td>ND, FRACTAL_NZ</td>
  </tr>
  <tr>
   <td>shape</td>
   <td>Input</td>
    <td>Target shape tensor. One <code>-1</code> is supported; when allowzero is 0, <code>0</code> indicates copying the corresponding input dimension.</td>
   <td>INT32, INT64</td>
   <td>1D</td>
  </tr>
   <tr>
    <td>axis</td>
    <td>Attribute</td>
    <td>Starting replacement dimension. The default value is 0.</td>
    <td>INT</td>
    <td>-</td>
   </tr>
   <tr>
    <td>num_axes</td>
     <td>Attribute</td>
     <td>Number of consecutive dimensions to be replaced. The default value is -1.</td>
     <td>INT</td>
     <td>-</td>
    </tr>
    <tr>
   <td>y</td>
   <td>Output</td>
   <td>Output tensor after reshaping. Its data type is the same as that of input x.</td>
   <td>Same as x</td>
   <td>ND, FRACTAL_NZ</td>
  </tr>
  </tbody>
 </table>

## Constraint Description

- The total number of output elements must be consistent with the total number of input elements.
- At most one <code>-1</code> can appear in shape.
- When allowzero is 0, <code>0</code> in shape copies the corresponding input dimension; when allowzero is not set or is a non-zero value, <code>0</code> participates in the computation as a literal value.

## Invocation Description

| Invocation Method | Sample Code | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_reshape](./examples/test_geir_reshape.cpp) | Invokes the reshape operator by building a graph through the [operator IR](./op_graph/reshape_proto.h). |
