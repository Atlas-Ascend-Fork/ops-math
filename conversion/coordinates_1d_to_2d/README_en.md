# Coordinates1DTo2D

English | [简体中文](./README.md)

## Product Support Status

| Product                                    | Supported |
| :------------------------------------------ | :------: |
| Ascend 950PR&950DT products                   |    √     |
| Atlas A3 products |    √     |
| Atlas A2 products |    √     |
| Atlas 200I/500 A2 inference products                  |    √     |
| Atlas inference products                          |    √     |
| Atlas training products                          |    √     |

## Function Description

- Operator function: converts 1D coordinates to 2D coordinates and computes the row and column indices based on the shape information.

## Parameter Description

<table style="table-layout: fixed; width: 1005px"><colgroup>
<col style="width: 140px">
<col style="width: 140px">
<col style="width: 180px">
<col style="width: 213px">
<col style="width: 100px">
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
    <td>1D coordinate index value.</td>
    <td>INT32, INT64, UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>shape</td>
    <td>Input</td>
    <td>Shape information, which contains 4 elements (N, D, H, W).</td>
    <td>INT32, INT64, UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>output_row</td>
    <td>Output</td>
    <td>Row index of the output.</td>
    <td>INT32, INT64, UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>output_col</td>
    <td>Output</td>
    <td>Column index of the output.</td>
    <td>INT32, INT64, UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>output_n</td>
    <td>Output</td>
    <td>Number of columns of the output (value of the W dimension).</td>
    <td>INT32, INT64, UINT64</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraint Description

- The number of elements of input[shape] must be 4
- The data type of input[x] must be the same as that of input[shape]
- shape[3] (the W dimension) cannot be 0

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph-mode API | [test_geir_coordinates_1d_to_2d](examples/test_geir_coordinates_1d_to_2d.cpp) | Invokes the Coordinates1DTo2D operator through the [operator IR](op_graph/coordinates_1d_to_2d_proto.h) API. |
