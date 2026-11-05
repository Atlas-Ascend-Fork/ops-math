# CompareAndBitpack

English | [简体中文](./README.md)

## Product Support Status

| Product                                        | Supported |
| :------------------------------------------ | :------: |
| Ascend 950PR&950DT products                   |    √     |
| Atlas A3 products |    √     |
| Atlas A2 products |    √     |
| Atlas 200I/500 A2 inference products                  |    √     |
| Atlas inference products                          |    √     |
| Atlas training products                          |    √     |

## Function Description

- Operator function: compares each element in the input tensor x with the threshold, and packs the comparison result (1 if greater, otherwise 0) into a bit-field output of the uint8 type. The comparison results of every 8 consecutive input elements are packed into one uint8 value.

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
    <td>Input tensor to be compared, with at least 1 dimension.</td>
    <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>threshold</td>
    <td>Input</td>
    <td>Scalar threshold used for comparison with the elements in x.</td>
    <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>Output</td>
    <td>Packed bit-field output tensor, whose innermost dimension is 1/8 of the innermost dimension of the input.</td>
    <td>UINT8</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraint Description

- threshold must be a scalar
- x must be a tensor with at least one dimension (vector or higher), and scalar inputs are not supported
- The size of the innermost dimension (the last dimension) of x must be divisible by 8
- The data types of x and threshold must be the same

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph-mode API | [test_geir_compare_and_bit_pack](examples/test_geir_compare_and_bit_pack.cpp) | Invokes the CompareAndBitpack operator through the [operator IR](op_graph/compare_and_bit_pack_proto.h) API. |
