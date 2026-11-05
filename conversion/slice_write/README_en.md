# SliceWrite

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

- Operator function: writes the value tensor to the specified position of the x tensor (with the offset specified by begin). This is an in-place operation, and the output shares the same memory as the input x.

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
    <td>Target tensor, which will be modified in place.</td>
    <td>FLOAT16, FLOAT, DOUBLE, INT32, INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>begin</td>
    <td>Input</td>
    <td>Start offset of the write operation. This parameter is a 1D tensor with a maximum of two elements (row_offset, col_offset).</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>value</td>
    <td>Input</td>
    <td>Tensor to be written. Its dimension must be the same as that of x.</td>
    <td>FLOAT16, FLOAT, DOUBLE, INT32, INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x</td>
    <td>Output</td>
    <td>Output result, which shares the same memory as the input x (in-place operation).</td>
    <td>FLOAT16, FLOAT, DOUBLE, INT32, INT64</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraint Description

- The dimension of x must be <= 2
- The dimension of value must be the same as the dimension of x
- begin must be a 1D tensor with two or fewer elements
- The value shape plus the begin offset must be within the x shape
- The input x and the output x must be the same memory

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph-mode API | [test_geir_slice_write](examples/test_geir_slice_write.cpp) | Invokes the SliceWrite operator through the [operator IR](op_graph/slice_write_proto.h) API. |
