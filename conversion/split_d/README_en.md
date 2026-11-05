# SplitD

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

- Operator function: evenly splits a tensor into num_split smaller tensors along the specified dimension split_dim. Unlike the Split operator, split_dim is provided as an attribute instead of an input.

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
    <td>Tensor to be split.</td>
    <td>FLOAT16, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>split_dim</td>
    <td>Attribute</td>
    <td>Dimension along which the tensor is split.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>num_split</td>
    <td>Attribute</td>
    <td>Number of tensors to be split.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y</td>
    <td>Output</td>
    <td>Output result, which is a dynamic output. The number of outputs is determined by num_split.</td>
    <td>FLOAT16, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraint Description

- split_dim must be in the range [0, number of dimensions of value). If it is negative, counting starts from the end
- The size of value on the split_dim dimension must be integer-divisible by num_split
- num_split must be greater than or equal to 1

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph-mode API | [test_geir_split_d](examples/test_geir_split_d.cpp) | Invokes the SplitD operator through the [operator IR](op_graph/split_d_proto.h) API. |
