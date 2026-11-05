# Split

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

- Operator function: evenly splits a tensor into num_split smaller tensors along the specified dimension split_dim.

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
    <td>FLOAT16, FLOAT32, INT8, INT16, INT64, INT32, UINT8, UINT16, UINT32, UINT64, BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>split_dim</td>
    <td>Input</td>
    <td>Dimension along which the tensor is split.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
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
    <td>Output result.</td>
    <td>FLOAT16, FLOAT32, INT8, INT16, INT64, INT32, UINT8, UINT16, UINT32, UINT64, BOOL</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph-mode API | [test_geir_split](examples/test_geir_split.cpp) | Invokes the Split operator through the [operator IR](op_graph/split_proto.h) API. |
