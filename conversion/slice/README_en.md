# Slice

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √      |
| <term>Atlas A2 products</term>    |    √      |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

## Function Description

- Operator function: extracts a slice from a tensor. This operation extracts a slice of "size" from the tensor "x", where the start position is specified by "offsets".

$$
y = x[offsets:offsets + size]
$$

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 310px">
<col style="width: 212px">
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
    <td>Input tensor</td>
    <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BF16, BOOL, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>offsets</td>
    <td>Input</td>
    <td>Start position of the slice</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>size</td>
    <td>Input</td>
    <td>Size of the slice</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>Output</td>
    <td>Output tensor</td>
    <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BF16, BOOL, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraint Description

- For each dimension n of the input x, the following condition must be satisfied: 0 <= offsets[i] <= offsets[i] + size[i] <= x_dim[i] for i in [0,n].

## Invocation Description

| Invocation Method | Sample Code                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_slice](examples/test_aclnn_slice.cpp) | Invokes the Slice operator through the [aclnnSlice](docs/aclnnSlice_en.md) API    |
