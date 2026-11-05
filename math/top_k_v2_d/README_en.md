# TopKV2D

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

- Operator function: finds the k largest or smallest elements in the input tensor and their indices along the specified dimension. Compared with TopK, TopKV2D has an additional assist_seq auxiliary input and is compatible with the TopKV2 operator of TensorFlow.

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
    <td>Input tensor.</td>
    <td>FLOAT16, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>k</td>
    <td>Input</td>
    <td>Number of elements to take.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>assist_seq</td>
    <td>Input</td>
    <td>Auxiliary sequence tensor.</td>
    <td>FLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sorted</td>
    <td>Attribute</td>
    <td>Whether to sort the output results. The default value is true.</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dim</td>
    <td>Attribute</td>
    <td>Dimension along which the operation is performed. The default value is -1 (the last dimension).</td>
    <td>Int</td>
    <td>-</td>
    </tr>
    <tr>
    <td>largest</td>
    <td>Attribute</td>
    <td>Whether to take the maximum values. The default value is true. If false, the minimum values are taken.</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>values</td>
    <td>Output</td>
    <td>Output top-k element values.</td>
    <td>FLOAT16, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>indices</td>
    <td>Output</td>
    <td>Indices of the output top-k elements in the input tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraint Description

- dim must be in the range [-number of dimensions of x, number of dimensions of x).
- k must be greater than or equal to 0 and less than or equal to the size of x on the dim dimension.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph-mode invocation | [test_geir_top_k_v2_d](examples/test_geir_top_k_v2_d.cpp) | Invokes the TopKV2D operator through the [operator IR](op_graph/top_k_v2_d_proto.h) API. |
