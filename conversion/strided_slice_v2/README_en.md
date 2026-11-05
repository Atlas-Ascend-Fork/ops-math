# StridedSliceV2

English | [简体中文](./README.md)

|Product             |  Supported  |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function Description

- Operator function: extracts a sub-tensor from the input tensor according to the specified start and end positions, axes, steps, and binary bit masks.

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
    <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BF16, BOOL, COMPLEX32, COMPLEX64, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>begin</td>
    <td>Input</td>
    <td>Start value of each dimension</td>
    <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BF16, BOOL, COMPLEX32, COMPLEX64, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>end</td>
    <td>Input</td>
    <td>End value of each dimension</td>
    <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BF16, BOOL, COMPLEX32, COMPLEX64, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>axes</td>
    <td>Input</td>
    <td>Specifies the axes to be sliced, and the remaining axes are fully taken by default.</td>
    <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BF16, BOOL, COMPLEX32, COMPLEX64, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>strides</td>
    <td>Input</td>
    <td>Interval at which values are taken at each point in each dimension</td>
    <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BF16, BOOL, COMPLEX32, COMPLEX64, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>begin_mask</td>
    <td>Attribute</td>
    <td>This value specifies the dimensions whose begin values are ignored</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>end_mask</td>
    <td>Attribute</td>
    <td>This value specifies the dimensions whose end values are ignored</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>ellipsis_mask</td>
    <td>Attribute</td>
    <td>All dimensions starting from the specified dimension are selected until the user-specified begin is encountered</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>new_axis_mask</td>
    <td>Attribute</td>
    <td>Adds a shape with dimension 1 at the specified bit</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>shrink_axis_mask</td>
    <td>Attribute</td>
    <td>Forces the dimension at the corresponding index to be reduced to 1</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>Output</td>
    <td>Output tensor</td>
    <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BF16, BOOL, COMPLEX32, COMPLEX64, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                                  |
|--------------|------------------------------------------------------------------------|---------------------------------------------------------------------|
| Graph-mode invocation | [test_geir_strided_slice_v2](./examples/test_geir_strided_slice_v2.cpp) | Invokes the StridedSliceV2 operator by building a graph through the [operator IR](./op_graph/strided_slice_v2_proto.h) |
