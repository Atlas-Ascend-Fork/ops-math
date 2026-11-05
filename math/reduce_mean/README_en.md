# ReduceMean

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: computes the average value of a multi-dimensional vector along the specified dimensions.
- Examples:
  - Example 1:
    Defines the specified computation dimension (Reduce axis) as the R axis and the non-specified dimension (Normal axis) as the A axis. As shown in the following figures, the computation is performed on a 2D matrix with the shape (2, 3). The average value is computed for the data on the first dimension, and the output is [2.5, 3.5, 4.5]. The average value is computed for the data on the second dimension, and the output is [2, 5].

    Figure 1 Example of ReduceMean computation along the first dimension
    ![alt text](./docs/fig/reduce_mean-by-dim-1.png)

    Figure 2 Example of ReduceMean computation along the last dimension
    ![alt text](./docs/fig/reduce_mean-by-dim-2.png)

## Parameter Description

<table class="tg" style="undefined;table-layout: fixed; width: 1034px">
<colgroup>
<col style="width: 82px">
<col style="width: 123px">
<col style="width: 352px">
<col style="width: 352px">
<col style="width: 125px">
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
    <td>Input tensor</td>
    <td>complex128, complex64, double, float32, float16, int64, int32, int16, int8 <br>
     uint64, uint32, uint16, uint8, bfloat16</td>
    <td>ND</td>
  </tr>
 <tr>
    <td>axes</td>
    <td>Input</td>
    <td>Dimensions to reduce. Supports int, list, tuple, or NoneType. The data type must be int32 or int64.<br>If the value is None, all dimensions are reduced. The value range is [-rank(x), rank(x)].</td>
    <td>int32, int64</td>
    <td>ND</td>
 </tr>
 <tr>
    <td>keep_dims</td>
    <td>Attribute</td>
    <td>Optional Boolean, default false. If true, the reduced dimension with length 1 is retained;<br>if false, the rank of the tensor is reduced by 1 for each entry in axis.</td>
    <td>bool</td>
    <td>ND</td>
 </tr>
 <tr>
    <td>noop_with_empty_axes</td>
    <td>Attribute</td>
    <td>Optional Boolean, default true. If true, no reduction is performed when axes = [];<br> if false, all dimensions are reduced when axes = [].</td>
    <td>bool</td>
    <td>ND</td>
 </tr>
 <tr>
    <td>y</td>
    <td>Output</td>
    <td>Output tensor, with the same type and format as the input tensor x</td>
    <td>Same as the input tensor x</td>
    <td>ND</td>
 </tr>
</tbody>
</table>

## Constraint Description

- The dimension of axes must be greater than 0.
- Dimensions of the input vector: a minimum of 1 dimension and a maximum of 8 dimensions.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation | [test_aclnn_mean_v2](./examples/test_aclnn_mean_v2.cpp) | Invokes the ReduceMean operator through the [aclnnMeanV2](./docs/aclnnMeanV2_en.md) API.    |
| aclnn invocation | [test_aclnn_global_average_pool](./examples/test_aclnn_global_average_pool.cpp) | Invokes the ReduceMean operator through the [aclnnGlobalAveragePool](./docs/aclnnGlobalAveragePool_en.md) API.    |
| Graph-mode invocation | [test_geir_reduce_mean](./examples/test_geir_reduce_mean.cpp)   | Invokes the ReduceMean operator by building a graph through the [operator IR](./op_graph/reduce_mean_proto.h). |
