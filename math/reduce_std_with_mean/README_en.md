# ReduceStdWithMean

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

> For Ascend 950PR/950DT, the kernel implementation is provided by this custom operator package (arch35). The other products are supported by the operator library built in CANN. Ascend 950PR/950DT, Atlas A3, and Atlas A2 support float16 / float32 / bfloat16. The other products support float16 / float32.

## Function Description

Given the externally pre-computed mean `mean`, computes the standard deviation (`invert=false`) of the input `x` along the specified dimension `dim` or the reciprocal of the standard deviation (`invert=true`). It is a key sub-step for computing the statistics of the BatchNorm/LayerNorm/GroupNorm normalization layers, and is the underlying implementation of PyTorch `torch.std`/`torch.std_mean`/`torch.batch_norm_stats` on the NPU.

Standard deviation mode (`invert=false`, with `correction` taking effect):

$$
y = \sqrt{\frac{\sum_{j=0}^{N-1}(x_j - mean_j)^2}{\max(0,\; N - correction)}}
$$

Reciprocal-of-standard-deviation mode (`invert=true`, always dividing by N and adding epsilon, ignoring correction):

$$
y = \frac{1}{\sqrt{\dfrac{1}{N}\sum_{j=0}^{N-1}(x_j - mean_j)^2 + \epsilon}}
$$

Here, $N=\prod_{d \in dim}\text{shape}[d]$ is the total number of elements in the reduction dimensions, $correction$ is the Bessel correction factor (0=population, 1=sample), and $\epsilon$ is a small value for numerical stability.

ReduceStdWithMean is an internal L0 operator of CANN. It is called in combination by the three upper-layer aclnn APIs `aclnnStd`, `aclnnStdMeanCorrection`, and `aclnnBatchNormStats`, and does not directly expose an independent aclnn API.

| Upper-layer API | Function | correction | invert | eps |
|:---|:---|:---|:---|:---|
| aclnnStd | Standalone standard deviation | Passed in by the user | false | — |
| aclnnStdMeanCorrection | Outputs (std, mean) at the same time | Passed in by the user | false | 0.001 (fixed) |
| aclnnBatchNormStats | Outputs (mean, invstd) | 0 (fixed) | true | Passed in by the user |

## Parameter Description

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
<td>Tensor for computing the standard deviation.</td>
<td>FLOAT16, FLOAT, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>mean</td>
<td>Input</td>
<td>Externally pre-computed mean (already broadcast to the shape of x). Its dtype is the same as that of x.</td>
<td>Same as x</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>Output</td>
<td>Output tensor. Its dtype is the same as that of x, and its shape is derived from dim + keepdim.</td>
<td>Same as x</td>
<td>ND</td>
</tr>
<tr>
<td>dim</td>
<td>Optional attribute</td>
<td>Reduction dimensions. The value range is [-rank(x), rank(x)-1]. Empty indicates all dimensions. The default value is {}.</td>
<td>ListInt</td>
<td>-</td>
</tr>
<tr>
<td>unbiased</td>
<td>Optional attribute</td>
<td>Bessel correction (legacy field). The default value is true.</td>
<td>Bool</td>
<td>-</td>
</tr>
<tr>
<td>keepdim</td>
<td>Optional attribute</td>
<td>Specifies whether to retain the reduced dimension as size 1. The default value is false.</td>
<td>Bool</td>
<td>-</td>
</tr>
<tr>
<td>invert</td>
<td>Optional attribute</td>
<td>If true, 1/sqrt(var+eps) is returned. If false, sqrt(var) is returned. The default value is false.</td>
<td>Bool</td>
<td>-</td>
</tr>
<tr>
<td>epsilon</td>
<td>Optional attribute</td>
<td>Term for numerical stability (used only when invert=true). It must be a non-negative floating-point number. The default value is 0.001.</td>
<td>Float</td>
<td>-</td>
</tr>
<tr>
<td>correction</td>
<td>Optional attribute</td>
<td>Correction factor: 0=population, 1=sample. The default value is 1.</td>
<td>Int</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraint Description

- The data types of x and mean must be the same, and only float16 / float32 / bfloat16 are supported. The output dtype is the same as that of x.
- The number of shape dimensions of x does not exceed 8.
- rank=0 scalar inputs are supported (x/mean is 0-D). If dim is empty, all dimensions are reduced and the output is a scalar (when keepdim=true, the output is 1-D with size 1).
- The shape of mean must be exactly the same as that of x (mean must have been broadcast to the shape of x).
- The value range of dim is [-rank(x), rank(x)-1]. Empty indicates reduction over all dimensions. dim cannot contain duplicate axes (duplicates are checked after negative axes are normalized).
- The shape of the output y must be the same as the result derived from dim + keepdim.
- correction supports only non-negative integers. When invert=true, correction is ignored and the division by N is always performed.
- epsilon supports only non-negative floating-point numbers (the default value is 0.001). An error is returned for epsilon<0 in the tiling phase.
- unbiased is a legacy field. If the correction attribute has a value, correction prevails. If it has no value, the value is converted as follows: unbiased=true→correction=1 and unbiased=false→correction=0.
- An empty tensor (N=0) outputs NaN. When N<=correction (invert=false), +Inf or NaN is output (+Inf for general inputs, and 0×Inf=NaN for inputs whose elements are all equal).
- When the input contains dynamic markers (-1/-2), the input is processed as a dynamic shape, and the specific semantics are defined by the downstream framework.

## Invocation Description

| Invocation Method | Sample Code | Description |
|---|---|---|
| GE graph-mode invocation | [test_geir_reduce_std_with_mean.cpp](examples/test_geir_reduce_std_with_mean.cpp) | Invokes the operator by building a graph through GE IR. For the operator IR prototype, see [op_graph/reduce_std_with_mean_proto.h](op_graph/reduce_std_with_mean_proto.h) |
| aclnn API | [test_aclnn_std_mean_correction.cpp](examples/test_aclnn_std_mean_correction.cpp) | Invokes the ReduceStdWithMean operator through the [aclnnStdMeanCorrection](docs/aclnnStdMeanCorrection_en.md) API. |
| aclnn API | [test_aclnn_batch_norm_stats.cpp](examples/test_aclnn_batch_norm_stats.cpp) | Invokes the ReduceStdWithMean operator through the [aclnnBatchNormStats](docs/aclnnBatchNormStats_en.md) API. |
| aclnn API | [test_aclnn_std.cpp](../reduce_std_v2/examples/test_aclnn_std.cpp) | Invokes the ReduceStdWithMean operator through the [aclnnStd](../reduce_std_v2/docs/aclnnStd_en.md) API (the implementation of this API is in `math/reduce_std_v2` in the same repository). |
