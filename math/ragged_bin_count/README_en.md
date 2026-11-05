# RaggedBinCount

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: For the ragged rows partitioned by `splits`, RaggedBinCount counts the number of occurrences or the weight sum of each bin. Let
  `B = numel(splits) - 1` and `M = size[0]`. Row `b` corresponds to
  `values[splits[b]:splits[b + 1]]`, and the output shape is `[B, M]`.

  For elements satisfying `0 <= values[i] < M`: when `binary_output=false` and `weights` is empty, `1.0` is accumulated at the corresponding position;
  when `binary_output=false` and `weights` is not empty, `weights[i]` is accumulated; when `binary_output=true`, a bin that has appeared
  outputs `1.0` and the weight values are ignored. Bins greater than or equal to `M` are ignored, and a negative bin is invalid input.

  Usage scenario: performs row-wise histogram statistics on a batch of variable-length sequences (ragged batch), for example, counting vocabulary hits or class weight sums by sample.

- Computation formula:

  Non-binary (`binary_output = false`):

$$
output_{b,m}
= \sum_{i=splits_b}^{splits_{b+1}-1} \mathbb{1}[values_i = m] \cdot w_i
$$

  where $w_i = weights_{flat}[i]$ (the number of elements of `weights` is not 0) or $w_i = 1.0$ (the number of elements of `weights` is 0).

  Binary (`binary_output = true`), ignoring `weights`:

$$
output_{b,m}
= \mathbb{1}\left[\exists\, i \in [splits_b,\ splits_{b+1}),\ values_i = m\right]
$$

  where $b \in [0, B)$ and $m \in [0, M)$, and $\mathbb{1}[\cdot]$ is the indicator function (1 if the condition holds, otherwise 0).

## Parameter Description

<table style="table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 150px">
  <col style="width: 160px">
  <col style="width: 650px">
  <col style="width: 180px">
  <col style="width: 160px">
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
      <td>splits</td>
      <td>Input</td>
      <td>Strictly 1D row-splits tensor, with at least 2 elements.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>Input</td>
      <td>0D, 1D, or 2D bin index tensor, flattened in the order of consecutive elements.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input</td>
      <td>1D with 1 element (that is, the shape is <code>[1]</code>). <code>size[0]</code> is the non-negative number of bins, and the dtype is the same as that of values.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>Input</td>
      <td>0D, 1D, or 2D weight tensor, which must be passed. When the number of elements is 0 (empty tensor), weights are not used, which is equivalent to all 1s. Otherwise, the number of elements must be the same as that of values, corresponding element-wise in the flattened order of values.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>binary_output</td>
      <td>Attribute</td>
      <td>Specifies whether to output the binary result. The default value is false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Strictly 2D, with the shape <code>[numel(splits)-1, size[0]]</code>.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- For all supported products, both static shapes and dynamic shapes use the ND format. The supported data type combinations and shape constraints are the same, and `values` supports 0 to 2 dimensions.

- During graph compilation, `-1` indicates an unknown dimension and `[-2]` indicates an unknown rank. The output rank is fixed to 2, and an output axis that cannot be determined by the length of `splits` or the constant `size` is recorded as `-1`. During execution, each input must be resolved to specific non-negative dimensions and satisfy the rank and element count constraints of each parameter.

- The data types of all inputs must match one of the rows in the following table as a complete set, and cross-row combinations are not allowed (the dtype of `size` must be the same as that of `values`):

| splits | values | size | weights | output | Data Format |
| --- | --- | --- | --- | --- | --- |
| INT64 | INT32 | INT32 | FLOAT | FLOAT | ND |
| INT64 | INT64 | INT64 | FLOAT | FLOAT | ND |

- `splits[0]` must be 0, `splits` must be monotonically non-decreasing, and `splits[-1]` must equal the number of elements of values.
- An out-of-range `splits` or a failure to satisfy the preceding data value constraints, as well as negative values in `values`, are invalid inputs. Do not rely on the output content of invalid inputs.
- The number of dimensions of `weights` itself must not exceed 2. Within this range, only the **number of elements** is validated, and the shape or number of dimensions is not required to be the same as that of `values`: a value of 0 elements means weights are not used, and a non-zero value must be the same as the number of elements of `values` (corresponding in the flattened order).
- Empty tensor support is described separately as follows:
  - When `size[0]` is 0: the output is an empty tensor `[numel(splits)-1, 0]`. No valid computation is delivered, and success is returned directly.
  - When the number of elements of `values` is 0: all elements of `splits` must be 0, and all positions of the output are `+0.0`.
  - When the number of elements of `weights` is 0 (for example, the shape is `[0]`, `[0,3]`, `[2,0]`, or `[0,0]`): it indicates no weights, which is equivalent to all 1s. All cases are valid.
  - Empty tensors are **not supported** for `splits`. The number of elements must be at least 2. Otherwise, validation fails and an error is returned.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE graph mode</td>
    <td><a href="./examples/test_geir_ragged_bin_count.cpp">test_geir_ragged_bin_count</a></td>
    <td>Invokes the RaggedBinCount operator by building a graph through the <a href="./op_graph/ragged_bin_count_proto.h">operator IR</a>.</td>
  </tr>
</tbody></table>

This operator does not provide an aclnn API with the same name.
