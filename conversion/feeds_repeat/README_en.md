# FeedsRepeat

English | [简体中文](./README.md)

## Product Support Status

| Product                                           | Supported |
|:---| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    ×     |
| <term>Atlas training products</term>                       |    ×     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function:

  For the input feeds, the data on dimension 0 of the corresponding feeds is copied the corresponding number of times according to the input feeds_repeat_times, and dimension 0 of the output y is padded to the size of output_feeds_size. The unfilled part (the part exceeding the actual number of copied rows) is set to 0.

## Parameter Description

<table class="tg" style="undefined;table-layout: fixed; width: 885px"><colgroup>
<col style="width: 137.333333px">
<col style="width: 104.333333px">
<col style="width: 369.333333px">
<col style="width: 205.333333px">
<col style="width: 68.333333px">
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
    <td>feeds</td>
    <td>Input</td>
    <td>Object to be processed.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>feeds_repeat_times</td>
    <td>Input</td>
    <td>Describes the number of repeat times for each row of feeds.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Tensor after the repeat. Its dtype is the same as that of feeds.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>output_feeds_size</td>
    <td>Attribute</td>
    <td>The value of dimension 0 of the output equals this value.</td>
    <td>INT</td>
    <td>-</td>
  </tr>
</tbody></table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraint Description

- The rank (number of dimensions) of feeds ranges from 1 to 8. Dimension 0 (N) must be the same as the number of elements of feeds_repeat_times, and N≥1 (empty Tensors, that is, N=0, are not supported). The product of the dimensions starting from dimension 1 of feeds (the number of elements in each row) must be ≥1.
- The data types of feeds and feeds_repeat_times are FLOAT/BFLOAT16/FLOAT16 and INT32/INT64 respectively. Other types are not supported.
- The total number of elements of feeds_repeat_times must equal dimension 0 (N) of feeds (its rank is unlimited and is usually one-dimensional). During graph building, its dimension can be a dynamic dimension (-1) or an unknown marker (in this case, the total number of elements cannot be verified, and the verification is postponed to the execution phase). During the execution phase, its specific dimension value must be a non-negative integer.
- Negative values in feeds_repeat_times are processed as 0.
- When the sum of the elements of feeds_repeat_times is greater than output_feeds_size, the repeated rows exceeding output_feeds_size are discarded (the output is truncated by output_feeds_size).
- The attribute output_feeds_size must be a non-negative integer. Negative values are rejected in the operator execution preparation phase (tiling).
- Dynamic shapes are supported: during graph building in GEIR graph mode, -1 (dynamic dimension) can be used for input dimensions, and the specific shape delivered at runtime prevails during execution. When feeds is [-2] (unknown rank), the output y is propagated as unknown rank [-2]. The execution phase requires that the specific dimension values of feeds and feeds_repeat_times be non-negative.
- Description: the -1/-2 markers take effect only in the graph building and shape inference phases. InferShape skips value verification for inputs containing the markers (or propagates them as unknown rank), and in the execution phase tiling completes the verification based on the specific shape delivered at runtime. Execution is performed only when the specific dimension values are non-negative.
- During operator implementation, feeds_repeat_times needs to be moved into the UB and undergo precision conversion and accumulation. Its number of bytes is limited by the on-chip UB capacity: tiling compares 2 times the byte usage plus a reserved amount of about 8 KB with the UB capacity, and rejects the input if the capacity is exceeded. In the current implementation, the number of bytes occupied by feeds_repeat_times cannot exceed about 92 KB on Atlas A2/A3 products and about 120 KB on Ascend 950PR&950DT products.
- Row offsets (the copy start position and zeroing position of each row) are obtained by accumulating feeds_repeat_times in float32. When a single or prefix-accumulated repeat count exceeds 2^24, row offsets may deviate. Avoid using repeat counts that exceed this magnitude.

## Invocation Description

| Invocation Method | Sample Code                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| GE graph-mode invocation | [test_geir_feeds_repeat](./examples/test_geir_feeds_repeat.cpp) | Invokes the feeds_repeat operator through the [feeds_repeat_proto](op_graph/feeds_repeat_proto.h) API. |
