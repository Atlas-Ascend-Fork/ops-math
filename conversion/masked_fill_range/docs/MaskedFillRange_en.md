# MaskedFillRange

## Product Support

| Product                                                       | Supported |
| :------------------------------------------------------------ | :-------: |
| <term>Ascend 950PR&950DT series products</term>                      |     √     |
| <term>Atlas A3 series products</term>      |     ×     |
| <term>Atlas A2 series products</term>            |     ×     |
| <term>Atlas 200I/500 A2 inference products</term>                   |     ×     |
| <term>Atlas inference series products</term>                        |     ×     |
| <term>Atlas training series products</term>                         |     ×     |

## Function Description

Fills the segments of the input tensor `x` along `axis` with the per-round value from `value`, according to the interval windows given by `start`/`end`.

The first dimension of `start`/`end`/`value` is the number of fill rounds `rounds`. Rounds are applied in order, with later rounds overwriting earlier ones. The second dimension of `start`/`end` is the window batch size `B`, and windows are indexed by row/batch (1-D consumes only window 0). Per-dimension semantics (`axis` normalized to `a ∈ [0, dim)` first):

| x rank | a | Fill behavior |
| ------ | - | ------------- |
| 1-D (D,) | 0 | Each round consumes only window 0 `[start[i,0], end[i,0])`, clamped to `[0, D)`; the hit segment is filled. When `B > 1`, the remaining window columns are ignored. |
| 2-D (N, D) | 0 or 1 | The fill always applies to the last dimension: row `r` takes window `[start[i,r], end[i,r])`, clamped to `[0, D)`; the hit segment is filled. |
| 3-D (N, C, D) | 0 | Window `[start[i,n], end[i,n])` indexed by `n` (direct comparison, no clamping); the hit plane `y[n, :, :]` is filled. |
| 3-D (N, C, D) | 1 | Window `[start[i,n], end[i,n])` indexed by `n` (direct comparison, no clamping); the hit row `y[n, c, :]` is filled. |
| 3-D (N, C, D) | 2 | Window `[start[i,n], end[i,n])` indexed by `n`, clamped to `[0, D)`; the hit segment `y[n, :, k]` is filled. |

## Operator Prototype

```cpp
REG_OP(MaskedFillRange)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32}))
    .INPUT(start, TensorType({DT_INT32}))
    .INPUT(end, TensorType({DT_INT32}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32}))
    .REQUIRED_ATTR(axis, Int)
    .OP_END_FACTORY_REG(MaskedFillRange)
```

## Parameters

- **Parameter Description**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 240px">
  <col style="width: 110px">
  <col style="width: 150px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Format</th>
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x（ge::Tensor）</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>Supports 1-D (D,), 2-D (N, D), and 3-D (N, C, D); the total number of elements must not exceed 1024 * 1024 * 1024. When any dimension is 0 (empty tensor), the operator performs no fill and y is returned with the same shape as x.</td>
      <td>FLOAT16, FLOAT, INT8, INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>start（ge::Tensor）</td>
      <td>Input</td>
      <td>Start position of the fill interval.</td>
      <td>2-D (rounds, B); the first dimension is the number of fill rounds (consistent with end/value), and the second is the window batch size. Windows are indexed by row/batch; see Function Description.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(rounds, B)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>end（ge::Tensor）</td>
      <td>Input</td>
      <td>End position of the fill interval (exclusive).</td>
      <td>Same constraints as start.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(rounds, B)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value（ge::Tensor）</td>
      <td>Input</td>
      <td>Fill value of each round.</td>
      <td>1-D (rounds,); the dtype must be the same as x.</td>
      <td>FLOAT16, FLOAT, INT8, INT32</td>
      <td>ND</td>
      <td>(rounds,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y（ge::Tensor）</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>Same shape and dtype as x; elements outside any hit window keep the value of x (bitwise unchanged).</td>
      <td>FLOAT16, FLOAT, INT8, INT32</td>
      <td>ND</td>
      <td>Same as x</td>
      <td>√</td>
    </tr>
  </tbody></table>

- **Attribute Description**

  <table style="table-layout: fixed; width: 1302px"><colgroup>
  <col style="width: 185px">
  <col style="width: 120px">
  <col style="width: 380px">
  <col style="width: 380px">
  <col style="width: 237px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Required/Optional</th>
      <th>Type</th>
      <th>Value Range</th>
      <th>Constraints</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>axis</td>
      <td>Required</td>
      <td>INT</td>
      <td>[-dim(x), dim(x)); negative values count from the last dimension</td>
      <td>For 2-D x the fill always applies to the last dimension; for 3-D x axes 0/1/2 are supported. Out-of-range values fail.</td>
    </tr>
  </tbody></table>

## Constraints

* The dtype of value must be the same as x; start and end must share the same dtype, which must be INT32.
* x supports only 1-D/2-D/3-D; start/end are 2-D and value is 1-D, all sharing the same first dimension (number of fill rounds).
* When any dimension of x is 0 (empty tensor, total number of elements is 0), the operator is a no-op: y is returned with the same (empty) shape as x without filling; start/end/value are still validated against the contracts above.
* The number of fill rounds must satisfy rounds ≤ 128, and the product of rounds and the effective batch size must not exceed 128; otherwise the operator fails. The effective batch size is min(B, required batches), where B is the second dimension of start and the required batches is 1 for 1-D x, the row count N for 2-D x, and the first dimension N for 3-D x (all C rows of the same n share one batch; see Function Description).
* When the second dimension of start is smaller than the number of rows/batches, the uncovered rows/batches are left unfilled; when larger, the extra window columns are ignored.
* When the fill applies to the last dimension, the window is clamped to `[0, D)`; for 3-D a=0/a=1 the window is compared directly without clamping (negative/out-of-range windows never hit).

## Example

For a complete compilable sample, see [test_geir_masked_fill_range.cpp](../examples/test_geir_masked_fill_range.cpp). Key steps:

1. Build the graph: create the `MaskedFillRange` operator node, connect the inputs and outputs, and set the `axis` attribute.

    ```cpp
    auto maskedFillRange1 = op::MaskedFillRange("maskedFillRange1");
    ADD_INPUT(1, x, DT_FLOAT, xShape, 1);      // x: (4, 8), all 1
    ADD_INPUT(2, start, DT_INT32, startShape, 2);  // start: (2, 4), all 2
    ADD_INPUT(3, end, DT_INT32, endShape, 6);      // end: (2, 4), all 6
    ADD_INPUT(4, value, DT_FLOAT, valueShape, 10); // value: (2,), all 10
    ADD_OUTPUT(1, y, DT_FLOAT, yShape);
    maskedFillRange1.set_attr_axis(-1);
    ```

2. Execute: run the graph with `ge::Session::AddGraph` + `RunGraph`.

3. Fetch the result: read `y` from the output tensor; in this sample the `[2, 6)` segment of each row is filled with 10.

    ```cpp
    ret = session->RunGraph(graph_id, input, output);
    ```
