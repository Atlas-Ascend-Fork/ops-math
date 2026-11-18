# SelectTopK

## Product Support Status

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | × |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

A2 (`ascend910b`) and A3 (`ascend910_93`) share the arch22 Host and Kernel implementation. Resource capacities are obtained from the runtime platform.
Sorted output on the Radix route requires SortWithIndex to sort FP16/BF16 values together with their original-axis indices on A2/A3.

## Function Description

SelectTopK selects the `k` largest or smallest elements within each row's valid range along the specified dimension. It can return values, indices, or both.

Let `N` be the length of the specified dimension. Range endpoints are clamped to `[0, N]`. For row $r$ with clamped range $[b_r, e_r)$, the number of valid results is

$$
valid_r = \min(k, \max(e_r - b_r, 0))
$$

The first $valid_r$ positions in each row contain the selected values and indices. The remaining positions contain fill values. Each valid index is the element's original index along the specified dimension plus `indices_offset`. The operator uses `index_fill` directly for padding indices. The ACLNN output shape is the input shape with the specified dimension's length replaced by `k`.

For example, select the largest elements from `[9, 2, 7, 1, 5]` within `[1, 4)`. Set `k=4` and sort by value in descending order. Use an index offset of 10. Pad values with `-99` and indices with `-1`. The output values are `[7, 2, 1, -99]`, and the output indices are `[12, 11, 13, -1]`.

## Parameter Description

The following table lists the operator prototype parameters. The ACLNN API also takes `dim`, output tensors, and workspace parameters. See
[aclnnSelectTopK](docs/aclnnSelectTopK.md) for details. The L0 input view is `[outer,N,inner]`, and the output view is `[outer,K,inner]`.

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 130px">
  <col style="width: 145px">
  <col style="width: 410px">
  <col style="width: 195px">
  <col style="width: 100px">
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
      <td>Input tensor to select from.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>begin</td>
      <td>Optional input</td>
      <td>Inclusive start of each row's range. Supply one endpoint per row or a single element to broadcast. The default is 0.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>end</td>
      <td>Optional input</td>
      <td>Exclusive end of each row's range. The broadcasting rules match those for begin. The default is N.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>Output</td>
      <td>Selected values or fill values, with the same data type as x.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Output</td>
      <td>Original-axis indices of selected elements plus the offset, or padding indices.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>Attribute</td>
      <td>Number of output elements per row. This number may exceed the valid range length.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>largest</td>
      <td>Attribute</td>
      <td>true selects the largest values and false selects the smallest. The default is true.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>order</td>
      <td>Attribute</td>
      <td>0 leaves the order of valid results unspecified. 1 sorts by value in descending order when largest is true and in ascending order otherwise. The default is 1. The value 2 is reserved.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>indices_dtype</td>
      <td>Attribute</td>
      <td>Data type of the output indices. Supports DT_INT32 and DT_INT64. The default is DT_INT64.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>indices_offset</td>
      <td>Attribute</td>
      <td>Offset added to valid original-axis indices. The default is 0.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_mode</td>
      <td>Attribute</td>
      <td>0 returns values only, 1 returns indices only, and 2 returns both. The default is 2.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value_fill</td>
      <td>Attribute</td>
      <td>Value used for padding when fewer than k elements are valid. The default is 0.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>index_fill</td>
      <td>Attribute</td>
      <td>Index used for padding when fewer than k elements are valid. The default is -1.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- The ACLNN API accepts inputs with 1 to 8 dimensions and supports noncontiguous input and output tensors. The length of the specified dimension and `k` must be in `[0, INT32_MAX]`.
- `indices_offset`, `index_fill`, and all original indices along the specified dimension after applying the offset must fit in the requested index data type.
- For equal values, the API gives no guarantee of stable index selection or ordering. When both outputs are requested, the values and indices correspond element by element.
- The outputs are empty if `k=0` or the row count is 0. If the specified dimension has length 0 and the outputs are nonempty, all output elements use the fill values.
- L2 calls Topk/RadixTopK for full-axis FP16/BF16 requests with both outputs and offset=0 when the Radix dispatch conditions hold.
  For `order=1`, SortWithIndex reorders values and original-axis indices together. The `order=0` path uses the unordered results directly.
- The Radix route may normalize NaN to `0x7fff`. Non-NaN values on that route and all values on the SelectTopK main path retain their original bit patterns.
  See the API documentation for the full dispatch conditions and the Radix dependencies.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| aclnn API | [test_aclnn_select_top_k](./examples/test_aclnn_select_top_k.cpp) | Invokes SelectTopK through the [aclnnSelectTopK API](./docs/aclnnSelectTopK.md). |
| L0 | [SelectTopK](op_api/select_top_k.h) | Builds computation tasks for the SelectTopK main path. |

Use these commands to build the package and run the CPU unit tests.

```bash
# Run from the ops-math repository root.
bash build.sh --pkg --soc=ascend910b --ops=select_top_k --vendor_name=select_topk -j4
# Use --soc=ascend910_93 for A3.
bash build.sh -u --ophost --soc=ascend910b --ops=select_top_k --gtest_filter='SelectTopK*' -j4
```

See [tests/README_en.md](tests/README_en.md) for test coverage. CPU tests do not replace validation of device functionality, bit accuracy, and full L2 performance.
