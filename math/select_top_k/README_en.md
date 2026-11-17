# SelectTopK

## Supported Products

| Product | Supported |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | × |
| Atlas A3 training series / Atlas A3 inference series | √ |
| Atlas A2 training series / Atlas A2 inference series | √ |
| Atlas 200I/500 A2 inference products | × |
| Atlas inference series | × |
| Atlas training series | × |

A2 (`ascend910b`) and A3 (`ascend910_93`) share the arch22 Host and Kernel implementation. Resource capacities are obtained from the runtime platform.
Sorted output on the Radix route requires SortWithIndex to sort FP16/BF16 values together with their original-axis indices on A2/A3.

## Function

- Selects the K largest or smallest elements within each row's valid range along the specified dimension. Supports values and indices, values only, or indices only.
- Supports per-row or broadcast ranges, offsets applied to original-axis indices, INT32/INT64 indices, noncontiguous inputs and outputs, and padding when fewer than K elements are valid.
- `order=0` does not guarantee the order of valid results. `order=1` sorts by value, descending for `largest=true` and ascending otherwise. `order=2` is reserved and currently unsupported.

For row r, clamp the range endpoints to `[0,N]`, obtaining b and e. The number of valid output elements is:

$$
valid_r = \min(K, \max(e_r-b_r, 0))
$$

The first `valid_r` entries contain selected values and their original-axis indices. Remaining entries contain fill values.
`indices_offset` is added only to valid indices; `index_fill` is written unchanged. The L2 output shape replaces the selected input dimension with K.

For input `[9,2,7,1,5]`, range `[1,4)`, K=4, largest values sorted first, offset 10, and value/index fills -99/-1,
the outputs are values=`[7,2,1,-99]` and indices=`[12,11,13,-1]`.

## Parameters

The following parameters belong to the operator prototype. The ACLNN interface additionally takes `dim`, output tensors, and workspace parameters; see
[aclnnSelectTopK](docs/aclnnSelectTopK.md). The L0 input view is `[outer,N,inner]`, and the output view is `[outer,K,inner]`.

| Name | Input/Output/Attribute | Description | Data type |
| :--- | :--- | :--- | :--- |
| x | Input | Values to select from, in ND format | FLOAT16, BFLOAT16, FLOAT |
| begin | Optional input | Inclusive original-axis start; one element broadcasts to all rows, otherwise one element per row; defaults to 0 | INT32 |
| end | Optional input | Exclusive original-axis end; same broadcasting rules as begin; defaults to N | INT32 |
| values | Output | Selected values, with the same type as x | FLOAT16, BFLOAT16, FLOAT |
| indices | Output | Original-axis indices or fill values | INT32, INT64 |
| k | Attribute | Output count per row; may exceed the valid range length | Int |
| largest | Attribute | true selects largest values, false selects smallest values; defaults to true | Bool |
| order | Attribute | 0: no ordering guarantee; 1: sort by value; defaults to 1 | Int |
| indices_dtype | Attribute | Output index type; defaults to INT64 | Int |
| indices_offset | Attribute | Added only to valid original-axis indices; defaults to 0 | Int |
| output_mode | Attribute | 0: values only; 1: indices only; 2: both; defaults to 2 | Int |
| value_fill | Attribute | Value padding when fewer than K elements are valid; defaults to 0 | Float |
| index_fill | Attribute | Index padding when fewer than K elements are valid; defaults to -1 | Int |

## Constraints

- ACLNN accepts inputs with 1 to 8 dimensions. The selected axis length and K must not exceed INT32_MAX. Indices, including offsets, must fit the requested index type.
- Index selection among equal values is not guaranteed to be stable. When both values and indices are returned, each pair must correspond to the same input element.
- `K=0` or zero rows produces empty outputs. An empty selected axis with nonempty outputs produces only fill values.
- Full-axis FP16/BF16 requests with both outputs and offset=0 use Topk/RadixTopK when the Radix gate is satisfied.
  For `order=1`, SortWithIndex reorders values and original-axis indices together; `order=0` uses the unordered results directly.
- The Radix route permits NaN normalization to `0x7fff`. Non-NaN values and the SelectTopK main path retain their original bit patterns.
  See the interface documentation for the full gate and companion Radix dependencies.

## Usage

| Method | Interface or example | Description |
| :--- | :--- | :--- |
| ACLNN | [aclnnSelectTopK](docs/aclnnSelectTopK.md#调用示例) | Two-stage API: query the workspace, then execute on the specified stream |
| Executable example | [test_aclnn_select_top_k.cpp](examples/test_aclnn_select_top_k.cpp) | Creates device tensors, runs standard mode, and checks values and indices |
| L0 | [SelectTopK](op_api/select_top_k.h) | Builds tasks for the SelectTopK main path |

Build and CPU unit test commands:

```bash
# Run from the ops-math repository root.
bash build.sh --pkg --soc=ascend910b --ops=select_top_k --vendor_name=select_topk -j4
# Use --soc=ascend910_93 for A3.
bash build.sh -u --ophost --soc=ascend910b --ops=select_top_k --gtest_filter='SelectTopK*' -j4
```

See [tests/README_en.md](tests/README_en.md) for test coverage. CPU tests do not replace device functionality, bit-accuracy, or complete L2 performance validation.
