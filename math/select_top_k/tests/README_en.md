# SelectTopK Unit Tests

Uses the repository's GTest, `InfershapeContextPara`, and `TilingContextPara` test framework, following the Host UT conventions of operators such as TopKV2 and Segsum.

`ut/op_api/test_aclnn_select_top_k_radix_policy.cpp` covers the minimum K, both N/K ratio boundaries, and the axis-length limit. It links directly to the gate implementation in the L2 API.

- `ut/op_host/test_select_top_k_infershape.cpp`: six shape-inference cases covering three-dimensional views, dynamic dimensions, empty outputs, padding when K exceeds the axis length, and invalid parameters.
- `ut/op_host/arch22/test_select_top_k_tiling.cpp`: six A2 tiling cases covering Top1, CopyAll, fused layouts, ranges/padding, order encoding, and INT32 offset boundaries; includes FP16, BF16, and FP32.

Run from the ops-math repository root:

```bash
source /path/to/cann/set_env.sh
bash build.sh -u --ophost --soc=ascend910b --ops=select_top_k --gtest_filter='SelectTopK*' -j4
bash build.sh -u --opapi --soc=ascend910b --ops=select_top_k --gtest_filter='SelectTopKRadixPolicy.*' -j4
```

The existing `add_all_modules_sources` collects the tests using `test_*_infershape.cpp`, `test_*_tiling.cpp`, and `test_aclnn_*.cpp`; no extra CMake target is needed. Tests invoke the registered shape-inference and tiling entry points on the CPU. They do not replace device functionality, bit-accuracy, or ATK performance acceptance.
