# SelectTopK 单元测试

采用仓库已有的 GTest、`InfershapeContextPara` 和 `TilingContextPara` 测试框架，参考 TopKV2、Segsum 等算子的 Host UT。

`ut/op_api/test_aclnn_select_top_k_radix_policy.cpp` 覆盖 K 下限、两档 N/K 比例边界和轴长上限，直接链接本目录 L2 中的门控实现。

- `ut/op_host/test_select_top_k_infershape.cpp`：6 个形状推导用例，覆盖三维视图、动态维、空输出、超长 K 填充及非法参数。
- `ut/op_host/arch22/test_select_top_k_tiling.cpp`：6 个 A2 tiling 用例，覆盖 Top1、CopyAll、融合布局、区间/填充、order 编码及 INT32 偏移边界；同时覆盖 FP16、BF16、FP32。

在 ops-math 仓库根目录执行：

```bash
source /path/to/cann/set_env.sh
bash build.sh -u --ophost --soc=ascend910b --ops=select_top_k --gtest_filter='SelectTopK*' -j4
bash build.sh -u --opapi --soc=ascend910b --ops=select_top_k --gtest_filter='SelectTopKRadixPolicy.*' -j4
```

现有 `add_all_modules_sources` 按 `test_*_infershape.cpp`、`test_*_tiling.cpp` 和 `test_aclnn_*.cpp` 收集上述测试文件，无需额外添加 CMake 目标。测试调用真实注册的推形/tiling 入口，在 CPU 上执行；它不替代设备功能、位精度或 ATK 性能验收。
