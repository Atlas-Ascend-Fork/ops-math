# SelectTopK

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | × |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

A2（`ascend910b`）与 A3（`ascend910_93`）共用 arch22 Host 和 Kernel 实现，资源大小由运行平台提供。
Radix 分流后的有序输出要求 SortWithIndex 在 A2/A3 上支持 FP16/BF16 值与原轴索引的同步排序。

## 功能说明

- 沿指定维度，从每行的有效区间选取最大或最小的 K 个元素，支持同时输出值和索引、仅值或仅索引。
- 支持逐行或广播区间、原轴索引偏移、INT32/INT64 索引、非连续输入与输出，以及有效元素不足 K 时的尾部填充。
- `order=0` 不保证有效结果顺序；`order=1` 按值排序，`largest=true` 时降序，否则升序。`order=2` 为预留值，当前不支持。

对第 r 行，将区间端点裁剪到排序轴范围 `[0,N]`，记为 b、e，有效输出数量为：

$$
valid_r = \min(K, \max(e_r-b_r, 0))
$$

前 `valid_r` 项保存选中值及其原轴索引，后续项使用填充值。有效索引加 `indices_offset`，
`index_fill` 原样写入。L2 输出形状为输入形状将排序维替换为 K。

例如输入 `[9,2,7,1,5]`，区间 `[1,4)`，K=4，选最大值并排序，偏移为10，值/索引填充为-99/-1，
输出 values=`[7,2,1,-99]`、indices=`[12,11,13,-1]`。

## 参数说明

以下为算子原型参数。ACLNN 层另提供 `dim`、输出张量和工作区参数，完整说明见
[aclnnSelectTopK](docs/aclnnSelectTopK.md)。L0 输入视图为 `[outer,N,inner]`，输出为 `[outer,K,inner]`。

| 参数名 | 输入/输出/属性 | 说明 | 数据类型 |
| :--- | :--- | :--- | :--- |
| x | 输入 | 待选择的值，ND 格式 | FLOAT16、BFLOAT16、FLOAT |
| begin | 可选输入 | 包含的原轴起点；一个元素广播或每行一个元素，缺省为0 | INT32 |
| end | 可选输入 | 不包含的原轴终点；广播规则同 begin，缺省为N | INT32 |
| values | 输出 | 选中值，类型与 x 一致 | FLOAT16、BFLOAT16、FLOAT |
| indices | 输出 | 原轴索引或填充值 | INT32、INT64 |
| k | 属性 | 每行输出数量，允许超过有效区间长度 | Int |
| largest | 属性 | true 取最大值，false 取最小值；默认true | Bool |
| order | 属性 | 0不排序，1按值排序；默认1 | Int |
| indices_dtype | 属性 | 索引输出类型；默认INT64 | Int |
| indices_offset | 属性 | 只作用于有效原轴索引；默认0 | Int |
| output_mode | 属性 | 0仅值、1仅索引、2双输出；默认2 | Int |
| value_fill | 属性 | 有效元素不足K时的值填充；默认0 | Float |
| index_fill | 属性 | 有效元素不足K时的索引填充；默认-1 | Int |

## 约束说明

- ACLNN 支持1～8维输入，排序轴长度及K不超过INT32_MAX；输出索引及其偏移不得超出指定索引类型范围。
- 相等值的索引选择不保证稳定；同时输出值与索引时，两者必须逐项对应。
- `K=0` 或行数为0时返回空输出；排序轴为空且输出非空时，全部结果使用填充值。
- 全轴、双输出、offset=0 的 FP16/BF16 请求满足 Radix 门控时，L2 调用 Topk/RadixTopK；
  `order=1` 再由 SortWithIndex 同时重排值及原轴索引，`order=0` 直接使用无序结果。
- Radix 分流接受 NaN 规范化为 `0x7fff`，非NaN值及 SelectTopK 主路径仍保持原值位模式。
  完整门控和配套 Radix 依赖见接口文档。

## 调用说明

| 调用方式 | 接口或样例 | 说明 |
| :--- | :--- | :--- |
| ACLNN | [aclnnSelectTopK](docs/aclnnSelectTopK.md#调用示例) | 两阶段接口，查询工作区后在指定流上执行 |
| 可执行样例 | [test_aclnn_select_top_k.cpp](examples/test_aclnn_select_top_k.cpp) | 创建设备张量，执行标准模式并校验值和索引 |
| L0 | [SelectTopK](op_api/select_top_k.h) | 构建 SelectTopK 主路径计算任务 |

构建及 CPU 单元测试入口：

```bash
# 在 ops-math 仓库根目录执行。
bash build.sh --pkg --soc=ascend910b --ops=select_top_k --vendor_name=select_topk -j4
# A3 使用 --soc=ascend910_93。
bash build.sh -u --ophost --soc=ascend910b --ops=select_top_k --gtest_filter='SelectTopK*' -j4
```

测试范围见 [tests/README.md](tests/README.md)。CPU 测试不替代设备功能、位精度和完整 L2 性能验证。
