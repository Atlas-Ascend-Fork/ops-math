# SelectTopK

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | × |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

A2（`ascend910b`）与 A3（`ascend910_93`）共用 arch22 Host 和 Kernel 实现，资源大小由运行平台提供。
Radix 分流后的有序输出要求 SortWithIndex 在 A2/A3 上支持 FP16/BF16 值与原轴索引的同步排序。

## 功能说明

SelectTopK 沿指定维度，从每行的有效区间内选取最大或最小的 `k` 个元素，支持返回值、索引或两者。

指定维度的长度为 `N`，区间端点裁剪到 `[0, N]`。设第 $r$ 行裁剪后的区间为 $[b_r, e_r)$，则有效结果数为

$$
valid_r = \min(k, \max(e_r - b_r, 0))
$$

有效值及其索引写在各行的前 $valid_r$ 个位置，其余位置使用填充值。有效索引为元素在指定维度上的原始索引加 `indices_offset`，填充索引直接使用 `index_fill`。ACLNN 输出形状由输入形状将指定维度的长度替换为 `k` 得到。

例如，从输入 `[9, 2, 7, 1, 5]` 的区间 `[1, 4)` 内选取最大值，设 `k=4` 并按值降序排列。索引偏移为 10，值和索引分别用 `-99` 和 `-1` 填充时，输出值为 `[7, 2, 1, -99]`，输出索引为 `[12, 11, 13, -1]`。

## 参数说明

以下为算子原型参数。ACLNN 层另提供 `dim`、输出张量和工作区参数，完整说明见
[aclnnSelectTopK](docs/aclnnSelectTopK.md)。L0 输入视图为 `[outer,N,inner]`，输出为 `[outer,K,inner]`。

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 130px">
  <col style="width: 145px">
  <col style="width: 410px">
  <col style="width: 195px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待选取元素的输入张量。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>begin</td>
      <td>可选输入</td>
      <td>各行区间的起点，包含该位置。支持每行一个端点或单元素广播，默认从 0 开始。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>end</td>
      <td>可选输入</td>
      <td>各行区间的终点，不包含该位置。广播规则同 begin，默认到 N 结束。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>输出</td>
      <td>选中的值或填充值，数据类型与 x 一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输出</td>
      <td>选中元素的原轴索引加偏移后的值，或填充索引。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>属性</td>
      <td>每行输出的元素个数，可以大于有效区间长度。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>largest</td>
      <td>属性</td>
      <td>true 选取最大值，false 选取最小值。默认为 true。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>order</td>
      <td>属性</td>
      <td>0 表示有效结果的顺序未指定。1 表示按值排序，largest 为 true 时降序，否则升序。默认为 1。取值 2 为预留值。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>indices_dtype</td>
      <td>属性</td>
      <td>输出索引的数据类型，支持 DT_INT32 和 DT_INT64，默认为 DT_INT64。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>indices_offset</td>
      <td>属性</td>
      <td>加到有效原轴索引上的偏移量，默认为 0。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_mode</td>
      <td>属性</td>
      <td>0 表示仅输出值，1 表示仅输出索引，2 表示同时输出值和索引。默认为 2。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value_fill</td>
      <td>属性</td>
      <td>有效元素不足 k 时的值填充，默认为 0。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>index_fill</td>
      <td>属性</td>
      <td>有效元素不足 k 时的索引填充，默认为 -1。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- ACLNN 接口支持 1～8 维输入和非连续的输入、输出张量。指定维度的长度与 `k` 均须在 `[0, INT32_MAX]` 内。
- `indices_offset`、`index_fill` 和指定维度上所有加偏移后的原始索引均须在指定索引类型的取值范围内。
- 对于相等的值，索引的选取和顺序均不保证稳定。同时输出值和索引时，两者逐项对应。
- `k=0` 或行数为 0 时输出为空。指定维度长度为 0 且输出非空时，输出全部使用填充值。
- 全轴、双输出、offset=0 的 FP16/BF16 请求满足 Radix 门控时，L2 调用 Topk/RadixTopK；
  `order=1` 再由 SortWithIndex 同时重排值及原轴索引，`order=0` 直接使用无序结果。
- Radix 分流接受 NaN 规范化为 `0x7fff`，非NaN值及 SelectTopK 主路径仍保持原值位模式。
  完整门控和配套 Radix 依赖见接口文档。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn API | [test_aclnn_select_top_k](./examples/test_aclnn_select_top_k.cpp) | 通过 [aclnnSelectTopK](./docs/aclnnSelectTopK.md) 接口调用 SelectTopK 算子。 |
| L0 | [SelectTopK](op_api/select_top_k.h) | 构建 SelectTopK 主路径计算任务 |

构建及 CPU 单元测试入口：

```bash
# 在 ops-math 仓库根目录执行。
bash build.sh --pkg --soc=ascend910b --ops=select_top_k --vendor_name=select_topk -j4
# A3 使用 --soc=ascend910_93。
bash build.sh -u --ophost --soc=ascend910b --ops=select_top_k --gtest_filter='SelectTopK*' -j4
```

测试范围见 [tests/README.md](tests/README.md)。CPU 测试不替代设备功能、位精度和完整 L2 性能验证。
