# ReduceStdWithMeanV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>         |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>         |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                         |    ×     |
| <term>Atlas 推理系列产品</term>                                |    ×     |
| <term>Atlas 训练系列产品</term>                                |    ×     |

## 功能说明

- 算子功能：沿给定维度 `dim` 归约计算标准差和均值，kernel 内使用**分块 `Welford`（Chan 并行合并）**算法单遍扫描输入，同时产出均值和标准差/标准差倒数。两个 L2 API 覆盖不同场景：`aclnnStdMeanCorrectionV2` 对应 `torch.std_mean`，计算标准差 + 均值，支持 Bessel 修正；`aclnnBatchNormStatsV2` 对应 BatchNorm 统计量，计算均值 + 标准差倒数（`1/sqrt(var+eps)`）。

- L2 API 内部管线（以 `aclnnStdMeanCorrectionV2` 为例）：

  ```text
  self ──→ (可选 Transpose：归约维移至最内层) ──→ ReduceStdWithMeanV2 ──→ stdOut
                                        ──→ meanOut（必要时逆转置）
  ```

  L0 kernel `l0op::ReduceStdWithMeanV2` 只读 `self`（无预计算 mean 输入），单遍读取 GM，块内树归约求精确块均值/块二阶矩，再用 Chan 公式跨 tile 合并，输出 `(std, mean)` 二元组。相比旧 Two-Pass 实现（`ReduceMean` → `Expand`/`BroadcastTo` 中间链 + 双遍读 GM），消除了中间 tensor 的 GM 读写与中间 kernel 链。

  **两阶段跨核路径**（"M 小 R 大"场景，如 `[64, 65536]`）：当 M ≤ 64 且 R > 2048（32B 对齐）时，
  R 按 `K = min(coreNum, ceil(R/2048))` 段拆到 K 个核：

  ```text
  self ──→ ReduceStdWithMeanV2Partial（每核一段，写 partial [M, K, 3] 三元组）
        ──→ ReduceStdWithMeanV2Merge（Chan 跨段合并 → std/mean，fp32）
        ──→（非 fp32 时 Cast 回原 dtype）──→ stdOut / meanOut
  ```

  Chan 合并对任意子集划分精确，两阶段结果与单 kernel 分块 `Welford` 数值等价（fp32 归约顺序
  差异 ~1e-7 相对）。tiling 数据含 `stage`（0/1/2）、`partialK`、`partialSeg` 字段。

- 计算公式（kernel 内）：

  $$
  \begin{aligned}
  \bar{x}_t &= \frac{\sum x_{tile}}{|tile|} \qquad (\text{块内树归约，精确}) \\
  M2_t &= \sum (x_{tile} - \bar{x}_t)^2 \\
  \delta &= \bar{x}_t - \bar{x} \\
  n &\mathrel{+}= |tile| \\
  \bar{x} &\mathrel{+}= \delta \cdot \frac{|tile|}{n} \\
  M2 &\mathrel{+}= M2_t + \delta^2 \cdot \frac{|tile| \cdot (n - |tile|)}{n} \\
  var &= \frac{M2}{\max(0,\ N - \text{correction})} \\
  output &= \begin{cases}
  \sqrt{var + eps}, & \text{invert} = \text{false} \\[6pt]
  \dfrac{1}{\sqrt{var + eps}}, & \text{invert} = \text{true}
  \end{cases}
  \end{aligned}
  $$

  其中 `N` 为归约维度元素总数。

## 参数说明

<table style="undefined;table-layout: fixed; width: 880px"><colgroup>
  <col style="width: 100px">
  <col style="width: 120px">
  <col style="width: 320px">
  <col style="width: 260px">
  <col style="width: 80px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (input)</td>
      <td>输入</td>
      <td>输入张量。支持 1-8 维。aclnnStdMeanCorrectionV2 使用参数名 self，aclnnBatchNormStatsV2 使用参数名 input。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>归约维度。支持单维或多维归约。L2 API 通过 Transpose 自动将非最内层归约维移至最内层后调度 kernel。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>correction</td>
      <td>属性</td>
      <td>Bessel 修正值。0 表示总体方差（除以 N），>= 1 表示样本方差（除以 N - correction）。当 correction 超过归约维度元素数时，std 输出按边界规则填 NAN/INF（mean 仍正常输出）。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdim</td>
      <td>属性</td>
      <td>是否保留归约维度。true 时输出与输入同维数（归约维度长度为 1），false 时压缩归约维度。仅 aclnnStdMeanCorrectionV2 使用。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>invert</td>
      <td>属性</td>
      <td>输出控制。false = 输出标准差（std），true = 输出标准差倒数（1/std）。aclnnStdMeanCorrectionV2 固定输出 std，aclnnBatchNormStatsV2 固定输出 1/std。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>数值稳定性常数，加在方差上再开根号，避免除零。aclnnStdMeanCorrectionV2 固定为 0.0f，aclnnBatchNormStatsV2 使用 DOUBLE 入参。</td>
      <td>FLOAT / DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stdOut (invstdOut)</td>
      <td>输出</td>
      <td>标准差结果。aclnnStdMeanCorrectionV2 输出标准差（std），参数名为 stdOut；aclnnBatchNormStatsV2 输出标准差倒数（invstd），参数名为 invstdOut。</td>
      <td>与 self 保持一致</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>meanOut</td>
      <td>输出</td>
      <td>均值结果。aclnnStdMeanCorrectionV2 输出归约后的均值，aclnnBatchNormStatsV2 输出沿 batch 维归约的均值。</td>
      <td>与 self 保持一致</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `self` (input) 的维度数（rank）必须在 1 到 8 之间。
- 支持多维归约，归约维度通过 `dim` 参数指定；非最内层归约由 L2 层 Transpose 自动处理。
- FLOAT16 / BFLOAT16 中间计算全部升精度到 FLOAT32 执行（Cast → Subs → Mul → ReduceSum），规避半精度中间累加精度损失。
- kernel 状态（count/mean/m2）为 fp32 标量，count 用 int64 精确计数，Chan 合并避免 E[x²]-E[x]² 相消。
- 仅支持 ND 格式。
- 仅支持 Atlas A2 训练系列产品/Atlas A2 推理系列产品（Ascend910B，DAV_2201），仅在 910B 上完成编译与精度/性能验证。
- RegBase（Ascend 950）路径在设计上保留原 `ReduceStdV2` / `ReduceVar`+`Add`+`Pow` 实现（依赖 `reduce_std_v2` 编译期符号，不改其代码），**尚未在 950 上验证**。

## 调用说明

本算子通过 aclnn 两段式接口（单算子）调用。

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn 调用（StdMeanCorrectionV2）</td>
    <td><a href="./examples/test_aclnn_std_mean_correction_v2.cpp">aclnn_std_mean_correction_v2</a></td>
    <td>通过 aclnn 两段式接口 <code>aclnnStdMeanCorrectionV2GetWorkspaceSize</code> + <code>aclnnStdMeanCorrectionV2</code>（声明见 <a href="./op_api/aclnn_std_mean_correction_v2.h">aclnn_std_mean_correction_v2.h</a>，由自定义算子包 <code>custom_math</code> 导出）计算标准差 + 均值。对应 PyTorch 的 <code>torch.std_mean(x, dim, correction=correction, keepdim=keepdim)</code>。</td>
  </tr>
  <tr>
    <td>aclnn 调用（BatchNormStatsV2）</td>
    <td><a href="./examples/test_aclnn_batch_norm_stats_v2.cpp">aclnn_batch_norm_status_v2</a></td>
    <td>通过 aclnn 两段式接口 <code>aclnnBatchNormStatsV2GetWorkspaceSize</code> + <code>aclnnBatchNormStatsV2</code>（声明见 <a href="./op_api/aclnn_batch_norm_stats_v2.h">aclnn_batch_norm_stats_v2.h</a>）计算 BatchNorm 统计量（均值 + 标准差倒数）。</td>
  </tr>
  <tr>
    <td>图模式调用</td>
    <td>-</td>
    <td>暂不支持。本算子仅提供 aclnn 接口。</td>
  </tr>
</tbody></table>

### 编译运行

#### 构建自定义算子包

```bash
# 在仓库根目录（source <CANN>/set_env.sh 后）
bash build.sh --pkg --experimental --soc=ascend910b --ops=reduce_std_with_mean_v2 -j16
```

#### UT 测试

```bash
# 在仓库根目录执行
bash build.sh -u --ophost --opapi --opkernel --ops=reduce_std_with_mean_v2 --experimental -j16
```

#### ST 测试（ATK）

```bash
# StdMeanCorrectionV2 ATK
atk aclnn --task accuracy --devices 0 \
  experimental/math/reduce_std_with_mean_v2/tests/st/aclnnStdMeanCorrection/atk_aclnnStdMeanCorrection.json

# BatchNormStatsV2 ATK
atk aclnn --task accuracy --devices 0 \
  -p experimental/math/reduce_std_with_mean_v2/tests/st/aclnnBatchNormStats/executor_aclnnBatchNormStats.py \
  experimental/math/reduce_std_with_mean_v2/tests/st/aclnnBatchNormStats/atk_aclnnBatchNormStats.json
```

#### 示例代码编译运行

```bash
# 在仓库根目录执行（需先完成算子包构建和安装）
bash build.sh --run_example reduce_std_with_mean_v2 eager cust --experimental --vendor_name=custom --soc=ascend910b
```
