# MaskedFillRange

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    ×     |
| <term>Atlas A2系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：按`start`/`end`给出的区间窗口，沿`axis`将输入Tensor`x`的对应段填充为`value`本轮的值。`start`/`end`/`value`的第一维为填充轮数`rounds`，逐轮顺序填充，后轮覆盖先轮。

- 计算公式（逐轮`i = 0 … rounds-1`，`y`初值为`x`，`axis`先归一化到`a ∈ [0, dim)`）：

  - 作用维为最后一维`D`（1D，2D，3D`a=2`），窗口按批次`n`索引：1D每轮仅取第0个窗口（`s=start[i,0]`）；2D行`r`即批次（`s=start[i,r]`）；3D`a=2`时同批次`n`的所有`C`行共享窗口列（`s=start[i,n]`、`e=end[i,n]`），窗口截断到`[0, D)`：

    $$
    y[..., k] =
    \begin{cases}
    value[i], \, \, \max(s, 0) \le k < \min(e, D)\\
    y[..., k], \, \, \text{otherwise}
    \end{cases}
    $$

  - 3D`a=0`，窗口按`n`取`s=start[i,n]`、`e=end[i,n]`（直接比较，不截断），整片填充：

    $$
    y[n, :, :] =
    \begin{cases}
    value[i], \, \, s \le n < e\\
    y[n, :, :], \, \, \text{otherwise}
    \end{cases}
    $$

  - 3D`a=1`，窗口按`n`取`s=start[i,n]`、`e=end[i,n]`（直接比较，不截断），整行填充：

    $$
    y[n, c, :] =
    \begin{cases}
    value[i], \, \, s \le c < e\\
    y[n, c, :], \, \, \text{otherwise}
    \end{cases}
    $$

  - 1D时每轮仅取第0个窗口（`start[i,0]`/`end[i,0]`），`start`第二维大于1时其余窗口列被忽略。

## 参数说明

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 180px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>x</td>
      <td>输入</td>
      <td>公式中的输入x。支持1-D (D,)、2-D (N, D)、3-D (N, C, D)。</td>
      <td>FLOAT16、FLOAT、INT8、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>start</td>
      <td>输入</td>
      <td>填充区间起点。shape为(rounds, B)，第一维为填充轮数，第二维为窗口批数；窗口按行/批次索引（见功能说明）。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>end</td>
      <td>输入</td>
      <td>填充区间终点（开区间）。约束同start。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>每轮填充值。shape为(rounds,)，dtype与x一致。</td>
      <td>FLOAT16、FLOAT、INT8、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出y。shape与dtype和x一致。</td>
      <td>FLOAT16、FLOAT、INT8、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>填充作用轴，取值范围为[-dim(x), dim(x))，负值表示从末维倒数。2D场景填充恒作用于最后一维，3D场景支持0/1/2三轴（见功能说明）。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

* value的dtype与x必须一致；start与end的dtype必须一致且为INT32。
* x的维度仅支持1-D/2-D/3-D；start/end为2-D，value为1-D，且第一维（填充轮数）一致。
* 填充轮数rounds需满足rounds ≤ 128，且rounds与有效批数的乘积不超过128。有效批数=start第二维B与批数需求取小：1D需求1、2D需求行数N、3D需求第一维N（3D时同n的C行共享一个批次）。
* x的元素总数不超过1024 * 1024 * 1024；任一维为0（空tensor）时算子不执行填充，y与x同形返回成功。
* start第二维小于行数/批数时，未覆盖的行/批次不填充；大于时多余窗口列被忽略。
* 作用维为最后一维时窗口区间截断到[0, D)；3D a=0/a=1时窗口直接比较不截断。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| GE图模式 | [test_geir_masked_fill_range](examples/test_geir_masked_fill_range.cpp) | 通过[geir](docs/MaskedFillRange.md)图模式调用MaskedFillRange算子。 |
