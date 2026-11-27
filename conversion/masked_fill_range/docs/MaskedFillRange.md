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

按`start`/`end`给出的区间窗口，沿`axis`将输入Tensor`x`的对应段填充为`value`本轮的值。

`start`/`end`/`value`的第一维为填充轮数`rounds`，逐轮顺序填充，后轮覆盖先轮；`start`/`end`的第二维为窗口批数`B`，窗口按行/批次索引（1D仅取第0个窗口）。各维度语义（`axis`先归一化到`a ∈ [0, dim)`）：

| x维度 | a | 填充行为 |
| ---- | ---- | ---- |
| 1-D (D,) | 0 | 每轮仅取第0个窗口`[start[i,0], end[i,0])`，截断到`[0, D)`，命中段填充；`B > 1`时其余窗口列被忽略。 |
| 2-D (N, D) | 0或1 | 填充恒作用于最后一维：行`r`取窗口`[start[i,r], end[i,r])`，截断到`[0, D)`，命中段填充。 |
| 3-D (N, C, D) | 0 | 窗口按`n`取`[start[i,n], end[i,n])`（直接比较，不截断），命中整片`y[n, :, :]`填充。 |
| 3-D (N, C, D) | 1 | 窗口按`n`取`[start[i,n], end[i,n])`（直接比较，不截断），命中整行`y[n, c, :]`填充。 |
| 3-D (N, C, D) | 2 | 窗口按`n`取`[start[i,n], end[i,n])`，截断到`[0, D)`，命中段`y[n, :, k]`填充。 |

## 算子原型

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

## 参数说明

- **参数说明**

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
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度（shape）</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x（ge::Tensor）</td>
      <td>输入</td>
      <td>输入tensor。</td>
      <td>支持1-D (D,)、2-D (N, D)、3-D (N, C, D)，元素总数不超过1024 * 1024 * 1024；任一维为0（空tensor）时算子不执行填充，y与x同形返回成功。</td>
      <td>FLOAT16、FLOAT、INT8、INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>start（ge::Tensor）</td>
      <td>输入</td>
      <td>填充区间起点。</td>
      <td>2-D (rounds, B)，第一维为填充轮数（与end/value一致），第二维为窗口批数；窗口按行/批次索引，见功能说明。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(rounds, B)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>end（ge::Tensor）</td>
      <td>输入</td>
      <td>填充区间终点（开区间）。</td>
      <td>约束同start。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(rounds, B)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value（ge::Tensor）</td>
      <td>输入</td>
      <td>每轮填充值。</td>
      <td>1-D (rounds,)，dtype与x一致。</td>
      <td>FLOAT16、FLOAT、INT8、INT32</td>
      <td>ND</td>
      <td>(rounds,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y（ge::Tensor）</td>
      <td>输出</td>
      <td>输出tensor。</td>
      <td>shape与dtype和x一致；未被命中的元素保持x原值（比特级不动）。</td>
      <td>FLOAT16、FLOAT、INT8、INT32</td>
      <td>ND</td>
      <td>同x</td>
      <td>√</td>
    </tr>
  </tbody></table>

- **属性说明**

  <table style="table-layout: fixed; width: 1302px"><colgroup>
  <col style="width: 185px">
  <col style="width: 120px">
  <col style="width: 380px">
  <col style="width: 380px">
  <col style="width: 237px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>必选/可选</th>
      <th>类型</th>
      <th>取值范围</th>
      <th>约束说明</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>axis</td>
      <td>必选</td>
      <td>INT</td>
      <td>[-dim(x), dim(x))，负值表示从末维倒数</td>
      <td>2-D场景填充恒作用于最后一维，3-D场景支持0/1/2三轴；越界返回失败。</td>
    </tr>
  </tbody></table>

## 约束说明

* value的dtype与x必须一致；start与end的dtype必须一致且为INT32。
* x仅支持1-D/2-D/3-D；start/end为2-D、value为1-D，三者第一维（填充轮数）一致。
* x任一维为0（空tensor，元素总数为0）时算子no-op：y与x同形（空）返回成功，不执行填充；start/end/value仍按上述契约校验。
* 填充轮数rounds ≤ 128，且rounds与有效批数的乘积不超过128，超限返回失败。有效批数=start第二维B与批数需求取小：1-D需求1、2-D需求行数N、3-D需求第一维N（同n的C行共享一个批次，见功能说明）。
* start第二维小于行数/批数时，未覆盖的行/批次不填充；大于时多余窗口列被忽略。
* 作用维为最后一维时窗口区间截断到`[0, D)`；3-D a=0/a=1时窗口直接比较不截断（负值/超界窗口自然不命中）。

## 调用示例

完整可编译样例参见[test_geir_masked_fill_range.cpp](../examples/test_geir_masked_fill_range.cpp)，关键步骤：

1. 构图：创建`MaskedFillRange`算子节点并连接输入输出、设置`axis`属性。

    ```cpp
    auto maskedFillRange1 = op::MaskedFillRange("maskedFillRange1");
    ADD_INPUT(1, x, DT_FLOAT, xShape, 1);      // x: (4, 8)全1
    ADD_INPUT(2, start, DT_INT32, startShape, 2);  // start: (2, 4)全2
    ADD_INPUT(3, end, DT_INT32, endShape, 6);      // end: (2, 4)全6
    ADD_INPUT(4, value, DT_FLOAT, valueShape, 10); // value: (2,)全10
    ADD_OUTPUT(1, y, DT_FLOAT, yShape);
    maskedFillRange1.set_attr_axis(-1);
    ```

2. 执行：`ge::Session::AddGraph` + `RunGraph`运行计算图。

3. 取结果：从输出Tensor读取`y`，本例中每行第`[2, 6)`段被填充为10。

    ```cpp
    ret = session->RunGraph(graph_id, input, output);
    ```
