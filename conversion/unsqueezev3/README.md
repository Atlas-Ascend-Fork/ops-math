# UnsqueezeV3

## 产品支持情况

| 产品                                           | 是否支持 |
| :------------------------------------------- | :--: |
| <term>Ascend 950PR&950DT系列产品</term>       |   √  |
| <term>Atlas A3系列产品</term> |   √  |
| <term>Atlas A2系列产品</term> |   √  |
| <term>Atlas 200I/500 A2推理产品</term>          |   √  |
| <term>Atlas推理系列产品</term>                    |   √  |
| <term>Atlas训练系列产品</term>                    |   √  |

## 功能说明

接口功能：在输入Tensor的指定位置插入维度值为1的维度，仅改变shape不改变数据。axes以输入的形式提供。示例：

axes指定时，在对应位置插入维度1：

```text
>>> x = torch.randn(2, 3)
>>> torch.unsqueeze(x, 1).shape
(2, 1, 3)
```

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1528px"><colgroup>
  <col style="width: 132px">
  <col style="width: 120px">
  <col style="width: 256px">
  <col style="width: 253px">
  <col style="width: 333px">
  <col style="width: 126px">
  <col style="width: 160px">
  <col style="width: 145px">
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
      <td>待插入维度的输入张量</td>
      <td>所有类型</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axes</td>
      <td>输入</td>
      <td><ul><li>待插入维度的位置列表，取值范围为[-(rank(x)+len(axes)), rank(x)+len(axes))</li><li>列表中的值不能重复</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>插入维度后的输出张量，数据类型与输入x一致</td>
      <td>所有类型</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- axes取值超出[-(rank(x)+len(axes)), rank(x)+len(axes))范围时，算子执行失败。
- axes包含重复值时，算子执行失败。
- axes为空时，输出shape与输入x一致。
- 编译阶段无法获取axes数据（如axes来自上游计算结果）时，输出shape置为-2。
- 输入shape为-2时，输出shape为-2。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_unsqueezev3](./examples/test_geir_unsqueezev3.cpp) | 通过[算子IR](./op_graph/unsqueezev3_proto.h)构图方式调用UnsqueezeV3算子。 |
