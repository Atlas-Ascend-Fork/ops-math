# SqueezeV3

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

接口功能：删除Tensor中维度值为1的维度。axes以可选输入的形式提供。示例：

axes为空时，删除所有维度值为1的维度：

```text
>>> x = torch.randn(2, 1, 3)
>>> torch.squeeze(x).shape
(2, 3)
```

axes指定时，仅删除指定的维度值为1的维度：

```text
>>> x = torch.randn(2, 1, 3)
>>> torch.squeeze(x, 1).shape
(2, 3)
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
      <td>待压缩的输入张量</td>
      <td>所有类型</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axes</td>
      <td>可选输入</td>
      <td><ul><li>待删除的维度列表，取值范围为[-rank(x), rank(x))</li><li>不指定或为空时，删除所有维度值为1的维度</li><li>指定时，仅删除指定的维度，指定的维度值必须为1</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>压缩后的输出张量，数据类型与输入x一致</td>
      <td>所有类型</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 指定axes时，axes对应维度的维度值必须为1或-1，否则算子执行失败。
- axes取值超出[-rank(x), rank(x))范围时，算子执行失败。
- 输入为动态shape（含-1）且axes为空时，无法确定-1维度是否为1，输出shape置为-2。
- 输入shape为-2时，输出shape为-2。
- 输入shape确定但axes在编译阶段无法获取（如axes来自上游计算结果）时，输出shape置为-2。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_squeezev3](./examples/test_geir_squeezev3.cpp) | 通过[算子IR](./op_graph/squeezev3_proto.h)构图方式调用SqueezeV3算子。 |
