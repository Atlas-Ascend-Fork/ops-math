# SqueezeV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

算子功能：删除Tensor中维度值为1的维度。axis以属性的形式提供。该算子为已废弃版本，请优先使用Squeeze算子。示例：

axis为空时，删除所有维度值为1的维度：

```text
>>> x = torch.randn(2, 1, 3)
>>> torch.squeeze(x).shape
(2, 3)
```

axis指定时，仅删除指定的维度值为1的维度：

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
      <td>axis</td>
      <td>属性</td>
      <td><ul><li>待删除的维度列表，取值范围为[-rank(x), rank(x))，支持负数索引</li><li>不指定或为空时，删除所有维度值为1的维度</li><li>指定时，仅删除指定的维度，指定的维度值必须为1</li></ul></td>
      <td>ListInt，默认值为[]</td>
      <td>-</td>
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

- 指定axis时，axis对应维度的维度值必须为1或-1，否则算子执行失败。
- axis取值超出[-rank(x), rank(x))范围，或axis按原始值去重后数量大于等于输入维数时，输出shape回退为输入shape（不执行squeeze）。
- 输入为动态shape（含-1）且axis为空时，无法确定-1维度是否为1，输出shape置为-2。
- 输入shape为-2时，输出shape为-2。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_squeezev2](./examples/test_geir_squeezev2.cpp) | 通过[算子IR](./op_graph/squeezev2_proto.h)构图方式调用SqueezeV2算子。 |
