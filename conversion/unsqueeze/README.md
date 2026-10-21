# Unsqueeze

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

接口功能：在Tensor shape中插入维度值为1的维度，仅改变shape，不改变数据。示例：

axes为空时，输出与输入shape一致：

```text
>>> x = tf.constant([[1, 2], [3, 4]])   # shape (2, 2)
>>> unsqueeze(x).shape
(2, 2)
```

axes指定插入位置：

```text
>>> x = tf.constant([[1, 2], [3, 4]])   # shape (2, 2)
>>> unsqueeze(x, [1]).shape
(2, 1, 2)
>>> unsqueeze(x, [0]).shape
(1, 2, 2)
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
      <td>待扩展的输入张量</td>
      <td>所有数据类型</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axes</td>
      <td>可选属性</td>
      <td><ul><li>插入维度值为1的位置列表，取值范围为[-rank(x)-len(axes), rank(x)+len(axes))</li><li>不指定或为空时，输出shape与输入一致</li><li>不允许包含重复值</li><li>默认值为[]</li></ul></td>
      <td>LIST_INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>扩展后的输出张量，数据类型与输入x一致</td>
      <td>所有数据类型</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- axes包含重复值时，算子执行失败。
- 输出维度数不能超过最大维度数限制（kMaxDimNum=25）。
- 输入为动态shape（含-1）时，-1维度按位置透传到输出。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_unsqueeze](./examples/test_geir_unsqueeze.cpp) | 通过[算子IR](./op_graph/unsqueeze_proto.h)构图方式调用Unsqueeze算子。 |
