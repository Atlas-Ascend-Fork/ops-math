# UnsqueezeV2

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

算子功能：在Tensor的shape中插入维度值为1的维度。axis以属性的形式提供。该算子为已废弃版本，请优先使用Unsqueeze算子。示例：

axis指定时，在指定位置插入维度值为1的维度：

```text
>>> x = torch.randn(2, 3)
>>> torch.unsqueeze(x, 1).shape
(2, 1, 3)
```

axis支持负数索引，-1表示在末尾插入：

```text
>>> x = torch.randn(2, 3)
>>> torch.unsqueeze(x, -1).shape
(2, 3, 1)
```

axis为空或不指定时，不插入任何维度，输出shape与输入shape一致。

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
      <td>待扩维的输入张量</td>
      <td>所有类型</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td><ul><li>待插入维度的位置列表，取值范围为[-rank(x)-len(axis), rank(x)+len(axis))，支持负数索引</li><li>位置不能重复（按负数索引归一化后的位置判定）</li><li>不指定或为空时，不插入任何维度，输出shape与输入shape一致</li></ul></td>
      <td>ListInt，默认值为[]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>扩维后的输出张量，数据类型与输入x一致</td>
      <td>所有类型</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- axis取值超出[-rank(x)-len(axis), rank(x)+len(axis))范围，或归一化后位置重复时，算子执行失败。
- 输出维数rank(x)+len(axis)超出框架最大维数限制时，算子执行失败。
- axis为空或不指定时，输出shape与输入shape一致。
- 输入shape为-2时，输出shape为-2。
- 输入含-1（未知维）时，未知维原样保留。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_unsqueezev2](./examples/test_geir_unsqueezev2.cpp) | 通过[算子IR](./op_graph/unsqueezev2_proto.h)构图方式调用UnsqueezeV2算子。 |
