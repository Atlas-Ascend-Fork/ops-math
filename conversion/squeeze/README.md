# Squeeze

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

接口功能：删除Tensor中维度值为1的维度。示例：

axis为空时，删除所有维度值为1的维度：

```text
>>> x = tf.constant([[[1], [2]], [[3], [4]]])   # shape (2, 2, 1)
>>> tf.squeeze(x).shape
(2, 2)
```

axis指定时，仅删除指定的维度值为1的维度：

```text
>>> x = tf.constant([[[1], [2]], [[3], [4]]])   # shape (2, 2, 1)
>>> tf.squeeze(x, [2]).shape
(2, 2)
>>> tf.squeeze(x, [0]).shape
(2, 1)
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
      <td>BOOL、FLOAT16、FLOAT、INT4、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、DOUBLE、COMPLEX64</td>
      <td>ND、FRACTAL_NZ</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td><ul><li>待删除的维度列表，取值范围为[-rank(x), rank(x))</li><li>不指定或为空时，删除所有维度值为1的维度</li><li>指定时，仅删除指定的维度，指定的维度值必须为1</li><li>默认值为[]</li></ul></td>
      <td>LIST_INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>压缩后的输出张量，数据类型与输入x一致</td>
      <td>BOOL、FLOAT16、FLOAT、INT4、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、DOUBLE、COMPLEX64</td>
      <td>ND、FRACTAL_NZ</td>
    </tr>
  </tbody></table>

## 约束说明

- 指定axis时，axis对应维度的维度值必须为1，否则算子执行失败。
- 输入为动态shape（含-1）且axis为空时，无法确定-1维度是否为1，输出shape置为-2。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_squeeze](./examples/test_geir_squeeze.cpp) | 通过[算子IR](./op_graph/squeeze_proto.h)构图方式调用Squeeze算子。 |
