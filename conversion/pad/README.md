# Pad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：按照输入paddings的配置，对输入x的每一维进行前后常量填充，填充值为0。兼容TensorFlow的Pad算子。
- 示例：

  ```text
  输入 x = [[0,1,2]]，shape为[1,3]
  paddings = [[1,1],[2,2]]

  输出 y = [[0,0,0,0,0,0,0],
            [0,0,0,1,2,0,0],
            [0,0,0,0,0,0,0]]，shape为[3,7]
  ```

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>待进行填充的原始tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT32、UINT8、INT16、INT8、COMPLEX64、INT64、QINT8、QUINT8、QINT32、QINT16、QUINT16、UINT16、COMPLEX128、UINT32、UINT64、BOOL、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E2M1、FLOAT4_E1M2。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>输入</td>
      <td>填充的配置，形状为[rank(x), 2]，第i行[a, b]表示对x第i维开头填充a个元素、末尾填充b个元素。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>进行填充后的tensor，数据类型与x相同。</td>
      <td>与x的数据类型相同</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- AICore路径支持的数据类型随平台不同：
  - <term>Atlas A2系列产品</term>：FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、UINT16、INT64；
  - 部分新一代平台在此基础上还支持：UINT8、UINT32、UINT64、BOOL、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E2M1、FLOAT4_E1M2；
  - 其余平台：FLOAT、FLOAT16、BFLOAT16、INT16、INT32、UINT16、INT64。
- 不在上述AICore支持列表中的数据类型（如DOUBLE、COMPLEX64、COMPLEX128、QINT8等）通过AICPU路径计算。

## 约束说明

- paddings的形状需为[rank(x), 2]，其中rank(x)为输入x的维度数。
- 如果x的数据类型为FLOAT4_E2M1或FLOAT4_E1M2，paddings的值应为偶数。
- 如果x的数据类型为HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN或FLOAT8_E8M0，paddings的值应为非负整数。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----|----|----|
| GE图模式 | [test_geir_pad](./examples/test_geir_pad.cpp) | 通过[算子IR](./op_graph/pad_proto.h)构图方式调用Pad算子。 |
