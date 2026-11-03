# TensorRedirect

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                              |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：将输入张量`x`的值传递到输出张量`output_x`，输出与输入的数据类型和shape一致。本算子用于图优化或引用重定向场景，不做类型转换、广播或数据布局变换。

- 计算公式：

$$
(\mathrm{output\_x})_i=x_i
$$

其中$i$遍历$x$的全部元素。

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
      <td>需要传递的输入张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_x</td>
      <td>输出</td>
      <td>传递结果张量。数据类型、shape和数据格式均与x一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 不提供原地调用接口。
- <term>Ascend 950PR&950DT系列产品</term>：
  - 支持参数说明中列出的全部数据类型。
  - 静态shape和动态shape均支持输入ND→输出ND；动态shape支持固定rank和动态rank。
  - rank取值范围为[0,8]，rank为0时表示标量；支持空Tensor。
  - 图推形中的`-1`表示未知维度，`{-2}`表示未知rank；实际执行时，输入Tensor的各维度必须为具体的非负整数。
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>、<term>Atlas推理系列产品</term>、<term>Atlas训练系列产品</term>：
  - 不支持BFLOAT16，其余数据类型与参数说明一致。
  - 静态shape支持输入ND→输出ND；不支持动态shape。
  - rank取值范围为[1,8]；不支持空Tensor。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | [test_geir_tensor_redirect](./examples/test_geir_tensor_redirect.cpp) | 通过[算子IR](./op_graph/tensor_redirect_proto.h)构图方式调用TensorRedirect算子。 |

本算子不提供同名ACLNN接口，仅支持上表的GE图模式调用。
