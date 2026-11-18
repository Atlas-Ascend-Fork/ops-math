# AdjacentDifference

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |    √     |
| <term>Atlas A3系列产品</term>     |    ×     |
| <term>Atlas A2系列产品</term>      |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：比较输入张量相邻元素的差异，若相邻元素相同，返回0，否则返回1。

- 计算公式：

$$
out_i = {(x_i == x_j ?\ 0 : 1)}.to(y_dtype) \ j = i - 1
$$

- 首元素定义：$out_0 = 0$。首元素无前相邻元素可比较，固定输出0。

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
      <td>待进行AdjacentDifference计算的入参，公式中的x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_dtype</td>
      <td>属性</td>
      <td>属性，指定输出张量y的数据类型，公式中的y_dtype。可选属性，未设置时默认为INT32（DT_INT32(3)）；合法取值为INT32（DT_INT32(3)）或INT64（DT_INT64(9)）。</td>
      <td>INT32、INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行AdjacentDifference计算的出参，公式中的out。shape与输入x相同，数据类型由属性y_dtype决定（INT32或INT64），且必须与y_dtype一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入x与输出y的shape相同；数据格式仅支持ND；输入rank范围为[0, 8]。支持动态shape与动态rank输入。
- 支持空Tensor输入（存在零长度维、元素数为0）：输出为相同shape的空Tensor。支持rank=0的标量输入（shape为()）：按单元素语义处理，输出标量0。
- 支持非连续的Tensor输入，参见[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)；输出y按连续存储分配。
- 输入x仅被读取，不会被修改；输出写入独立分配的y，不支持原地修改。输入与输出的存储区域不应重叠，重叠时行为未定义。
- 输出y的数据类型必须与属性y_dtype一致（y为INT32时y_dtype须为DT_INT32(3)，y为INT64时y_dtype须为DT_INT64(9)）；y_dtype取值仅允许DT_INT32(3)/DT_INT64(9)。
- 浮点类型输入（FLOAT、FLOAT16、BFLOAT16）按数值相等语义比较相邻元素：NaN与任意值（含NaN自身）比较均不相等，对应位置输出1；+Inf与+Inf、-Inf与-Inf比较相等，对应位置输出0。
- 异常场景（图模式下在图编译阶段校验拒绝并报错，错误信息以GE框架日志为准，本算子不提供独立的公开错误码）：
  - y的数据类型与y_dtype不一致；
  - y_dtype取值不为DT_INT32(3)/DT_INT64(9)；
  - 输入x的数据类型不在支持列表内。

## 调用说明

| 调用方式 | 样例代码 |   说明    |
| :-------- | :---------- | :------------------------ |
|  图模式  |[test_geir_adjacent_difference](examples/test_geir_adjacent_difference.cpp) | 通过[算子IR](./op_graph/adjacent_difference_proto.h)构图方式调用AdjacentDifference算子。 |
