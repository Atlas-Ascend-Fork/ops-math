# StridedSliceAssignV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |    √     |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |
| <term>Kirin X90处理器系列产品</term>                          |    √     |
| <term>Kirin 9030处理器系列产品</term>                         |    √     |

## 功能说明

- 算子功能：StridedSliceAssignV2是一种张量切片赋值操作，将输入张量input_value的内容赋值给目标张量var中由begin、end、strides指定的切片位置，切片以外的区域保持var原值不变。输出var与输入var同名，为in-place语义，输出shape与输入var的shape相同。

- 计算公式：

$$
out[b_0:e_0:s_0, b_1:e_1:s_1, \dots] = input\_value
$$

切片第i维的长度为：$sliceLen[i] = \max(0, \lceil\frac{end[i] - begin[i]}{strides[i]}\rceil)$，其中$\lceil x \rceil$表示对$x$向上取整。$begin$和$end$为经过负索引归一化与边界截断后的取值：当$x < 0$时$x = x + dim$。正stride时截断到$[0, dim]$区间；负stride（仅Ascend 950系列产品支持）时，$begin$截断到$[0, dim-1]$、$end$截断到$[-1, dim]$区间。

## 参数说明

<table style="table-layout: fixed; width: 1162px"><colgroup>
<col style="width: 139px">
<col style="width: 148px">
<col style="width: 532px">
<col style="width: 267px">
<col style="width: 76px">
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
    <td>var</td>
    <td>输入</td>
    <td>待切片赋值的目标tensor。</td>
    <td>FLOAT16、FLOAT、BFLOAT16、INT32、INT64、DOUBLE、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input_value</td>
    <td>输入</td>
    <td>替换切片内容的tensor，dtype需与var保持一致，shape需严格等于计算得出的切片shape。</td>
    <td>FLOAT16、FLOAT、BFLOAT16、INT32、INT64、DOUBLE、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>begin</td>
    <td>输入</td>
    <td>切片位置的起始索引，1维常量tensor，长度不超过var的维数。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>end</td>
    <td>输入</td>
    <td>切片位置的终止索引，1维常量tensor，长度不超过var的维数。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>strides</td>
    <td>输入</td>
    <td>切片的步长，1维常量tensor，长度不超过var的维数。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>axes</td>
    <td>输入</td>
    <td>可选参数，指定切片作用的轴，1维常量tensor，长度与begin相同。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>var</td>
    <td>输出</td>
    <td>切片赋值后的结果tensor，与输入var同名，shape与dtype和输入var一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16、INT32、INT64、DOUBLE、INT8</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- begin、end、strides为值依赖输入，axes为可选值依赖输入，dtype均为INT64，必须能在host侧获取具体数值。
- 输入var支持1D~8D，var的rank必须与input_value的rank相同；Atlas A2/A3及Kirin系列产品要求var每个维度必须大于0，Ascend 950系列产品允许var含0长度维（按空切片no-op处理，输出等于输入var）。
- begin、end、strides三者的长度必须相等，且不超过var的维数；axes的长度必须与begin相同。
- strides各维取值不能为0。
- input_value的dtype必须与var一致，shape必须严格等于切片计算后的shape，不支持广播。
- axes中每个轴必须在[-dim, dim)范围内，且不能重复。axes缺省时，切片作用于前len(begin)个轴（等价于axes=[0, 1, ..., len(begin)-1]），其余未列轴按整轴切片（begin=0、end=dim、strides=1）处理。
- 输出var与输入var为in-place语义，输出shape等于输入var的shape。
- 本算子为纯数据搬运算子，结果位级精确，无精度损失。
- Atlas A2/A3及Kirin系列产品：strides必须为正数（不支持负stride），且最内维（最后一维）对应的strides必须为1。
- Ascend 950系列产品：strides可以为负数，支持负stride反向切片，各轴strides可正可负混合使用；该系列产品不要求最内维strides必须为1。
- Kirin X90/Kirin 9030处理器系列产品：不支持BFLOAT16。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn调用 | [test_aclnn_strided_slice_assign_v2](./examples/test_aclnn_strided_slice_assign_v2.cpp) | 通过[aclnnStridedSliceAssignV2](./docs/aclnnStridedSliceAssignV2.md)接口方式调用StridedSliceAssignV2算子。 |
| 图模式调用 | [test_geir_strided_slice_assign_v2](./examples/test_geir_strided_slice_assign_v2.cpp) | 通过[算子IR](./op_graph/strided_slice_assign_v2_proto.h)构图方式调用StridedSliceAssignV2算子。 |
