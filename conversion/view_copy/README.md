# ViewCopy

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：将src张量按src_size、src_stride、src_storage_offset描述的视图复制到dst张量按dst_size、dst_stride、dst_storage_offset描述的视图，并返回dst。
- 计算公式：

  $$
  \text{dst}\left[\text{dst\_storage\_offset} + \sum_{i=0}^{n-1} \text{dst\_index}_i \cdot \text{dst\_stride}_i\right]
  =
  \text{src}\left[\text{src\_storage\_offset} + \sum_{i=0}^{n-1} \text{src\_index}_i \cdot \text{src\_stride}_i\right]
  $$

  其中，$n$为视图维度数；$\text{dst\_index}_i$、$\text{src\_index}_i$分别为dst、src视图在第$i$维的索引，取值范围为$0 \le \text{dst\_index}_i < \text{dst\_size}_i$、$0 \le \text{src\_index}_i < \text{src\_size}_i$，且因dst_size与src_size形状一致，对应维度的索引逐元素相等；$\text{dst\_storage\_offset}$、$\text{src\_storage\_offset}$为视图起始偏移量；$\text{dst\_stride}_i$、$\text{src\_stride}_i$为第$i$维步长。方括号内为张量存储区中的元素下标。

## 参数说明

<table style="table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 180px">
  <col style="width: 130px">
  <col style="width: 380px">
  <col style="width: 290px">
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
      <td>dst</td>
      <td>输入</td>
      <td>待写入的目标张量，按dst_size、dst_stride、dst_storage_offset描述其视图。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、UINT8、INT8、UINT16、INT16、UINT32、INT32、UINT64、INT64、BOOL、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_size</td>
      <td>输入</td>
      <td>dst张量的视图shape，1D tensor，元素非负，元素个数与dst_stride一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_stride</td>
      <td>输入</td>
      <td>dst张量的视图stride，1D tensor，元素非负，元素个数与dst_size一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_storage_offset</td>
      <td>输入</td>
      <td>dst张量的视图起始偏移量，1D tensor，元素非负且不超出dst张量元素范围。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src</td>
      <td>输入</td>
      <td>待读取的源张量，按src_size、src_stride、src_storage_offset描述其视图。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、UINT8、INT8、UINT16、INT16、UINT32、INT32、UINT64、INT64、BOOL、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src_size</td>
      <td>输入</td>
      <td>src张量的视图shape，1D tensor，元素非负，元素个数与src_stride一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src_stride</td>
      <td>输入</td>
      <td>src张量的视图stride，1D tensor，元素非负，元素个数与src_size一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src_storage_offset</td>
      <td>输入</td>
      <td>src张量的视图起始偏移量，1D tensor，元素非负且不超出src张量元素范围。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst</td>
      <td>输出</td>
      <td>复制结果，与输入dst为同一块内存（原地写入）。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、UINT8、INT8、UINT16、INT16、UINT32、INT32、UINT64、INT64、BOOL、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- dst_size与src_size形状一致。
- dst_size与dst_stride维度数一致；src_size与src_stride维度数一致。
- dst_size、src_size的元素非负。
- dst_stride、src_stride的元素非负。
- dst_storage_offset、src_storage_offset非负，且不超出对应张量的元素范围。
- 对每个轴i，dst_storage_offset + (dst_size[i] - 1) * dst_stride[i]不超过dst张量元素数；src同理。
- aclnn接口层的dtype、类型转换和确定性约束见[aclnnInplaceCopy](docs/aclnnInplaceCopy.md)。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_view_copy](examples/test_aclnn_view_copy.cpp) | 通过[aclnnInplaceCopy](docs/aclnnInplaceCopy.md)接口方式调用ViewCopy算子。 |
| 图模式调用 | - | 通过[算子IR](op_graph/view_copy_proto.h)构图方式调用ViewCopy算子。 |
