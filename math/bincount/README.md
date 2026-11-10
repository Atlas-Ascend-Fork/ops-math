# Bincount

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算非负整数数组中每个数的频率。minlength为输出tensor的最小size；当weights为空指针时，self中的self[i]每出现一次，则其频率加1，当weights不为空时，self[i]每出现一次，其频率加weights[i]，最后存放到out中下标为self[i]的位置；因此out大小为(self的最大值+1)与minlength中的最大值。

- 计算公式：

  如果n是self在位置i上的值，如果指定了weights，则

  $$
  out[self_i] = out[self_i] + weights_i
  $$

  否则：

  $$
  out[self_i] = out[self_i] + 1
  $$

## 参数说明

<table style="table-layout: fixed; width: 910px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>array</td>
      <td>输入</td>
      <td>输入tensor，公式中的self_i。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>指定输出tensor最小长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>输入tensor的权重，公式中的weights_i。</td>
      <td>FLOAT</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>bins</td>
      <td>输出</td>
      <td>输出tensor，公式中的out_i。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- 上表为算子IR（GE图模式）当前注册并可执行的数据类型。
- aclnn API方式调用时支持更宽的数据类型范围，详见[aclnnBincount](./docs/aclnnBincount.md)。
- 算子各层声明的weights、bins数据类型域存在差异：算子IR原型声明为FLOAT、INT32、INT64、DOUBLE，算子信息库注册为FLOAT，tiling校验接受FLOAT、INT32、INT64。数据类型域的统一口径以上表为当前可执行范围。

## 约束说明

- 输入size的值必须大于或等于输入array的最大值+1
- 输出bins的长度与size的值必须相等
- 输入array的每个元素取值必须落在[0, 输出长度)区间内，该约束由调用方保证
- 输出长度（GE图模式下为size的值，aclnn API方式下为max(self的最大值+1, minlength)）不得超过2147483647
- 空Tensor：aclnn API方式下，当self为空Tensor且minlength大于0时，out各元素置0；当self为空Tensor且minlength小于或等于0时，不执行计算
- 动态shape：aclnn API方式要求self、weights、out均为具体shape，不接受-1、-2动态维；GE图模式下算子注册支持动态shape，但输入size为ValueDepend输入，图编译期必须能够取到其常量值，否则tiling以失败返回
- 各产品的确定性计算约束详见[aclnnBincount](./docs/aclnnBincount.md)的约束说明章节

## 错误码与拒绝行为

- aclnn API方式：返回码含义参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)，各场景见[aclnnBincount](./docs/aclnnBincount.md)返回值章节。
  - ACLNN_ERR_PARAM_NULLPTR（161001）：self或out为空指针。
  - ACLNN_ERR_PARAM_INVALID（161002）：self、weights、out的数据类型或数据格式不在支持范围内；self不是1维；weights与self的数据格式或shape不一致；输出长度超过2147483647。
- GE图模式/内核方式：算子不定义专有错误码，所有校验失败统一以ge::GRAPH_FAILED返回，由GE框架上报通用错误码。校验失败场景包括：array、weights、bins任一不是1维；array或size的数据类型不是INT32；weights或bins的数据类型不在FLOAT、INT32、INT64范围内；weights与bins的数据类型不一致；输入size的常量值在编译期不可获取；size的值为负数；bins的长度与size的值不相等。

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_bincount](./examples/test_aclnn_bincount.cpp) | 通过[aclnnBincount](./docs/aclnnBincount.md)接口方式调用Bincount算子。 |
| GE图模式 | - | 通过[算子IR](./op_graph/bincount_proto.h)构图方式调用Bincount算子。 |

另提供TensorFlow图解析插件（[bincount_tf_plugin](./framework/bincount_tf_plugin.cpp)），
用于将TensorFlow图中的Bincount节点映射为本算子IR。

本算子未提供TensorFlow、PyTorch的端到端调用接口与绑定，也未提供对应的参考实现资产。
PyTorch场景下使用的bincount为torch_npu内建实现，不属于本算子交付范围。
