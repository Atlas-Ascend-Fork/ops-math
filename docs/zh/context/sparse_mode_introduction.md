# sparseMode介绍

本文档介绍sparseMode的稀疏模式取值，说明sparseMode取0至8时的attenMask遮蔽场景、preTokens与nextTokens取值要求及外切场景参数配置方法，适用于需要为融合算子配置attenMask的开发人员。在大模型领域，sparseMode（稀疏模式）通常指模型架构或计算公式中参数或激活的稀疏性设计，与稠密模式（DenseMode）相对。

本节将介绍常用的sparseMode和对应的场景说明。

| sparseMode | 含义                                  | 备注               |
| ---------- | --------------------- | ------------------ |
| 0          | defaultMask模式。                     | -    |
| 1          | allMask模式。                         | -    |
| 2          | leftUpCausal模式。                    | -    |
| 3          | rightDownCausal模式。                 | -    |
| 4          | band模式。                            | -    |
| 5          | prefix非压缩模式。                    | varlen场景不支持。 |
| 6          | prefix压缩模式。                      | -       |
| 7          | varlen外切场景，rightDownCausal模式。 | 仅varlen场景支持。 |
| 8          | varlen外切场景，leftUpCausal模式。    | 仅varlen场景支持。 |

attenMask的工作原理为，在Mask为True的位置遮蔽query(Q)与key(K)的转置矩阵乘积的值，示意如下：

![attenMask工作原理示意图](../figures/QK_transpose_diagram.png)

$QK^T$矩阵在attenMask为True的位置会被遮蔽，效果如下：

![遮蔽效果示意图](../figures/masked_QK_diagram.png)

## sparseMode=0

sparseMode为0时，代表defaultMask模式。

- 不传mask：如果attenMask未传入则不做mask操作，attenMask取值为None，忽略preTokens和nextTokens取值。Masked $QK^T$矩阵示意如下：

  ![sparseMode=0且不传入mask时的Masked原理图](../figures/sparsemode_0_masked_matrix.png)

- nextTokens取值为0，preTokens大于等于Sq，表示causal场景，attenMask应传入下三角矩阵，此时preTokens和nextTokens之间的部分需要计算，Masked $QK^T$矩阵示意如下：

  ![sparseMode=0在nextTokens=0、preTokens≥Sq的causal场景下的Masked原理图](../figures/sparsemode_0_masked_matrix_1.png)

  attenMask应传入下三角矩阵，示意如下：

  ![causal场景应传入的下三角attenMask矩阵示意图](../figures/attenmask_lower_triangle.png)

- preTokens小于Sq，nextTokens小于Skv，且都大于等于0，表示band场景，此时preTokens和nextTokens之间的部分需要计算。Masked $QK^T$矩阵示意如下：

  ![sparseMode=0在preTokens<Sq、nextTokens<Skv且均≥0的band场景下的Masked原理图](../figures/sparsemode_0_masked_matrix_2.png)

  attenMask应传入band形状矩阵，示意如下：

  ![band场景应传入的band形状attenMask矩阵示意图](../figures/attenmask_band_matrix.png)

- nextTokens为负数，以preTokens=9、nextTokens=-3为例，preTokens和nextTokens之间的部分需要计算。Masked $QK^T$示意如下：

  **说明：nextTokens为负数时，preTokens取值必须大于等于nextTokens的绝对值，且nextTokens的绝对值小于Skv。**

  ![sparseMode=0在nextTokens为负数原理图](../figures/sparsemode_0_masked_matrix_3.png)

- preTokens为负数，以nextTokens=7、preTokens=-3为例，preTokens和nextTokens之间的部分需要计算。Masked $QK^T$示意如下：

  **说明：preTokens为负数时，nextTokens取值必须大于等于preTokens的绝对值，且preTokens的绝对值小于Sq。**

  ![sparseMode=0在preTokens为负数原理图](../figures/sparsemode_0_masked_matrix_4.png)

## sparseMode=1

sparseMode为1时，代表allMask，即传入完整的attenMask矩阵。

该场景下忽略nextTokens、preTokens取值，Masked $QK^T$矩阵示意如下：

![sparseMode=1（allMask）的Masked原理图](../figures/sparsemode_1_masked_matrix.png)

## sparseMode=2

sparseMode为2时，代表leftUpCausal模式的mask，对应以左上顶点划分的下三角场景（参数起点为左上角）。

该场景下忽略preTokens、nextTokens取值，Masked $QK^T$矩阵示意如下：

![sparseMode=2（leftUpCausal）的Masked原理图](../figures/sparsemode_2_masked_matrix.png)

传入的attenMask为优化后的压缩下三角矩阵（2048\*2048），压缩下三角矩阵示意（下同）：

![leftUpCausal场景应传入的优化后压缩下三角attenMask矩阵原理图](../figures/attenmask_compressed_lower_triangle.png)

## sparseMode=3

sparseMode为3时，代表rightDownCausal模式的mask，对应以右下顶点划分的下三角场景（参数起点为右下角）。

该场景下忽略preTokens、nextTokens取值。attenMask为优化后的压缩下三角矩阵（2048\*2048），Masked $QK^T$矩阵示意如下：

![sparseMode=3（rightDownCausal）的Masked原理图](../figures/sparsemode_3_masked_matrix.png)

## sparseMode=4

sparseMode为4时，代表band场景，即计算preTokens和nextTokens之间的部分，参数起点为右下角，preTokens和nextTokens之间需要有交集。attenMask为优化后的压缩下三角矩阵（2048\*2048）。Masked $QK^T$矩阵示意如下：

![sparseMode=4（band）的Masked原理图](../figures/sparsemode_4_masked_matrix.png)

## sparseMode=5

sparseMode为5时，代表prefix非压缩场景，即在rightDownCausal的基础上，左侧加上一个长为Sq，宽为N的矩阵，N的值由可选输入prefix获取，例如下图中表示batch=2场景下prefix传入数组[4,5]，每个batch轴的N值可以不一样，参数起点为左上角。

该场景下忽略preTokens、nextTokens取值，attenMask矩阵数据格式须为BNSS或B1SS，Masked $QK^T$矩阵示意如下：

![sparseMode=5（prefix非压缩）的Masked原理图](../figures/sparsemode_5_masked_matrix.png)

attenMask应传入矩阵示意如下：

![prefix非压缩场景应传入的attenMask矩阵示意图](../figures/attenmask_matrix.png)

## sparseMode=6

sparseMode为6时，代表prefix压缩场景，即prefix场景时，attenMask为优化后的压缩下三角+矩形的矩阵（3072\*2048）：其中上半部分为[2048, 2048]的下三角矩阵，下半部分为[1024, 2048]的矩形矩阵，矩形矩阵左半部分全0，右半部分全1，attenMask应传入矩阵示意如下。该场景下忽略preTokens、nextTokens取值。

![sparseMode=6（prefix压缩）的Masked原理图](../figures/sparsemode_6_masked_matrix.png)

## sparseMode=7

sparseMode为7时，表示varlen且为长序列外切场景（即长序列在模型脚本中进行多卡切分query的sequence length）；用户需要确保外切前为使用sparseMode 3的场景；当前mode下用户需要设置preTokens和nextTokens（起点为右下顶点），且需要保证参数正确，否则会存在精度问题。

Masked $QK^T$矩阵示意如下，在第二个batch对query进行切分，key和value不切分，4x6的mask矩阵被切分成2x6和2x6的mask，分别在卡1和卡2上计算：

- 卡1的最后一块mask为band类型的mask，配置preTokens=6（保证大于等于最后一个Skv），nextTokens=-2，actual_seq_qlen应传入{3,5}，actual_seq_kvlen应传入{3,9}。
- 卡2的mask类型切分后不变，sparseMode为3，actual_seq_qlen应传入{2,7,11}，actual_seq_kvlen应传入{6,11,15}。

![sparseMode=7（varlen外切、rightDownCausal）的Masked原理图](../figures/sparsemode_7_masked_matrix.png)

**说明**

- sparseMode=7，band表示的是最后一个非空tensor的Batch的sparse类型；如果只有一个batch，用户需按照band模式的要求来配置参数；sparseMode=7时，用户需要输入2048x2048的下三角mask作为该融合算子的输入。
- 基于sparseMode=3进行外切产生的band模式的sparse参数应符合以下条件：
  - preTokens >= last_Skv。
  - last_Sq-last_Skv <= nextTokens <= 0。
  - 当前模式下不支持可选输入pse。
- 非band模式的batch应满足：Sq <= Skv。

## sparseMode=8

sparseMode为8时，表示varlen且为长序列外切场景；用户需要确保外切前为使用sparseMode 2的场景；当前mode下用户需要设置preTokens和nextTokens（起点为右下顶点），且需要保证参数正确，否则会存在精度问题。

Masked $QK^T$矩阵示意如下，在第二个batch对query进行切分，key和value不切分，5x4的mask矩阵被切分成2x4和3x4的mask，分别在卡1和卡2上计算：

- 卡1的mask类型切分后不变，sparseMode为2，actual_seq_qlen应传入{3,5}，actual_seq_kvlen应传入{3,7}。
- 卡2的第一块mask为band类型的mask，配置preTokens=4（保证大于等于第一个Skv），nextTokens=1，actual_seq_qlen应传入{3,8,12}，actual_seq_kvlen应传入{4,9,13}。

![sparseMode=8（varlen外切、leftUpCausal）的Masked原理图](../figures/sparsemode_8_masked_matrix.png)

**说明**：

- sparseMode=8，band表示的是第一个非空tensor的Batch的sparse类型；如果只有一个batch，用户需按照band模式的要求来配置参数；sparseMode=8时，用户需要输入2048x2048的下三角mask作为该融合算子的输入。

- 基于sparseMode=2进行外切产生的band模式的sparse参数应符合以下条件：
  - preTokens >= first_Skv。
  - nextTokens >= first_Sq - first_Skv，根据实际情况进行配置。
  - 当前模式下不支持可选输入pse。
