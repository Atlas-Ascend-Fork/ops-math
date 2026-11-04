# AdaCast

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                              |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：面向 HDRnet（High Dynamic Range Network）实时图像增强网络的输入预处理算子。将传感器输出的 uint16 HDR 图像数据转换为 float16 张量，并按白电平（pixel）完成归一化，等价于把「整型→浮点类型转换」与「按除数缩放」两步融合为单次逐元素运算。
- 计算公式：

$$
out_{i} = \mathrm{Cast\_FP16}^{sat+round}\bigl(\mathrm{Cast\_FP32}(self_{i}) \times (1 / pixel)\bigr)
$$

其中末步先做饱和（clip 到 fp16 值域 [-65504, 65504]），再按 round-to-nearest-even（就近取偶）方式舍入转换到 float16。

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
      <td>待归一化的 HDR uint16 图像数据，公式中的self_i，shape 支持 1D~4D。</td>
      <td>UINT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>归一化后的 float16 张量，公式中的out_i，shape 与 x 完全一致。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pixel</td>
      <td>可选属性</td>
      <td><ul><li>白电平归一化基数，缩放系数为 1/pixel。</li><li>默认值为 65535。</li><li>取值必须为正整数（pixel &gt; 0），非正数时算子编译报错。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 数据格式仅支持 ND。
- 输入 x 的 rank 范围为 [1, 4]（0-D 标量与 rank>4 均拒绝），输出 y 的 shape 必须与 x 完全一致（element-wise 语义，不支持广播）。
- 类型组合固定为 UINT16 → FLOAT16，不支持其他 dtype 组合。
- pixel 为可选属性，未传入时使用默认值 65535；取值必须为正整数（pixel > 0），非正数时算子编译报错。
- 支持空 Tensor（0 元素），host 侧提前返回，不下发 kernel。
- 默认确定性实现，相同输入始终产生相同输出。

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| GE图模式 | [test_geir_ada_cast.cpp](./examples/test_geir_ada_cast.cpp) | 通过 GE-IR（`ge::Session` / `RunGraph`）构图方式调用 AdaCast 算子；算子 IR 原型见 [ada_cast_proto.h](./op_graph/ada_cast_proto.h)。样例默认不参与构建（`ENABLE_GEIR_EXAMPLE` 未在仓库 CMake 中定义），需按样例注释在客户端工程中显式编译，并链接 libgraph / libge_runner。 |

> 注：本算子仅提供 GE 图模式接口；`CMakeLists.txt` 中 `ACLNNTYPE aclnn_exclude` 且无 `op_api/`，未提供 ACLNN 与 PyTorch/TensorFlow 接口。
