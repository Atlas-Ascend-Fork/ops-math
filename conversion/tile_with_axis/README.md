# TileWithAxis

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>                                 |    √     |
| <term>Atlas A2系列产品</term>                                 |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                        |    √     |
| <term>Atlas推理系列产品</term>                                |    √     |
| <term>Atlas训练系列产品</term>                                |    √     |

## 功能说明

- 算子功能：沿指定维度复制输入Tensor数据，扩展输出Tensor。

- 计算示例：
  - 输入x = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]，shape = (2, 3, 2)
  - axis = 1，tiles = 2
  - 输出y = [[[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12], [7, 8], [9, 10], [11, 12]]]，shape = (2, 6, 2)

- 计算公式：
  输出y的shape与输入x相同，仅axis维度变为 `x.shape[axis] * tiles`。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| x      | 输入          | 输入张量。 | BFLOAT16、FLOAT16、FLOAT、INT32、INT8、UINT8 | ND |
| y      | 输出          | 输出张量，与x具有相同的数据类型和格式。 | 与x一致 | ND |
| axis   | 可选属性       | 指定复制的维度，默认为1。 | INT | - |
| tiles  | 属性          | 复制次数，必须大于0。 | INT | - |

## 约束说明

- axis必须在输入Tensor的维度范围内（支持负数索引，如axis = -1表示最后一维）。
- tiles必须大于0。
- 输出Tensor在axis维度的大小必须等于输入Tensor在axis维度的大小乘以tiles。
- 输入Tensor维度不超过8维。
- 支持标量输入（rank = 0）：输出shape为[tiles]，即输入标量值复制tiles份；此时axis不生效。
- 输入Tensor任一维允许为0（空Tensor）：输出对应维仍为0，输出Tensor为空。

## 调用说明

| 调用方式   | 样例代码 | 说明 |
|------------|----------|------|
| GE图模式 | [test_geir_tile_with_axis](./examples/test_geir_tile_with_axis.cpp) | 通过[算子IR](./op_graph/tile_with_axis_proto.h)构图方式调用TileWithAxis算子。 |
