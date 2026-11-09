# SliceLastDim

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                              |    ×     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入张量的最后一维进行切片，通过start、end指定切片范围，stride指定步长。

- 计算示例：
  - 输入x shape = (2, 3, 10)，start = 0，end = 10，stride = 2
  - 输出y shape = (2, 3, 5)
  - y[i, j, k] = x[i, j, start + k \* stride] = x[i, j, 2 \* k]

- 计算公式：
  输出y的shape与输入x相同，仅最后一维变为：

  ```text
  y.shape[-1] = ceil((end - start) / stride)
  y[..., k] = x[..., start + k * stride]
  ```

  start和end支持负数索引（相对最后一维长度），start会被截断到[0, lastDim]，end会被截断到[0, lastDim]。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| x      | 输入          | 输入张量。 | INT8、INT16、INT32、INT64、FLOAT16、FLOAT、BFLOAT16 | ND |
| y      | 输出          | 输出张量，与x具有相同的数据类型和格式。 | 与x一致 | ND |
| start  | 属性          | 最后一维切片的起始索引，支持负数索引。 | INT | - |
| end    | 属性          | 最后一维切片的结束索引，支持负数索引。 | INT | - |
| stride | 可选属性      | 切片步长，默认为1，必须 >= 1。 | INT | - |

## 约束说明

- 输入张量x的维度范围为[1, 8]。
- stride必须 >= 1；当输出非空时，stride还须满足stride < 2^40，超出时在Host Tiling阶段被校验拒绝。
- start和end支持负数索引，负数时加上最后一维长度转换为正索引；start截断到[0, lastDim]，end截断到[0, lastDim]。
- 支持空轴输入：输入x的非末维为0，或切片结果为空（如end <= start）时，输出为对应shape的空张量：Host侧Tiling短路返回，Kernel直接跳过搬运。
- 行跨距上限：连续搬运的行跨距，即 (输入末维长度 - 输出末维长度) × 数据类型字节数，须小于2^40字节。
- 动态shape支持情况：支持动态Shape（维度值以-1标记的动态维度）与动态编译静态：非末维的-1维度原样传播到输出，末维为-1时输出末维推导为-1。输入为未知rank（shape以-2标记）时输出透传未知rank，最终形状由运行期按真实输入形状推导。

### 异常校验

| 违反条件 | 校验阶段 | 错误表现 |
| :------- | :------- | :------- |
| x的维度数（rank）不在[1, 8]内 | 图编译推导 / Host Tiling | 返回GRAPH_FAILED，呈现`EZ0012`并记录shape dim校验日志 |
| stride < 1，或输出非空时stride >= 2^40 | 图编译推导 / Host Tiling | 返回GRAPH_FAILED，呈现`EZ0026`并记录stride校验日志 |
| x的数据类型不在支持集（INT8、INT16、INT32、INT64、FLOAT16、FLOAT、BFLOAT16）内 | 图编译推导 / Host Tiling | 返回GRAPH_FAILED，呈现`EZ0020`并记录dtype校验日志 |

> 说明：错误码由图编译或运行框架对校验失败进行封装，不同集成通路可能只返回GRAPH_FAILED并保留具体日志；部分非法输入在到达算子校验前可能被框架层检查先拦截（如rank越界时部分通路的`E80012`），此时以框架层报码为准。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|----------|----------|------|
| 图模式调用 | [test_geir_slice_last_dim](examples/test_geir_slice_last_dim.cpp) | 通过[算子IR](./op_graph/slice_last_dim_proto.h)构图方式调用SliceLastDim算子。 |
