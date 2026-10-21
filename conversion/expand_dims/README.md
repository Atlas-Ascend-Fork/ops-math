# ExpandDims

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

- 算子功能：在输入张量shape的指定位置插入一个大小为1的维度，仅改变张量的shape，不改变底层数据。算子由图引擎在主机侧（Host CPU）执行，作为内存复用（reuse-input）算子处理，输出直接视图输入内存。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 输入张量。 | BOOL, FLOAT16, FLOAT, INT4, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, DOUBLE, COMPLEX64 | ND, FRACTAL_NZ |
| axis | 输入 | 插入维度的位置索引，标量或单元素张量，支持负数索引（`[-rank-1, rank]`），值依赖输入。 | INT32, INT64 | ND, FRACTAL_NZ |
| y | 输出 | 插入维度后的输出张量，数据类型与 `x` 一致。 | 与 `x` 相同 | ND, FRACTAL_NZ |

## 约束说明

- `axis` 的取值范围为 `[-(rank(x)-1), rank(x)]`，超界则推理失败。
- `axis` 必须为标量或单元素张量。
- `axis` 的数据类型必须为INT32或INT64。
- 若 `axis` 的值在编译期不可知，输出shape为输入rank+1个未知维度（-1）。
- 支持的数据类型以 [`expand_dims_proto.h`](op_graph/expand_dims_proto.h) 中的IR定义为准。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_expand_dims](examples/test_geir_expand_dims.cpp) | 通过[算子IR](op_graph/expand_dims_proto.h)构图方式调用ExpandDims算子。 |
