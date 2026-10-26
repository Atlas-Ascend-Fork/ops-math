# RefIdentity

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：功能与[Identity](../../conversion/identity/README.md)一致——输出张量`y`与输入张量`x`的形状、数据类型和内容相同；区别在于RefIdentity的输入为可被ref类算子覆盖的引用张量，用于引用（ref）类型数据。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `x` | 输入 | 输入张量。 | float16、float、double、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool、bfloat16 | ND |
| `y` | 输出 | 与输入`x`形状、数据类型和内容相同的张量。 | 与`x`相同 | ND |

## 约束说明

- 输入`x`与输出`y`的数据类型和形状保持一致。
- 支持动态shape与动态rank。

## 调用说明

RefIdentity算子暂无原型（REG_OP）与推导实现，暂不提供调用样例；如您想要贡献该算子的完整实现，请参考[贡献流程](../../CONTRIBUTING.md)。
