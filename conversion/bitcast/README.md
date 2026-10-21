# Bitcast

## 产品支持情况

| 产品                                           | 是否支持 |
| :------------------------------------------- | :--: |
| <term>Ascend 950PR&950DT系列产品</term>       |   √  |
| <term>Atlas A3系列产品</term> |   √  |
| <term>Atlas A2系列产品</term> |   √  |
| <term>Atlas 200I/500 A2推理产品</term>          |   √  |
| <term>Atlas推理系列产品</term>                    |   √  |
| <term>Atlas训练系列产品</term>                    |   √  |

## 功能说明

- 算子功能：将输入tensor按照比特位重新解释为目标数据类型，不改变底层数据的存储内容，仅改变dataType和shape。
- 计算规则：根据输入x的数据类型与目标数据类型的比特位大小关系，调整输出shape：
  - 当 `sizeof(xDtype) == sizeof(targetDtype)`：shape不变。
  - 当 `sizeof(xDtype) > sizeof(targetDtype)`（高精度转低精度）：在shape尾部扩展1维，维度值 = `sizeof(xDtype) / sizeof(targetDtype)`。
  - 当 `sizeof(xDtype) < sizeof(targetDtype)`（低精度转高精度）：将shape尾部降1维，需满足xShape最后一维 = `sizeof(targetDtype) / sizeof(xDtype)`。
- keep_dim属性：`keep_dim=true` 时不改变维度数，而是在尾轴上乘/除倍数；`keep_dim=false` 时通过增/降维来调整。
- 数据类型大小单位：常规类型按字节计算；4bit类型（INT4、FLOAT4_E2M1、FLOAT4_E1M2等）按比特位计算，可与字节类型混合换算（如INT8 ↔ INT4倍数为2）。

## 参数说明

| 参数名   | 输入/输出/属性 | 描述                                                                 | 数据类型                                                                                                                                                              | 数据格式                                         |
| :------- | :------------- | :------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------- |
| x        | 输入           | 需要进行bitcast的输入tensor。                                        | BOOL、FLOAT16、FLOAT、INT4、INT8、INT32、UINT32、UINT8、INT64、UINT64、INT16、UINT16、DOUBLE、COMPLEX64                                                               | ND、FRACTAL_NZ、FRACTAL_NZ_C0_8、FRACTAL_NZ_C0_16、FRACTAL_NZ_C0_32 |
| type     | 属性           | 目标数据类型。                                                       | INT（ge::DataType枚举值，如INT4、FLOAT4_E2M1、FLOAT4_E1M2、HIFLOAT8）                                                                                               | -                                                |
| keep_dim | 属性           | 是否保持维度数不变。true时尾轴乘/除倍数，false时通过增/降维调整。默认false。 | BOOL                                                                                                                                                                  | -                                                |
| y        | 输出           | bitcast后的输出tensor，dtype由type属性决定。                         | INT4、FLOAT4_E2M1、FLOAT4_E1M2、HIFLOAT8                                                                                                                              | ND、FRACTAL_NZ、FRACTAL_NZ_C0_16、FRACTAL_NZ_C0_32 |

## 约束说明

- x的数据类型与目标数据类型的比特位大小必须能整除，否则算子执行失败。
- 当 `sizeof(xDtype) < sizeof(targetDtype)` 且 `keep_dim=false` 时，x不能为标量，且xShape最后一维必须等于 `sizeof(targetDtype) / sizeof(xDtype)`。
- 当 `keep_dim=true` 时，x不能为标量。
- 不支持STRING、STRING_REF、UNDEFINED、MAX等数据类型，以及大小未知的类型（GetSizeByDataType返回 -1）。
- x的shape range（动态shape场景）的min与max维度数必须一致。

## 调用说明

无
