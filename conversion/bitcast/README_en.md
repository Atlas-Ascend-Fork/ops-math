# Bitcast

English | [简体中文](./README.md)

## Product Support Status

| Product                                           | Supported |
| :------------------------------------------- | :--: |
| <term>Ascend 950PR&950DT products</term>       |   √  |
| <term>Atlas A3 products</term> |   √  |
| <term>Atlas A2 products</term> |   √  |
| <term>Atlas 200I/500 A2 inference products</term>          |   √  |
| <term>Atlas inference products</term>                    |   √  |
| <term>Atlas training products</term>                    |   √  |

## Function Description

- Operator function: reinterprets the input tensor as the target data type at the bit level, without changing the storage content of the underlying data, and only changes dataType and shape.
- Computation rule: adjusts the output shape based on the bit size relationship between the data type of the input x and the target data type:
  - When `sizeof(xDtype) == sizeof(targetDtype)`: the shape remains unchanged.
  - When `sizeof(xDtype) > sizeof(targetDtype)` (high precision to low precision): one dimension is appended to the end of the shape, and the dimension value is `sizeof(xDtype) / sizeof(targetDtype)`.
  - When `sizeof(xDtype) < sizeof(targetDtype)` (low precision to high precision): the last dimension of the shape is reduced by 1, and the last dimension of xShape must be `sizeof(targetDtype) / sizeof(xDtype)`.
- keep_dim attribute: when `keep_dim=true`, the number of dimensions remains unchanged, and the tail axis is multiplied or divided by the multiple; when `keep_dim=false`, the shape is adjusted by adding or reducing dimensions.
- Data type size unit: sizes of regular types are calculated in bytes; sizes of 4-bit types (INT4, FLOAT4_E2M1, FLOAT4_E1M2, and so on) are calculated in bits, and can be converted together with byte types (for example, the INT8 ↔ INT4 ratio is 2).

## Parameter Description

| Parameter Name   | Input/Output/Attribute | Description                                                                 | Data Type                                                                                                                                                              | Data Format                                         |
| :------- | :------------- | :------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------- |
| x        | Input           | Input tensor for the bitcast computation.                                        | BOOL, FLOAT16, FLOAT, INT4, INT8, INT32, UINT32, UINT8, INT64, UINT64, INT16, UINT16, DOUBLE, COMPLEX64                                                               | ND, FRACTAL_NZ, FRACTAL_NZ_C0_8, FRACTAL_NZ_C0_16, FRACTAL_NZ_C0_32 |
| type     | Attribute           | Target data type.                                                       | INT (ge::DataType enumeration values, such as INT4, FLOAT4_E2M1, FLOAT4_E1M2, and HIFLOAT8)                                                                                               | -                                                |
| keep_dim | Attribute           | Whether to keep the number of dimensions unchanged. When true, the tail axis is multiplied or divided by the multiple; when false, the shape is adjusted by adding or reducing dimensions. The default value is false. | BOOL                                                                                                                                                                  | -                                                |
| y        | Output           | Output tensor after bitcast, whose dtype is determined by the type attribute.                         | INT4, FLOAT4_E2M1, FLOAT4_E1M2, HIFLOAT8                                                                                                                              | ND, FRACTAL_NZ, FRACTAL_NZ_C0_16, FRACTAL_NZ_C0_32 |

## Constraint Description

- The bit sizes of the data type of x and the target data type must be integer-divisible. Otherwise, the operator fails to execute.
- When `sizeof(xDtype) < sizeof(targetDtype)` and `keep_dim=false`, x cannot be a scalar, and the last dimension of xShape must be `sizeof(targetDtype) / sizeof(xDtype)`.
- When `keep_dim=true`, x cannot be a scalar.
- Data types such as STRING, STRING_REF, UNDEFINED, and MAX, and types with unknown sizes (GetSizeByDataType returns -1) are not supported.
- For the shape range of x (dynamic shape scenarios), the numbers of dimensions of min and max must be the same.

## Invocation Description

None
