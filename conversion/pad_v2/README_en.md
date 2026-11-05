# PadV2

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term>       |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                       |    √     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

## Function Description

- **Operator function**: pads the input tensor with a constant of a specified value.
- **Example**:

  ```text
  Input tensor([[0,1,2]])
  paddings([[0,0],[2,2]])
  constant_values(0)

  The output is ([[0,0,0,1,2,0,0]])
  ```

## Parameter Description

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Original tensor to be padded.</td>
      <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, FLOAT4_E1M2.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>Input</td>
      <td>Padding configuration. shape=[N, 2], where N is the number of dimensions of x.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>constant_values</td>
      <td>Input</td>
      <td>Padding constant value, which is a scalar tensor.</td>
      <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, FLOAT4_E1M2.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Tensor after padding.</td>
      <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, FLOAT4_E1M2.</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas training products</term>, <term>Atlas inference products</term>, and <term>Atlas 200I/500 A2 inference products</term>: the data types BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, and FLOAT4_E1M2 are not supported.
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: the data types HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, and FLOAT4_E1M2 are not supported.

### Output Shape Computation

```text
y.shape[d] = x.shape[d] + paddings[d][0] + paddings[d][1]
```

## Constraint Description

1. **Dimension constraints**:
   - The number of dimensions of x must be in the range [1, 8]
   - The first dimension of paddings must be equal to the number of dimensions of x
   - The second dimension of paddings must be equal to 2

2. **Data type constraints**:
   - x, constant_values, and y must use the same data type
   - paddings must use INT32 or INT64

3. **fp8/fp4 type constraints**:
   - When x is HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, or FLOAT4_E1M2, each dimension of paddings must be a non-negative value
   - When x is FLOAT4_E2M1 or FLOAT4_E1M2, the shape of the trailing axis of the input must be an even number, and the paddings of the trailing axis must be an even number

4. **Padding constraints**:
   - For negative padding (slice), the output shape = x.shape[d] + left + right >= 0

4. **paddings parameter constraints**
   - The shape of paddings must be [rank, 2], where rank is the number of dimensions of input x (1 to 8)
   - Each row [left, right] indicates the number of paddings in the corresponding dimension
     - left: the number of elements padded at the beginning of the dimension
     - right: the number of elements padded at the end of the dimension
   - The value of paddings can be:
     - A positive number: indicates padding
     - A negative number: indicates slice (cropping)
     - Zero: indicates no padding

## Invocation Description

| Invocation Method | Sample Code | Description |
| ------------| ---------------------------------------------------------| ----------------------------------------------------------------|
| GE graph mode | [test_geir_pad_v2.cpp](./examples/arch35/test_geir_pad_v2.cpp) | Invokes the PadV2 operator through the [operator IR](op_graph/pad_v2_proto.h) API |
