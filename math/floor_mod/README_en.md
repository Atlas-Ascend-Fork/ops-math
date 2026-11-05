# FloorMod

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: broadcasts the scalar self into a tensor with the same shape as the tensor other, and then converts each element into the remainder obtained by dividing it by the corresponding element of other. The result has the same sign as the divisor other, and the absolute value of the result is less than the absolute value of other.
  The actual computation of remainder(self, other) is equivalent to the following formula:

  $$
  out_i = self - floor(self / other_i) * other_i
  $$

- Example:

```text
self = 5.0   # float
other = tensor([[-1, -2],
                [-3, -4]]).type(int32)
result = remainder(self, other)

# Value of result
# tensor([[ 0., -1.],
#         [-1., -3.]])  float

# For the element -4 in other, the computation result is 5 - floor(5 / -4) * -4 = -3
# It can be seen that the absolute value of the final result -3 is less than the absolute value of the original -4.
```

## Parameter Description

<table style="table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>x1</td>
      <td>Input</td>
      <td>Input tensor self_i in the formula</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input tensor other_i in the formula</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor out_i in the formula</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Invocation Description

| Invocation Method | Invocation Sample                                             | Description                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_remainder_scalar_tensor](./examples/test_aclnn_remainder_scalar_tensor.cpp) | Invokes the FloorMod operator through the [aclnnRemainderScalarTensor](./docs/aclnnRemainderScalarTensor_en.md) API  |
