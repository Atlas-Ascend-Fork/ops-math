# IsClose

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |     ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: returns a new tensor with Boolean elements, which indicates whether the given self and other are close to each other. If the values are close, it returns True; otherwise, it returns False.
- Computation formula:
  Returns a new tensor out whose data type is BOOL, indicating whether each element of the input self is "close" to the corresponding element of the input other.
  closeness is defined as:

  $$
  \left | self_{i}-other_{i}\right | \le  atol + rtol\times \left | other_{i} \right |
  $$

  The preceding formula holds when both self and other are finite values. When both self and other are non-finite, the result is True only when they are equal. If equal_nan is True, NaNs are considered close; if equal_nan is False, NaNs are considered not close.

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
      <td>x1</td>
      <td>Input</td>
      <td>Input parameter for the is_close computation, which is self_i in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input parameter for the is_close computation, which is other_i in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rtol</td>
      <td>Input</td>
      <td>Input parameter for the is_close computation, which is rtol in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>atol</td>
      <td>Input</td>
      <td>Input parameter for the is_close computation, which is atol in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>equal_nan</td>
      <td>Input</td>
      <td>Input parameter for the is_close computation.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the is_close computation.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_isclose](./examples/test_aclnn_isclose.cpp) | Invokes the IsClose operator through the [aclnnIsClose](./docs/aclnnIsClose_en.md) API. |
