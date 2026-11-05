# MulAddn

English | [简体中文](./README.md)

## Product Support Status

|Product         |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |    √     |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function:

  Implements the fused computation of n>=2 mul operations and addn, reducing data movement time and memory usage.

- Computation formula:

  The inputs x1 and x2 are variable-length inputs, each being a list of N tensors. The shape of each tensor in the x1 input is [B, M, 1], and the shape of each tensor in the x2 input is [B, 1, K]. y is the output, with the shape [B, M, K]. N corresponds to the number n of the addn operator and also to the number of fused mul operators.

$$
x1 * x2 = y
$$

$$
x1 = [[B, M, 1], [B, M, 1],...,[B, M, 1]] (N [B, M, 1] tensors in total)
$$

$$
x2 = [[B, 1, K],[B, 1, K],...,[B, 1, K]] (N [B, 1, K] tensors in total)
$$

$$
[[B, M, 1], [B, M, 1],...,[B, M, 1]] * [[B, 1, K],[B, 1, K],...,[B, 1, K]] = y
$$

## Parameter Description

<table style="table-layout: fixed; width: 855px"><colgroup>
  <col style="width: 164px">
  <col style="width: 138px">
  <col style="width: 300px">
  <col style="width: 133px">
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
      <td>Input x1 in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input x2 in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>N</td>
      <td>Attribute</td>
      <td>Number of fused mul operators.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>y in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraint Description

- K<=2040;
- The shapes of the inputs x1 and x2 are [B,M,1] and [B,1,K], respectively;
- N>=2;

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| Graph-mode invocation | [test_geir_mul_addn](./examples/test_geir_mul_addn.cpp)   | Invokes the MulAddn operator by building a graph through the [operator IR](./op_graph/mul_addn_proto.h). |
