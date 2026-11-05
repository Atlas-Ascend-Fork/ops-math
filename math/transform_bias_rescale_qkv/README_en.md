# TransformBiasRescaleQkv

English | [简体中文](./README.md)

## Product Support Status

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function:
  The TransformBiasRescaleQkv operator is an API for processing the query (Query), key (Key), and value (Value) vectors in the multi-head attention mechanism. It adjusts the bias (Bias) and rescale (Rescale) factors of these vectors to optimize the attention computation process.

- Computation formula:
  The element-wise computation process is as follows:

  $$

  q_o=(q_i+q_{bias})/\sqrt{dim\_per\_head}\\

  $$
  $$

  k_o=k_i+k_{bias}\\

  $$
  $$

    v_o=v_i+v_{bias}

  $$

  In the formula:

  - dim_per_head is the dimension of each attention head.
  - q<sub>o</sub>, k<sub>o</sub>, and v<sub>o</sub> are the output elements of the query (Query), key (Key), and value (Value) vectors respectively.
  - q<sub>i</sub>, k<sub>i</sub>, and v<sub>i</sub> are the input elements of the query (Query), key (Key), and value (Value) vectors respectively.
  - q<sub>bias</sub>, k<sub>bias</sub>, and v<sub>bias</sub> are the offsets of the input elements of the query (Query), key (Key), and value (Value) vectors respectively.

## Parameter Description

<table style="table-layout: fixed; width: 937px"><colgroup>
  <col style="width: 126px">
  <col style="width: 135px">
  <col style="width: 293px">
  <col style="width: 266px">
  <col style="width: 117px">
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
      <td>qkv</td>
      <td>Input</td>
      <td>Inputs q<sub>i</sub>, k<sub>i</sub>, and v<sub>i</sub> in the formula.</td>
      <td>BFLOAT16, FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>qkvBias</td>
      <td>Input</td>
      <td>Inputs q<sub>bias</sub>, k<sub>bias</sub>, and v<sub>bias</sub> in the formula.</td>
      <td>BFLOAT16, FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numHeads</td>
      <td>Attribute</td>
      <td><ul><li>Number of input heads.</li><li>The value must be greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qOut</td>
      <td>Output</td>
      <td>q<sub>o</sub> in the formula.</td>
      <td>BFLOAT16, FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kOut</td>
      <td>Output</td>
      <td>k<sub>o</sub> in the formula.</td>
      <td>BFLOAT16, FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vOut</td>
      <td>Output</td>
      <td>v<sub>o</sub> in the formula.</td>
      <td>BFLOAT16, FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraint Description

  - The data types of the inputs qkv and qkvBias and the outputs qOut, kOut, and vOut must be the same.
  - If the input value is NaN, the output is also NaN. If the input is Inf, the output is also Inf.
  - If the input is -Inf, the output is also -Inf.

## Invocation Description

| Invocation Method   | Sample Code | Description  |
| ------------ | ------------ | ------------ |
| aclnn invocation  | [test_aclnn_transform_bias_rescale_qkv](./examples/test_aclnn_transform_bias_rescale_qkv.cpp) | Invokes the TransformBiasRescaleQkv operator through the [aclnnTransformBiasRescaleQkv](./docs/aclnnTransformBiasRescaleQkv_en.md) API.   |
| Graph-mode invocation | -   | Invokes the TransformBiasRescaleQkv operator by building a graph through the [operator IR](./op_graph/transform_bias_rescale_qkv_proto.h). |
