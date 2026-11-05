# KlDivV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

* Operator function: computes the KL divergence.
* Computation formula:
  * Defines loss_pointwise to store intermediate results.

    $$
    loss\_pointwise_i=\begin{cases}
    NaN & \text{ if }&logTarget=false \text{ and } target_i <= 0,  \\
    target_i * \left ( \log{(target_i)}- self_i  \right )  & \text{ if }& logTarget=false, \\
    \exp^ {target_i} * \left ( target_i- self_i \right )  & \text{ else. }
    \end{cases}
    $$

  * The computation formula of out is:

    $$
    out=\begin{cases}
    \bar{loss\_pointwise}  & \text{ if }& reduction= 1, \\
    \sum loss\_pointwise & \text{ elif }& reduction= 2,\\
    \frac{\sum loss\_pointwise}{self.size(0)} & \text{ elif }& reduction= 3,\\
    loss\_pointwise & \text{ else. }
    \end{cases}
    $$

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
      <td>x</td>
      <td>Input</td>
      <td>Input tensor x in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>Input</td>
      <td>Input tensor x in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Optional attribute</td>
      <td><ul><li>reduction in the formula.</li><li>The default value is mean.</li></ul></td>
      <td>STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>log_target</td>
      <td>Optional attribute</td>
      <td><ul><li>logTarget in the formula.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor y in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_kl_div_v2](./examples/test_aclnn_kl_div_v2.cpp) | Invokes the KlDivV2 operator through the [aclnnKlDiv](./docs/aclnnKlDiv_en.md) API.    |
| Graph-mode invocation | [test_geir_kl_div_v2](./examples/test_geir_kl_div_v2.cpp)   | Invokes the KlDivV2 operator by building a graph through the [operator IR](./op_graph/kl_div_v2_proto.h). |
