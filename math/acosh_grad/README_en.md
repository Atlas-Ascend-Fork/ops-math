# AcoshGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √      |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |   √     |

## Function Description

- Operator function: computes the backward gradient of the Acosh (inverse hyperbolic cosine) operator.
- Operator formula:

  $$
  z_i = dy_i \cdot \dfrac{1}{\sinh(y_i)}
  $$

  where:
  - $y_i$ is the output tensor of the forward Acosh operator, and its value range is expected to be $\geq 0$;
  - $dy_i$ is the gradient passed from upstream;
  - $z_i$ is the gradient with respect to the original input tensor, which equals the upstream gradient multiplied by $1/\sinh(y_i)$.

- When y is zero, the output may contain Inf or NaN.

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>y</td>
      <td>Input</td>
      <td>The output tensor of the forward Acosh operator. The value range is expected to fall within [0, +∞).</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>The upstream gradient, with the same element count and dtype as y. Its shape may differ from y.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>Output</td>
      <td>The gradient with respect to the original input tensor, whose shape and dtype are the same as those of y.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- y and dy must have the same element count and dtype. Their shapes may differ; the output shape matches y.
- In graph mode, the broadcast shape of y and dy must equal the shape of y. Equal input element counts are still required.
- ND format is supported.
- Dynamic dimensions (-1) and unknown rank (-2) are supported.
- Empty inputs produce empty outputs when the above input constraints are satisfied.
- No limit is imposed on the rank of the inputs or outputs.
- Direct kernel calls require contiguous input tensors.

## Invocation Description

| Invocation Method | Invocation Sample                                                                      | Description                                                                                       |
| -------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Graph-mode invocation | [test_geir_acosh_grad](./examples/test_geir_acosh_grad.cpp)             | Invokes the AcoshGrad operator by building a graph through the [operator IR](./op_graph/acosh_grad_proto.h).                 |

For the [arch35 example](./examples/arch35/test_geir_acosh_grad.cpp) and the default example, see [Quick Operator Invocation (GE Graph Mode, Chinese)](../../docs/zh/invocation/quick_op_invocation.md) for compilation and execution.
