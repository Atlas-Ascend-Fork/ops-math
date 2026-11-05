# KLDiv

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

## Function Description

- Operator function: computes the KL divergence loss (Kullback-Leibler Divergence).

- Shell operator positioning: KLDiv is a shell routing operator. Its registered name KLDiv exactly matches the op type of the legacy GE graph and is used for compatibility with legacy graph migration. If you have new computation requirements, use KLDivV2 instead. The computation kernel of this operator purely forwards the log_target=False DAG of KLDivV2 and always computes with the log_target=False semantics.

- Computation formula (always log_target=False):

$$
loss_i=\begin{cases}
0 & \text{if } target_i=0 \\
NaN & \text{if } target_i<0 \\
target_i \times \left(\log(target_i)-x_i\right) & \text{else}
\end{cases}
$$

- The output y is determined by reduction:

$$
y=\begin{cases}
loss & \text{if } reduction=\text{"none"} \\
\sum_i loss_i & \text{if } reduction=\text{"sum"} \\
\frac{\sum_i loss_i}{numel} & \text{if } reduction=\text{"mean"} \\
\frac{\sum_i loss_i}{dim_0} & \text{if } reduction=\text{"batchmean"}
\end{cases}
$$

- When reduction is "none", y has the same shape as x; when reduction is "mean", "sum", or "batchmean", y is a scalar (with an empty shape).

## Parameter Description

<table style="table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 200px">
  <col style="width: 90px">
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
      <td>Input tensor x (log probabilities) in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>Input</td>
      <td>Input tensor target (probabilities) in the formula, which has the same shape and dtype as x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Required attribute</td>
      <td><ul><li>Specifies the reduction mode of the output.</li><li>Only none, mean, sum, and batchmean are supported.</li><li>There is no default value, and it must be explicitly set when building a graph.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor y in the formula. When reduction is "none", it has the same shape as x; for other values, it is a scalar.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- The operator prototype does not declare the log_target attribute (the v1 prototype does not have this attribute either), and the operator always computes with the log_target=False semantics. The log_target attribute carried in a legacy graph is ignored as an unknown attribute.

## Constraint Description

- x and target must have the same shape and dtype.
- reduction is a required attribute, and only none, mean, sum, and batchmean are supported.
- The prototype declares DT_DOUBLE support on paper, but the dtype whitelist in the tiling phase rejects double. DT_DOUBLE is actually not supported.
- The output is 0 at positions where target equals 0, and NaN at positions where target is less than 0 (value range constraint, which is not verified by tiling/infershape; callers must ensure the input semantics are valid).
- FLOAT16 and BFLOAT16 inputs are promoted to FLOAT32 for computation inside the kernel, and the results are cast back to the original dtype by round-to-nearest-even.
- Output shape: when reduction is "none", y has the same shape as x; when reduction is "mean", "sum", or "batchmean", y is a scalar.
- This operator is a shell routing operator and has no aclnn API: legacy graph migration uses the atc graph compilation path. For PyTorch adaptation, use the aclnnKlDiv API in the kl_div_v2 directory of ops-math.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/test_geir_kl_div.cpp">test_geir_kl_div</a></td>
    <td>Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">operator invocation</a> to complete the operator compilation and verification.</td>
  </tr>
</tbody>
</table>

- The graph-mode example invokes this operator in the op::KLDiv graph-building mode through the <a href="./op_graph/kl_div_proto.h">operator IR</a>.
