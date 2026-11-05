# AmpUpdateScale

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :----------------------------------------- | :----:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: implements the dynamic Scale update logic in AMP (Automatic Mixed Precision) training. It dynamically adjusts the loss scale based on the current scale value, the growth tracker counter, and whether Inf/NaN is found.
- Computation formula:

  $$
  \text{updated\_scale} = \begin{cases}
  \text{current\_scale} \times \text{backoff\_factor} & \text{if found\_inf} \neq 0 \\
  \text{current\_scale} \times \text{growth\_factor} & \text{if growth\_tracker + 1 = growth\_interval and new\_scale is finite} \\
  \text{current\_scale} & \text{otherwise}
  \end{cases}
  $$

  $$
  \text{updated\_growth\_tracker} = \begin{cases}
  0 & \text{if found\_inf} \neq 0 \text{ or growth triggered} \\
  \text{growth\_tracker} + 1 & \text{otherwise}
  \end{cases}
  $$

  where:
  - $\text{current\_scale}$: the current loss scale value
  - $\text{found\_inf}$: a flag that indicates whether Inf/NaN is detected (0 indicates normal, and non-0 indicates that Inf/NaN is found)
  - $\text{growth\_tracker}$: a counter of the number of steps in which Inf/NaN does not occur consecutively
  - $\text{growth\_factor}$: the scale growth factor (usually 2.0)
  - $\text{backoff\_factor}$: the scale backoff factor (usually 0.5)
  - $\text{growth\_interval}$: the interval in steps for triggering scale growth

- Usage scenario: dynamic loss scaling in AMP training, which is used to prevent gradient underflow in FP16/BF16 mixed-precision training.

## Parameter Description

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 170px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 220px">
  <col style="width: 160px">
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
      <td>current_scale</td>
      <td>Input</td>
      <td>The current loss scale value.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>growth_tracker</td>
      <td>Input</td>
      <td>A counter of the number of steps in which Inf/NaN does not occur consecutively.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>found_inf</td>
      <td>Input</td>
      <td>A flag that indicates whether Inf/NaN is detected. 0 indicates normal, and non-0 indicates that Inf/NaN is found.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>growth_factor</td>
      <td>Attribute</td>
      <td>The scale growth factor, usually set to 2.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>backoff_factor</td>
      <td>Attribute</td>
      <td>The scale backoff factor, usually set to 0.5.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>growth_interval</td>
      <td>Attribute</td>
      <td>The interval in steps for triggering scale growth.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>updated_scale</td>
      <td>Output</td>
      <td>The updated loss scale value.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updated_growth_tracker</td>
      <td>Output</td>
      <td>The updated growth tracker counter.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Shape constraint: All input and output tensors are scalars, and the shape is [1].
- Data type constraint: The data type of current_scale must be the same as those of found_inf and updated_scale.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| aclnn invocation | [test_aclnn_amp_update_scale](./examples/test_aclnn_amp_update_scale.cpp) | Invokes the AmpUpdateScale operator through the [aclnnAmpUpdateScale](./docs/aclnnAmpUpdateScale_en.md) API. |
