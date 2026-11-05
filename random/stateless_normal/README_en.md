# StatelessNormal

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term>    |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: generates a random number tensor that follows the normal distribution based on the specified mean and standard deviation (stdev). It uses the counter-based Philox4x32-10 random number generation algorithm and the Box-Muller transform, strictly matching GPU precision (the same seed + offset produces the same random sequence), and supports mean and stdev in the form of a scalar or a tensor.

- Computation formula:

  $$
  y = BoxMuller(Philox(seed, offset)) \times stdev + mean
  $$

  Here, mean and stdev are a scalar or a tensor. When they are tensors, only item 0 participates in the computation. All output elements share this set of mean and standard deviation values, and the remaining elements do not participate in the computation.

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
      <td>shape</td>
      <td>Input</td>
      <td>Shape of the output tensor.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>Input</td>
      <td>Seed of the Philox random number generator.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>Offset of the Philox random number generator, which must be a multiple of 4.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Mean of the normal distribution, which can be a scalar or a tensor. When it is a tensor, only item 0 participates in the computation, and all output elements share this mean.</td>
      <td>FLOAT, FLOAT16, BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>stdev</td>
      <td>Input</td>
      <td>Standard deviation of the normal distribution, which can be a scalar or a tensor. When it is a tensor, only item 0 participates in the computation, and all output elements share this standard deviation.</td>
      <td>FLOAT, FLOAT16, BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Outputs random values that follow the normal distribution with the specified shape.</td>
      <td>FLOAT, BF16, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- offset must be a multiple of 4.
- The data types of mean and stdev must be in lockstep with the output y.
- When mean and stdev are tensors, only item 0 participates in the computation. All output elements share this set of mean and standard deviation values, and the remaining elements do not participate in the computation.
- Only the SIMT programming model is supported (Ascend 950 series).

## Invocation Description

| Invocation Method | Sample Code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_stateless_normal.cpp](./examples/arch35/test_aclnn_stateless_normal.cpp) | Invokes the StatelessNormal operator through the [aclnnInplaceNormal](../dsa_random_normal/docs/aclnnInplaceNormal_en.md) API, where both mean and stdev are scalars. |
