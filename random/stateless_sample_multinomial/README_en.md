# StatelessSampleMultinomial

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: This operator is an internal service operator used when the `aclnnMultinomialGetWorkspaceSize` and `aclnnMultinomialTensorGetWorkspaceSize` APIs build the computation flow. It is used for the with-replacement multinomial distribution sampling path on Ascend 950. Based on the input cumulative distribution and random seed, the operator draws `num_samples` samples from each multinomial distribution and stores the category indexes of the samples in the output tensor.
- Computation formula:

  For the $j$-th draw of the $d$-th distribution, a random number is generated:

  $$
  u_{d,j} \sim U(0, 1]
  $$

  The output is the smallest category index $k$ that satisfies the following condition:

  $$
  x_{d,k-1} < u_{d,j} \le x_{d,k}
  $$

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
      <td>Input cumulative distribution tensor, with the shape (C) or (N, C). The last dimension represents the cumulative probability of categories.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>norm_probs</td>
      <td>Optional Input</td>
      <td>Normalized probability tensor, with the same shape and data type as x. When it is passed in, it is used for the fallback of zero-probability categories in the sampling results.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>Input</td>
      <td>Seed of the random number generator, which affects the generated random number sequence.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>Offset of the random number generator, which affects the position in the generated random number sequence.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_samples</td>
      <td>Attribute</td>
      <td>Number of samples drawn from each multinomial distribution.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Category indexes of the output samples, with the shape (num_samples) or (N, num_samples).</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

1. `x` is a 1D or 2D tensor, and its last dimension is the category dimension whose length does not exceed $2^{24}$.
2. `norm_probs` is an optional input. When it is passed in, its shape and data type are the same as those of `x`, and `x` is the result of computing the cumulative sum of `norm_probs` along the last dimension. When it is not passed in, the fallback of zero-probability categories is performed by comparing adjacent elements of `x`.
3. `num_samples` is a positive integer.
4. `offset` is an integer multiple of 4.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn invocation | [test_aclnn_multinomial](./examples/test_aclnn_multinomial.cpp) | The StatelessSampleMultinomial service operator is invoked internally when the computation flow is built through the [aclnnMultinomial](./docs/aclnnMultinomial_en.md) API. |
| aclnn invocation | [test_aclnn_multinomial_tensor](./examples/test_aclnn_multinomial_tensor.cpp) | The StatelessSampleMultinomial service operator is invoked internally when the computation flow is built through the [aclnnMultinomialTensor](./docs/aclnnMultinomialTensor_en.md) API. |
