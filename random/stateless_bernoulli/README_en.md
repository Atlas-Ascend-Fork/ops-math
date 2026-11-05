# stateless_bernoulli

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

- Operator function:
    Extracts binary random numbers (0 or 1) from the Bernoulli distribution. prob is the probability of generating a binary random number, and the input tensor specifies the shape.

- Computation formula:

  $$
  out∼Bernoulli(prob)
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
      <td>shape</td>
      <td>Input</td>
      <td>Shape of the random numbers.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>prob</td>
      <td>Input</td>
      <td>Keep probability.</td>
      <td>FLOAT16, BF16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>Input</td>
      <td>Obtains the random seed.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>Step size for obtaining values.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Obtains the output tensor.</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL, FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Data type of the output.</td>
      <td>INT (ge::DataType enumeration value, default DT_FLOAT)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>probMode</td>
      <td>Attribute (Optional)</td>
      <td>Semantic mode of prob: 1 = scalar semantics (prob is a single-element tensor, and its value is used as the probability); 0 = tensor semantics (prob is a probability tensor whose shape must be broadcastable to the output shape). Default 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- When probMode=0, the shape of prob must be broadcastable to the output shape. Output tensors with more than 8 dimensions are not supported.
- When probMode=1, prob must be a single-element tensor.

## Invocation Description

| Invocation Method | Sample Code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_bernoulli_tensor](./examples/arch35/test_aclnn_bernoulli_tensor.cpp) | Invokes the stateless_bernoulli operator through the [aclnnBernoulliTensor](./docs/aclnnBernoulliTensor&aclnnInplaceBernoulliTensor_en.md) API. |
| aclnn invocation | [test_aclnn_bernoulli](./examples/arch35/test_aclnn_bernoulli.cpp) | Invokes the stateless_bernoulli operator through the [aclnnBernoulli](./docs/aclnnBernoulli&aclnnInplaceBernoulli_en.md) API. |
