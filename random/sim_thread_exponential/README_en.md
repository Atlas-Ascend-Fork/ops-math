# SimThreadExponential

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: generates random numbers that follow the exponential distribution with the parameter lambda and fills them into the self tensor.
- Computation formula:

  $$
  f(x) = -\frac{1}{\lambda} \ln(1 - u), u \sim \text{Uniform}(0, 1]
  $$

## Parameter Description

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 180px">
  <col style="width: 213px">
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
      <td>self</td>
      <td>Input/Output</td>
      <td>Input and output tensor.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>count</td>
      <td>Attribute</td>
      <td>Number of generated random number elements.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambda</td>
      <td>Attribute</td>
      <td>lambda in the formula.</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>Attribute</td>
      <td>Random number seed.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Attribute</td>
      <td>Random number offset value.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- lambda must be greater than 0
- offset must be a multiple of 4 (Ascend 950)
- count must be equal to the total number of elements in the self tensor (Ascend 950)
- Only the ND format is supported

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn invocation | [test_sim_thread_exponential](tests/ut/op_kernel/test_sim_thread_exponential.cpp) | Invokes the SimThreadExponential operator through the [aclnnSimThreadExponential](docs/aclnnSimThreadExponential_en.md) API. |
