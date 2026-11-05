# HansEncode

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: performs PDF statistics on the bytes containing the exponent bits of a tensor, performs lossless compression according to the PDF distribution statistics, and stores the compressed result in the memory on the device or offloads it to the host side.

## Parameter Description

<table class="tg" style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 60px">
  <col style="width: 60px">
  <col style="width: 310px">
  <col style="width: 150px">
  <col style="width: 60px">
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
      <td>input_tensor</td>
      <td>Input</td>
      <td>Indicates the input tensor to be compressed. The data element size supports only multiples of 64 and must be greater than or equal to 32768.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pdf</td>
      <td>Output</td>
      <td>Indicates the probability density distribution of the bytes containing the exponent bits of inputTensor. The shape is required to be (1, 256). The value of each element indicates the number of times its corresponding index appears in input.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mantissa</td>
      <td>Output</td>
      <td>Indicates the output mantissa part.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>fixed</td>
      <td>Output</td>
      <td>Indicates the fixed-length part of the exponent bit compression of input.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>Indicates the variable-length part of the exponent bit compression of input.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>statistic</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates whether to perform PDF statistics.</li><li>The default value is false.</li></ul></td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reshuffle</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates whether to perform memory reshuffling on the encoding results of all cores.</li><li>The default value is false.</li></ul></td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_hans_encode](./examples/test_aclnn_hans_encode.cpp) | Invokes the HansEncode operator through the [aclnnHansEncode](./docs/aclnnHansEncode_en.md) API.    |
