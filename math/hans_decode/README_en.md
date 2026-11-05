# HansDecode

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

- Operator function: decodes the compressed tensor based on the PDF, and restores the tensor by reassembling it based on the mantissa.

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
      <td>mantissa</td>
      <td>Input</td>
      <td>Indicates the mantissa part of the input tensor to be decompressed.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pdf</td>
      <td>Input</td>
      <td>Indicates the probability density distribution of the bytes containing the exponent bits used during compression. The shape is required to be (1, 256). The value of each element indicates the number of times its corresponding index appears in input.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>fixed</td>
      <td>Input</td>
      <td>Indicates the first output segment of compression.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Input</td>
      <td>Indicates the part that exceeds the fixed space during compression.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Indicates the storage location of the exponent after decompression.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reshuff</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates whether to refill on contiguous memory.</li><li>The default value is false.</li></ul></td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- The Ascend 950 implementation uses SIMD/SIMT collaborative decoding.
- For Ascend 950, the data types of mantissa, fixed, var, and out must be the same as those of the original tensors during encoding.
- For Ascend 950, fixed must contain at least 512 bytes of common header; the input code stream and the reshuff attribute must be consistent with the encoding side.
- For the test commands and coverage limitations of Ascend 950, see [Verification Description](../hans_encode/tests/README_en.md).

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_hans_decode](./examples/test_aclnn_hans_decode.cpp) | Invokes the HansDecode operator through the [aclnnHansDecode](./docs/aclnnHansDecode_en.md) API.    |
