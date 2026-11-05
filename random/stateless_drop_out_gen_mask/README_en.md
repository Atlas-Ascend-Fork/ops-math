# stateless_drop_out_gen_mask

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
    Computes the random number input parameters key and counter based on seed, seed1, and offset. Then, it calls the philox_random algorithm to generate random numbers, calls the uniform algorithm to normalize the random numbers of the uint32 type, and finally calls the compare_scalar comparison function to output the mask result.

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
      <td>Obtains the size of the input shape.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>prob</td>
      <td>Input</td>
      <td>Obtains the keep probability.</td>
      <td>FLOAT16, BF16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>Input</td>
      <td>Obtains the random seed.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed1</td>
      <td>Input</td>
      <td>Obtains the random seed.</td>
      <td>INT32, INT64</td>
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
      <td>out</td>
      <td>Output</td>
      <td>Obtains the output tensor.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Sample Code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_stateless_drop_out_gen_mask](examples/arch35/test_aclnn_stateless_drop_out_gen_mask.cpp) | Invokes the stateless_drop_out_gen_mask operator through the [aclnnDropoutGenMask](../dsa_gen_bit_mask/docs/aclnnDropoutGenMask_en.md) API. |
