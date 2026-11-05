# ChunkCat

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR&950DT products                       |    √     |
| Atlas A3 products     |    √     |
| Atlas A2 products      |    √     |
| Atlas 200I/500 A2 inference products                      |    ×     |
| Atlas inference products                              |    ×     |
| Atlas training products                              |    ×     |

## Function Description

Splits all tensors in tensors into num_chunks chunks along dimension dim, concatenates them along the dimension after dim, and finally converts the result to the data type of out.

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
      <td>x</td>
      <td>Input</td>
      <td>List of input tensors.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td>Dimension along which tensors are split into chunks.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_chunks</td>
      <td>Attribute</td>
      <td>Number of chunks to split into.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

* The data type or data format of elements in the x list is not within the supported range.
* The data types of elements in the x list are inconsistent.
* The shapes of tensors in the x list are not within 1 to 8 dimensions.
* dim is not 0.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_chunk_cat](examples/test_aclnn_chunk_cat.cpp) | Invokes the ChunkCat operator through the [aclnnChunkCat](docs/aclnnChunkCat_en.md) API. |
