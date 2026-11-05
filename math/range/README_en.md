# Range

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: obtains values starting from start to end at the interval of step and saves them to the output 1-dimension tensor, where the data range is [start,end).
- Computation formula:
  $$
  out_{i+1} = out_i+step
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
      <td>start</td>
      <td>Input</td>
      <td>Start position of the range for obtaining values.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>limit</td>
      <td>Input</td>
      <td>End position of the range for obtaining values.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>delta</td>
      <td>Input</td>
      <td>Step length for obtaining values.</td>
      <td>FLOAT16, FLOAT, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>is_closed</td>
      <td>Input Attribute</td>
      <td>Specifies whether limit is included to form a closed interval.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output of the increasing sequence.</td>
      <td>FLOAT16, FLOAT, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn API | [test_aclnn_range](examples/test_aclnn_range.cpp) | Invokes the Range operator through the [aclnnRange](docs/aclnnRange_en.md) API. |
| aclnn API | [test_aclnn_arange](examples/test_aclnn_arange.cpp) | Invokes the Range operator through the [aclnnArange](docs/aclnnArange_en.md) API. |
