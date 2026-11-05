# OneHot

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √    |
| <term>Atlas A3 products</term>       |    √    |
| <term>Atlas A2 products</term>       |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √    |
| <term>Atlas inference products</term>                             |    √    |
| <term>Atlas training products</term>                              |    √    |

## Function Description

- Operator function: for the input self of length n, obtains an output out with n*k elements after the one_hot computation, where the value of k is numClasses.
- Computation formula:
  $$
  out[i][j]=\left\{
  \begin{aligned}
  onValue,\quad self[i] = j \\
  offValue, \quad self[i] \neq j
  \end{aligned}
  \right.
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
      <td>Indicates the index tensor.</td>
      <td>UINT8, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>depth</td>
      <td>Input</td>
      <td>Number of classes.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>on_value</td>
      <td>Input</td>
      <td>Fill value for the index position.</td>
      <td>FLOAT16, FLOAT, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>off_value</td>
      <td>Input</td>
      <td>Fill value for non-index positions.</td>
      <td>FLOAT16, FLOAT, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Input attribute</td>
      <td>Dimension where the encoding vector is inserted.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Indicates the one-hot tensor.</td>
      <td>FLOAT16, FLOAT, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn API | [test_aclnn_one_hot](examples/test_aclnn_one_hot.cpp) | Invokes the OneHot operator through the [aclnnOneHot](docs/aclnnOneHot_en.md) API. |
