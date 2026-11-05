# StatelessRandomUniformV3

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

- Operator function: returns a random number tensor, where the random numbers are drawn from independent uniform distributions.

- Computation formula:

  $$
  UniformResult = StatelessRandomUniformV2 * (to - from) + from
  $$

  $$
  RandomResult = StatelessRandomUniformV2 * to - (StatelessRandomUniformV2 * from - from)
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
      <td>Shape of the input tensor.</td>
      <td>INT64, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>Key used by the counter-based random number generation algorithm, with the shape [1].</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counter</td>
      <td>Input</td>
      <td>Initial counter value used by the counter-based random number generation algorithm. It is a 1D tensor with at least 2 elements.</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>from</td>
      <td>Input</td>
      <td>Inclusive lower bound of the random range.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>to</td>
      <td>Input</td>
      <td>Exclusive upper bound of the random range.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Outputs random values with the specified shape.</td>
      <td>FLOAT, BF16, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

None
