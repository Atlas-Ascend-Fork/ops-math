# StatelessRandomChoiceWithMask

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
    Selects the corresponding index list based on the elements whose values are true in the input x. Then, it calls philox_random with seed and offset to generate a number of random numbers equal to the length of the index list, and uses the random numbers to apply the shuffling algorithm to the index list, obtaining the randomly shuffled index list. count items are returned.
    If count is 0, the actual number of items in the index list is output. If the list contains fewer than count items, it is padded with 0 to count items. The output mask marks whether each index is a valid index.

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
      <td>Input x value.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>count</td>
      <td>Input</td>
      <td>Expected number of output indexes.</td>
      <td>INT32</td>
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
      <td>out</td>
      <td>Output</td>
      <td>Indexes of the shuffled elements whose values are true in the input x.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Output</td>
      <td>Marks whether the index at each position is a valid index.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                    | Description                                                           |
|--------------|---------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_stateless_random_choice_with_mask](./examples/test_geir_stateless_random_choice_with_mask.cpp) | Invokes the StatelessRandomChoiceWithMask operator by building a graph through the [operator IR](./op_graph/stateless_random_choice_with_mask_proto.h). |
