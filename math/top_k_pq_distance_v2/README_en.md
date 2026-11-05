# TopKPQDistanceV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: for each batch row of the input pq_distance, takes the k largest or smallest values among the element_num elements of the row and their column indices, and outputs them to topk_distance and topk_index respectively. grouped_extreme_distance provides the extreme value of each group and is used only to accelerate pruning, without changing the computation result.

- Computation formula: let the elements of a batch row sorted in ascending order be $s_0 \le s_1 \le \dots \le s_{n-1}$, where $n$ is element_num. Take the extreme values according to order, where $i=0,1,\dots,k-1$:
  - When order is "ASC", the k smallest values are taken and output in ascending order: $\mathrm{topk\_distance}[i]=s_i$;
  - When order is "DES", the k largest values are taken and output in descending order: $\mathrm{topk\_distance}[i]=s_{n-1-i}$.

  $\mathrm{topk\_index}[i]$ is the original column index of the corresponding element in the pq_distance of that row.

- Computation process: the preceding result is quickly obtained with the help of grouped extreme-value pruning.
  1. Select several groups that most likely contain the top k extreme values based on grouped_extreme_distance, and sort the groups by quality.
  2. Initialize a heap of size k with the elements in these groups, then traverse the remaining groups. With grouped extreme-value pruning, the heap is updated only when a better result is possible.
  3. Pop the elements from the heap in sequence to obtain the ordered topk_distance and topk_index.

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>pq_distance</td>
      <td>Input</td>
      <td>Distance tensor, 2D, with the shape `[batch, element_num]`.</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grouped_extreme_distance</td>
      <td>Input</td>
      <td>Extreme value of each group, 2D, with the shape `[batch, element_num / group_size]`. When order is "ASC", the minimum value of each group is taken; when order is "DES", the maximum value of each group is taken. The data type must be the same as that of pq_distance.</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>topk_distance</td>
      <td>Output</td>
      <td>k largest or smallest element values on the last dimension, with the shape `[batch, k]`. The data type is the same as that of pq_distance.</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>topk_index</td>
      <td>Output</td>
      <td>Indices of the elements in topk_distance on the last dimension of pq_distance, with the shape `[batch, k]`.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>order</td>
      <td>Attribute</td>
      <td>Optional attribute that indicates the sorting method. The value is "ASC" or "DES". The default value is "ASC".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>k</td>
      <td>Attribute</td>
      <td>Required attribute that indicates the number of largest or smallest elements to take. The value must be greater than 0 and not greater than element_num.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_size</td>
      <td>Attribute</td>
      <td>Required attribute that indicates the group size for computing grouped extreme values. The value must be greater than 0, and element_num must be an integer multiple of group_size.</td>
      <td>INT</td>
      <td>-</td>
    </tr>

  </tbody></table>

## Constraint Description

- grouped_extreme_distance must match the value of order: when order is "ASC", it should be the minimum value of each group; when order is "DES", it should be the maximum value of each group. Otherwise, pruning may not produce the expected result.
- When elements with equal distance values exist, the sorting is non-stable, and the order between equal elements is uncertain.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_top_k_pq_distance_v2](./examples/test_geir_top_k_pq_distance_v2.cpp)   | Invokes the TopKPQDistanceV2 operator by building a graph through the [operator IR](./op_graph/top_k_pq_distance_v2_proto.h). |
