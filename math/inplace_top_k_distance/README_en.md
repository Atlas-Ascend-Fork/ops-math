# InplaceTopKDistance

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

- Operator function: merges and re-sorts a group of sorted distance data (topk_pq_distance) with a group of new distance data (pq_distance), selects the top k elements, and writes the result back to topk_pq_distance, topk_pq_index, and topk_pq_ivf in place. Here, k is the number of elements in topk_pq_distance. This operator has no output, and the computation result updates the preceding inputs in place.

- Computation formula: sorts all elements of topk_pq_distance and pq_distance in ascending order of distance values, denoted as $s_0 \le s_1 \le \dots \le s_{m-1}$, where $m$ is the total number of elements of the two tensors and $k$ is the number of elements in topk_pq_distance. Writes back the first k elements according to order, where $i=0,1,\dots,k-1$:
  - If order is "asc", selects the smallest k elements and writes them back in ascending order: $\mathrm{topk\_pq\_distance}[i]=s_i$;
  - If order is set to any other value, selects the largest k elements and writes them back in descending order: $\mathrm{topk\_pq\_distance}[i]=s_{m-1-i}$.

  The index and bucket ID (topk_pq_index and topk_pq_ivf) of each element are written back together with its distance value.

- Computation process:
  1. Forms a triplet for each element of topk_pq_distance together with its corresponding topk_pq_index and topk_pq_ivf, and forms a triplet for each element of pq_distance together with its corresponding pq_index and the scalar pq_ivf.
  2. Merges the two groups of triplets and sorts them in ascending order of distance values.
  3. Selects the first or last k triplets according to the preceding formula and writes them back.

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
      <td>topk_pq_distance</td>
      <td>Input</td>
      <td>Sorted distance tensor, which is updated in place after the computation. The number of its elements is k, the number of top data elements to select.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>topk_pq_index</td>
      <td>Input</td>
      <td>Index corresponding to topk_pq_distance, which is updated in place after the computation. The number of elements is the same as that of topk_pq_distance.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>topk_pq_ivf</td>
      <td>Input</td>
      <td>Bucket ID corresponding to topk_pq_distance, which is updated in place after the computation. The number of elements is the same as that of topk_pq_distance.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pq_distance</td>
      <td>Input</td>
      <td>New distance tensor, which will be merged and re-sorted with topk_pq_distance. Its data type must be the same as that of topk_pq_distance.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pq_index</td>
      <td>Input</td>
      <td>Index corresponding to pq_distance. The number of elements is the same as that of pq_distance.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pq_ivf</td>
      <td>Input</td>
      <td>Bucket ID corresponding to pq_distance, which is a scalar or a one-dimensional Tensor containing one element. All elements of pq_distance share this bucket ID.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>order</td>
      <td>Attribute</td>
      <td>Optional attribute, indicating the sorting mode of topk_pq_distance. If the value is "asc", the result is written back in ascending order; if the value is set to any other value, the result is written back in descending order. The default value is "asc".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>

  </tbody></table>

## Constraint Description

- If elements with equal distance values exist, the sorting is unstable, and the order between equal elements is uncertain.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_inplace_top_k_distance](./examples/test_geir_inplace_top_k_distance.cpp)   | Invokes the InplaceTopKDistance operator by building a graph through the [operator IR](./op_graph/inplace_top_k_distance_proto.h). This operator has no output, and the sample drives the execution as a graph target node. |
