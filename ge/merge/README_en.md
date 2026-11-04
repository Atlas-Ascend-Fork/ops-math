# Merge

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

Merge is a basic control flow operator. It forwards the first tensor in the input `x` that becomes available to the output `y`, and gives the index of this tensor in the inputs in `value_index`. Merge usually works with the [Switch](../switch/README_en.md) operator to implement branch convergence: Switch distributes data to different branches based on conditions, and then Merge converges the data of the available branches back to the main flow.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 996px"><colgroup>
  <col style="width: 102px">
  <col style="width: 168px">
  <col style="width: 203px">
  <col style="width: 403px">
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
      <td>Dynamic input</td>
      <td>Input tensors, one of which becomes available. All input tensors should use the same data type; shape inference merges the dimension information of all inputs.</td>
      <td>float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Available tensor, whose data type is consistent with that of x.</td>
      <td>float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_index</td>
      <td>Output</td>
      <td>Scalar, which is the index of the selected input tensor in the input list.</td>
      <td>DT_INT32</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- The number of inputs `x` must be greater than or equal to 1; otherwise, shape inference fails.
- The data types of all inputs `x` must be the same; data type inference checks the input types one by one and returns a failure if they are inconsistent. The data type of the output `y` is the same as that of the inputs, and `value_index` is `DT_INT32`.
- If the numbers of dimensions of inputs `x` are inconsistent or there is an unknown-rank input, the output `y` is inferred as unknown rank (`[-2]`), and inference failure is not returned due to inconsistent ranks.
- For a dimension, if the values of the inputs are inconsistent or there is an unknown dimension (-1), the dimension is inferred as -1 (unknown).

## Invocation Description

The Merge operator is a basic control flow operator. It is usually used as a branch convergence node in a graph and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_merge](./examples/test_geir_merge.cpp) | Invokes the Merge operator by building a graph through the [operator IR](./op_graph/merge_proto.h) |
