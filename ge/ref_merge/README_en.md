# RefMerge

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

RefMerge is a basic control flow operator. Its function is the same as [Merge](../merge/README_en.md): it forwards the first tensor that becomes available in the input `x` to the output `y`, and gives the index of the tensor in the inputs in `value_index`. The difference is that RefMerge is used for reference (ref) type data. RefMerge usually works with the [RefSwitch](../refswitch/README_en.md) operator to converge branches.

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
      <td>Input tensors, one of which becomes available. All input tensors should use the same data type. Shape inference merges the dimension information of all inputs.</td>
      <td>float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The available tensor, whose data type is the same as that of x.</td>
      <td>float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_index</td>
      <td>Output</td>
      <td>A scalar, which is the index of the selected input tensor in the input list.</td>
      <td>DT_INT32</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- The number of inputs `x` must be greater than or equal to 1. Otherwise, shape inference fails.
- The data types of all inputs `x` must be the same. Data type inference checks the type of each input one by one, and returns a failure if the types are inconsistent. The data type of the output `y` is the same as that of the inputs, and `value_index` is `DT_INT32`.
- If the numbers of dimensions of the inputs `x` are inconsistent or an input of unknown rank exists, the output `y` is inferred as unknown rank (`[-2]`), and inference does not fail due to inconsistent ranks.
- For a dimension, if the values of the inputs are inconsistent or an unknown dimension (-1) exists, the dimension is inferred as -1 (unknown).

## Invocation Description

The RefMerge operator is a basic control flow operator. It is usually used as a branch convergence node in a graph, and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_ref_merge](./examples/test_geir_ref_merge.cpp) | Invokes the RefMerge operator by building a graph through the [operator IR](./op_graph/ref_merge_proto.h). |
