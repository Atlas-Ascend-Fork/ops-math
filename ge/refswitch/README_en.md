# RefSwitch

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

RefSwitch is a basic control flow operator. Its function is the same as [Switch](../switch/README_en.md): it forwards the input tensor `data` to the corresponding output port based on the Boolean scalar input `pred`. When `pred` is `true`, the data is forwarded to `output_true`; otherwise, it is forwarded to `output_false`. The shapes and data types of both outputs are the same as those of the input `data`. The difference is that RefSwitch is used for reference (ref) type data, and usually works with the [RefMerge](../ref_merge/README_en.md) operator to implement branch control.

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
      <td>data</td>
      <td>Input</td>
      <td>The reference tensor to be forwarded.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pred</td>
      <td>Input</td>
      <td>A Boolean scalar that determines the output port to which the data is forwarded.</td>
      <td>DT_BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_false</td>
      <td>Output</td>
      <td>When pred is false, data is forwarded to this output. The shape and data type are the same as those of data.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_true</td>
      <td>Output</td>
      <td>When pred is true, data is forwarded to this output. The shape and data type are the same as those of data.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- The input `pred` must be a Boolean scalar (rank 0 or unknown rank). Otherwise, shape inference fails.
- The data type of the input `pred` must be `DT_BOOL`. Otherwise, data type inference fails.

## Invocation Description

The RefSwitch operator is a basic control flow operator. It is usually used as a branch selection node in a graph, and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_refswitch](./examples/test_geir_refswitch.cpp) | Invokes the RefSwitch operator by building a graph through the [operator IR](./op_graph/refswitch_proto.h). |
