# MemSetV2

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
|<term>Ascend 950PR&950DT products</term>     |    √     |
|<term>Atlas A3 products</term>|×|
|<term>Atlas A2 products</term>|×|
|<term>Atlas 200I/500 A2 inference products</term>|×|
|<term>Atlas inference products</term>|×|
|<term>Atlas training products</term>|×|

## Function Description

- Operator function: initializes the output and workspace specified by downstream operators to specified values.

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
      <td>Tensor to be initialized, which is passed by the framework.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64<br>
      BF16, FLOAT16, FLOAT, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values_int</td>
      <td>Attribute</td>
      <td>Specifies the initial value of the int type of the tensor at the corresponding position.</td>
      <td>int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>values_float</td>
      <td>Attribute</td>
      <td>Specifies the initial value of the float type of the tensor at the corresponding position.</td>
      <td>float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>Tensor to be initialized, which is passed by the framework. The output of this operator is the input, and initialization is performed in place.</td>
      <td>Same as the input Tensor x</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- None.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_mem_set_v2](./examples/test_geir_mem_set_v2.cpp)   | Invokes the MemSetV2 operator by building a graph through the [operator IR](./op_graph/mem_set_v2_proto.h). |
