# TopKV2

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                      |    √    |
| <term>Atlas A3 products</term>      |    √    |
| <term>Atlas A2 products</term>      |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √    |
| <term>Atlas inference products</term>                              |    √    |
| <term>Atlas training products</term>                              |    √    |

## Function Description

- API function: returns the k extreme values and their indices of the input tensor on the specified dimension. The TopKV2 operator supports specifying the value of k through an input tensor and supports a configurable output data type of indices.

- Computation description:
  - When largest=True, the k largest values on the specified dimension and their indices are returned.
  - When largest=False, the k smallest values on the specified dimension and their indices are returned.
  - When sorted=True, the output results are sorted in descending order (largest=True) or ascending order (largest=False).
  - When sorted=False, the output results are not sorted.

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
      <td>Input tensor for the TopK computation. Dimensions 1-8 are supported.</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>Input</td>
      <td>0D scalar tensor that indicates the number of extreme values to output on the computation dimension. The value range is [0, x.size(dim)].</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>Output</td>
      <td>Output values of the TopK computation. The data type is the same as that of x. The shape is the same as k on the sorting axis and the same as x on non-sorting axes.</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Output</td>
      <td>Output indices of the TopK computation. The data type is specified by the indices_dtype attribute. The shape is the same as k on the sorting axis and the same as x on non-sorting axes.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted</td>
      <td>Attribute</td>
      <td>Optional Boolean. The default value is True. True indicates that the output results are sorted, and False indicates that the output results are not sorted.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td>Optional integer. The default value is -1. It indicates the computation dimension. The value range is [-x.dim(), x.dim()).</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>largest</td>
      <td>Attribute</td>
      <td>Optional Boolean. The default value is True. True indicates that the k largest elements are returned, and False indicates that the k smallest elements are returned.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>indices_dtype</td>
      <td>Attribute</td>
      <td>Optional integer. The default value is DT_INT32(3). It indicates the data type of the output indices. DT_INT32(3) or DT_INT64(9) is supported.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sort_policy</td>
      <td>Attribute</td>
      <td>Optional integer. The default value is 0. It indicates the sorting policy for TopK candidate results. 0 indicates that the original sorting behavior is used, and 1 indicates that the Bitonic Small TopK path is enabled in the scenario where sorted=True and k is small (2≤k≤32).</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>: The BFLOAT16 data type is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_top_k_v2](./examples/test_geir_topkv2.cpp)   | Invokes the top_k_v2 operator by building a graph through the [operator IR](./op_graph/top_k_v2_proto.h). |
