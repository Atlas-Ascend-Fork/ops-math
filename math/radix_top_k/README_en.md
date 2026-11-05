# RadixTopK

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                      |    ×    |
| <term>Atlas A3 products</term>      |    √    |
| <term>Atlas A2 products</term>      |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                              |    ×    |
| <term>Atlas training products</term>                              |    ×    |

## Function Description

- API function: returns the k extreme values and indices of the input tensor along the specified dimension. The RadixTopK operator supports specifying the value of k through an input tensor.

- Computation description:
  - When largest=True, returns the k largest values and their indices along the specified dimension
  - When largest=False, returns the k smallest values and their indices along the specified dimension

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
      <td>Input tensor for the TopK computation, supporting 1 to 8 dimensions.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>Input</td>
      <td>0D scalar tensor, indicating the number of extreme values output along the computation dimension. The value range is [0, x.size(dim)].</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>Output</td>
      <td>Output values of the TopK computation. The data type is the same as that of x. The sorting axis of the shape is the same as that of k, and non-sorting axes are the same as those of x.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Output</td>
      <td>Output indices of the TopK computation. The data type is specified by the indices_dtype attribute. The sorting axis of the shape is the same as that of k, and non-sorting axes are the same as those of x.</td>
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
  </tbody></table>

## Constraint Description

- This operator is implemented by selection. The output is the index order of the selected elements on the sorting axis, and orderliness is not guaranteed.
