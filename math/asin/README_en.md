# Asin

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: computes the arcsine of each element in the input tensor and outputs a tensor whose shape is the same as that of the input.
- Computation formula:

  $$
  y_{i} = \arcsin(x_{i})
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
      <td>x</td>
      <td>Input</td>
      <td>The input tensor for the arcsine computation, which is x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The arcsine computation result. The shape is the same as that of x, and it is y in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- For the data types, type conversion rules, and dimension constraints supported by the aclnn API, see [aclnnAsin&aclnnInplaceAsin](docs/aclnnAsin&aclnnInplaceAsin_en.md).

## Constraint Description

- The shapes of x and y are the same.
- In graph-mode (GEIR) invocation, the data types of x and y must be the same. The Tiling check returns a failure when the data types of the input and output are different. For the type conversions supported by the aclnn API, see [aclnnAsin&aclnnInplaceAsin](docs/aclnnAsin&aclnnInplaceAsin_en.md).
- For the dtype, type conversion, empty tensor, and dimension constraints at the aclnn API layer, see [aclnnAsin&aclnnInplaceAsin](docs/aclnnAsin&aclnnInplaceAsin_en.md).

## Invocation Description

| Invocation Method | Sample Code | Description |
| :--- | :--- | :--- |
| aclnn invocation | [test_aclnn_asin](examples/test_aclnn_asin.cpp), [test_aclnn_asin_fp16](examples/test_aclnn_asin_fp16.cpp) | Invokes the Asin operator through the [aclnnAsin](docs/aclnnAsin&aclnnInplaceAsin_en.md) API. |
| aclnn in-place invocation | - | Invokes the Asin operator through the [aclnnInplaceAsin](docs/aclnnAsin&aclnnInplaceAsin_en.md) API. |
| Graph-mode invocation | [test_geir_asin](examples/test_geir_asin.cpp), [test_geir_asin_const_folding](examples/test_geir_asin_const_folding.cpp) | Invokes the Asin operator by building a graph through the [operator IR](op_graph/asin_proto.h). |
