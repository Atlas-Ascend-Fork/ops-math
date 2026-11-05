# CosineSimilarity

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: computes the cosine similarity of two input tensors along the specified dimension. The cosine similarity measures the cosine of the angle between two vectors, falls within the range [-1, 1], and is commonly used to measure how similar two vectors are.

- Computation formula:

$$
\text{cosine\_similarity}(x_1, x_2, \text{dim}, \text{eps}) = \frac{\sum(x_1 \cdot x_2, \text{dim})}{\max(\sqrt{\sum(x_1^2, \text{dim})}, \text{eps}) \cdot \max(\sqrt{\sum(x_2^2, \text{dim})}, \text{eps})}
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
      <td>input_x1</td>
      <td>Input</td>
      <td>The first input tensor, whose rank ranges from 1 to 8.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_x2</td>
      <td>Input</td>
      <td>The second input tensor, which must have exactly the same shape as input_x1.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_y</td>
      <td>Output</td>
      <td>The output tensor after reduction along the dim dimension. Its shape is the input shape with the dimension corresponding to dim removed.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td>The dimension along which the cosine similarity is computed. The value range is [-rank, rank-1], and the default value is 1.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td>A non-negative numerical stability parameter that prevents division by zero. The default value is 1e-8.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- Only the FLOAT (float32) data type is supported for the inputs and output.
- The two inputs must have the same rank, and the rank ranges from 1 to 8. 0D tensors are not supported.
- Each dimension of the two inputs must have the same length, that is, `input_x1.shape == input_x2.shape`.
- No form of broadcasting is supported, including expanding dimensions of length 1 and left-padding 1s for inputs of different ranks.
- A shape that contains a dimension of length 0 is not supported.
- `dim` must be within the `[-rank, rank-1]` range for the input rank. A negative index is processed as the corresponding positive dimension, for example, `dim=-1` indicates the last dimension.
- The output shape is the common shape of the two inputs with the dimension corresponding to `dim` removed.
- `eps` must be a non-negative value.
- If both inputs have a rank of 1, `dim` must be explicitly specified within the `[-1, 0]` range.

## Invocation Description

| Invocation Method   | Sample Code | Description  |
| ------------ | ------------ | ------------ |
| Graph-mode invocation | [test_geir_cosine_similarity](./examples/test_geir_cosine_similarity.cpp) | Invokes the CosineSimilarity operator by building a graph through the [operator IR](./op_graph/cosine_similarity_proto.h) |
