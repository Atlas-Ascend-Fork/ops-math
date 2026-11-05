# SquareSumAll

English | [简体中文](./README.md)

## Product Support Status

| Product                                                      | Supported |
| :-------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                    |    √     |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>   |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                   |    √     |
| <term>Atlas inference products</term>                           |    √     |
| <term>Atlas training products</term>                           |    √     |

## Function Description

- Operator function: computes the sum of squares of all elements of the two input tensors `x1` and `x2` that have the same shape, and outputs two scalars `y1` and `y2`.

- Computation formula:

$$
y_1=\sum_{i=0}^{N-1}x_{1,i}^{2},\qquad
y_2=\sum_{i=0}^{N-1}x_{2,i}^{2}
$$

Here, $N$ is the total number of elements in the input tensor.

## Parameter Description

<table style="table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 420px">
  <col style="width: 120px">
  <col style="width: 410px">
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
      <td>x1</td>
      <td>Input</td>
      <td>Denotes the first tensor for which the sum of squares is to be computed, corresponding to x1 in the formula.</td>
      <td>FLOAT</td>
      <td>FRACTAL_Z, C1HWNCoC0, NC1HWC0, ND, NCHW, NHWC</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Denotes the second tensor for which the sum of squares is to be computed, corresponding to x2 in the formula.</td>
      <td>FLOAT</td>
      <td>FRACTAL_Z, C1HWNCoC0, NC1HWC0, ND, NCHW, NHWC</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td><ul><li>Denotes the sum of squares of all elements of x1, corresponding to y1 in the formula.</li><li>The output is a 0-dimension Tensor (rank 0) containing one FLOAT element.</li></ul></td>
      <td>FLOAT</td>
      <td>FRACTAL_Z, C1HWNCoC0, NC1HWC0, ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td><ul><li>Denotes the sum of squares of all elements of x2, corresponding to y2 in the formula.</li><li>The data type and shape are the same as those of the output y1.</li></ul></td>
      <td>FLOAT</td>
      <td>FRACTAL_Z, C1HWNCoC0, NC1HWC0, ND</td>
    </tr>
  </tbody></table>

### Product Difference Description

The "input format → output format" combinations in the following table apply to both x1→y1 and x2→y2, and the two input paths must use the same data format.

| Product | Static shape format combinations | Dynamic shape format combinations | Rank restrictions |
| :--- | :--- | :--- | :--- |
| <term>Ascend 950PR&950DT products</term> | ND→ND, NCHW→ND, NHWC→ND | ND→ND, NCHW→ND, NHWC→ND | The rank of ND inputs ranges from [0, 8]; the rank of NCHW and NHWC inputs must be 4. |
| <term>Atlas A3 products</term><br><term>Atlas A2 products</term><br><term>Atlas 200I/500 A2 inference products</term><br><term>Atlas inference products</term><br><term>Atlas training products</term> | FRACTAL_Z→FRACTAL_Z, C1HWNCoC0→C1HWNCoC0, NC1HWC0→NC1HWC0, ND→ND | ND→ND | The rank of inputs ranges from [0, 8]. |

## Constraint Description

- During graph compilation, `-1` denotes an unknown dimension and `[-2]` denotes an unknown rank; the two outputs are always rank-0 scalars. During execution, the input shape must be parsed first, and then the following non-empty, same-shape, and format constraints must be met.

- Empty tensors are not supported for x1 or x2. A rank of 0 indicates a scalar input with one element. If the rank is greater than 0, the length of each dimension must be greater than 0.
- The execution-time rank and the length of each dimension of x1 and x2 must be exactly the same, and broadcasting is not supported. The shapes `[2, 3]` and `[6]` are invalid even if the number of elements is the same.
- All four parameters of the operator node must use the FLOAT type, and the data formats must meet the valid combinations in the product difference description.
- NaN, Inf, and overflow of the square of finite values propagate naturally according to FLOAT semantics. The two accumulators and the two outputs do not pollute each other; a non-finite value in `x1` does not affect `y2`, and vice versa.
- This operator provides only GE graph-mode registration, with the build configuration set to `aclnn_exclude`, and does not provide aclnn APIs.

## Invocation Description

| Invocation Method   | Invocation Sample                                                                            | Description                                                                                    |
| :--------- | :---------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------- |
| Graph-mode invocation | [test_geir_square_sum_all](./examples/arch35/test_geir_square_sum_all.cpp)          | Invokes the SquareSumAll operator by building a graph through the [operator IR](./op_graph/square_sum_all_proto.h).            |
