# DropOutV3

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function Description

- Operator function: during training, randomly sets the elements in the input to zero with the probability p, and scales the output by a factor of 1/(1-p).
- Computation formula:
  $$
  out_i=\begin{cases}0,&\text { with probability }p \\\frac{1}{1-p}input_i,&\text { with probability }1-p\end{cases}
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
      <td>input</td>
      <td>Input</td>
      <td>Input elements.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>noise_shape</td>
      <td>Input</td>
      <td>Reserved parameter. Use a null pointer for this input parameter.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>Input</td>
      <td>Probability of setting elements to zero. The value range is [0, 1].</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>Input</td>
      <td>Seed of the random number, which affects the generated random number sequence.</td>
      <td>INT64, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>Offset of the random number, which affects the position in the generated random number sequence.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output data.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Output</td>
      <td>Mask data of the bit type stored in the UINT8 type.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn API | [test_aclnn_drop_out_v3](./examples/arch35/test_aclnn_drop_out_v3.cpp) | Invokes the DropOutV3 operator through the [aclnn_drop_out_v3](docs/aclnnDropoutV3_en.md) API. |
| aclnn API | [test_aclnn_drop_out_v3_tensor](./examples/arch35/test_aclnn_dropout_v3_tensor.cpp) | Invokes the DropOutV3 operator through the [aclnn_drop_out_v3_tensor](docs/aclnnDropoutV3Tensor.md) API. |
| Graph-mode invocation | [test_geir_drop_out_v3](./examples/arch35/test_geir_drop_out_v3.cpp)   | Invokes the DropOutV3 operator by building a graph through the [operator IR](./op_graph/drop_out_v3_proto.h). |
