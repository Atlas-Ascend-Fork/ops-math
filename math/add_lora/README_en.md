# AddLora

English | [简体中文](./README.md)

## Product Support Status

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     ×    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- Operator function:

  Multiplies the input x by the corresponding weightA and weightB respectively based on the input index indices, then accumulates the results to the input y and outputs the result.

- Computation formula:

  Given the input tensor x whose last dimension has a length of 2d, the AddLora function performs the following computation:

  1. Rearranges x based on the indices in indices, so that the x values corresponding to the same group of weights are placed together.

  2. Iterates over each LoRa group, and performs matrix multiplication between the corresponding x and weightA:

     $$
     Z1 = x_{i} \cdot weightA[i, layerIdx, :, :]
     $$
  3. Performs matrix multiplication between the obtained Z1 and weightB:

     $$
     Z2 = Z1 \cdot weightB[i, layerIdx, :, :] \times scale
     $$
  4. Finally accumulates the Z2 output to y:

     $$
     \text{out} = y[:, yOffset: yOffset+ySliceSize] + Z2
     $$

## Parameter Description

<table style="table-layout: fixed; width: 855px"><colgroup>
  <col style="width: 164px">
  <col style="width: 138px">
  <col style="width: 300px">
  <col style="width: 133px">
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
      <td>y</td>
      <td>Input</td>
      <td>The input y in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>The input x in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weightB</td>
      <td>Input</td>
      <td>The input weightB in the formula.</td>
      <td>FLOAT16</td>
      <td>ND, NZ</td>
    </tr>
     <tr>
      <td>indices</td>
      <td>Input</td>
      <td>The input indices in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weightAOptional</td>
      <td>Input</td>
      <td>The input weightA in the formula.</td>
      <td>FLOAT16</td>
      <td>ND, NZ</td>
    </tr>
    <tr>
      <td>layerIdx</td>
      <td>Attribute</td>
      <td><ul><li>Indicates the layer index.</li><li>The value must be less than the second dimension L of weightB.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Attribute</td>
      <td>The scaling factor.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOffset</td>
      <td>Attribute</td>
      <td><ul><li>The offset when y is updated.</li><li>The value must be less than the second dimension H3 of y.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ySliceSize</td>
      <td>Attribute</td>
      <td><ul><li>The range when y is updated.</li><li>The value must be less than or equal to the second dimension H3 of y.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The out in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A2 products</term>: The data formats of weightB and weightAOptional support ND.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_add_lora](./examples/test_aclnn_add_lora.cpp) | Invokes the AddLora operator through the [aclnnAddLora](./docs/aclnnAddLora_en.md) API.    |
| Graph-mode invocation | [test_geir_add_lora](./examples/test_geir_add_lora.cpp)   | Invokes the AddLora operator by building a graph through the [operator IR](./op_graph/add_lora_proto.h). |
