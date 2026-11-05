# DropOutV3Grad

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √     |
| <term>Atlas A3 products</term> |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                  |    ×     |
| <term>Atlas inference products</term>                          |    ×     |
| <term>Atlas training products</term>                          |    ×     |

## Function Description

- Operator function: the Dropout operator dedicated to the backward process. During training, the elements in the input gradY are scaled up by scale or set to zero according to the values of the corresponding bits in mask.
- Computation formula:

  $$
  gradX_i = gradY_i * mask_i * scale
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
      <td>grad_y</td>
      <td>Input</td>
      <td>Input gradY_i in the formula, which is the backward gradient input. Dimensions 0-8 are supported for the shape.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>Input mask_i in the formula, which is the mask bit corresponding to the i-th element of gradY. The value is 0 (discard) or 1 (keep). The data is of the bit type stored in the UINT8 type, and the shape must be (align(number of elements of grad_y, 128)/8).</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Input scale in the formula, which is the scaling factor used to compute the scaling ratio of the output data. The range is not verified, and the computation is performed directly as gradY*mask*scale. In normal services, scale comes from the forward 1/(1-p), that is, it is 0 or greater than or equal to 1.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>Output</td>
      <td>gradX_i in the formula, which is the backward gradient output. The data type must be a data type to which grad_y can be converted, and the shape must be the same as that of grad_y.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

1. The range of scale is not verified, and any value is computed directly as gradY$\times$mask$\times$scale. In normal services, scale comes from the forward 1/(1-p), that is, it is 0 or greater than or equal to 1.
2. The data type of mask is UINT8, and its shape must satisfy the following condition:

   $$
   \text{mask\_shape} = \frac{\text{align}(\text{num}(grad\_y), 128)}{8}
   $$

3. Dimensions 0-8 are supported.

## Invocation Description

| Invocation Method  | Invocation Sample                                                                         | Description                                                                                |
| --------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| aclnn invocation | [test_aclnn_drop_out_v3_grad](./examples/arch35/test_aclnn_drop_out_v3_grad.cpp) | Invokes the DropOutV3Grad operator through the [aclnnDropoutV3Grad](docs/aclnnDropoutV3Grad_en.md) API. |
