# UnfoldGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    ×     |
| <term>Atlas training products</term>                       |    ×     |

## Function Description

- Operator function: implements the backward function of the Unfold operator, and computes the corresponding gradients.

  The Unfold operator computes all slices of size $size$ along dimension $dim$ of the input self. The step between two slices is given by $step$. If $sizedim$ is the size of dimension $dim$ of the input self, the size of dimension $dim$ in the returned tensor will be $(sizedim-size)/step+1$. An additional dimension of size $size$ is appended to the returned tensor.

  The shape of the input gradOutput of the UnfoldGrad operator is the shape of the forward output of Unfold. The input inputSizes is the shape of the forward input self of Unfold. The shape of the output gradIn of the UnfoldGrad operator is the shape of the forward input self of Unfold.

  Example:

  ```text
  >>> x = torch.arange(1., 8)
  >>> x
  tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
  >>> x.unfold(0, 2, 1)
  tensor([[ 1.,  2.],
          [ 2.,  3.],
          [ 3.,  4.],
          [ 4.,  5.],
          [ 5.,  6.],
          [ 6.,  7.]])
  >>> grad = x.unfold(0, 2, 2)
  tensor([[ 1.,  2.],
          [ 3.,  4.],
          [ 5.,  6.]])
  >>> res = torch.ops.aten.unfold_backward(grad, [7], 0, 2, 2)
  tensor([1, 2, 3, 4, 5, 6, 0])
  ```

## Parameter Description

| Parameter Name     | Input/Output/Attribute | Description                                                         | Data Type                 | Data Format |
| ---------- | -------------- | ------------------------------------------------------------ | ------------------------ | -------- |
| gradOutput | Input tensor       | Represents the gradient update coefficients. The shape is (..., (sizedim-size)/step+1, size). The dim-th dimension of gradOutput must equal $(inputSizes[dim]-size)/step+1$, and the size of gradOut must equal the size of inputSizes plus 1. | FLOAT, FLOAT16, BFLOAT16 | ND       |
| inputSizes | Input array       | Represents the shape of the output tensor, with the value (..., sizedim). The size of inputSizes is less than or equal to 8. | INT64                    | ND       |
| dim        | Input           | $dim$ in the formula. Indicates the dimension in which the unfold occurs. $dim$ is greater than or equal to 0 and less than len($inputSizes$). | INT64                    | -        |
| size       | Input           | $size$ in the formula. Indicates the size of each unfolded slice. $size$ must satisfy that $size$ is greater than 0 and $size$ is less than or equal to the $dim$-th dimension of $inputSizes$. | INT64                    | -        |
| step       | Input           | $step$ in the formula. Indicates the step between slices. $step$ must be greater than 0. | INT64                    |          |
| gradIn     | Output           | Represents the corresponding gradient of Unfold. The data type must be the same as that of gradOut.            | FLOAT, FLOAT16, BFLOAT16 |          |

## Constraint Description

1. The shape of gradOut satisfies the following constraints:
    - The dim-th dimension of gradOut equals (inputSizes[dim]-size)/step+1.
    - The size of gradOut equals the size of inputSizes plus 1.
2. Requirements for dim, size, and step:
    - size is greater than 0 and less than or equal to the dim-th dimension of inputSizes.
    - step is greater than 0.
    - dim is greater than or equal to 0 and less than len(inputSizes).

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_unfold_grad](tests/ut/op_kernel/test_unfold_grad.cpp) | Invokes the UnfoldGrad operator through the [aclnnUnfoldGrad](docs/aclnnUnfoldGrad_en.md) API. |
