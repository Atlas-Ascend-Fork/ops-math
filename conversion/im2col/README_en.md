# Im2col

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: image to columns. Sliding local window data is converted into column vectors and concatenated into a large tensor. Sliding windows are extracted from a batched input tensor. Consider a batched input tensor with shape (N, C, H, W) or (C, H, W), where N is the batch dimension, C is the channel dimension, and H, W indicate the image size. This operation flattens each sliding block of kernel_size within the spatial dimensions of the input into a column (that is, the last dimension) of a 3-D output tensor of (N, C $\times \prod$(kernel_size), L) or a 2-D output tensor of (C $\times \prod$(kernel_size), L), and L is the total number of such blocks.
- Computation formula:
$L = \prod_{d} \lfloor \frac{spatial\_size[d] + 2 \times padding[d] - dilation[d] \times (kernel\_size[d] -1)-1}{stride[d]} + 1 \rfloor$, where spatial_size is composed of H and W of the aforementioned input tensor.

## Parameter Description

<table style="table-layout: fixed; width: 966px"><colgroup>
<col style="width: 144px">
<col style="width: 166px">
<col style="width: 290px">
<col style="width: 264px">
<col style="width: 102px">
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
    <td>self</td>
    <td>Input tensor</td>
    <td>Input tensor with a 3-dimensional or 4-dimensional shape.</td>
    <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BFLOAT16, FLOAT16, FLOAT, DOUBLE, BOOL, COMPLEX32, COMPLEX64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>kernelSize</td>
    <td>Input array</td>
    <td>Kernel size. The size is 2. kernelSize[0] indicates the 'H' direction, and kernelSize[1] indicates the 'W' direction.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dilation</td>
    <td>Input array</td>
    <td>Dilation parameter. The size is 2. dilation[0] indicates the 'H' direction, and dilation[1] indicates the 'W' direction.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>Input array</td>
    <td>Padding size. The size is 2. padding[0] indicates the H direction, and padding[1] indicates the W direction.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stride</td>
    <td>Input array</td>
    <td>Stride. The size is 2. stride[0] indicates the H direction, and stride[1] indicates the W direction.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>Output tensor</td>
    <td>Output tensor whose shape is derived from the parameters.</td>
    <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BFLOAT16, FLOAT16, FLOAT, DOUBLE, BOOL, COMPLEX32, COMPLEX64</td>
    <td>ND</td>
  </tr>
  </tbody>
  </table>

  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: only FLOAT, FLOAT16, and BFLOAT16 are supported.

## Constraint Description

- The input tensor must be 3-dimensional or 4-dimensional.
- The size of kernelSize, dilation, padding, and stride must be 2.
- The values of kernelSize, dilation, and stride must be greater than 0.
- The value of padding cannot be less than 0.

## Invocation Description

| Invocation Method  | Sample Code                                              | Description                                                         |
| --------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| aclnn invocation | [test_aclnn_im2col](./examples/test_aclnn_im2col.cpp) | Invokes the Im2col operator through the [aclnnIm2col](./docs/aclnnIm2col_en.md) API. |
| Graph-mode invocation | [test_geir_im2col](./examples/test_geir_im2col.cpp) | Invokes the Im2col operator by building a graph through the [operator IR](./op_graph/im2col_proto.h). |
