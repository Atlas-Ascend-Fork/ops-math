# ViewCopy

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: copies the src tensor as the view described by src_size, src_stride, and src_storage_offset to the dst tensor as the view described by dst_size, dst_stride, and dst_storage_offset, and returns dst.
- Computation formula:

  $$
  \text{dst}\left[\text{dst\_storage\_offset} + \sum_{i=0}^{n-1} \text{dst\_index}_i \cdot \text{dst\_stride}_i\right]
  =
  \text{src}\left[\text{src\_storage\_offset} + \sum_{i=0}^{n-1} \text{src\_index}_i \cdot \text{src\_stride}_i\right]
  $$

  where $n$ is the number of view dimensions; $\text{dst\_index}_i$ and $\text{src\_index}_i$ are the indices of the dst and src views in the $i$-th dimension respectively, with the value ranges $0 \le \text{dst\_index}_i < \text{dst\_size}_i$ and $0 \le \text{src\_index}_i < \text{src\_size}_i$. Since dst_size and src_size have the same shape, the indices of corresponding dimensions are element-wise equal; $\text{dst\_storage\_offset}$ and $\text{src\_storage\_offset}$ are the view start offsets; $\text{dst\_stride}_i$ and $\text{src\_stride}_i$ are the strides of the $i$-th dimension. The content in square brackets is the element subscript in the tensor storage area.

## Parameter Description

<table style="table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 180px">
  <col style="width: 130px">
  <col style="width: 380px">
  <col style="width: 290px">
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
      <td>dst</td>
      <td>Input</td>
      <td>Destination tensor to be written to. Its view is described by dst_size, dst_stride, and dst_storage_offset.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, BOOL, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_size</td>
      <td>Input</td>
      <td>View shape of the dst tensor, a 1D tensor, whose elements are non-negative and whose number of elements is the same as that of dst_stride.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_stride</td>
      <td>Input</td>
      <td>View stride of the dst tensor, a 1D tensor, whose elements are non-negative and whose number of elements is the same as that of dst_size.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_storage_offset</td>
      <td>Input</td>
      <td>View start offset of the dst tensor, a 1D tensor, whose elements are non-negative and do not exceed the element range of the dst tensor.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src</td>
      <td>Input</td>
      <td>Source tensor to be read. Its view is described by src_size, src_stride, and src_storage_offset.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, BOOL, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src_size</td>
      <td>Input</td>
      <td>View shape of the src tensor, a 1D tensor, whose elements are non-negative and whose number of elements is the same as that of src_stride.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src_stride</td>
      <td>Input</td>
      <td>View stride of the src tensor, a 1D tensor, whose elements are non-negative and whose number of elements is the same as that of src_size.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src_storage_offset</td>
      <td>Input</td>
      <td>View start offset of the src tensor, a 1D tensor, whose elements are non-negative and do not exceed the element range of the src tensor.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst</td>
      <td>Output</td>
      <td>Copy result, which shares the same memory as the input dst (in-place write).</td>
      <td>BFLOAT16, FLOAT16, FLOAT, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, BOOL, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- dst_size and src_size have the same shape.
- The number of dimensions of dst_size is the same as that of dst_stride; the number of dimensions of src_size is the same as that of src_stride.
- The elements of dst_size and src_size are non-negative.
- The elements of dst_stride and src_stride are non-negative.
- dst_storage_offset and src_storage_offset are non-negative and do not exceed the element range of the corresponding tensors.
- For each axis i, dst_storage_offset + (dst_size[i] - 1) * dst_stride[i] does not exceed the number of elements of the dst tensor. The same applies to src.
- For the dtype, type conversion, and determinism constraints at the aclnn API layer, see [aclnnInplaceCopy](docs/aclnnInplaceCopy_en.md).

## Invocation Description

| Invocation Method | Sample Code | Description |
| :--- | :--- | :--- |
| aclnn invocation | [test_aclnn_view_copy](examples/test_aclnn_view_copy.cpp) | Invokes the ViewCopy operator through the [aclnnInplaceCopy](docs/aclnnInplaceCopy_en.md) API. |
| Graph-mode invocation | - | Invokes the ViewCopy operator by building a graph through the [operator IR](op_graph/view_copy_proto.h). |
