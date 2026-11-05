# Tile

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

API function: replicates the input tensor along each dimension for the number of times specified in repeats. Example:
Assume that the input tensor is [[a,b],[c,d],[e,f]], that is, the shape is [3,2], and repeats is (2,4). Then the generated tensor has the shape [6,8], and the values are as follows:

```text
>>> x = torch.tensor([[a,b],[c,d],[e,f]])
>>> x.repeat(2,4)
tensor([[a,b,a,b,a,b,a,b],
        [c,d,c,d,c,d,c,d],
        [e,f,e,f,e,f,e,f],
        [a,b,a,b,a,b,a,b],
        [c,d,c,d,c,d,c,d],
        [e,f,e,f,e,f,e,f],
        ])
```

When repeats is (2,4,2), that is, the number of elements in repeats is greater than the number of dimensions in the tensor, the output tensor is equivalent to the following operations: first, expand the shape of the input tensor to the number of dimensions equal to the number of elements in repeats: [1,3,2]; then, expand the tensor along each dimension according to the value in repeats, that is, the output tensor has the shape [2,12,4]. The result is as follows:

```text
>>> x.repeat(2,4,2)
tensor([[[a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f]],

        [[a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f]]])
```

The following conditions must be met during computation:
The number of parameters in repeats cannot be less than the number of dimensions of the input tensor.
The values in repeats must be greater than or equal to 0.

## Parameter Description

  <table style="table-layout: fixed; width: 1528px"><colgroup>
  <col style="width: 132px">
  <col style="width: 120px">
  <col style="width: 256px">
  <col style="width: 253px">
  <col style="width: 333px">
  <col style="width: 126px">
  <col style="width: 160px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Instructions</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>-</td>
      <td>-</td>
      <td>FLOAT, DOUBLE, FLOAT16, COMPLEX64, COMPLEX128, UINT8, INT8, INT16, INT32, INT64, UINT16, UINT32, UINT64, BOOL, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>≤8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>repeats</td>
      <td>Input</td>
      <td>-</td>
      <td>Indicates the number of times to repeat the input tensor along each dimension. The number of parameters is not greater than 8. Currently, the scenario of repeating more than 4 dimensions at the same time is not supported. For details about the constraints, see Constraint Description.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>-</td>
      <td>-</td>
      <td>Same as self</td>
      <td>ND</td>
      <td>≤8</td>
      <td>√</td>
    </tr>
  </tbody></table>

## Constraint Description

- Deterministic computation:
  - By default, the implementation is deterministic.

## Invocation Example

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_repeat](./examples/test_aclnn_repeat.cpp) | Invokes the Tile operator through the [aclnnRepeat](docs/aclnnRepeat_en.md) API. |
| Graph-mode invocation | [test_geir_tile](./examples/test_geir_tile.cpp) | Invokes the Tile operator by building a graph through the [operator IR](./op_graph/tile_proto.h). |
