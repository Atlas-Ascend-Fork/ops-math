# ReduceAny

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                         |    √     |
| <term>Atlas A3 products</term>      |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

For each dimension in the given dimension `dim`, if any element corresponding to that dimension in the input tensor computes to True, True is returned; otherwise, False is returned. If `keepdim` is True, the dimensions of the output tensor are the same as those of the input. Otherwise, the `dim` dimensions are squeezed, reducing the output tensor by `len(dim)` dimensions.

For example, the shape of the input tensor is $(A\times B \times C \times D)$, and the value of `dim` is [0, 2]. Then the shape of the output tensor is $(B \times D)$, and the output tensor has two fewer dimensions than the input.

## Parameter Description

<table style="table-layout: fixed; width: 1494px"><colgroup>
    <col style="width: 154px">
    <col style="width: 125px">
    <col style="width: 259px">
    <col style="width: 334px">
    <col style="width: 124px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Data Type</th>
        <th>Data Format</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self</td>
        <td>Input</td>
        <td>Input tensor.</td>
        <td>BOOL, INT8, UINT8, INT16, INT32, INT64, BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>Dimensions to squeeze.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepDim</td>
        <td>Input</td>
        <td>Specifies whether the dimensions of the reduce axis are retained.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output tensor.</td>
        <td>BOOL</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace that needs to be allocated on the Device side.</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation process.</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
    </table>

- <term>Atlas inference products</term>, <term>Atlas training products</term>, and <term>Atlas 200I/500 A2 inference products</term>: the data type BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method   | Sample Code                                                                         | Description                                               |
| ---------------- |------------------------------------------------------------------------------|--------------------------------------------------|
| aclnn API  | [test_aclnn_any.cpp](examples/test_aclnn_any.cpp) | Invokes the ReduceAny operator through the [aclnnAny](docs/aclnnAny_en.md) API. |
