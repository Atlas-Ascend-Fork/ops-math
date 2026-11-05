# ReduceAll

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                           |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- API function: for each dimension in the given dimension `dim`, if all elements corresponding to that dimension in the input tensor compute to True, True is returned; otherwise, False is returned. If `keepdim` is True, the dimensions of the output tensor are the same as those of the input. Otherwise, the `dim` dimensions are squeezed, reducing the output tensor by `len(dim)` dimensions.
- For example, the shape of the input tensor is $(A\times B \times C \times D)$, and the value of `dim` is [0, 2]. If `keepdim` is False, the shape of the output tensor is $(B \times D)$, and the output tensor has two fewer dimensions than the input. If `keepdim` is True, the shape of the output tensor is $(1 \times B \times 1 \times D)$, and the dimensions of the output tensor are the same as those of the input.

## Parameter Description

 <table style="table-layout: fixed; width: 1499px"><colgroup>
    <col style="width: 153px">
    <col style="width: 120px">
    <col style="width: 269px">
    <col style="width: 283px">
    <col style="width: 119px">
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
        <td>BOOL, UINT8, INT8, INT16, INT32, INT64, FLOAT, FLOAT16, BFLOAT16, DOUBLE</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>Dimensions to reduce.</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepdim</td>
        <td>Input</td>
        <td>Specifies whether the dimensions of the reduce axis are retained.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output tensor.</td>
        <td>BOOL, UINT8</td>
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
| Graph-mode invocation  | [test_geir_reduce_all.cpp](examples/test_geir_reduce_all.cpp) | Invokes the ReduceAll operator in GE IR graph mode. |
| aclnn invocation  | [test_aclnn_all.cpp](examples/test_aclnn_all.cpp) | Invokes the ReduceAll operator through the [aclnnAll](docs/aclnnAll_en.md) API. |
