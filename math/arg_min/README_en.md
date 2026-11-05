# ArgMin

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

Returns the indices of the minimum values of a tensor along the specified axis and saves them to out. If keepdim is false, the corresponding axis is not retained; if keepdim is true, the dimension value of the specified axis is retained as 1.

## Function Prototype

Each operator is divided into [a two-phase API](../../docs/en/context/two_phase_api.md). You must first call the "aclnnArgMinGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnArgMin" API to perform the computation.

```Cpp
aclnnStatus aclnnArgMinGetWorkspaceSize(
  const aclTensor* self,
  int64_t          dim,
  bool             keepdim,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnArgMin(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnArgMinGetWorkspaceSize

- **Parameter Description**:

  <table style="table-layout: fixed; width: 1499px"><colgroup>
  <col style="width: 165px">
  <col style="width: 125px">
  <col style="width: 294px">
  <col style="width: 247px">
  <col style="width: 261px">
  <col style="width: 124px">
  <col style="width: 138px">
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
      <td>Input tensor.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Input</td>
      <td>The dimension on which the minimum value is computed.</td>
      <td>The value range is [-self.dim(), self.dim()).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdim</td>
      <td>Input</td>
      <td>Specifies whether the dimensions of the output tensor shape are the same as those of the input tensor.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>-</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas training products</term>: The data type BFLOAT16 is not supported.
  - <term>Atlas inference products</term>: The data type BFLOAT16 is not supported. When the input data type of self is INT8, INT16, INT32, INT64, or UINT8, the dimensions of the input tensor shape support a maximum of 7 dimensions.

- **Return Values**:

  aclnnStatus: returns a status code. For details, see [aclnn Return Codes](../../docs/en/context/aclnn_return_code.md).

  The first-phase API completes input parameter validation and reports an error in the following scenarios:

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 288px">
  <col style="width: 114px">
  <col style="width: 747px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed-in self or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types and data formats of self and out are not within the supported range.</td>
    </tr>
    <tr>
      <td>The deduced data type cannot be converted to the type of the specified output out.</td>
    </tr>
    <tr>
      <td>dim exceeds the dimensions of the tensor array.</td>
    </tr>
    <tr>
      <td>The shape of the passed-in out does not satisfy InferShape.</td>
    </tr>
  </tbody>
  </table>

## aclnnArgMin

- **Parameter Description**:

    <table style="table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 153px">
    <col style="width: 124px">
    <col style="width: 872px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter Name</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>The memory address of the workspace allocated on the Device side.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>The size of the workspace allocated on the Device side, obtained by the first-phase API aclnnArgMinGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>The op executor, which contains the operator computation process.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream that executes tasks.</td>
      </tr>
    </tbody>
    </table>

- **Return Values**:

  aclnnStatus: returns a status code. For details, see [aclnn Return Codes](../../docs/en/context/aclnn_return_code.md).

## Constraint Description

- Deterministic computation:
  - aclnnArgMin uses a deterministic implementation by default.

- For the input self, the scenario where the reduce axis is 0 is not supported

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_arg_min](./examples/test_aclnn_arg_min.cpp) | Invokes the ArgMin operator through the [aclnnArgMin](./docs/aclnnArgMin_en.md) API. |
