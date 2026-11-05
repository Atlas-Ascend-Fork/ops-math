# ArgMaxWithValue

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

Operator function: returns the maximum values of a tensor along the specified dimension and their index positions. The maximum values are saved to out, and the indices of the maximum values are saved to indices. If keepdim is false, the corresponding axis is not retained; if keepdim is true, the dimension value of the specified axis is retained as 1.

## Function Prototype

Each operator is divided into [a two-phase API](../../docs/en/context/two_phase_api.md). You must first call the "aclnnMaxDimGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnMaxDim" API to perform the computation.

- `aclnnStatus aclnnMaxDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepdim, aclTensor *out, aclTensor *indices, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnMaxDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMaxDimGetWorkspaceSize

- **Parameter Description**:

  - self(aclTensor*, computation input): an aclTensor on the Device side. Supports [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md), and the supported [data format](../../docs/en/context/data_format.md) is ND. Supports 1 to 8 dimensions.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: The supported data types are FLOAT, FLOAT16, INT64, and BOOL
     * <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The supported data types are FLOAT, FLOAT16, INT64, BOOL, and BFLOAT16
     * <term>Ascend 950PR&950DT products</term>: The supported data types are FLOAT, FLOAT16, INT64, BOOL, BFLOAT16, and INT32

  - dim(int64_t, computation input): the specified dimension. The data type is INT64, and the value range is [-self.dim(), self.dim()).

  - keepdim(bool, computation input): specifies whether to retain the dimension of the reduce axis. The data type is BOOL.

  - out(aclTensor*, computation output): an aclTensor on the Device side, and its data type is the same as that of self. Supports [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md), and the supported [data format](../../docs/en/context/data_format.md) is ND. If keepdim is false, the output dimension is the dimension of self minus 1; if keepdim is true, the output dimension is equal to the dimension of self.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: The supported data types are FLOAT, FLOAT16, INT64, and BOOL
     * <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The supported data types are FLOAT, FLOAT16, INT64, BOOL, and BFLOAT16
     * <term>Ascend 950PR&950DT products</term>: The supported data types are FLOAT, FLOAT16, INT64, BOOL, BFLOAT16, and INT32

  - indices(aclTensor*, computation output): an aclTensor on the Device side. Supports [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md), and the supported [data format](../../docs/en/context/data_format.md) is ND.
     * <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The supported data types are BOOL, FLOAT32, FLOAT16, INT8, INT16, UINT16, UINT8, INT32, INT64, UINT32, UINT64, DOUBLE, COMPLEX64, COMPLEX128, and BFLOAT16.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: The supported data types are BOOL, FLOAT32, FLOAT16, INT8, INT16, UINT16, UINT8, INT32, INT64, UINT32, UINT64, DOUBLE, COMPLEX64, and COMPLEX128.

  - workspaceSize(uint64_t*, output parameter): returns the size of the workspace that needs to be allocated on the Device side.

  - executor(aclOpExecutor**, output parameter): returns the op executor, which contains the operator computation process.

- **Return Values**:

  aclnnStatus: returns a status code. For details, see [aclnn Return Codes](../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API completes input parameter validation and reports an error in the following scenarios:
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed-in self, out, and indices are null pointers.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types of self, out, and indices are not within the supported range.
                                  2. The data formats of self, out, and indices are not within the supported range.
                                  3. dim exceeds the dimension range of the input self.
```

## aclnnMaxDim

- **Parameter Description**:

  - workspace(void*, input parameter): the memory address of the workspace allocated on the Device side.

  - workspaceSize(uint64_t, input parameter): the size of the workspace allocated on the Device side, obtained by the first-phase API aclnnMaxDimGetWorkspaceSize.

  - executor(aclOpExecutor*, input parameter): the op executor, which contains the operator computation process.

  - stream(aclrtStream, input parameter): specifies the Stream that executes tasks.

- **Return Values**:

  aclnnStatus: returns a status code. For details, see [aclnn Return Codes](../../docs/en/context/aclnn_return_code.md).

## Constraint Description

- Deterministic computation:
  - aclnnMaxDim uses a deterministic implementation by default.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_max_dim](./examples/test_aclnn_max_dim.cpp) | Invokes the ArgMaxWithValue operator through the [aclnnMaxDim](./docs/aclnnMaxDim_en.md) API. |
