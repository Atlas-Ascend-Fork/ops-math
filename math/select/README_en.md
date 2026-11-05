# select

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

+ Operator function: selects elements from self or other based on the condition and returns them (broadcast is supported).
+ The computation formula is as follows:

$$
out_i=where(self_i,other_i,condition_i)=\begin{cases}
  self_i, & \text{if condition}_i \\
  other_i, & \text{otherwise}
   \end{cases}
$$

## Function Prototype

Each operator uses a [two-phase API](../../docs/en/context/two_phase_api.md). Call the "aclnnSWhereGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnSWhere" API to execute the computation.

- `aclnnStatus aclnnSWhereGetWorkspaceSize(const aclTensor *condition, const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSWhere(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSWhereGetWorkspaceSize

* **Parameters**:
  - condition (aclTensor*, compute input): Input `condition` in the formula, which is an aclTensor on the device. The data types support UINT8 and BOOL. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the shape of condition and those of self and other must meet the [broadcast relationship](../../docs/en/context/broadcast_relationship.md). [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, the supported [data format](../../docs/en/context/data_format.md) is ND, and the number of data dimensions cannot exceed 8.

  - self (aclTensor*, compute input): Input `self` in the formula. Its data type and the data type of other must meet the data type deduction rules (see [Deduction Relationship](../../docs/en/context/deduction_relationship.md)). It is an aclTensor on the device, and its shape and the shapes of other and condition must meet the [broadcast relationship](../../docs/en/context/broadcast_relationship.md). [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, the supported [data format](../../docs/en/context/data_format.md) is ND, and the number of data dimensions cannot exceed 8.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The data types support FLOAT,  INT32,  UINT64,  INT64,  UINT32,  FLOAT16,  UINT16,  INT16,  INT8,  UINT8,  DOUBLE,  BOOL,  COMPLEX64,  COMPLEX128,  BFLOAT16.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data types support FLOAT,  INT32,  UINT64,  INT64,  UINT32,  FLOAT16,  UINT16,  INT16,  INT8,  UINT8,  DOUBLE,  BOOL,  COMPLEX64,  COMPLEX128.

  - other (aclTensor*, compute input): Input `other` in the formula. Its data type and the data type of self must meet the data type deduction rules (see [Deduction Relationship](../../docs/en/context/deduction_relationship.md)). It is an aclTensor on the device, and its shape and the shapes of self and condition must meet the [broadcast relationship](../../docs/en/context/broadcast_relationship.md). [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, the supported [data format](../../docs/en/context/data_format.md) is ND, and the number of data dimensions cannot exceed 8.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The data types support FLOAT,  INT32,  UINT64,  INT64,  UINT32,  FLOAT16,  UINT16,  INT16,  INT8,  UINT8,  DOUBLE,  BOOL,  COMPLEX64,  COMPLEX128,  BFLOAT16.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data types support FLOAT,  INT32,  UINT64,  INT64,  UINT32,  FLOAT16,  UINT16,  INT16,  INT8,  UINT8,  DOUBLE,  BOOL,  COMPLEX64,  COMPLEX128.

  - out (aclTensor \*, compute output): Output `out` in the formula. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported. It is an aclTensor on the device, and its shape must be the shape of self, other, and condition after broadcast. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, the supported [data format](../../docs/en/context/data_format.md) is ND, and the number of data dimensions cannot exceed 8.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The data types support FLOAT,  INT32,  UINT64,  INT64,  UINT32,  FLOAT16,  UINT16,  INT16,  INT8,  UINT8,  DOUBLE,  BOOL,  COMPLEX64,  COMPLEX128,  BFLOAT16.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data types support FLOAT,  INT32,  UINT64,  INT64,  UINT32,  FLOAT16,  UINT16,  INT16,  INT8,  UINT8,  DOUBLE,  BOOL,  COMPLEX64,  COMPLEX128.

  - workspaceSize (uint64_t \*, output parameter): Returns the size of the workspace to be allocated on the Device side.
  - executor(aclOpExecutor\*\*, output parameter): Returns the op executor, which contains the operator computation flow.
* **Return value**:
  aclnnStatus: Return status code. For details, see [aclnn Return Codes](../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API performs input parameter validation and returns an error in the following scenarios:
Return 161001 (ACLNN_ERR_PARAM_NULLPTR): The input self, other, condition, or out is a null pointer.
Return 161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type and dimensions of self, other, or condition are not within the supported range.
                                      2. Data type deduction cannot be performed between self and other.
                                      3. The broadcast deduction of self, other, and condition fails, or the broadcast result is different from the shape of out.
```

## aclnnSWhere

* **Parameters**:

  - workspace (void \*, input parameter): Memory address of the workspace allocated on the Device side.
  - workspaceSize (uint64_t, input parameter): Size of the workspace allocated on the Device side, obtained by calling the first-phase API aclnnSWhereGetWorkspaceSize.
  - executor(aclOpExecutor \*, input parameter): op executor, which contains the operator computation flow.
  - stream (aclrtStream, input parameter): Stream for executing tasks.

* **Return value**:
  aclnnStatus: Return status code. For details, see [aclnn Return Codes](../../docs/en/context/aclnn_return_code.md).

## Constraint Description

- Deterministic computation:
  - aclnnSWhere defaults to a deterministic implementation.
