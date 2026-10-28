# aclnnEqScalar&aclnnInplaceEqScalar

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:56:55.520Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Compares whether the value of each element in self is equal to the value of other, and writes the comparison result of each element in self with the value of other into out.
- Formula:

  $$
  out_i = (self_i == \mathit{other} )  ?  [True] : [False]
  $$

## Function Prototype

- aclnnEqScalar and aclnnInplaceEqScalar implement the same functionality. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.

  - aclnnEqScalar: Requires creating a new output tensor object to store the computation result.
  - aclnnInplaceEqScalar: Does not require creating a new output tensor object. The computation result is stored directly in the memory of the input tensor.

- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnEqScalarGetWorkspaceSize" or "aclnnInplaceEqScalarGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnEqScalar" or "aclnnInplaceEqScalar" to execute the computation.

```Cpp
aclnnStatus aclnnEqScalarGetWorkspaceSize(
  const aclTensor *self,
  const aclScalar *other,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnEqScalar(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

```Cpp
aclnnStatus aclnnInplaceEqScalarGetWorkspaceSize(
  const aclTensor *selfRef,
  const aclScalar *other,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInplaceEqScalar(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnEqScalarGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 168px">
  <col style="width: 136px">
  <col style="width: 213px">
  <col style="width: 200px">
  <col style="width: 373px">
  <col style="width: 115px">
  <col style="width: 142px">
  <col style="width: 147px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension(shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (const aclTensor*)</td>
      <td>Input</td>
      <td>self in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT32, UINT64</td>
      <td>ND</td>
      <td>No more than 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other (const aclScalar*)</td>
      <td>Input</td>
      <td>other in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT32, UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>The out in the formula.</td>
      <td>Data types that can be converted to BOOL (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT64, UINT32, UINT16</td>
      <td>ND</td>
      <td>Same shape as self.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
    - self and other satisfy [TensorScalar deduction relationship](../../../docs/en/context/deduction_relationship.md).
    - self and other do not support the UINT32 data type.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - self and other satisfy [deduction relationship](../../../docs/en/context/deduction_relationship.md).
    - UINT32 and UINT64 data types are not supported for self and other.
    - out does not support the UINT64, UINT32, and UINT16 data types.
  <!-- end id8 -->
  <!-- npu="910" id9 -->
  - <term>Atlas training products</term>:
    - self and other satisfy [deduction relationship](../../../docs/en/context/deduction_relationship.md).
    - BFLOAT16 data type is not supported.
    - UINT64, UINT32, UINT16, and BFLOAT16 data types are not supported for out.
  <!-- end id9 -->

- **Return value**

  aclnnStatus: return status. For details, see [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 272px">
  <col style="width: 114px">
  <col style="width: 764px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>When self, other, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of self, other, or out is out of the supported range.</td>
    </tr>
    <tr>
      <td>The data types of self and other do not satisfy the data type deduction rules.</td>
    </tr>
    <tr>
      <td>The shapes of self and out are different.</td>
    </tr>
    <tr>
      <td>The dimensions of self and out are greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnEqScalar

- **Parameters:**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 873px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>The workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Input</td>
      <td>The workspace size allocated on the device, obtained by calling the first-phase API aclnnEqScalarGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor*)</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceEqScalarGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1502px"><colgroup>
  <col style="width: 167px">
  <col style="width: 135px">
  <col style="width: 267px">
  <col style="width: 159px">
  <col style="width: 372px">
  <col style="width: 115px">
  <col style="width: 141px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef</td>
      <td>Input/Output</td>
      <td>selfRef in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT32, UINT64</td>
      <td>ND</td>
      <td>No more than 8 dimensions.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other (const aclScalar*)</td>
      <td>Input</td>
      <td>other in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT32, UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied for on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>:
    - selfRef and other satisfy the [TensorScalar deduction relationship](../../../docs/en/context/deduction_relationship.md).
    - selfRef and other do not support the UINT32 data type.
  <!-- end id10 -->
  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - selfRef and other satisfy the [deduction relationship](../../../docs/en/context/deduction_relationship.md).
    - selfRef and other do not support the UINT32 or UINT64 data type.
  <!-- end id11 -->
  <!-- npu="910" id12 -->
  - <term>Atlas training products</term>:
    - selfRef and other satisfy the [deduction relationship](../../../docs/en/context/deduction_relationship.md).
    - BFLOAT16 data type is not supported.
  <!-- end id12 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 272px">
  <col style="width: 114px">
  <col style="width: 764px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>When selfRef or other is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types of selfRef and other are not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of selfRef and other do not satisfy the data type deduction rules.</td>
    </tr>
    <tr>
      <td>The dimension of selfRef is greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceEqScalar

- **Parameters:**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 873px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by the first-phase API aclnnInplaceEqScalarGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor*)</td>
      <td>Input</td>
      <td>Executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnEqScalar&aclnnInplaceEqScalar default to a deterministic implementation.

## Example

The following example code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

**aclnnEqScalar sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_eq_scalar.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
  // Boilerplate for resource initialization.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            aclrtResetDevice(deviceId); aclFinalize(); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the host to the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
      *deviceAddr);
  return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, void** selfDeviceAddr, void** outDeviceAddr,
    aclTensor** self, aclScalar** other, aclTensor** out)
{
  std::vector<double> selfHostData = {0, 1, 1.2, 0.3, 4.1, 5, 1.6, 7};
  std::vector<char> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  double otherValue = 1.2;

  // Create the self tensor.
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the other scalar.
  *other = aclCreateScalar(&otherValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(*other != nullptr, return ret);

  // Create the out tensor.
  ret = CreateAclTensor(outHostData, outShape, &(*outDeviceAddr), aclDataType::ACL_BOOL, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpAPI(
    aclTensor* self, aclScalar* other, aclTensor* out, void** workspaceAddrOut, uint64_t& workspaceSize,
    void* outDeviceAddr, std::vector<int64_t>& outShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  // Obtain the workspace size.
  auto ret = aclnnEqScalarGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEqScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate workspace (freed in main).
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // Call the operator.
  ret = aclnnEqScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEqScalar failed. ERROR: %d\n", ret); return ret);

  // Synchronize.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy the output.
  auto size = GetShapeSize(outShape);
  std::vector<char> resultData(size, 0);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(char),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InitAcl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclTensor* out = nullptr;

  ret = CreateInputs(selfShape, outShape, &selfDeviceAddr, &outDeviceAddr, &self, &other, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpAPI(self, other, out, &workspaceAddr, workspaceSize, outDeviceAddr, outShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Release tensors / scalars.
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyTensor(out);

  // Release device memory.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```

**aclnnInplaceEqScalar sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_eq_scalar.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
  // Boilerplate for resource initialization.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            aclrtResetDevice(deviceId); aclFinalize(); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
      *deviceAddr);
  return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(std::vector<int64_t>& selfShape, void** selfDeviceAddr, aclTensor** self, aclScalar** other)
{
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  double otherValue = 2.0;

  // Create the self aclTensor.
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create other aclScalar.
  *other = aclCreateScalar(&otherValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(*other != nullptr, return ACL_ERROR_INVALID_PARAM);

  return ACL_SUCCESS;
}

aclError ExecOpAPI(
    aclTensor* self, aclScalar* other, void* selfDeviceAddr, std::vector<int64_t>& selfShape, aclrtStream stream,
    void** workspaceAddrOut)
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  auto ret = aclnnInplaceEqScalarGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceEqScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate the workspace.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  *workspaceAddrOut = workspaceAddr;

  // Execute.
  ret = aclnnInplaceEqScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceEqScalar failed. ERROR: %d\n", ret); return ret);

  // Synchronize.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy the output.
  auto size = GetShapeSize(selfShape);
  std::vector<double> resultData(size);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;

  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> selfShape = {4, 2};
  void* selfDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclScalar* other = nullptr;

  ret = CreateInputs(selfShape, &selfDeviceAddr, &self, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  void* workspaceAddr = nullptr;
  ret = ExecOpAPI(self, other, selfDeviceAddr, selfShape, stream, &workspaceAddr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Destroy.
  aclDestroyTensor(self);
  aclDestroyScalar(other);

  aclrtFree(selfDeviceAddr);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  // Release.
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```
