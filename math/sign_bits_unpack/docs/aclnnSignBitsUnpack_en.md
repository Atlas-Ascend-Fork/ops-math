# aclnnSignBitsUnpack

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:06:07.951Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

Unpacks uint8-type 1-bit Adam into float32 or float16.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnSignBitsUnpackGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process. Then call the "aclnnSignBitsUnpack" API to perform the computation.

```Cpp
aclnnStatus aclnnSignBitsUnpackGetWorkspaceSize(
  const aclTensor* self,
  int64_t          size,
  aclDataType      dtype,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnSignBitsUnpack(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnSignBitsUnpackGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 190px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 160px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimensions (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>A 1D tensor used for computation.</td>
      <td>Empty tensors are supported.</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>size (int64_t)</td>
      <td>Input</td>
      <td>Dimension handling. The first dimension of the output tensor after reshape.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype (aclDataType)</td>
      <td>Input</td>
      <td>Specifies the data type of the quantization output tensor.</td>
      <td>ACL_FLOAT16, ACL_FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>The data type is determined by the dtype parameter.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 300px">
  <col style="width: 134px">
  <col style="width: 716px">
  </colgroup>
  <thead>
    <tr>
      <th>Return value</th>
      <th>Error code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If self or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type \ data format of self or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>size is less than or equal to 0, or (number of elements in self) * 8 % size != 0.</td>
    </tr>
    <tr>
      <td>The data type of out is inconsistent with dtype.</td>
    </tr>
    <tr>
      <td>self is not a 1-dimensional tensor.</td>
    </tr>
    <tr>
      <td>If the first dimension of out is inconsistent with size.</td>
    </tr>
  </tbody>
  </table>

## aclnnSignBitsUnpack

- **Parameters:**

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnSignBitsUnpackGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the computation flow of the operator.</td>
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
  - aclnnSignBitsUnpack defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <memory>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sign_bits_unpack.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate, resource initialization
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host side to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (boilerplate) Device/stream initialization. Refer to the acl API manual.
  // Fill in deviceId according to the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction according to the API.
  std::vector<int64_t> selfShape = {2};
  std::vector<int64_t> outShape = {2, 8};

  int64_t outsize = 2;
  aclDataType dataType = ACL_FLOAT;

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  std::vector<uint8_t> selfHostData = {159, 15};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_UINT8, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSignBitsUnpack.
  ret = aclnnSignBitsUnpackGetWorkspaceSize(self, outsize, dataType, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSignBitsUnpackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSignBitsUnpack.
  ret = aclnnSignBitsUnpack(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSignBitsUnpack failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait for the task execution to complete synchronously.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the code based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. Release device resources.
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
