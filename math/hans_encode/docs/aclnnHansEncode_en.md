# aclnnHansEncode

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T00:17:22.189Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

Performs PDF statistics on the byte where the exponent bits of the inputTensor reside (optional), and performs lossless compression based on the PDF distribution. The compressed result can be stored in the device HBM or offloaded to the host side.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnHansEncodeGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnHansEncode" API to execute the computation.

- `aclnnStatus aclnnHansEncodeGetWorkspaceSize(const aclTensor *inputTensor, aclTensor *pdfRef, bool statistic, bool reshuff, const aclTensor *mantissaOut, const aclTensor *fixedOut, const aclTensor *varOut, uint64_t *workspaceSize, aclOpExecutor **executor);`
- `aclnnStatus aclnnHansEncode(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnHansEncodeGetWorkspaceSize

- **Parameters:**

  - inputTensor(aclTensor*, computed input): Represents the tensor to be compressed on the Device side. The data type supports FLOAT16, BFLOAT16, and FLOAT32. The size of each data element must be a multiple of 64 and greater than or equal to 32768. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
  - pdfRef(aclTensor*, computed input/computed output): Represents the probability density distribution of the bytes containing the exponent bits of inputTensor on the Device side. The data type supports INT32. The shape must be (1, 256). [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
  - statistic(bool, computed input): Specifies whether to perform PDF statistics.
  - reshuff(bool, computed input): Specifies whether to perform memory reshuffling on the encoding results of all cores.
  - mantissaOut(aclTensor*, computed output): Represents the mantissa part of the output on the Device side. The data type supports FLOAT16, BFLOAT16, and FLOAT32 and must be the same as that of inputTensor. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
  - fixedOut(aclTensor*, computed output): Represents the first output segment of the compression on the Device side. The data type supports FLOAT16, BFLOAT16, and FLOAT32 and must be the same as that of inputTensor. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
  - varOut(aclTensor*, computed output): Represents the output of the compressed data exceeding fixedOut on the Device side. The data type supports FLOAT16, BFLOAT16, and FLOAT32 and must be the same as that of inputTensor. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
  - workspaceSize(uint64_t*, output): Returns the size of the workspace to be requested on the Device side.
  - executor(aclOpExecutor**, output): Returns the op executor, which contains the operator computation process.

- **Return value**

  aclnnStatus: Return status code. For details, see [aclnnReturnCode](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 286px">
  <col style="width: 124px">
  <col style="width: 740px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Values</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>inputTensor, pdf, mantissaOut, fixedOut, or varOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The length of pdf or mantissa is invalid.</td>
    </tr>
    <tr>
      <td>The accumulated coding space is too small, and compression may overflow. The following formula must be satisfied: (size(fixedOut) + size(mantissaOut)) &gt;= (len(inputTensor) + len(inputTensor) / 64 + 8448 * processCoreDim).</td>
    </tr>
    <tr>
      <td>size(fixedOut) is less than 512 Byte, and compression metadata cannot be stored.</td>
    </tr>
    <tr>
      <td>The number of input elements is not a multiple of 64 or is less than 32768.</td>
    </tr>
    <tr>
      <td>The data types of inputTensor, pdf, and outputMantissaTensorOut are not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnHansEncode

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
      <td>Workspace memory size allocated on the Device side, obtained by calling the first-phase API aclnnHansEncodeGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnHansEncode defaults to a deterministic implementation.

## Example

The following is sample code for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_hans_encode.h"
#include "aclnn_hans_decode.h"

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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the acl API.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs. Customize the construction based on the API.
  std::vector<float> inputHost(65536, 0);
  std::vector<float> mantissaHost(49152, 0);
  std::vector<float> fixedHost(16384, 0);
  std::vector<float> varHost(16384, 0);
  std::vector<int32_t> pdfHost(256, 0);
  std::vector<float> recoverHost(65536, 0);
  bool statistic = true;
  bool reshuff = false;
  int64_t outHostAddr = -1;
  int64_t outHostLength = 0;

  void* inputAddr = nullptr;
  void* outMantissaAddr = nullptr;
  void* outFixedAddr = nullptr;
  void* outVarAddr = nullptr;
  void* pdfAddr = nullptr;
  void* recoverAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* outMantissa = nullptr;
  aclTensor* outFixed = nullptr;
  aclTensor* pdf = nullptr;
  aclTensor* outVar = nullptr;
  aclTensor* recover = nullptr;
  // Create the output aclTensor.
  ret = CreateAclTensor(inputHost, {1, 65536}, &inputAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(mantissaHost, {1, 49152}, &outMantissaAddr, aclDataType::ACL_FLOAT, &outMantissa);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(fixedHost, {1, 16384}, &outFixedAddr, aclDataType::ACL_FLOAT, &outFixed);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(varHost, {1, 16384}, &outVarAddr, aclDataType::ACL_FLOAT, &outVar);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(pdfHost, {1, 256}, &pdfAddr, aclDataType::ACL_INT32, &pdf);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(recoverHost, {1, 65536}, &recoverAddr, aclDataType::ACL_FLOAT, &recover);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API. Replace the API name with the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API aclnnHansEncode.
  ret = aclnnHansEncodeGetWorkspaceSize(input, pdf, statistic, reshuff, outMantissa, outFixed, outVar, &workspaceSize,
                                        &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHansEncodeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnHansEncode.
  ret = aclnnHansEncode(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHansEncode failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value, copy the result on the Device side to the Host side, and modify the code based on the actual API definition.
  auto size = 16384 * sizeof(float);
  std::vector<float> resultData(16384, 0);
  ret = aclrtMemcpy(resultData.data(), size, outFixedAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < 128; i++) {
    int32_t intVal = *reinterpret_cast<int32_t*>(&resultData[i]);
    LOG_PRINT("result header[%ld] is: %d\n", i, intVal);
  }

  uint64_t workspaceSizeDecode = 0;
  aclOpExecutor* executorDecode;
  // Call the first-phase API aclnnHansDecode.
  ret = aclnnHansDecodeGetWorkspaceSize(outMantissa, outFixed, outVar, pdf, reshuff, recover, &workspaceSizeDecode,
                                        &executorDecode);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHansDecodeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddrDecode = nullptr;
  if (workspaceSizeDecode > 0) {
    ret = aclrtMalloc(&workspaceAddrDecode, workspaceSizeDecode, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnHansDecode.
  ret = aclnnHansDecode(workspaceAddrDecode, workspaceSizeDecode, executorDecode, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHansDecode failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  std::vector<float> recoverData(65536, 0);
  ret = aclrtMemcpy(recoverData.data(), 65536 * sizeof(recoverData[0]), recoverAddr, 65536 * sizeof(recoverData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < 256; i++) {
    LOG_PRINT("reco[%ld] is: %f org is: %f\n", i, recoverData[i], inputHost[i]);
  }

  // 6. Release aclTensor, and modify the code based on the actual API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(outMantissa);
  aclDestroyTensor(outFixed);
  aclDestroyTensor(outVar);
  aclDestroyTensor(pdf);
  aclDestroyTensor(recover);

  // 7. Release device resources, and modify the code based on the actual API definition.
  aclrtFree(inputAddr);
  aclrtFree(outMantissaAddr);
  aclrtFree(outFixedAddr);
  aclrtFree(outVarAddr);
  aclrtFree(pdfAddr);
  aclrtFree(recoverAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  if (workspaceSizeDecode > 0) {
    aclrtFree(workspaceAddrDecode);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
