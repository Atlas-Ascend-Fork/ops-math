# aclnnSvd

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:22:32.833Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Computes the singular value decomposition of one or more matrices.

  When the input tensor has more than two dimensions, the high-dimension tensor is treated as a batch of matrices. For an input tensor with shape (..., M, N), the dimensions (...) before the second-to-last dimension are batch dimensions, and singular value decomposition is performed independently for each (M, N) matrix.

- Formula:

$$
\mathbf{input} = \mathbf{U} \times \mathrm{diag}(\boldsymbol{sigma}) \times \mathbf{V}^T
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnSvdGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the operator computation process. Then call "aclnnSvd" to execute the computation.

```c++
aclnnStatus aclnnSvdGetWorkspaceSize(
    const aclTensor *input,
    const bool       fullMatrices,
    const bool       computeUV,
    aclTensor       *sigma,
    aclTensor       *u,
    aclTensor       *v,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnSvd(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnSvdGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1544px"><colgroup>
  <col style="width: 141px">
  <col style="width: 120px">
  <col style="width: 354px">
  <col style="width: 461px">
  <col style="width: 141px">
  <col style="width: 101px">
  <col style="width: 81px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>Tensor that requires singular value decomposition, corresponding to input in the formula.</td>
      <td>Empty tensors are not supported.<br>The shape must have at least two dimensions.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>fullMatrices</td>
      <td>Input</td>
      <td>Input parameter that specifies whether to fully compute output tensors u and v.</td>
      <td>Controls whether to compute the full SVD decomposition.<br>If set to true, the full u and v are output.<br>If set to false, only the economy version is computed to save memory and computing resources.<br>If the shape of input is [..., M, N], let K=min(M, N).<br>If set to true, the output shapes are:<br>u: [..., M, M], sigma: [..., K], v: [..., N, N].<br>If set to false, the output shapes are:<br>u: [..., M, K], sigma: [..., K], v: [..., N, K].</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>computeUV</td>
      <td>Input</td>
      <td>Input parameter, indicating whether to compute output tensors u and v.</td>
      <td>If set to true, output tensors sigma, u, and v are all computed.<br>If set to false, only sigma is computed, and the shapes of u and v are no longer validated.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>Output</td>
      <td>Output tensor, corresponding to sigma in the formula.</td>
      <td>The shape must be derived based on the shape of input and the fullMatrices parameter.<br>The data type must be the same as that of input.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>u</td>
      <td>Output</td>
      <td>Output tensor corresponding to U in the formula.</td>
      <td>The shape is derived based on the shape of input and the fullMatrices parameter.<br>The data type must be the same as that of input.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>v</td>
      <td>Output</td>
      <td>Output tensor corresponding to V in the formula.</td>
      <td>The shape is derived from the shape of input and the fullMatrices parameter.<br>The data type must be the same as that of input.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
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

  <table style="table-layout: fixed; width: 1145px"><colgroup>
  <col style="width: 296px">
  <col style="width: 135px">
  <col style="width: 714px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>input, sigma, u, or v is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of input, sigma, u, and v are not within the supported range.</td>
    </tr>
    <tr>
      <td>The data type of input is inconsistent with that of sigma, u, and v.</td>
    </tr>
    <tr>
      <td>The dimension of input is less than 2 or greater than 8.</td>
    </tr>
    <tr>
      <td>The shapes of sigma, u, and v are inconsistent with the shapes inferred from input.</td>
    </tr>
  </tbody>
  </table>

## aclnnSvd

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
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the device, obtained from the first-phase API aclnnSvdGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnSvd defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_svd.h"

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

struct SVDTensors {
  void* inputDeviceAddr = nullptr;
  void* uDeviceAddr = nullptr;
  void* sigmaDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* u = nullptr;
  aclTensor* sigma = nullptr;
  aclTensor* v = nullptr;
  std::vector<int64_t> uShape = {2, 2};
  std::vector<int64_t> sigmaShape = {2};
  std::vector<int64_t> vShape = {3, 3};
};

struct SVDWorkspace {
  void* addr = nullptr;
  uint64_t size = 0;
};


int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate for resource initialization.
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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}



int SetupAndExecuteSVD(aclrtStream stream, SVDTensors& tensors, SVDWorkspace& workspace) {
  auto ret = 0;

  // Construct the inputs and outputs as required by the API.
  std::vector<int64_t> inputShape = {2, 3};
  std::vector<float> inputHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> uHostData = {0, 0, 0, 0};
  std::vector<float> sigmaHostData = {0, 0};
  std::vector<float> vHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  bool fullMatrices = true;
  bool computeUV = true;


  // Create the input aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &tensors.inputDeviceAddr, aclDataType::ACL_FLOAT, &tensors.input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the u aclTensor.
  ret = CreateAclTensor(uHostData, tensors.uShape, &tensors.uDeviceAddr, aclDataType::ACL_FLOAT, &tensors.u);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the sigma aclTensor.
  ret = CreateAclTensor(sigmaHostData, tensors.sigmaShape, &tensors.sigmaDeviceAddr, aclDataType::ACL_FLOAT, &tensors.sigma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the v aclTensor.
  ret = CreateAclTensor(vHostData, tensors.vShape, &tensors.vDeviceAddr, aclDataType::ACL_FLOAT, &tensors.v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Call the aclnnSvd operator API.
  aclOpExecutor* executor;
  // Call the first-phase API aclnnSvdGetWorkspaceSize.
  ret = aclnnSvdGetWorkspaceSize(tensors.input, fullMatrices, computeUV, tensors.sigma, tensors.u, tensors.v,
                                 &workspace.size, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSvdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  workspace.addr = nullptr;
  if (workspace.size > 0) {
    ret = aclrtMalloc(&workspace.addr, workspace.size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API aclnnSvd.
  ret = aclnnSvd(workspace.addr, workspace.size, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSvd failed. ERROR: %d\n", ret); return ret);

  // Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  return 0;
}

int ProcessAndCleanupSVD(SVDTensors& tensors, SVDWorkspace& workspace) {
  auto ret = 0;

  // Obtain the output values and copy the results from device to host. Modify the code based on the actual API definition.
  auto uSize = GetShapeSize(tensors.uShape);
  std::vector<float> uData(uSize, 0);
  ret = aclrtMemcpy(uData.data(), uData.size() * sizeof(uData[0]), tensors.uDeviceAddr,
                    uSize * sizeof(uData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outTensor U from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < uSize; i++) {
    LOG_PRINT("u[%ld] is: %f\n", i, uData[i]);
  }

  auto sigmaSize = GetShapeSize(tensors.sigmaShape);
  std::vector<float> sigmaData(sigmaSize, 0);
  ret = aclrtMemcpy(sigmaData.data(), sigmaData.size() * sizeof(sigmaData[0]), tensors.sigmaDeviceAddr,
                    sigmaSize * sizeof(sigmaData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outTensor sigma from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < sigmaSize; i++) {
    LOG_PRINT("sigma[%ld] is: %f\n", i, sigmaData[i]);
  }

  auto vSize = GetShapeSize(tensors.vShape);
  std::vector<float> vData(vSize, 0);
  ret = aclrtMemcpy(vData.data(), vData.size() * sizeof(vData[0]), tensors.vDeviceAddr,
                    vSize * sizeof(vData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outTensor V from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < vSize; i++) {
    LOG_PRINT("v[%ld] is: %f\n", i, vData[i]);
  }

  // Release aclTensor and aclScalar. Modify the code based on the actual API definition.
  aclDestroyTensor(tensors.input);
  aclDestroyTensor(tensors.u);
  aclDestroyTensor(tensors.sigma);
  aclDestroyTensor(tensors.v);

  // Release device resources.
  aclrtFree(tensors.inputDeviceAddr);
  aclrtFree(tensors.uDeviceAddr);
  aclrtFree(tensors.sigmaDeviceAddr);
  aclrtFree(tensors.vDeviceAddr);
  if (workspace.size > 0) {
    aclrtFree(workspace.addr);
  }

  return 0;
}

int ExecuteSVDOperator(aclrtStream stream) {
  SVDTensors tensors;
  SVDWorkspace workspace;

  auto ret = SetupAndExecuteSVD(stream, tensors, workspace);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = ProcessAndCleanupSVD(tensors, workspace);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return 0;
}


int main() {
  // Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // Execute the SVD operation.
  ret = ExecuteSVDOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteSVDOperator failed. ERROR: %d\n", ret); return ret);

  // Reset the device and finalize ACL.
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
