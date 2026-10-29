# aclnnCdistBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:40:01.978Z -->

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

- Description: Completes the backward propagation of aclnnCdist.
- Formula:

  $$
  \begin{aligned}
  out&=grad \cdot y' \\
  &= grad \cdot \left( \sqrt[p]{\sum (x_1 - x_2)^p} \right)' \\
  &= grad \cdot \frac{1}{p} \times \left( \sum (x_1 - x_2)^p \right)^{\frac{1}{p}-1} \times p \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \left( \sum (x_1 - x_2)^p \right)^{\frac{-(p-1)}{p}} \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \left( \sum (x_1 - x_2)^p \right)^{\frac{1}{p} \times (-(p-1))} \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \frac{diff^{p-1}}{cdist^{p-1}} \\
  &= grad \cdot \frac{diff \times |diff|^{p-2}}{cdist^{p-1}}
  \end{aligned}
  $$

  - $\mathrm{diff} = x_1 - x_2$ : variable difference
  - $\mathrm{cdist} = \sqrt[p]{\sum (x_1 - x_2)^p}$ : $p$-norm distance

  The closed-form expression above applies only when $cdist > 0$. For each supported $p$ mode, when $cdist = 0$ ($x_1$ and $x_2$ coincide for that pair), the pair's contribution to the gradient of $x_1$ is defined as 0.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnCdistBackwardGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size according to the procedure, and then call the "aclnnCdistBackward" API to perform the computation.

```Cpp
aclnnStatus aclnnCdistBackwardGetWorkspaceSize(
    const aclTensor *grad,
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *cdist,
    float            p,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCdistBackward(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnCdistBackwardGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1510px"><colgroup>
    <col style="width: 153px">
    <col style="width: 120px">
    <col style="width: 250px">
    <col style="width: 140px">
    <col style="width: 150px">
    <col style="width: 119px">
    <col style="width: 280px">
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
        <th>Non-Contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>grad (aclTensor*)</td>
        <td>Input</td>
        <td>grad in the formula.</td>
        <td>-</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2-7 dimensions.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1 (aclTensor*)</td>
        <td>Input</td>
        <td>x1 in the formula.</td>
        <td>Data type is the same as grad.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>Dimension is the same as grad. Shape is the same as grad except the last dimension.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2 (aclTensor*)</td>
        <td>Input</td>
        <td>x2 in the formula.</td>
        <td>Data type is the same as grad.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>The dimensions are the same as those of grad. The penultimate dimension is the same as the last dimension of grad. The last dimension is the same as the last dimension of x1. The remaining dimensions must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with grad.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>cdist (aclTensor*)</td>
        <td>Input</td>
        <td>cdist in the formula.</td>
        <td>Same data type as grad.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>Same shape as grad.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>p(float)</td>
        <td>Attribute</td>
        <td>p in the formula.</td>
        <td>-</td>
        <td>float</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out(aclTensor*)</td>
        <td>Output</td>
        <td>out in the formula.</td>
        <td>Data type is the same as grad.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>Shape is the same as x1.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor that contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
    </table>
- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1110px"><colgroup>
    <col style="width: 291px">
    <col style="width: 112px">
    <col style="width: 707px">
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
        <td>grad, x1, x2, or cdist is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="6">161002</td>
        <td>The data type of grad, x1, x2, or cdist is not within the supported range.</td>
      </tr>
      <tr>
        <td>The shape of grad, x1, x2, or cdist is not within the supported range.</td>
      </tr>
    </tbody>
    </table>

## aclnnCdistBackward

- **Parameters:**

  <table style="table-layout: fixed; width: 1110px"><colgroup>
    <col style="width: 153px">
    <col style="width: 124px">
    <col style="width: 833px">
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
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the device, obtained by the first-phase API aclnnCdistBackwardGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, which contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for task execution.</td>
      </tr>
    </tbody>
    </table>
- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

Defaults to a deterministic computation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cdist_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // Boilerplate for resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory
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
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    int B=5, P=7, Q=9, M=11;
    // 2. Construct inputs and outputs. Custom construction is required based on the API.
    std::vector<int64_t> gradShape = {B, P, Q};
    std::vector<int64_t> x1Shape = {B, P, M};
    std::vector<int64_t> x2Shape = {B, Q, M};
    std::vector<int64_t> cdistShape = {B, P, Q};
    std::vector<int64_t> outShape = {B, P, M};
    void *gradDeviceAddr = nullptr;
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *cdistDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *grad = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *cdist = nullptr;
    aclTensor *out = nullptr;
    float p = 0.5;
    std::vector<float> gradHostData(B * P * Q, 1);
    std::vector<float> x1HostData(B * P * M, 2);
    std::vector<float> x2HostData(B * Q * M, 4);
    std::vector<float> cdistHostData(B * P * Q, 0.5);
    std::vector<float> outHostData(B * P * M, 1);
    // Create the grad aclTensor.
    ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the x1 aclTensor.
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the x2 aclTensor.
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the cdist aclTensor.
    ret = CreateAclTensor(cdistHostData, cdistShape, &cdistDeviceAddr, aclDataType::ACL_FLOAT, &cdist);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API of aclnnCdistBackward.
    ret = aclnnCdistBackwardGetWorkspaceSize(grad, x1, x2, cdist, p, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCdistBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnCdistBackward.
    ret = aclnnCdistBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCdistBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify it according to the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        outDeviceAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release the aclTensor and aclScalar. Modify them according to the specific API definition.
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(cdist);
    aclDestroyTensor(grad);
    aclDestroyTensor(out);

    // 7. Release device resources.
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(cdistDeviceAddr);
    aclrtFree(gradDeviceAddr);
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
