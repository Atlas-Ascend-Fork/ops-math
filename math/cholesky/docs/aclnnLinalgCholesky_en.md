# aclnnLinalgCholesky

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:40:28.352Z -->

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

- Description: Computes the Cholesky decomposition of one or more real symmetric positive-definite matrices.
- When upper is false, out is lower triangular and satisfies:

  $$
  self = out \times out^{H}
  $$

- When upper is true, out is upper triangular and satisfies:

  $$
  self = out^{H} \times out
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnLinalgCholeskyGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the "aclnnLinalgCholesky" API to perform the computation.

```Cpp
aclnnStatus aclnnLinalgCholeskyGetWorkspaceSize(
  const aclTensor   *self,
  bool              upper,
  aclTensor         *out,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnLinalgCholesky(
  void               *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor      *executor,
  const aclrtStream  stream)
```

## aclnnLinalgCholeskyGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1537px"><colgroup>
  <col style="width: 200px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 130px">
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
      <th>Dimensions (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>The input self in the formula.</td>
      <td>The last two dimensions of shape must be equal, and self must be a symmetric positive-definite matrix.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>upper (bool)</td>
      <td>Input</td>
      <td>Selects upper-triangular or lower-triangular Cholesky decomposition.</td>
      <td><ul><li>True: Computes the upper triangular Cholesky decomposition.</li><li>False: Computes the lower triangular Cholesky decomposition.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output out in the formula.</td>
      <td><ul><li>Same data type as self.</li><li>Same shape as self.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
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
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  This API does not check whether the data in self is symmetric and positive definite.

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
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
      <td>self, out, workspaceSize, or executor is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>self and out have inconsistent data types.</td>
    </tr>
    <tr>
      <td>The storage formats of self and out differ, or a private format is used.</td>
    </tr>
    <tr>
      <td>The data type of self is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data type of out is not in the supported range.</td>
    </tr>
    <tr>
      <td>self has more than 8 dimensions.</td>
    </tr>
    <tr>
      <td>self and out have different shapes.</td>
    </tr>
    <tr>
      <td>self has fewer than 2 dimensions.</td>
    </tr>
    <tr>
      <td>The last two dimensions of self are not equal.</td>
    </tr>
    <tr>
      <td>The size of the last dimension of self exceeds 8192.</td>
    </tr>
  </tbody>
  </table>

## aclnnLinalgCholesky

- **Parameters:**

  <table style="table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
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
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the device, obtained by the first-phase API aclnnLinalgCholeskyGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor containing the operator computation process.</td>
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

  If self is not positive definite, computation may fail and the error may be reported when the stream is synchronized.

## Constraints

- Dimension: the rank of self must be in [2, 8], the last two dimensions must be equal to form a square matrix, and the last dimension M must be in [0, 8192]. Otherwise 161002 (ACLNN_ERR_PARAM_INVALID) is returned.
- Data type: FLOAT, FLOAT16, BFLOAT16.
- Empty tensor: an empty tensor that satisfies the parameter constraints is accepted. The first phase returns success with workspaceSize set to 0.
- Determinism: the computation is deterministic.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <new>
#include <memory>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_linalg_cholesky.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

// Guards attempt ordinary cleanup before submission or after a successful stream wait.
using StreamPtr = std::unique_ptr<std::remove_pointer<aclrtStream>::type, decltype(&aclrtDestroyStream)>;
using DeviceMemPtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
void DestroyUnusedExecutor(aclOpExecutor* executor)
{
    if (executor == nullptr) {
        return;
    }
    auto ret = aclSetAclOpExecutorRepeatable(executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclSetAclOpExecutorRepeatable failed. ERROR: %d\n", ret);
        return;
    }
    ret = aclDestroyAclOpExecutor(executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclDestroyAclOpExecutor failed. ERROR: %d\n", ret);
    }
}

using ExecutorPtr = std::unique_ptr<aclOpExecutor, decltype(&DestroyUnusedExecutor)>;

// Standalone failure boundary: never release submitted buffers before a successful wait.
class PendingStreamGuard {
public:
    explicit PendingStreamGuard(aclrtStream stream) noexcept : stream_(stream) {}
    PendingStreamGuard(const PendingStreamGuard&) = delete;
    PendingStreamGuard& operator=(const PendingStreamGuard&) = delete;

    void MarkSubmitted() noexcept { pending_ = true; }
    void RecordApiStatus(int status) noexcept
    {
        apiStatus_ = status;
        apiStatusKnown_ = true;
    }

    aclError Synchronize(bool fromCleanup = false) noexcept
    {
        const auto syncStatus = aclrtSynchronizeStream(stream_);
        if (syncStatus != ACL_SUCCESS) {
            // A failed wait does not establish completion. This standalone process stops
            // without normal destruction; it does not claim graceful resource recovery.
            std::fprintf(stderr,
                         "Pending stream failed: api2_status_known=%d api2_status=%d "
                         "sync_status=%d cleanup=%d; exiting without normal cleanup\n",
                         static_cast<int>(apiStatusKnown_), apiStatus_, static_cast<int>(syncStatus),
                         static_cast<int>(fromCleanup));
            std::fflush(stderr);
            std::_Exit(EXIT_FAILURE);
        }
        pending_ = false;
        return syncStatus;
    }

    ~PendingStreamGuard() noexcept
    {
        if (pending_) {
            (void)Synchronize(true);
        }
    }

private:
    aclrtStream stream_;
    bool pending_ = false;
    bool apiStatusKnown_ = false;
    int apiStatus_ = 0;
};

int Init(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet)
{
    // Boilerplate: resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    initialized = true;
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    deviceSet = true;
    aclrtStream rawStream = nullptr;
    ret = aclrtCreateStream(&rawStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    stream.reset(rawStream);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    DeviceMemPtr& deviceAddr, TensorPtr& tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory, then hand it to the deviceAddr guard immediately
    void* rawDeviceAddr = nullptr;
    auto ret = aclrtMalloc(&rawDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    deviceAddr.reset(rawDeviceAddr);
    // Call aclrtMemcpy to copy host data to the device memory.
    ret = aclrtMemcpy(deviceAddr.get(), size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor, then hand it to the tensor guard immediately
    aclTensor* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), deviceAddr.get());
    CHECK_RET(rawTensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    tensor.reset(rawTensor);
    return 0;
}

int RunExample()
{
    // 1. (Boilerplate) Device/stream initialization. For details, see the ACL API Reference.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    bool initialized = false;
    bool deviceSet = false;
    // Ordinary scope exit attempts Device/context cleanup only after pending work is settled.
    std::shared_ptr<void> aclGuard(nullptr, [&](void*) {
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (initialized) {
            aclFinalize();
        }
    });
    StreamPtr stream(nullptr, &aclrtDestroyStream);
    auto ret = Init(deviceId, stream, initialized, deviceSet);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output as required by the API.
    std::vector<int64_t> selfShape = {3, 3};
    std::vector<int64_t> outShape = {3, 3};
    DeviceMemPtr selfDeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr outDeviceAddr(nullptr, &aclrtFree);
    TensorPtr self(nullptr, &aclDestroyTensor);
    TensorPtr out(nullptr, &aclDestroyTensor);
    std::vector<float> selfHostData = {6, -3, 1, -3, 2, 0, 1, 0, 4};
    std::vector<float> outHostData(9, 0);
    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, aclDataType::ACL_FLOAT, selfDeviceAddr, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, aclDataType::ACL_FLOAT, outDeviceAddr, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);
    aclOpExecutor* executor = nullptr;
    // Call the first-phase API of aclnnLinalgCholesky.
    ret = aclnnLinalgCholeskyGetWorkspaceSize(self.get(), false, out.get(), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgCholeskyGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    ExecutorPtr executorGuard(executor, &DestroyUnusedExecutor);
    // Request device memory based on the workspaceSize calculated by the first-phase API,
    // then hand it to the workspaceAddr guard
    if (workspaceSize > static_cast<uint64_t>(0)) {
        void* rawWorkspaceAddr = nullptr;
        ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddr.reset(rawWorkspaceAddr);
    }
    // Call the second-phase API of aclnnLinalgCholesky.
    PendingStreamGuard pendingStream(stream.get());
    pendingStream.MarkSubmitted();
    ret = aclnnLinalgCholesky(workspaceAddr.get(), workspaceSize, executorGuard.release(), stream.get());
    pendingStream.RecordApiStatus(ret);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgCholesky failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = pendingStream.Synchronize();

    // 5. Obtain the output values and copy the result from device memory to host memory. Modify it according to the
    // specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr.get(),
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Submitted work must synchronize successfully before ordinary resource destruction.
    // A synchronization error terminates this standalone process without claiming cleanup success.
    // Pending-executor cleanup failures are logged; recovery is not guaranteed.
    return 0;
}

int main()
{
    try {
        return RunExample();
    } catch (const std::bad_alloc& error) {
        std::fprintf(stderr, "Host allocation failed: %s\n", error.what());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Host exception: %s\n", error.what());
    }
    return EXIT_FAILURE;
}
```
