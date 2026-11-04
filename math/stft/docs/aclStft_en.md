# aclStft

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

- API function: computes the Fourier transform of the input within a sliding window.
- Computation formula:

  - When normalized=False:

    $$
    X[w,m]=\sum_{k=0}^{winLength-1}window[k]*self[m*hopLength+k]*exp(-j*\frac{2{\pi}wk}{nFft})
    $$

  - When normalized=True:

    $$
    X[w,m]=\frac{1}{\sqrt{nFft}}(\sum_{k=0}^{winLength-1}window[k]*self[m*hopLength+k]*exp(-j*\frac{2{\pi}wk}{nFft}))
    $$

  Here:

  - $w$ is the frequency point of the FFT.
  - $m$ is the index of the sliding window.
  - $self$ is a 1D or 2D Tensor. If $self$ is 1D, it is a time-series sampling sequence. If $self$ is 2D, it is multiple time-series sampling sequences.
  - $hopLength$ is the sliding window size.
  - $window$ is a 1D Tensor and the window function of the STFT (for example, hann_window), whose length is $winLength$.
  - $exp(-j*\frac{2{\pi}wk}{nFft})$ is the twiddle factor.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclStftGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclStft" API to execute the computation.

```Cpp
aclnnStatus aclStftGetWorkspaceSize(
  const aclTensor *self,
  const aclTensor *windowOptional,
  aclTensor       *out,
  int64_t          nFft,
  int64_t          hopLength,
  int64_t          winLength,
  bool             normalized,
  bool             onesided,
  bool             returnComplex,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclStft(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclStftGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
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
      <td>self</td>
      <td>Input</td>
      <td>Input to be computed, corresponding to `self` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape is [L]/[B, L], where L is the length of the time-series sampling sequence and B is the number of time-series sampling sequences.</li><li>Inputs Inf, -Inf, and NaN are not supported.</li></ul></td>
      <td>FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>1-2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>windowOptional</td>
      <td>Input</td>
      <td>Must be a 1D Tensor, corresponding to `window` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type is the same as that of `self`.</li><li>The shape is [winLength], where winLength is the length of the STFT window function.</li></ul></td>
      <td>FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Result of the Fourier transform of self within the window, corresponding to `X` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>If returnComplex=True, `out` is a complex Tensor with the shape [N, T] or [B, N, T].</li><li>If returnComplex=False, `out` is a real Tensor with the shape [N, T, 2] or [B, N, T, 2].</li></ul>Here, N=nFft(onesided=False) or (nFft // 2 + 1)(onesided=True); T is the number of sliding windows, T = (L - nFft) // hopLength + 1.</td>
      <td>FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>3-4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>nFft</td>
      <td>Input</td>
      <td>Number of FFT points (greater than 0), corresponding to `nFft` in the formula.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hopLength</td>
      <td>Input</td>
      <td>Denotes the spacing between sliding windows (greater than 0), corresponding to `hopLength` in the formula.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLength</td>
      <td>Input</td>
      <td>Denotes the size of window (greater than 0), corresponding to `winLength` in the formula.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>normalized</td>
      <td>Input</td>
      <td>Denotes whether to normalize the Fourier transform result.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>onesided</td>
      <td>Input</td>
      <td>Denotes whether to return the full result or half of the result.</td>
      <td>If the data type of the input `self` is COMPLEX64 or COMPLEX128, only False is supported.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnComplex</td>
      <td>Input</td>
      <td>Denotes whether the return value is a complex Tensor or a Tensor with the real and imaginary parts separated.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device side.</td>
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
  </tbody>
  </table>

- **Return value**

  aclnnStatus: Return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input self or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data format of self is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of self and windowOptional are inconsistent.</td>
    </tr>
    <tr>
      <td>The data types of self, windowOptional, and out are not within the range supported by the platform.</td>
    </tr>
    <tr>
      <td>Invalid values are passed in for nFft, hopLength, or winLength.</td></tr>
    <tr>
      <td>The dimensions of self, windowOptional, and out are not within the supported range.</td>
    </tr>
    <tr>
      <td>When the data type of the input self is COMPLEX64 or COMPLEX128, the value of onesided is True.</td>
    </tr>
  </tbody></table>

## aclStft

- **Parameters:**

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
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
      <td>Memory address of the workspace allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device side, obtained by calling the first-phase API aclStftGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: Return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The input self differs from that of the PyTorch API: the input self of the PyTorch API is the original input, whereas the input self of aclStftGetWorkspaceSize is the result obtained after the original input is padded by the PyTorch frontend.
- When the shape of the input self is [B, L], the computation of the current API may time out if the result of the following formula is large.

  $$
  B * ((L - nFft) / hopLength + 1) * nFft
  $$

- nFft <= L.
- winLength <= nFft.
- When normalized=True,

  $$
  STFT(w,m)=\frac{1}{\sqrt{N}}X[w,m]
  $$

- The mapping between the input and output data types of self, windowOptional, returnComplex, and out is as follows:

  |self|windowOptional|returnComplex|out|
  |-------|-------|-------|-------|
  |FLOAT32| FLOAT32        | True          | COMPLEX64  |
  | DOUBLE     | DOUBLE         | True          | COMPLEX128 |
  |COMPLEX64|COMPLEX64|True|COMPLEX64|
  |COMPLEX128|COMPLEX128|True|COMPLEX128|
  |FLOAT32| FLOAT32        | False     | FLOAT32 |
  | DOUBLE     | DOUBLE         | False     | DOUBLE |
  |COMPLEX64|COMPLEX64|False|FLOAT32|
  |COMPLEX128|COMPLEX128|False|DOUBLE|

- Deterministic computation:
  - aclStft defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/acl_stft.h"

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
int Init(int32_t deviceId, aclrtStream* stream)
{
    // Fixed code for resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}
template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device side
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host side to the device-side memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Compute the strides of the contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Call the aclCreateTensor API to create an aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}
int main()
{
    // 1. (Fixed code) Initialize the device/stream. For details, refer to the list of ACL APIs.
    // Set deviceId based on the actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output. Customize the construction based on the API definition.
    std::vector<int64_t> selfShape = {5};
    std::vector<int64_t> windowShape = {4};
    std::vector<int64_t> outShape = {3, 1, 2};
    void* selfDeviceAddr = nullptr;
    void* windowDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* window = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {1, 6, 8, 5, 7};
    std::vector<float> windowHostData = {1, 1, 1, 1};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0};
    // Create the self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the window aclTensor
    ret = CreateAclTensor(windowHostData, windowShape, &windowDeviceAddr, aclDataType::ACL_FLOAT, &window);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    int n_fft = 4;
    int hop_length = 2;
    int win_length = 4;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    // 3. Call the CANN operator library API. Change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API aclStft
    ret = aclStftGetWorkspaceSize(
        self, window, out, n_fft, hop_length, win_length, normalized, onesided, returnComplex, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclStftGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API aclStft
    ret = aclStft(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclStft failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed code) Synchronize and wait for the task execution to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value, and copy the result from the device-side memory to the host side. Modify the code based on the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    // 6. Release the aclTensor. Modify the code based on the specific API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(window);
    aclDestroyTensor(out);
    // 7. Release the device resources. Modify the code based on the specific API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(windowDeviceAddr);
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
