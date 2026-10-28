# aclnnAdds

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:07:10.255Z -->

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Performs addition.
- Formula:

  $$
  out_i = self_i+alpha \times other
  $$

## Function Prototype

- aclnnAdds and aclnnInplaceAdds implement the same function. The differences in usage are as follows. Select a proper operator based on your actual use case.

  - aclnnAdds: A new output tensor object is required to store the computation result.
  - aclnnInplaceAdds: Requires no new output tensor object and directly stores the computation result in the memory of the input tensor.

- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnAddsGetWorkspaceSize" or "aclnnInplaceAddsGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnAdds" or "aclnnInplaceAdds" API to perform the computation.

```Cpp
  aclnnStatus aclnnAddsGetWorkspaceSize(
  const aclTensor* self,
  const aclScalar* other,
  const aclScalar* alpha,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
  aclnnStatus aclnnAdds(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

```Cpp
aclnnStatus aclnnInplaceAddsGetWorkspaceSize(
  const aclTensor *selfRef,
  const aclScalar *other,
  const aclScalar *alpha,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInplaceAdds(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddsGetWorkspaceSize

- **Parameters:**

    <table style="table-layout: fixed; width: 1510px"><colgroup>
    <col style="width: 153px">
    <col style="width: 120px">
    <col style="width: 219px">
    <col style="width: 235px">
    <col style="width: 386px">
    <col style="width: 119px">
    <col style="width: 132px">
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
        <th>Shape</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self</td>
        <td>Input</td>
        <td>Input self in the formula.</td>
        <td>-</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
        <td>ND</td>
        <td>No higher than 8 dimensions</td>
        <td>√</td>
      </tr>
      <tr>
        <td>other</td>
        <td>Input</td>
        <td>Input other in the formula.</td>
        <td>-</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>Input</td>
        <td>The alpha in the formula.</td>
        <td>The data type must be convertible to the derived data types of self and other.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>out in the formula.</td>
        <td>The data type must be a data type convertible from the data types of self and other after derivation (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
        <td>ND</td>
        <td>Same as self</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be applied on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns an op executor containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
    </table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas inference products</term> and <term>Atlas training products</term>:
    - BFLOAT16 data type is not supported.
    - The data types of self and other satisfy the [deduction relationship](../../../docs/en/context/deduction_relationship.md).
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The data types of self and other satisfy the [deduction relationship](../../../docs/en/context/deduction_relationship.md).
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: The data types of other and self satisfy the [TensorScalar deduction relationship](../../../docs/en/context/deduction_relationship.md).
  <!-- end id9 -->

- **Return value**

  aclnnStatus: Return status code, see specifically [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

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
        <td>The input self, other, out, or alpha is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="6">161002</td>
        <td>The data type of self is not supported.</td>
      </tr>
      <tr>
        <td>Data type derivation cannot be performed for self and other.</td>
      </tr>
      <tr>
        <td>The derived data type cannot be converted to the type of out.</td>
      </tr>
      <tr>
        <td>alpha cannot be converted to the derived data type of self and other.</td>
      </tr>
      <tr>
        <td>The shape of self is inconsistent with that of out.</td>
      </tr>
      <tr>
        <td>The dimensions of self and out are greater than 8.</td>
      </tr>
    </tbody>
    </table>

## aclnnAdds

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
        <td>Workspace memory address applied on the side of the Device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size applied on the side of the Device, obtained by calling the first-phase API aclnnAddsGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor of the op, containing the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for task execution.</td>
      </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code, see specifically [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceAddsGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1549px"><colgroup>
  <col style="width: 160px">
  <col style="width: 137px">
  <col style="width: 259px">
  <col style="width: 229px">
  <col style="width: 350px">
  <col style="width: 123px">
  <col style="width: 143px">
  <col style="width: 148px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data type</th>
      <th>Data format</th>
      <th>Dimensions (shape)</th>
      <th>Non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef</td>
      <td>Input/Output</td>
      <td>Input selfRef in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, BFLOAT16</td>
      <td>ND</td>
      <td>No more than 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Other in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>The alpha in the formula.</td>
      <td>The data type must be convertible to the derived data type of selfRef and other.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be applied on the side of Device.</td>
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

  <!-- npu="910" id10 -->
  - <term>Atlas training products</term>:
    - BFLOAT16 data type is not supported.
    - selfRef and other satisfy the [deduction relationship](../../../docs/en/context/deduction_relationship.md), and must be convertible data types after derivation (see [mutual conversion relationship](../../../docs/en/context/conversion_relationship.md)).
  <!-- end id10 -->
  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - selfRef and other satisfy the [deduction relationship](../../../docs/en/context/deduction_relationship.md), and must be convertible data types after derivation (see [mutual conversion relationship](../../../docs/en/context/conversion_relationship.md)).
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>:
    - selfRef and other satisfy the [TensorScalar mutual derivation relationship](../../../docs/en/context/deduction_relationship.md), and must be convertible data types after derivation (see [mutual conversion relationship](../../../docs/en/context/conversion_relationship.md)).
  <!-- end id12 -->

- **Return value**

  aclnnStatus: return status code, see specifically [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 283px">
  <col style="width: 120px">
  <col style="width: 747px">
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
      <td>The input selfRef, other, or alpha is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of selfRef is not within the supported range.</td>
    </tr>
    <tr>
      <td>Data type deduction between selfRef and other fails.</td>
    </tr>
    <tr>
      <td>The derived data type cannot be converted to the type of selfRef.</td>
    </tr>
    <tr>
      <td>alpha cannot be converted to the derived data type of selfRef and other.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceAdds

- **Parameters:**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>The workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size applied on the Device side, obtained by calling the first-phase API aclnnInplaceAddsGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code, see specifically [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnAdds&aclnnInplaceAdds default to a deterministic implementation.

  The other parameter has no precision loss for float. For int32 and int64 data types, precision loss may occur when the other parameter is greater than 2^24. It is recommended to use [aclnnAdd](./aclnnAdd&aclnnInplaceAdd_en.md).

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"

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
    // Boilerplate. Resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for device-side memory.
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

int main()
{
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API.
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclScalar* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> outHostData(8, 0);
    float otherValue = 2.0f;
    float alphaValue = 1.2f;
    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the other aclScalar.
    other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
    CHECK_RET(other != nullptr, return ret);
    // Create alpha aclScalar.
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return ret);
    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAdds.
    ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAdds.
    ret = aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdds failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result on the device-side memory to the host side. Modify it according to the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // aclnnInplaceAdds API call example
    // 3. Call the CANN operator library API.
    LOG_PRINT("\ntest aclnnInplaceAdds\n");
    // Call the first-phase API of aclnnInplaceAdds.
    ret = aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnInplaceAdds.
    ret = aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAdds failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Wait synchronously for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device side to the host side. Modify the code according to the specific API definition.
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release the aclTensor and aclScalar. Modify the code according to the specific API definition.
    aclDestroyTensor(self);
    aclDestroyScalar(other);
    aclDestroyScalar(alpha);
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
