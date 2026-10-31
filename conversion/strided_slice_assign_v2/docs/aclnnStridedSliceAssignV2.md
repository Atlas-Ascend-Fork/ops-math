# aclnnStridedSliceAssignV2

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

StridedSliceAssignV2是一种张量切片赋值操作，它可以将张量inputValue的内容，赋值给目标张量varRef中由begin、end、strides指定的切片位置，切片以外的区域保持varRef原值不变。本算子为in-place语义，输出与输入varRef同名，shape与输入varRef相同。

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持负stride反向切片。
<!-- end id7 -->

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnStridedSliceAssignV2GetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnStridedSliceAssignV2`接口执行计算。

```cpp
aclnnStatus aclnnStridedSliceAssignV2GetWorkspaceSize(
    aclTensor         *varRef,
    const aclTensor   *inputValue,
    const aclIntArray *begin,
    const aclIntArray *end,
    const aclIntArray *strides,
    const aclIntArray *axesOptional,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnStridedSliceAssignV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnStridedSliceAssignV2GetWorkspaceSize

- **参数说明：**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 240px">
  <col style="width: 110px">
  <col style="width: 150px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>varRef（aclTensor*）</td>
      <td>输入/输出</td>
      <td>输入Tensor。</td>
      <td>-</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT32、INT64、DOUBLE、INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputValue（aclTensor*）</td>
      <td>输入</td>
      <td>替换切片内容的tensor。</td>
      <td>shape与rank需要与varRef计算得出的切片shape保持一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT32、INT64、DOUBLE、INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>begin（aclIntArray*）</td>
      <td>输入</td>
      <td>切片位置的起始索引。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>end（aclIntArray*）</td>
      <td>输入</td>
      <td>切片位置的终止索引。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides（aclIntArray*）</td>
      <td>输入</td>
      <td>切片步长。</td>
      <td>strides取值可以为正数或负数。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axesOptional（aclIntArray*）</td>
      <td>输入</td>
      <td>可选参数，切片的轴。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

<!-- npu="910b,A3" id8 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：strides不支持负数，varRef最后一维对应的strides取值必须为1。
<!-- end id8 -->

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的varRef、inputValue、begin、end或strides是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入和输出的数据类型不在支持的范围之内。</td>
    </tr>
  </tbody>
  </table>

## aclnnStridedSliceAssignV2

- **参数说明**：
  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnStridedSliceAssignV2GetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnStridedSliceAssignV2默认确定性实现。
- inputValue的shape第i维的计算公式为：$inputValueShape[i] = \max(0, \lceil\frac{end[i] - begin[i]}{strides[i]} \rceil)$，其中$\lceil x\rceil$ 表示对 $x$向上取整。$end$ 和 $begin$ 为经过负索引归一化与边界截断后的取值：当 $x < 0$ 时 $x = x + varShape[i]$。正stride时 $begin[i]$ 与 $end[i]$ 均截断到 $[0, varShape[i]]$ 区间。
<!-- npu="950" id9 -->
- <term>Ascend 950PR&950DT系列产品</term>：负stride时，$begin[i]$ 截断到 $[0, varShape[i]-1]$、$end[i]$ 截断到 $[-1, varShape[i]]$ 区间。
<!-- end id9 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_strided_slice_assign_v2.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
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
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1.（固定写法）device/stream初始化，参考
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2.构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> varRefShape = {4, 3};
  std::vector<int64_t> inputValueShape = {2, 2};

  void* varRefDeviceAddr = nullptr;
  void* inputValueDeviceAddr = nullptr;

  aclTensor* varRef = nullptr;
  aclTensor* inputValue = nullptr;

  std::vector<float> varRefHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<float> inputValueHostData = {-1, -2, -3, -4};
  std::vector<int64_t> beginData = {1, 0};
  std::vector<int64_t> endData = {4, 2};
  std::vector<int64_t> stridesData = {2, 1};
  std::vector<int64_t> axesData = {0, 1};

  // 创建varRef aclTensor
  ret = CreateAclTensor(varRefHostData, varRefShape, &varRefDeviceAddr, aclDataType::ACL_FLOAT, &varRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建inputValue aclTensor
  ret = CreateAclTensor(inputValueHostData, inputValueShape, &inputValueDeviceAddr, aclDataType::ACL_FLOAT, &inputValue);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建begin aclIntArray
  aclIntArray *begin = aclCreateIntArray(beginData.data(), beginData.size());
  // 创建end aclIntArray
  aclIntArray *end = aclCreateIntArray(endData.data(), endData.size());
  // 创建strides aclIntArray
  aclIntArray *strides = aclCreateIntArray(stridesData.data(), stridesData.size());
  // 创建axes aclTensor
  aclIntArray *axes = aclCreateIntArray(axesData.data(), axesData.size());

  // 3.调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnStridedSliceAssignV2第一段接口
  ret = aclnnStridedSliceAssignV2GetWorkspaceSize(varRef, inputValue, begin, end, strides, axes, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSliceAssignV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnStridedSliceAssignV2第二段接口
  ret = aclnnStridedSliceAssignV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSliceAssignV2 failed. ERROR: %d\n", ret); return ret);
  // 4.（固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(varRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varRefDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6.释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(varRef);
  aclDestroyTensor(inputValue);
  aclDestroyIntArray(begin);
  aclDestroyIntArray(end);
  aclDestroyIntArray(strides);
  aclDestroyIntArray(axes);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(varRefDeviceAddr);
  aclrtFree(inputValueDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
