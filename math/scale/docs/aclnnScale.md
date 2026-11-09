# aclnnScale

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
- <term>Atlas推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：计算输入的scale和bias的结果。

- 计算公式：

  若不输入bias，则

  $$
  y=x*scale
  $$

  若输入bias，则

  $$
  y=x*scale + bias
  $$

  说明：scale/bias支持跟X的broadcast，scale/bias的shape规则如下

  - 当scaleFromBlob为True时（axis转换为正数，numAxes为-1时表示到最后轴）：

    numAxes=0时，scale必须包含且仅包含一个元素，rank为0～8，允许`()`、`[1]`、`[1,1]`等单元素形状按标量广播；x的rank为1～8，axis等通用约束仍须满足。

    numAxes=-1时，scaleShape为xShape[axis:]；numAxes>0时，须满足axis + numAxes ≤ rank(xShape)，scaleShape为xShape[axis:axis + numAxes]。

    bias存在时，其shape和dtype必须与scale一致。

  - 当scaleFromBlob为False时（axis转换为正数，numAxes不参与shape推导，但仍须满足numAxes>=-1）：

    scaleShape为xShape[axis:axis + rank(scaleShape)]

    biasShape为xShape[axis:axis + rank(scaleShape)]

    此分支须满足axis + rank(scaleShape) ≤ rank(xShape)。scale为`()`或`[1]`时按标量广播；其余scaleShape必须逐维等于上述x窗口，不能用size-1维替代窗口中的非1维。bias存在时shape必须与scale相同。例如xShape=[2,4]、scaleShape=[4]、axis=1合法；scaleShape=[1,4]、axis=0会在第一阶段返回ACLNN_ERR_PARAM_INVALID（161002）。
  - 示例：

    - scaleFromBlob = True：

      xShape = [a, b, c, d, e, f] axis = 3 numAxes = 2  --> scaleShape = [d, e]

      xShape = [a, b, c, d, e, f] axis = 3 numAxes = 3  --> scaleShape = [d, e, f]

      xShape = [a, b, c, d, e, f] axis = 3 numAxes = -1 --> scaleShape = [d, e, f]

    - scaleFromBlob = False：

      xShape = [a, b, c, d, e, f] axis = 3 rank(scaleShape) = 2 --> scaleShape = [d, e]

      xShape = [a, b, c, d, e, f] axis = 3 rank(scaleShape) = 3 --> scaleShape = [d, e, f]

      xShape = [a, b, c, d, e, f] axis = 3 rank(scaleShape) = 1 --> scaleShape = [d]

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnScaleGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnScale”接口执行计算。

```Cpp
aclnnStatus aclnnScaleGetWorkspaceSize(
  const aclTensor* x,
  const aclTensor* scale,
  const aclTensor* bias,
  int64_t          axis,
  int64_t          numAxes,
  bool             scaleFromBlob,
  aclTensor*       y,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnScale(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnScaleGetWorkspaceSize

- **参数说明：**

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
      <td>x（const aclTensor*）</td>
      <td>输入</td>
      <td>算子输入的Tensor。</td>
      <td>不得为空指针。shape各维度的长度必须为非负数；支持零元素Tensor（shape中至少一维为0），但仍须满足shape、dtype、axis等参数约束；参数校验通过后返回同shape的空输出。秩为0的x没有合法axis，属于非法输入。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale（const aclTensor*）</td>
      <td>输入</td>
      <td>算子输入的Tensor。</td>
      <td>不得为空指针。数据类型需要与x的数据类型相同，shape各维度的长度必须为非负数且须满足broadcast要求（参见<a href="#功能说明">功能说明</a>）；仅当合法shape本身包含0维长时支持零元素Tensor，x非空时不能传入零元素scale。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias（const aclTensor*）</td>
      <td>可选输入</td>
      <td>算子输入的Tensor。</td>
      <td>可传nullptr表示不提供偏置。指针非空时，数据类型和shape必须与scale一致，即使bias为零元素Tensor也须满足此要求；零元素bias不等同于nullptr。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis（int64_t）</td>
      <td>输入</td>
      <td>指定进行scale的起始轴。</td>
      <td>取值范围[-x_rank, x_rank)（x_rank表示x的shape维度）。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numAxes（int64_t）</td>
      <td>输入</td>
      <td>指定进行scale的轴长度。</td>
      <td>True/False分支均要求取值>=-1。仅当scaleFromBlob为True时，numAxes=-1表示从axis轴开始scale到最后一轴；False时不参与shape推导。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleFromBlob（bool）</td>
      <td>输入</td>
      <td>指定要scaleFromBlob类型。</td>
      <td><ul><li>True：scale from blob，使用numAxes + axis进行scale；</li><li>False：scale from input scale，从axis开始按scale的rank推导shape；numAxes不参与shape推导，但仍须>=-1。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y（aclTensor*）</td>
      <td>输出</td>
      <td>输出Tensor。</td>
      <td>不得为空指针。支持空Tensor。数据类型必须与x相同，shape必须与x完全一致（维数以及每一维的长度均相同）。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>必须为有效的非空输出地址。实际workspace大小以本接口返回值为准，可能包含非连续输入转换、中间结果和输出回拷所需的临时空间。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>必须为有效的非空输出地址。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="310p" id7 -->
  - <term>Atlas推理系列产品</term>：不支持BFLOAT16。
  <!-- end id7 -->

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 300px">
  <col style="width: 134px">
  <col style="width: 716px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>码值</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的x、scale、y、workspaceSize或executor是空指针。</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>x的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>bias指针非空时，bias与scale的数据类型不一致。</td>
    </tr>
    <tr>
      <td>scale与x的数据类型不一致。</td>
    </tr>
    <tr>
      <td>y与x的数据类型不一致。</td>
    </tr>
    <tr>
      <td>x和y的shape不一致。</td>
    </tr>
    <tr>
      <td>bias指针非空时，bias与scale的shape不一致。</td>
    </tr>
    <tr>
      <td>x的秩不在1～8范围内，scale或bias的秩大于8，或者x、scale、bias、y的shape包含负维度长度。</td>
    </tr>
    <tr>
      <td>axis的取值不在[-x_rank, x_rank)范围内。</td>
    </tr>
    <tr>
      <td>numAxes的取值小于-1。</td>
    </tr>
    <tr>
      <td>scaleFromBlob为True，numAxes等于0且scale的元素总数不等于1。</td>
    </tr>
    <tr>
      <td>scaleFromBlob为True且numAxes>0时，axis转换为正数之后与numAxes相加，大于x_rank。</td>
    </tr>
    <tr>
      <td>scale的shape与预期不符（预期shape推导参考功能说明）。</td>
    </tr>
  </tbody>
  </table>

## aclnnScale

- **参数说明：**

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnScaleGetWorkspaceSize获取。</td>
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

- aclnnScale默认确定性实现。
- x、scale、bias、y的数据类型必须一致；bias为nullptr时不参与校验。
- y的shape必须与x完全一致，包含维数及每一维的长度。
- bias不为nullptr时，其shape必须与scale完全一致。
- scale的shape必须满足[功能说明](#功能说明)中的广播窗口规则。
- 支持[非连续的Tensor](../../../docs/zh/context/non_contiguous_tensor.md)，维数不超过8。
- 参数表中的ND为公开保证的数据格式。接口不对所有非ND格式统一报错；非ND描述可能由框架转换后执行，不代表任意非ND格式组合均受支持。
- 可执行的Tensor规模受设备可用内存及框架资源限制；本接口未规定统一的2³¹元素或4GiB字节上限。

<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT系列产品</term>：FLOAT16、BFLOAT16输入使用FLOAT计算，结果转换回输入的数据类型；FLOAT输入直接计算。
<!-- end id8 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scale.h"

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

using StreamPtr = std::unique_ptr<std::remove_pointer<aclrtStream>::type, decltype(&aclrtDestroyStream)>;
using DeviceMemPtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;

int Init(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet) {
  // 固定写法，资源初始化
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
                    DeviceMemPtr& deviceAddr, TensorPtr& tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  void* rawDeviceAddr = nullptr;
  auto ret = aclrtMalloc(&rawDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  deviceAddr.reset(rawDeviceAddr);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(deviceAddr.get(), size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  aclTensor* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                       aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), deviceAddr.get());
  CHECK_RET(rawTensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
  tensor.reset(rawTensor);
  return 0;
}

int main() {
  // 1.（固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  bool initialized = false;
  bool deviceSet = false;
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

  // 2.构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> tensor1Shape = {4};
  std::vector<int64_t> tensor2Shape = {4};
  std::vector<int64_t> outShape = {4, 2};
  DeviceMemPtr selfDeviceAddr(nullptr, &aclrtFree);
  DeviceMemPtr tensor1DeviceAddr(nullptr, &aclrtFree);
  DeviceMemPtr tensor2DeviceAddr(nullptr, &aclrtFree);
  DeviceMemPtr outDeviceAddr(nullptr, &aclrtFree);
  TensorPtr self(nullptr, &aclDestroyTensor);
  TensorPtr tensor1(nullptr, &aclDestroyTensor);
  TensorPtr tensor2(nullptr, &aclDestroyTensor);
  TensorPtr out(nullptr, &aclDestroyTensor);

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> tensor1HostData = {2, 2, 2, 2};
  std::vector<float> tensor2HostData = {2, 2, 2, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t axis = 0;
  int64_t numAxes = 1;
  bool fromBlob = true;

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, aclDataType::ACL_FLOAT, selfDeviceAddr, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建tensor1 aclTensor
  ret = CreateAclTensor(tensor1HostData, tensor1Shape, aclDataType::ACL_FLOAT, tensor1DeviceAddr, tensor1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建tensor2 aclTensor
  ret = CreateAclTensor(tensor2HostData, tensor2Shape, aclDataType::ACL_FLOAT, tensor2DeviceAddr, tensor2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, aclDataType::ACL_FLOAT, outDeviceAddr, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3.调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnScale第一段接口
  ret = aclnnScaleGetWorkspaceSize(self.get(), tensor1.get(), tensor2.get(), axis, numAxes, fromBlob, out.get(),
                                   &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScaleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);
  if (workspaceSize > 0) {
    void* rawWorkspaceAddr = nullptr;
    ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddr.reset(rawWorkspaceAddr);
  }
  // 调用aclnnScale第二段接口
  ret = aclnnScale(workspaceAddr.get(), workspaceSize, executor, stream.get());
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScale failed. ERROR: %d\n", ret); return ret);

  // 4.（固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream.get());
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr.get(), size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("resultData[%ld] is: %f\n", i, resultData[i]);
  }

  return 0;
}
```
