# SpaceToBatch

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :-------: |
| <term>Ascend 950PR&950DT系列产品</term>                             | √        |
| <term>Atlas A3系列产品</term>     | √        |
| <term>Atlas A2系列产品</term> | √        |
| <term>Atlas 200I/500 A2推理产品</term>                      |     ×     |
| <term>Atlas推理系列产品</term>                             | √        |
| <term>Atlas训练系列产品</term>                              | √        |

## 功能说明

- 算子功能：将空间维度的数据按块重新排列到批次维度，并对空间维度补零。

- 功能描述：
  该算子首先根据paddings参数对输入的空间维度进行zero-padding，然后将padded空间划分为`block_size × block_size`的块，每个块搬移到批次维度。输出批次维度变为输入的`block_size × block_size`倍，空间维度相应缩小。

- 计算公式：
  设输入x为4D NHWC张量`[N, H_in, W_in, C]`，block_size = bs，paddings = `[[pad_top, pad_bottom], [pad_left, pad_right]]`。

  H_padded = H_in + pad_top + pad_bottom，W_padded = W_in + pad_left + pad_right，

  H_out = H_padded / bs，W_out = W_padded / bs。

  要求H_padded和W_padded均能被bs整除。

  输出y形状为`[N * bs * bs, H_out, W_out, C]`。

  坐标映射（输出 → 输入）：

  ```text
  n     = n_out % N
  rem   = n_out / N
  bh    = rem / bs
  bw    = rem % bs
  h_in  = h_out * bs + bh - pad_top
  w_in  = w_out * bs + bw - pad_left

  若0 ≤ h_in < H_in且0 ≤ w_in < W_in：
    y[n_out, h_out, w_out, c] = x[n, h_in, w_in, c]
  否则：
    y[n_out, h_out, w_out, c] = 0
  ```

  其中输出批次维顺序为 `n_out = n + (bh * bs + bw) * N`，即 batch 维 n 变化最快。

## 参数说明

<table style="table-layout: fixed; width: 1480px">
  <colgroup>
    <col style="width: 177px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示输入张量，4D NHWC张量 [N, H_in, W_in, C]，支持多种数据类型</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、INT64、BFLOAT16、FLOAT16、FLOAT、DOUBLE</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>输入</td>
      <td>表示空间维度zero-padding量，形状为 [2, 2] 或一维 4 元素（[4]/[1,4]），值为 [[pad_top, pad_bottom], [pad_left, pad_right]]，扁平顺序为 [pad_top, pad_bottom, pad_left, pad_right]</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>属性</td>
      <td>表示空间块的尺寸大小，为必需属性，必须是大于0的整数。H_padded和W_padded必须能被block_size整除</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示输出张量，与输入x具有相同的数据类型。输出形状为 [N*block_size*block_size, H_out, W_out, C]</td>
      <td>与x一致</td>
      <td>NHWC</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入张量x必须为4D NHWC格式。
- block_size必须大于0。
- paddings必须包含4个元素，形状可为 [2,2] 或一维4元素；每个元素 >= 0。
- H_padded = H_in + pad_top + pad_bottom必须能被block_size整除。
- W_padded = W_in + pad_left + pad_right必须能被block_size整除。
- block_size为编译期常量（算子属性），非运行时tensor输入。
- 支持空Tensor（存在零长度维）：允许N、H_in、W_in、C中任一维为0，输出shape按公式正常推导；输出为空时Tiling直接返回，不进行切分。
- paddings必须在执行阶段（Tiling）可读取为具体常量值（共4个元素），仅提供其形状不足以完成Tiling；若无法读取具体值，Tiling返回失败。
- 动态shape场景下，InferShape阶段允许H、W等空间维暂时未知，但在进入执行阶段（Tiling）前，x的形状必须已具体化且paddings的值可读取，否则Tiling返回失败。
- 非法输入（x非4D、paddings元素个数不为4或含负值、block_size≤0、H_padded或W_padded不能被block_size整除等）会在InferShape或Tiling阶段返回失败（对应构图/编译期或执行期报错），不会静默产生错误结果。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|  GE图模式  |[test_geir_space_to_batch](examples/test_geir_space_to_batch.cpp) | 通过[算子IR](./op_graph/space_to_batch_proto.h)构图方式调用SpaceToBatch算子。 |

### GE图模式示例构建与运行

> 环境准备（CANN 环境变量、仓库依赖等）请参考仓级文档：[快速入门](../../docs/QUICKSTART.md)、[算子调用指导](../../docs/zh/invocation/quick_op_invocation.md)。本示例使用 `DT_INT32`，不要求覆盖全部数据类型。

1. 在仓库根目录编译并安装算子包（以 ascend950 为例）：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --soc=ascend950 --ops=space_to_batch -j16
./build_out/cann-ops-math-*linux*.run
```

2. 编译并运行示例：

```bash
bash build.sh --run_example space_to_batch graph --soc=ascend950
```

> 运行前提：需要可用的 NPU 设备（示例固定使用 `ge.exec.deviceId=0`，即 0 号设备）；示例通过相对路径 `./dump`、`./tc_ge_irrun_test_0008_npu_input_*.bin`、`./tc_ge_irrun_test_0008_npu_output_*.bin` 写出文件，因此需在可写目录下运行（使用 `bash build.sh --run_example` 时工作目录为 `build/`）。

3. 预期输出与产物：
- 标准输出包含 `Session run ir compute graph success`、`Finalize ir graph session success`，并逐元素打印 `result[j] is: ...`。
- 可执行文件与数据文件生成在构建目录 `build/` 下：
  - `build/test_geir_space_to_batch`：示例可执行文件；
  - `build/tc_ge_irrun_test_0008_npu_input_0.bin`、`build/tc_ge_irrun_test_0008_npu_input_1.bin`：输入 x 与 paddings 的原始数据；
  - `build/tc_ge_irrun_test_0008_npu_output_0.bin`：输出 y 的原始数据（`DT_INT32`，形状 `[4,2,2,2]`）；
  - `build/dump*`：`aclgrphDumpGraph` 生成的图 dump。
