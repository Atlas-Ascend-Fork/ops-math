# SpaceToBatch

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :-------: |
| <term>Ascend 950PR&950DT products</term>                             | √        |
| <term>Atlas A3 products</term>     | √        |
| <term>Atlas A2 products</term> | √        |
| <term>Atlas 200I/500 A2 inference products</term>                      |     ×     |
| <term>Atlas inference products</term>                             | √        |
| <term>Atlas training products</term>                              | √        |

## Function Description

- Operator function: rearranges blocks of data in the spatial dimensions into the batch dimension, and pads the spatial dimensions with zeros.

- Function Description:
  This operator first performs zero-padding on the spatial dimensions of the input according to the paddings parameter. Then it divides the padded spatial data into blocks of `block_size × block_size`, and moves each block to the batch dimension. The batch dimension of the output becomes `block_size × block_size` times that of the input, and the spatial dimensions are reduced accordingly.

- Computation formula:
  Assume that the input x is a 4D NHWC tensor `[N, H_in, W_in, C]`, block_size = bs, and paddings = `[[pad_top, pad_bottom], [pad_left, pad_right]]`.

  H_padded = H_in + pad_top + pad_bottom, W_padded = W_in + pad_left + pad_right,

  H_out = H_padded / bs, W_out = W_padded / bs.

  H_padded and W_padded must both be divisible by bs.

  The shape of the output y is `[N * bs * bs, H_out, W_out, C]`.

  Coordinate mapping (output → input):

  ```text
  n     = n_out % N
  rem   = n_out / N
  bh    = rem / bs
  bw    = rem % bs
  h_in  = h_out * bs + bh - pad_top
  w_in  = w_out * bs + bw - pad_left

  If 0 ≤ h_in < H_in and 0 ≤ w_in < W_in:
    y[n_out, h_out, w_out, c] = x[n, h_in, w_in, c]
  Otherwise:
    y[n_out, h_out, w_out, c] = 0
  ```

  The output batch dimension order is `n_out = n + (bh * bs + bw) * N`, which means the batch dimension n changes fastest.

## Parameter Description

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
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor, which is a 4D NHWC tensor [N, H_in, W_in, C] and supports multiple data types</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, INT64, BFLOAT16, FLOAT16, FLOAT, DOUBLE</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>Input</td>
      <td>Zero-padding amounts of the spatial dimensions. The shape is [2, 2] or a one-dimensional array of 4 elements ([4]/[1,4]). The value is [[pad_top, pad_bottom], [pad_left, pad_right]], and the flat order is [pad_top, pad_bottom, pad_left, pad_right]</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>Attribute</td>
      <td>Size of the spatial block. This parameter is a required attribute and must be an integer greater than 0. H_padded and W_padded must be divisible by block_size</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, which has the same data type as the input x. The output shape is [N*block_size*block_size, H_out, W_out, C]</td>
      <td>Same as x</td>
      <td>NHWC</td>
    </tr>
  </tbody>
</table>

## Constraint Description

- The input tensor x must be in 4D NHWC format.
- block_size must be greater than 0.
- paddings must contain 4 elements. The shape can be [2, 2] or a one-dimensional array of 4 elements. Each element must be >= 0.
- H_padded = H_in + pad_top + pad_bottom must be divisible by block_size.
- W_padded = W_in + pad_left + pad_right must be divisible by block_size.
- block_size is a compile-time constant (operator attribute), not a runtime tensor input.
- Empty tensors are supported (a zero-length dimension exists). N, H_in, W_in, or C can be 0, and the output shape is derived normally according to the formula. When the output is empty, tiling returns directly without splitting.
- paddings must be readable as concrete constant values (4 elements in total) during the execution phase (tiling). Providing only its shape is insufficient for tiling. If the concrete values cannot be read, tiling returns a failure.
- In dynamic shape scenarios, InferShape allows spatial dimensions such as H and W to be unknown temporarily. Before entering the execution phase (tiling), the shape of x must be concrete and the values of paddings must be readable. Otherwise, tiling returns a failure.
- Invalid inputs (for example, x is not 4D, paddings does not contain 4 elements or contains negative values, block_size is less than or equal to 0, or H_padded or W_padded is not divisible by block_size) cause a failure during InferShape or tiling. The failure corresponds to a graph construction, compilation, or execution error, and the operator does not silently produce incorrect results.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|  GE graph mode  |[test_geir_space_to_batch](examples/test_geir_space_to_batch.cpp) | Invokes the SpaceToBatch operator by building a graph through the [operator IR](./op_graph/space_to_batch_proto.h). |

### GE Graph Mode Sample Build and Run

> For environment preparation (CANN environment variables, repository dependencies, and so on), refer to the repository-level documents: [Quick Start](../../docs/QUICKSTART.md) and [Operator Invocation](../../docs/zh/invocation/quick_op_invocation.md). This sample uses `DT_INT32` and does not cover all data types.

1. Build and install the operator package in the repository root directory (ascend950 is used as an example):

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --soc=ascend950 --ops=space_to_batch -j16
./build_out/cann-ops-math-*linux*.run
```

2. Build and run the sample:

```bash
bash build.sh --run_example space_to_batch graph --soc=ascend950
```

> Prerequisites: A usable NPU device is required (the sample uses `ge.exec.deviceId=0`, that is, device 0). The sample writes files through the relative paths `./dump`, `./tc_ge_irrun_test_0008_npu_input_*.bin`, and `./tc_ge_irrun_test_0008_npu_output_*.bin`. Therefore, run the sample in a writable directory (when you use `bash build.sh --run_example`, the working directory is `build/`).

3. Expected output and artifacts:
- The standard output contains `Session run ir compute graph success` and `Finalize ir graph session success`, and prints `result[j] is: ...` for each element.
- The executable file and data files are generated in the build directory `build/`:
  - `build/test_geir_space_to_batch`: sample executable file.
  - `build/tc_ge_irrun_test_0008_npu_input_0.bin` and `build/tc_ge_irrun_test_0008_npu_input_1.bin`: raw data of the input x and paddings.
  - `build/tc_ge_irrun_test_0008_npu_output_0.bin`: raw data of the output y (`DT_INT32`, shape `[4,2,2,2]`).
  - `build/dump*`: graph dump generated by `aclgrphDumpGraph`.
