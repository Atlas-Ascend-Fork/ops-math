# aclnnNpuFormatCast

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:24:45.321Z -->

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

- **Description**:

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
    - Completes the conversion from the ND [data format](../../../docs/en/context/data_format.md) to the FRACTAL_NZ [data format](../../../docs/en/context/data_format.md) with a specified C0 size. C0 is the size of the last dimension of the FRACTAL_NZ [data format](../../../docs/en/context/data_format.md) and is determined by `additionalDtype`.
    - Completes the conversion from the FRACTAL_NZ [data format](../../../docs/en/context/data_format.md) with a specified C0 size to the ND [data format](../../../docs/en/context/data_format.md). The supported NZ formats include: FRACTAL_NZ, FRACTAL_NZ_C0_2, FRACTAL_NZ_C0_4, FRACTAL_NZ_C0_16, and FRACTAL_NZ_C0_32.
    - When the data type of srcTensor is FLOAT8_E8M0, the format of srcTensor is ND/NCL/NCHW, and dstFormat is FRACTAL_NZ, conversion of the MX scale to the Ascend-affine storage shape is supported.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
    - Implements the conversion between ND←→[NZ](../../../docs/en/context/data_format.md), NCL←→[NZ](../../../docs/en/context/data_format.md), NCHW←→[NZ](../../../docs/en/context/data_format.md), and NCDHW←→[NZ](../../../docs/en/context/data_format.md). C0 is the size of the last dimension of the [NZ](../../../docs/en/context/data_format.md) data format. Calculation method: C0 = 32B / `ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype))`.
    - Implements the conversion between NCDHW←→[NDC1HWC0](../../../docs/en/context/data_format.md), NCDHW←→[FRACTAL_Z_3D](../../../docs/en/context/data_format.md), NCHW←→[NC1HWC0](../../../docs/en/context/data_format.md), NHWC←→[NC1HWC0](../../../docs/en/context/data_format.md), NCHW←→[FRACTAL_Z](../../../docs/en/context/data_format.md), HWCN←→[FRACTAL_Z](../../../docs/en/context/data_format.md), NDHWC←→[NDC1HWC0](../../../docs/en/context/data_format.md), and DHWCN←→[FRACTAL_Z_3D](../../../docs/en/context/data_format.md). C0 is strongly related to the microarchitecture, and its value equals the size of the cube unit, for example, 16. C1 is obtained by splitting the C dimension by C0: C1=C/C0. If the result is not evenly divisible, the last portion of data is padded to C0. Calculation method: C0 = 32B / `ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype))` (for example, the enumeration value of additionalDtype for FP16 is 1, and the size of the corresponding FP16 data is 2 Byte).
  <!-- end id8 -->

- **Calculation Process**:

  `aclnnNpuFormatCastCalculateSizeAndFormat` calculates the shape and actual data format of the destination tensor dstTensor after conversion based on the source tensor srcTensor, the data type `additionalDtype`, and the data format dstFormat of the destination tensor. This information is used to construct dstTensor. Then, call `aclnnNpuFormatCast` to convert srcTensor into the destination tensor dstTensor with the actual data format.

## Function Prototype

Call `aclnnNpuFormatCastCalculateSizeAndFormat` first to calculate the shape and actual data format of dstTensor, and then call the [two-phase API](../../../docs/en/context/two_phase_api.md). For the two-phase API, first call the `aclnnNpuFormatCastGetWorkspaceSize` API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the `aclnnNpuFormatCast` API to execute the computation.

```c++
aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(
    const aclTensor* srcTensor,
    const int        dstFormat,
    const int        additionalDtype,
    int64_t**        dstShape,
    uint64_t*        dstShapeSize,
    int*             actualFormat)
```

```c++
aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(
    const aclTensor* srcTensor,
    aclTensor*       dstTensor,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```c++
aclnnStatus aclnnNpuFormatCast(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnNpuFormatCastCalculateSizeAndFormat

- **Parameters**

  <table style="table-layout: fixed; width: 1665px;">
    <colgroup>
        <col style="width: 180px">
        <col style="width: 120px">
        <col style="width: 200px">
        <col style="width: 310px">
        <col style="width: 290px">
        <col style="width: 300px">
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
          <th>Dimension (shape)</th>
          <th>Non-contiguous Tensor</th>
      </tr>
  </thead>
  <tbody>
        <tr>
            <td>srcTensor (aclTensor*)</td>
            <td>Input</td>
            <td>Source tensor of the conversion.</td>
            <td>-</td>
            <td>INT8 (2), UINT8 (4), INT32 (3), UINT32 (8), FLOAT (0), FLOAT16 (1), BFLOAT16 (27), FLOAT8_E4M3FN<sup>2</sup> (36), FLOAT4_E2M1<sup>2</sup> (40), FLOAT4_E1M2<sup>2</sup> (41), HIFLOAT8<sup>2</sup> (34)</td>
            <td>ACL_FORMAT_ND (2), ACL_FORMAT_NZ (29), ACL_FORMAT_NCL (47), ACL_FORMAT_NCHW<sup>1</sup> (0), ACL_FORMAT_NHWC<sup>1</sup> (1), ACL_FORMAT_HWCN<sup>1</sup> (16), ACL_FORMAT_NCDHW<sup>1</sup> (30), ACL_FORMAT_NDHWC<sup>1</sup> (27), ACL_FORMAT_DHWCN<sup>1</sup> (31), ACL_FORMAT_NDC1HWC0<sup>1</sup> (32), ACL_FORMAT_FRACTAL_Z_3D<sup>1</sup> (33), ACL_FORMAT_NC1HWC0<sup>1</sup> (3), ACL_FORMAT_FRACTAL_Z<sup>1</sup> (4).</td>
            <td>2-6</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dstFormat (int)</td>
            <td>Input</td>
            <td>Enumeration value corresponding to the data format of the output tensor. The value in parentheses in Instruction is the corresponding enumeration value, which is the actual input.</td>
            <td>ACL_FORMAT_ND (2), ACL_FORMAT_NZ (29), ACL_FORMAT_NCL (47), ACL_FORMAT_NCHW<sup>1</sup> (0), ACL_FORMAT_NHWC<sup>1</sup> (1), ACL_FORMAT_HWCN<sup>1</sup> (16), ACL_FORMAT_NCDHW<sup>1</sup> (30), ACL_FORMAT_NDHWC<sup>1</sup> (27), ACL_FORMAT_DHWCN<sup>1</sup> (31), ACL_FORMAT_NDC1HWC0<sup>1</sup> (32), ACL_FORMAT_FRACTAL_Z_3D<sup>1</sup> (33), ACL_FORMAT_NC1HWC0<sup>1</sup> (3), ACL_FORMAT_FRACTAL_Z<sup>1</sup> (4).</td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>additionalDtype (int)</td>
            <td>Input</td>
            <td>Basic data type used to infer the C0 size when converting to a private data format. Each specific data type corresponds to an enumeration value. The content in parentheses in the Instruction indicates the corresponding enumeration value, and the actual input is this enumeration value.</td>
            <td>If -1 is passed, additionalDtype defaults to the data type of srcTensor.</td>
            <td>INT8 (2), UINT8 (4), INT32 (3), UINT32 (8), FLOAT (0), FLOAT16 (1), BFLOAT16 (27), FLOAT8_E4M3FN<sup>2</sup> (36), FLOAT4_E2M1<sup>2</sup> (40), HIFLOAT8<sup>2</sup> (34), -1</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dstShape (int64_t**)</td>
            <td>Output</td>
            <td>Pointer used to output the shape array of dstTensor. The memory pointed to by this pointer is allocated by this API and released by the caller.</td>
            <td>-</td>
            <td>None</td>
            <td>None</td>
            <td>4-8</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dstShapeSize(uint64_t*)</td>
            <td>Output</td>
            <td>Pointer used to output the size of the shape array of dstTensor.</td>
            <td>-</td>
            <td>None</td>
            <td>None</td>
            <td>None</td>
            <td>-</td>
        </tr>
        <tr>
            <td>actualFormat(int*)</td>
            <td>Output</td>
            <td>Pointer for outputting the actual data format of dstTensor.</td>
            <td>The value in parentheses indicates the corresponding enumeration value.</td>
            <td>None</td>
            <td>ACL_FORMAT_ND (2), ACL_FORMAT_NZ (29), ACL_FORMAT_NCL (47), ACL_FORMAT_NCHW<sup>1</sup> (0), ACL_FORMAT_NHWC<sup>1</sup> (1), ACL_FORMAT_HWCN<sup>1</sup> (16), ACL_FORMAT_NCDHW<sup>1</sup> (30), ACL_FORMAT_NDHWC<sup>1</sup> (27), ACL_FORMAT_DHWCN<sup>1</sup> (31), ACL_FORMAT_NDC1HWC0<sup>1</sup> (32), ACL_FORMAT_FRACTAL_Z_3D<sup>1</sup> (33), ACL_FORMAT_NC1HWC0<sup>1</sup> (3), ACL_FORMAT_FRACTAL_Z<sup>1</sup> (4).</td>
            <td>None</td>
            <td>-</td>
        </tr>
    </tbody>
    </table>

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: In the data type column of the preceding table, superscript "1" indicates the data types or data formats not supported by the corresponding series.
  <!-- end id9 -->
  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: In the data type column of the preceding table, superscript "2" indicates the data types or data formats not supported by the corresponding series.
  <!-- end id10 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table>
    <thead>
      <tr>
        <th style="width: 291px">Return Value</th>
        <th style="width: 135px">Error Code</th>
        <th style="width: 724px">Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="1"> ACLNN_ERR_PARAM_NULLPTR </td>
        <td rowspan="1"> 161001 </td>
        <td>srcTensor is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="4"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="4"> 161002 </td>
        <td>The data format of srcTensor is out of the specified range.</td>
      </tr>
      <tr>
        <td>The data format of dstFormat is out of the specified range.</td>
      </tr>
      <tr>
        <td>The data type of additionalDtype is not within the specified range.</td>
      </tr>
      <tr>
        <td>The number of dimensions of the view shape of srcTensor is not within the range of [2, 6].</td>
      </tr>
      <tr>
        <td rowspan="2"> ACLNN_ERR_RUNTIME_ERROR </td>
        <td rowspan="2"> 361001 </td>
        <td>The product model is not supported.</td>
      </tr>
      <tr>
        <td>The conversion format is not supported.</td>
      </tr>
    </tbody>
  </table>

## aclnnNpuFormatCastGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1665px;">
    <colgroup>
        <col style="width: 180px">
        <col style="width: 120px">
        <col style="width: 300px">
        <col style="width: 200px">
        <col style="width: 290px">
        <col style="width: 300px">
        <col style="width: 130px">
        <col style="width: 145px">
  </colgroup>
  <thead>
      <tr>
          <th>Parameter</th>
          <th>Input/Output</th>
          <th>Description</th>
          <td>Instruction</td>
          <th>Data Type</th>
          <th>Data Format</th>
          <th>Dimensions (shape)</th>
          <td>Non-contiguous Tensor</td>
      </tr>
  </thead>
  <tbody>
        <tr>
            <td>srcTensor (aclTensor*)</td>
            <td>Input</td>
            <td>Input tensor. The input data supports only contiguous tensors.</td>
            <td>The values in brackets are the corresponding enumeration values.</td>
            <td>INT8(2), UINT8(4), INT32(3), UINT32(8), FLOAT(0), FLOAT16(1), BFLOAT16(27), FLOAT8_E4M3FN<sup>2</sup>(36), FLOAT4_E2M1<sup>2</sup>(40), FLOAT4_E1M2<sup>2</sup>(41), HIFLOAT8<sup>2</sup>(34)</td>
            <td>ND, NZ, NCDHW, NDC1HWC0, FRACTAL_Z_3D, NCL<sup>2</sup></td>
            <td>2-6</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dstTensor (aclTensor*)</td>
            <td>Input</td>
            <td>Destination tensor after conversion. Only contiguous tensors are supported.</td>
            <td>The corresponding enumeration values are given in brackets.</td>
            <td>INT8(2), UINT8(4), INT32(3), UINT32(8), FLOAT(0), FLOAT16(1), BFLOAT16(27), FLOAT8_E4M3FN<sup>2</sup>(36), FLOAT4_E2M1<sup>2</sup>(40), HIFLOAT8<sup>2</sup>(34)</td>
            <td>ACL_FORMAT_ND(2), ACL_FORMAT_NZ(29), ACL_FORMAT_NCL(47), ACL_FORMAT_NCHW<sup>1</sup>(0), ACL_FORMAT_NHWC<sup>1</sup>(1), ACL_FORMAT_HWCN<sup>1</sup>(16), ACL_FORMAT_NCDHW<sup>1</sup>(30), ACL_FORMAT_NDHWC<sup>1</sup>(27), ACL_FORMAT_DHWCN<sup>1</sup>(31), ACL_FORMAT_NDC1HWC0<sup>1</sup>(32), ACL_FORMAT_FRACTAL_Z_3D<sup>1</sup>(33), ACL_FORMAT_NC1HWC0<sup>1</sup>(3), ACL_FORMAT_FRACTAL_Z<sup>1</sup>(4).</td>
            <td>4-8</td>
            <td>-</td>
        </tr>
        <tr>
            <td>workspaceSize(uint64_t*)</td>
            <td>Input</td>
            <td>Size of the workspace to be requested on the Device.</td>
            <td>-</td>
            <td>None</td>
            <td>None</td>
            <td>None</td>
            <td>-</td>
        </tr>
        <tr>
            <td>executor (aclOpExecutor**)</td>
            <td>Input</td>
            <td>Op executor that contains the operator computation process.</td>
            <td>-</td>
            <td>None</td>
            <td>None</td>
            <td>None</td>
            <td>-</td>
        </tr>
    </tbody>
    </table>

  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: Superscript "1" in the data type column of the table above indicates data types or data formats not supported by the series.
  <!-- end id11 -->
  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Superscript "2" in the data type column of the table above indicates data types or data formats not supported by the series.
  <!-- end id12 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table>
    <thead>
      <tr>
        <th style="width: 291px">Return Value</th>
        <th style="width: 135px">Error Code</th>
        <th style="width: 724px">Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="1"> ACLNN_ERR_PARAM_NULLPTR </td>
        <td rowspan="1"> 161001 </td>
        <td>The input srcTensor or dstTensor is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="4"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="4"> 161002 </td>
        <td>The data type or data format of srcTensor is not in the supported range.</td>
      </tr>
      <tr>
        <td>The data type or data format of dstTensor is not in the supported range.</td>
      </tr>
      <tr>
        <td>srcTensor or dstTensor is a non-contiguous tensor.</td>
      </tr>
      <tr>
        <td>The view shape dimension of srcTensor is not in the range of [2, 6], or the storage shape dimension of dstTensor is not in the range of [4, 8].<sup>2</sup></td>
      </tr>
      <tr>
        <td rowspan="1"> ACLNN_ERR_RUNTIME_ERROR </td>
        <td rowspan="1"> 361001 </td>
        <td>The product model is not supported.</td>
      </tr>
    </tbody>
  </table>

  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>: The superscript "1" in the data type column of the preceding table indicates the interception type not supported by this series.
  <!-- end id13 -->
  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The superscript "2" in the data type column of the preceding table indicates the intercepted types that are not supported by the series.
  <!-- end id14 -->

## aclnnNpuFormatCast

- **Parameters**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
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
      <td>The workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The size of the workspace memory allocated on the device, obtained by calling the first-phase API aclnnNpuFormatCastGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnNpuFormatCast defaults to a deterministic implementation.

- The input and output support the following data type combinations:

  <!-- npu="950" id15 -->
  <details>
  <summary><term>Ascend 950PR&950DT products</term></summary>

  - aclnnNpuFormatCastCalculateSizeAndFormat API:

      | srcTensor | srcTensor [Data Format](../../../docs/en/context/data_format.md) | dstFormat | additionalDtype              | actualFormat                    |
      | --------- | -------------------------------------------------------- | --------- | ---------------------------- | ------------------------------- |
      | INT8      | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_INT8(2)                  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | INT32     | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT16(1), ACL_BF16(27) | ACL_FORMAT_FRACTAL_NZ_C0_16(50) |
      | FLOAT     | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT16(1), ACL_BF16(27) | ACL_FORMAT_FRACTAL_NZ_C0_16(50) |
      | FLOAT     | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT8_E4M3FN(36) | ACL_FORMAT_FRACTAL_NZ_C0_32(51) |
      | FLOAT16      | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT16(1) | ACL_FORMAT_FRACTAL_NZ(29) |
      | BFLOAT16     | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_BF16(27)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT8_E4M3FN     | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT8_E4M3FN(36)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT8_E8M0     | ACL_FORMAT_ND(2)/ACL_FORMAT_NCL(47)/ACL_FORMAT_NCHW(0) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT8_E8M0   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT4_E2M1 | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT8_E4M3FN(36)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT4_E2M1 | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT4_E2M1(40)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT4_E1M2 | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT4_E1M2(41)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | HIFLOAT8 | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | HIFLOAT8(34)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT4_E2M1 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_FLOAT4_E2M1(40) |  ACL_FORMAT_ND(2) |
      | FLOAT4_E2M1 | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2) | ACL_FLOAT4_E2M1(40) |  ACL_FORMAT_ND(2) |
      | INT8 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_INT8(2)  |  ACL_FORMAT_ND(2) |
      | INT8 | ACL_FORMAT_FRACTAL_NZ_C0_16(50) | ACL_FORMAT_ND(2) | ACL_INT8(2)  |  ACL_FORMAT_ND(2) |
      | INT8 | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2) | ACL_INT8(2)  |  ACL_FORMAT_ND(2) |
      | UINT8 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_UINT8(4)  |  ACL_FORMAT_ND(2) |
      | UINT8 | ACL_FORMAT_FRACTAL_NZ_C0_16(50) | ACL_FORMAT_ND(2) | ACL_UINT8(4)  |  ACL_FORMAT_ND(2) |
      | UINT8 | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2) | ACL_UINT8(4)  |  ACL_FORMAT_ND(2) |
      | FLOAT8_E4M3FN | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_FLOAT8_E4M3FN(36)  |  ACL_FORMAT_ND(2) |
      | FLOAT16 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_FLOAT16(1) |  ACL_FORMAT_ND(2) |
      | BFLOAT16 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_BF16(27) |  ACL_FORMAT_ND(2) |
      | INT32 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_INT32(3) |  ACL_FORMAT_ND(2) |
      | INT32 | ACL_FORMAT_FRACTAL_NZ_C0_2(52) | ACL_FORMAT_ND(2) | ACL_INT32(3) |  ACL_FORMAT_ND(2) |
      | INT32 | ACL_FORMAT_FRACTAL_NZ_C0_4(53) | ACL_FORMAT_ND(2) | ACL_INT32(3) |  ACL_FORMAT_ND(2) |
      | INT32 | ACL_FORMAT_FRACTAL_NZ_C0_16(50) | ACL_FORMAT_ND(2) | ACL_INT32(3) |  ACL_FORMAT_ND(2) |
      | INT32 | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2) | ACL_INT32(3) |  ACL_FORMAT_ND(2) |
      | FLOAT | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_FLOAT(0) |  ACL_FORMAT_ND(2) |
      | FLOAT | ACL_FORMAT_FRACTAL_NZ_C0_2(52) | ACL_FORMAT_ND(2) | ACL_FLOAT(0) |  ACL_FORMAT_ND(2) |
      | FLOAT | ACL_FORMAT_FRACTAL_NZ_C0_4(53) | ACL_FORMAT_ND(2) | ACL_FLOAT(0) |  ACL_FORMAT_ND(2) |
      | FLOAT | ACL_FORMAT_FRACTAL_NZ_C0_16(50) | ACL_FORMAT_ND(2) | ACL_FLOAT(0) |  ACL_FORMAT_ND(2) |
      | FLOAT | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2) | ACL_FLOAT(0) |  ACL_FORMAT_ND(2) |

  - aclnnNpuFormatCastGetWorkspaceSize API:

      | srcTensor | dstTensor Data Type | srcTensor[data format](../../../docs/en/context/data_format.md) | dstTensor[data format](../../../docs/en/context/data_format.md)            |
      | --------- | ----------------- | ------------------------------------------------------- | ------------------------------------------------------------------ |
      | INT8      | INT8              | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29)       |
      | INT32     | INT32             | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ_C0_16(50) |
      | FLOAT     | FLOAT             | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ_C0_16(50)/ACL_FORMAT_FRACTAL_NZ_C0_32(51) |
      | FLOAT16   | FLOAT16           | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29)       |
      | BFLOAT16  | BFLOAT16          | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29)       |
      | FLOAT8_E4M3FN  | FLOAT8_E4M3FN   | ACL_FORMAT_ND(2)  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | FLOAT8_E8M0  | FLOAT8_E8M0   | ACL_FORMAT_ND(2)/ACL_FORMAT_NCL(47)/ACL_FORMAT_NCHW(0)  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | FLOAT4_E2M1  | FLOAT4_E2M1       | ACL_FORMAT_ND(2)  | ACL_FORMAT_FRACTAL_NZ_C0_32(51)       |
      | FLOAT4_E2M1  | FLOAT4_E2M1       | ACL_FORMAT_ND(2)  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | FLOAT4_E1M2  | FLOAT4_E1M2       | ACL_FORMAT_ND(2)  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | HIFLOAT8  | HIFLOAT8             | ACL_FORMAT_ND(2)  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | INT8      | INT8                 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2)       |
      | INT8      | INT8                 | ACL_FORMAT_FRACTAL_NZ_C0_16(50) | ACL_FORMAT_ND(2)       |
      | INT8      | INT8                 | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2)       |
      | UINT8     | UINT8                | ACL_FORMAT_FRACTAL_NZ(29)       | ACL_FORMAT_ND(2)       |
      | UINT8     | UINT8                | ACL_FORMAT_FRACTAL_NZ_C0_16(50) | ACL_FORMAT_ND(2)       |
      | UINT8     | UINT8                | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2)       |
      | INT32     | INT32             | ACL_FORMAT_FRACTAL_NZ(29)          | ACL_FORMAT_ND(2)  |
      | INT32     | INT32             | ACL_FORMAT_FRACTAL_NZ_C0_2(52)     | ACL_FORMAT_ND(2)  |
      | INT32     | INT32             | ACL_FORMAT_FRACTAL_NZ_C0_4(53)     | ACL_FORMAT_ND(2)  |
      | INT32     | INT32             | ACL_FORMAT_FRACTAL_NZ_C0_16(50)    | ACL_FORMAT_ND(2)  |
      | INT32     | INT32             | ACL_FORMAT_FRACTAL_NZ_C0_32(51)    | ACL_FORMAT_ND(2)  |
      | FLOAT     | FLOAT             | ACL_FORMAT_FRACTAL_NZ(29)          | ACL_FORMAT_ND(2)  |
      | FLOAT     | FLOAT             | ACL_FORMAT_FRACTAL_NZ_C0_2(52)     | ACL_FORMAT_ND(2)  |
      | FLOAT     | FLOAT             | ACL_FORMAT_FRACTAL_NZ_C0_4(53)     | ACL_FORMAT_ND(2)  |
      | FLOAT     | FLOAT             | ACL_FORMAT_FRACTAL_NZ_C0_16(50)    | ACL_FORMAT_ND(2)  |
      | FLOAT     | FLOAT             | ACL_FORMAT_FRACTAL_NZ_C0_32(51)    | ACL_FORMAT_ND(2)  |
      | FLOAT16   | FLOAT16           | ACL_FORMAT_FRACTAL_NZ(29)          | ACL_FORMAT_ND(2)       |
      | BFLOAT16  | BFLOAT16          | ACL_FORMAT_FRACTAL_NZ(29)          | ACL_FORMAT_ND(2)        |
      | FLOAT8_E4M3FN  | FLOAT8_E4M3FN    |  ACL_FORMAT_FRACTAL_NZ(29)     | ACL_FORMAT_ND(2)        |
      | FLOAT4_E2M1  | FLOAT4_E2M1        |  ACL_FORMAT_FRACTAL_NZ(29)     | ACL_FORMAT_ND(2)        |
      | FLOAT4_E2M1  | FLOAT4_E2M1        |  ACL_FORMAT_FRACTAL_NZ_C0_32(51)  | ACL_FORMAT_ND(2)        |

  - C0 calculation method: $C0=\frac{32B}{size\ of\ additionalDtype}$

      | additionalDtype | C0 |
      | --------------- | -- |
      | ACL_INT8(2)     | 32 |
      | ACL_FLOAT16(1)  | 16 |
      | ACL_BF16(27)    | 16 |
      | ACL_FLOAT8_E4M3FN(36)    | 32 |
      | ACL_HIFLOAT8(34)    | 32 |

  - FLOAT8_E8M0 special scenarios:
    - When the data type of srcTensor is FLOAT8_E8M0, the format of srcTensor is ND/NCL/NCHW, and dstFormat is FRACTAL_NZ, only 3-dimensional or 4-dimensional scale inputs are supported. The storage shapes after conversion are `(ceil(N/16), ceil(K/64), 16, 2)` or `(g, ceil(N/16), ceil(K/64), 16, 2)`, respectively.
    - The capability of converting FLOAT8_E8M0 scale to the Ascend-affine storage shape does not support graph mode. Only the single-operator mode and direct aclnn call scenario are supported.

  - Currently unsupported special scenarios:
    - When the data type of srcTensor is the same as additionalDtype, the format of srcTensor is ND, and the type is FLOAT16 or BFLOAT16, if the dimensions are represented as [k, n], the scenario where k is 1 is not yet supported.
    - It is not supported to perform any operation that modifies the tensor, such as contiguous, pad, and slice, after calling this API to convert to the Ascend-affine [data format](../../../docs/en/context/data_format.md) FRACTAL_NZ;
    - When the shape of any of the last two dimensions of srcTensor is 1, it is also not allowed to perform any operation that modifies the tensor, including transpose, after converting to the Ascend-affine [data format](../../../docs/en/context/data_format.md) FRACTAL_NZ.
  </details>
  <!-- end id15 -->

  <!-- npu="A3,910b" id16 -->
  <details>
  <summary><term>Atlas A3 products</term>, <term>Atlas A2 products</term></summary>

  - aclnnNpuFormatCastCalculateSizeAndFormat API parameters:

      | srcTensor | dstFormat                 | additionalDtype              | actualFormat                    |
      | --------- | ------------------------- | ---------------------------- | ------------------------------- |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_ND(2)     | ACL_FORMAT_FRACTAL_NZ(29) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_NCL(47)    | ACL_FORMAT_FRACTAL_NZ(29) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32   | ACL_FORMAT_FRACTAL_NZ(29) |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_NCHW(0)     | ACL_FORMAT_FRACTAL_NZ(29) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_FRACTAL_NZ(29)    |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_NCDHW(30)     | ACL_FORMAT_FRACTAL_NZ(29) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_FRACTAL_NZ(29)    |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_FRACTAL_NZ(29)    | ACL_FORMAT_ND(2) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32           | ACL_FORMAT_ND(2)       |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_FRACTAL_NZ(29)    | ACL_FORMAT_NCL(47) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32           | ACL_FORMAT_NCL(47)       |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_FRACTAL_NZ(29)    | ACL_FORMAT_NCHW(0) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32           | ACL_FORMAT_NCHW(0)       |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_FRACTAL_NZ(29)    | ACL_FORMAT_NCDHW(30) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32         | ACL_FORMAT_NCDHW(30)     |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_NCDHW(30)    | ACL_FORMAT_NDC1HWC0(32) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32           | ACL_FORMAT_NDC1HWC0(32)  |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_NDC1HWC0(32)    | ACL_FORMAT_NCDHW(30) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32           | ACL_FORMAT_NCDHW(30)     |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_NCDHW(30)    |ACL_FORMAT_FRACTAL_Z_3D(33) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_FRACTAL_Z_3D(33)  |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_FRACTAL_Z_3D(33)    |ACL_FORMAT_NCDHW(30) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NCDHW(30)  |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_DHWCN(31)    | ACL_FORMAT_FRACTAL_Z_3D(33) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_FRACTAL_Z_3D(33) |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_NCHW(0)   | ACL_FORMAT_NC1HWC0(3) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NC1HWC0(3) |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_NC1HWC0(3)   | ACL_FORMAT_NCHW(0) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NCHW(0) |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_NHWC(1)   | ACL_FORMAT_NC1HWC0(3) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NC1HWC0(3) |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_NC1HWC0(3)   | ACL_FORMAT_NHWC(1) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NHWC(1) |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_NCHW(0)   | ACL_FORMAT_FRACTAL_Z(4) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_FRACTAL_Z(4) |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_FRACTAL_Z(4)   | ACL_FORMAT_NCHW(0) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NCHW(0) |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_HWCN(16)   | ACL_FORMAT_FRACTAL_Z(4) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_FRACTAL_Z(4) |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_FRACTAL_Z(4)   | ACL_FORMAT_HWCN(16) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_HWCN(16) |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_NDHWC(27)   | ACL_FORMAT_NDC1HWC0(32) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NDC1HWC0(32) |
      | Data type: INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32; data format: ACL_FORMAT_NDC1HWC0(32)   | ACL_FORMAT_NDHWC(27) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NDHWC(27) |

  - aclnnNpuFormatCastGetWorkspaceSize API:

      | srcTensor | dstTensor Data Type | dstTensor Data Format               |
      | --------- | ----------------- | ------------------------------- |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_FRACTAL_NZ(29)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_ND(2)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NCDHW(30)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NDC1HWC0(32)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_FRACTAL_Z_3D(33)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NCL(47)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NCHW(0)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NC1HWC0(3)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NHWC(1)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_FRACTAL_Z(4)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_HWCN(16)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NDHWC(27)       |

  - C0 calculation method: `C0 = 32B / ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype))`

      | srcTensor Base Type | C0 |
      | --------------- | -- |
      | ACL_FLOAT(0)、ACL_INT32(3)、ACL_UINT32(8)     | 8 |
      | ACL_FLOAT16(1)、ACL_BF16(27)  | 16 |
      | ACL_INT8(2)、ACL_UINT8(4)、ACL_HIFLOAT8(34)    | 32 |

  - Currently unsupported special scenarios:
    - After calling this API to convert to the Ascend-affinity [data format](../../../docs/en/context/data_format.md) FRACTAL_NZ, operations that can modify the tensor, such as contiguous, pad, and slice, are not supported;
    - After converting to the Ascend-affinity [data format](../../../docs/en/context/data_format.md) FRACTAL_NZ, any operation that modifies the tensor, including transpose, is not allowed.
  </details>
  <!-- end id16 -->

## Example

<!-- npu="950" id17 -->
- <term>Ascend 950PR&950DT products</term>:

  The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_npu_format_cast.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  extern "C" aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(const aclTensor* srcTensor, const int dstFormat, const int additionalDtype,  int64_t** dstShape, uint64_t* dstShapeSize, int* actualFormat);
  extern "C" aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor* srcTensor, aclTensor* dstTensor,uint64_t* workspaceSize, aclOpExecutor** executor);
  extern "C" aclnnStatus aclnnNpuFormatCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

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

      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape, uint64_t* storageShapeSize, void** deviceAddr,
                                aclDataType dataType, aclTensor** tensor, aclFormat format) {
      auto size = hostData.size() * sizeof(T);
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy data from the host to the device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                  format, *storageShape, *storageShapeSize, *deviceAddr);
      return 0;
  }

  int main() {
      // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Customize the construction based on the API.
      int64_t k = 64;
      int64_t n = 128;
      int64_t srcDim0 = k;
      int64_t srcDim1 = n;
      int dstFormat = 29;
      aclDataType srcDtype = aclDataType::ACL_INT32;
      aclDataType additionalDtype = aclDataType::ACL_FLOAT16;

      std::vector<int64_t> srcShape = {srcDim0, srcDim1};
      void* srcDeviceAddr = nullptr;
      void* dstDeviceAddr = nullptr;
      aclTensor* srcTensor = nullptr;
      aclTensor* dstTensor= nullptr;
      std::vector<int32_t> srcHostData(k * n, 1);
      for (size_t i = 0; i < k; i++) {
          for (size_t j = 0; j < n; j++) {
              srcHostData[i * n + j] = (j + 1) % 128;
          }
      }

      std::vector<int32_t> dstTensorHostData(k * n, 1);

      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      int actualFormat;

      // Create src  aclTensor.
      ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, srcDtype, &srcTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;

      // Calculate the shape and format of the target tensor
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, 29, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);

      ret = CreateAclTensorWithFormat(dstTensorHostData, srcShape, &dstShape, &dstShapeSize, &dstDeviceAddr, srcDtype, &dstTensor, static_cast<aclFormat>(actualFormat));
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API aclnnNpuFormatCastGetWorkspaceSize
      ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      // Call the second-phase API aclnnNpuFormatCast
      ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Synchronize and wait for the task execution to complete
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from device memory to the host
      auto size = 1;
      for (size_t i = 0; i < dstShapeSize; i++) {
          size *= dstShape[i];
      }

      std::vector<int32_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dstDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
      }

      // 6. Release dstShape, aclTensor, and aclScalar
      delete[] dstShape;
      aclDestroyTensor(srcTensor);
      aclDestroyTensor(dstTensor);

      // 7. Release device resources
      aclrtFree(srcDeviceAddr);
      aclrtFree(dstDeviceAddr);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```
<!-- end id17 -->

<!-- npu="A3,910b" id18 -->
- x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.

  The sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_npu_format_cast.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  extern "C" aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(const aclTensor* srcTensor, const int dstFormat, const int additionalDtype,  int64_t** dstShape, uint64_t* dstShapeSize, int* actualFormat);
  extern "C" aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor* srcTensor, aclTensor* dstTensor,uint64_t* workspaceSize, aclOpExecutor** executor);
  extern "C" aclnnStatus aclnnNpuFormatCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

  int Init(int32_t deviceId, aclrtStream* stream) {
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
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
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
      // Modify the format of src here.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                                  shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape, uint64_t* storageShapeSize, void** deviceAddr,
                                aclDataType dataType, aclTensor** tensor, aclFormat format) {
      auto size = hostData.size() * sizeof(T);
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

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                  format, *storageShape, *storageShapeSize, *deviceAddr);
      return 0;
  }

  int main() {
      // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Custom construction is required based on the API.

      int dstFormat = 32;
      // Modify the target format here.
      aclDataType srcDtype = aclDataType::ACL_INT32;
      int additionalDtype = -1;

      // std::vector<int64_t> srcShape = {srcDim0 , srcDim1};
      int64_t N = 1;
      int64_t C = 17;
      int64_t D = 1;
      int64_t H = 2;
      int64_t W = 2;

      std::vector<int64_t> srcShape = {N, C, D, H, W};
      void* srcDeviceAddr = nullptr;
      void* dstDeviceAddr = nullptr;
      aclTensor* srcTensor = nullptr;
      aclTensor* dstTensor= nullptr;
      std::vector<int32_t> srcHostData(N * C * D * H * W, 1);

      int num = 0;
      for (int n = 0; n < N; ++n) {
          for (int c = 0; c < C; ++c) {
              for (int d = 0; d < D; ++d) {
                  for (int h = 0; h < H; ++h) {
                      for (int w = 0; w < W; ++w) {
                          // Arranged in row-major order. Compute the linear index.
                          int index = (((n * C + c) * D + d) * H + h) * W + w;
                          srcHostData[index] = num;
                          num++;
                      }
                  }
              }
          }
      }

      std::vector<int32_t> dstTensorHostData(N * C * D * H * W, 1);

      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      int actualFormat;

      // Create src  aclTensor.
      ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, srcDtype, &srcTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;
      std::cout << "init actualFormat = " << actualFormat << std::endl;
      // Compute the shape and format of the target tensor.
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, dstFormat, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);

      std::cout << "actualFormat = " << actualFormat << std::endl;
      std::cout << "&dstShape = " << &dstShape << std::endl;
      std::cout << "dstShape = [ ";
      for (int64_t i = 0; i < dstShapeSize; ++i) {
          std::cout << dstShape[i] << " ";
      }
      std::cout << "]" << std::endl;

      ret = CreateAclTensorWithFormat(dstTensorHostData, srcShape, &dstShape, &dstShapeSize, &dstDeviceAddr, srcDtype, &dstTensor, static_cast<aclFormat>(actualFormat));
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API aclnnNpuFormatCastGetWorkspaceSize.
      ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize computed by the first-phase API.

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      // Call the second-phase API aclnnNpuFormatCast.
      ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Wait synchronously for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host memory.
      auto size = 1;
      for (size_t i = 0; i < dstShapeSize; i++) {
          size *= dstShape[i];
      }

      std::vector<int32_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dstDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
      }

      // 6. Release dstShape, aclTensor, and aclScalar.
      delete[] dstShape;
      aclDestroyTensor(srcTensor);
      aclDestroyTensor(dstTensor);

      // 7. Release device resources.
      aclrtFree(srcDeviceAddr);
      aclrtFree(dstDeviceAddr);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```
<!-- end id18 -->
