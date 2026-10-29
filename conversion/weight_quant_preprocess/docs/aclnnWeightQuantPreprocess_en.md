# aclnnWeightQuantPreprocess

<!-- codespell:ignore outWeight -->
<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:09:33.330Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

Performs parameter preprocessing for pseudo-quantization Matmul (including QuantBatchMatmulV5, GroupedMatmul-pseudo-quantization, and WeightQuantBatchMatmulV2). The supported data flows are as follows:

- MM_MX_A8W4 data flow: MX quantization scenario of the QuantBatchMatmulV4 operator (x is FLOAT8_E4M3FN, weight is compact 4-bit FLOAT4_E2M1), where weight is converted from ND to FRACTAL_NZ format;
- GMM_MX_A8W4 data flow: MX quantization scenario of the GroupedMatmul operator (x is FLOAT8_E4M3FN, weight is compact 4-bit FLOAT4_E2M1); weight is converted from ND to FRACTAL_NZ format;
- MM_A16S4 data flow: pertensor/perchannel/pergroup quantization scenarios of the WeightQuantBatchMatmulV2 operator (x is FLOAT16/BFLOAT16, weight is compact 4-bit INT4); the pertensor (transpose/non-transpose) and perchannel/pergroup transpose scenarios are ND direct copy pass-through, and the perchannel/pergroup non-transpose scenario converts weight from ND to FRACTAL_NZ format;
- MM_A16F4 data flow: pergroup quantization scenario of the WeightQuantBatchMatmulV2 operator (x is FLOAT16/BFLOAT16, weight is compact 4-bit FLOAT4_E2M1); only the non-transpose scenario is supported, and weight is converted from ND to FRACTAL_NZ format;
- MM_A16MXF4 data flow: MX quantization scenario of the WeightQuantBatchMatmulV2 operator (x is FLOAT16/BFLOAT16, weight is compact 4-bit FLOAT4_E2M1); the transpose scenario is ND direct copy pass-through, and the non-transpose scenario converts weight from ND to FRACTAL_NZ format.

During the conversion, weightScale, weightOffsetOptional, and biasOptional are synchronized as needed.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md).

1. Call `aclnnWeightQuantPreprocessGetWorkspaceSize` to obtain the workspace size and executor;
2. Call `aclnnWeightQuantPreprocess` to perform the computation.

**Note**: You need to construct the output tensor by yourself. For details, see the shape calculation formula in Constraints.

```c++
aclnnStatus aclnnWeightQuantPreprocessGetWorkspaceSize(
    const aclTensor  *weight,
    const aclTensor  *weightScale,
    const aclTensor  *weightOffsetOptional,
    const aclTensor  *biasOptional,
    const aclDataType xDtype,
    const aclDataType xScaleDtype,
    const int64_t     kGroupSize,
    aclTensor        *outWeight,
    aclTensor        *outWeightScale,
    aclTensor        *outWeightOffsetOptional,
    aclTensor        *outBiasOptional,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)

aclnnStatus aclnnWeightQuantPreprocess(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnWeightQuantPreprocessGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1195px"><colgroup>
  <col style="width: 220px">
  <col style="width: 90px">
  <col style="width: 160px">
  <col style="width: 270px">
  <col style="width: 180px">
  <col style="width: 95px">
  <col style="width: 120px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">weight(aclTensor *)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Weight matrix of Matmul.</td>
      <td class="tg-0pky">Does not support null tensor.</td>
      <td class="tg-0pky">float4_e2m1/int4</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2-3</td>
      <td class="tg-0pky">Only the transpose-only scenario is supported.</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightScale(aclTensor *)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Dequantization scale parameter of the weight.</td>
      <td class="tg-0pky">Does not support null tensors.</td>
      <td class="tg-0pky">float8_e8m0/float16/bfloat16</td>
      <td class="tg-0pky">ND/NCL/NCHW</td>
      <td class="tg-0pky">1-4</td>
      <td class="tg-0pky">Only supported in the transpose-only scenario.</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightOffsetOptional(aclTensor *)</td>
      <td class="tg-0pky">Optional input</td>
      <td class="tg-0pky">Dequantization offset parameter of the weight.</td>
      <td class="tg-0pky">The MM_A16S4 data flow supports pass-through (requires the same viewShape and data type as weightScale); other data flows do not support it, and it must be nullptr.</td>
      <td class="tg-0pky">float16/bfloat16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-2</td>
      <td class="tg-0pky">Supported only in the transpose-only scenario.</td>
    </tr>
    <tr>
      <td class="tg-0pky">biasOptional(aclTensor *)</td>
      <td class="tg-0pky">Optional input</td>
      <td class="tg-0pky">Bias matrix of Matmul.</td>
      <td class="tg-0pky">Does not support null tensors and must be contiguous.</td>
      <td class="tg-0pky">float16/bfloat16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-2</td>
      <td class="tg-0pky">Not supported</td>
    </tr>
    <tr>
      <td class="tg-0pky">xDtype(aclDataType)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Data type of the Matmul activation matrix.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">aclDataType</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">xScaleDtype(aclDataType)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Data type of the activation quantization scale parameter.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">aclDataType</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">kGroupSize(int64_t)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Group size along the K dimension of the weight in pergroup quantization.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">int64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">outWeight(aclTensor *)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Weight after preprocessing.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">int8/int4/fp8_e4m3/hif8/fp4_e2m1</td>
      <td class="tg-0pky">ND/NZ</td>
      <td class="tg-0pky">2-5</td>
      <td class="tg-0pky">Supported in the transpose-only scenario.</td>
    </tr>
    <tr>
      <td class="tg-0pky">outWeightScale(aclTensor *)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">weightScale after preprocessing.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">float16/bfloat16/fp8_e8m0</td>
      <td class="tg-0pky">ND/NCL/NCHW</td>
      <td class="tg-0pky">1-4</td>
      <td class="tg-0pky">Supported in transpose-only scenario.</td>
    </tr>
    <tr>
      <td class="tg-0pky">outWeightOffsetOptional(aclTensor *)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">weightOffset after preprocessing.</td>
      <td class="tg-0pky">When weightOffsetOptional is not null, this parameter must not be nullptr and must have the same viewShape, storageShape, and data type as weightOffsetOptional. In other cases, it must be nullptr.</td>
      <td class="tg-0pky">float16/bfloat16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-2</td>
      <td class="tg-0pky">Supported in transpose-only scenario.</td>
    </tr>
    <tr>
      <td class="tg-0pky">outBiasOptional(aclTensor *)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">bias after preprocessing.</td>
      <td class="tg-0pky">Must be contiguous.</td>
      <td class="tg-0pky">float16/bfloat16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-2</td>
      <td class="tg-0pky">Not supported</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize(uint64_t *)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Size of the workspace required for computation.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">uint64*</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor(aclOpExecutor **)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Executor containing the operator's computation flow.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">aclOpExecutor**</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Return Value</th>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">weight, weightScale, outWeight, or outWeightScale is a null pointer; or biasOptional is not null but outBiasOptional is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="6">161002</td>
      <td class="tg-0pky">The input data type combination is not supported and cannot match any supported data flow (MM_MX_A8W4/GMM_MX_A8W4/MM_A16S4/MM_A16F4/MM_A16MXF4).</td>
    </tr>
    <tr>
      <td class="tg-0lax">weight, weightScale, outWeight, or outWeightScale is an empty tensor; or biasOptional/outBiasOptional is an empty tensor if provided.</td>
    </tr>
    <tr>
      <td class="tg-0lax">The data type and data format of weight, weightScale, biasOptional, outWeight, outWeightScale, or outBiasOptional are not within the supported range.</td>
    </tr>
    <tr>
      <td class="tg-0lax">The shape or storage shape of weight, weightScale, biasOptional, outWeight, outWeightScale, or outBiasOptional does not meet the verification conditions.</td>
    </tr>
    <tr>
      <td class="tg-0lax">The stride of weight or weightScale does not meet the transpose requirement, or biasOptional/outBiasOptional is non-contiguous if provided.</td>
    </tr>
    <tr>
      <td class="tg-0lax">weightOffsetOptional or outWeightOffsetOptional is not null (for the MM_MX_A8W4/GMM_MX_A8W4/MM_A16F4/MM_A16MXF4 data flows; the MM_A16S4 data flow supports pass-through and requires the same viewShape and data type as weightScale), or kGroupSize is not equal to 32 (for the MM_MX_A8W4/GMM_MX_A8W4/MM_A16MXF4 data flows); the MM_A16S4/MM_A16F4 data flows also require kGroupSize to match the quantization granularity of weightScale.</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_RUNTIME_ERROR</td>
      <td class="tg-0pky">361001</td>
      <td class="tg-0pky">The product model is not supported.</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td class="tg-0pky">561101</td>
      <td class="tg-0pky">Internal error. Failed to create the executor.</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_INNER_NULLPTR</td>
      <td class="tg-0pky">561103</td>
      <td class="tg-0pky">workspaceSize or executor is a null pointer, or the internal graph construction API returns a null pointer.</td>
    </tr>
  </tbody>
  </table>

## aclnnWeightQuantPreprocess

- **Parameters**

  | Name     | Input/Output | Description                                                  |
  | ------------- | ------------ | ------------------------------------------------------------ |
  | workspace     | Input        | Device-side allocated workspace memory address.            |
  | workspaceSize | Input        | Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnWeightQuantPreprocessGetWorkspaceSize. |
  | executor      | Input        | Op executor that contains the operator computation flow.   |
  | stream        | Input        | Stream on which the task is executed.                               |

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism description: aclnnWeightQuantPreprocess defaults to a deterministic implementation.

<details>
<summary><strong>MM_MX_A8W4 data flow</strong> (MM indicates Matmul; MX_A8W4 indicates that the data type of x is FLOAT8_E4M3FN, the data type of weight is FLOAT4_E2M1, and the quantization mode is Mx)</summary>

- **weight**
  - Data type: FLOAT4_E2M1
  - Format: ND
  - K % kGroupSize == 0
  - view shape: 2-D `{K, N}`
  - storage shape: `{N, K}` (transposed)
  - stride: `[1, K]` (last two dimensions transposed)
  - Does not support null tensor.

- **weightScale**
  - Data type: FLOAT8_E8M0
  - Format: ND/NCL
  - view shape: 3-D `{ceildiv(K, 64), N, 2}`
  - storage shape: `{N, ceildiv(K, 64), 2}` (transposed)
  - stride: `[2, 2*ceildiv(K,64), 1]` (dimensions 0 and 1 swapped)
  - Does not support null tensor.

- **weightOffsetOptional**
  - Currently does not support this parameter; it must be nullptr
  - outWeightOffsetOptional must also be nullptr

- **biasOptional**
  - Data type: float16/bfloat16
  - Format: ND
  - Must be contiguous
  - Does not support null tensor (if provided)

- **kGroupSize**
  - Must be equal to 32

- **xDtype**
  - FLOAT8_E4M3FN

- **xScaleDtype**
  - FLOAT8_E8M0

- **outWeight**
  - Data type: same as weight
  - Format: FRACTAL_NZ_C0_32
  - view shape: same as weight view shape `{K, N}`
  - storage shape: 4-D `{ceildiv(K, 32), ceildiv(N, 16), 16, 32}`

- **outWeightScale**
  - Data type: same as weightScale
  - Format: ND
  - view shape: same as weightScale view shape
  - storage shape: same as weightScale storage shape

- **outBiasOptional**
  - Data type: same as biasOptional
  - Format: ND
  - Must be contiguous
  - view shape: same as biasOptional view shape
  - storage shape: same as biasOptional storage shape

</details>

<details>
<summary><strong>GMM_MX_A8W4 data flow</strong> (GMM indicates GroupedMatmul; MX_A8W4 indicates that the data type of x is FLOAT8_E4M3FN, the data type of weight is FLOAT4_E2M1, and Mx quantization mode is used)</summary>

- **weight**
  - Data type: FLOAT4_E2M1
  - Format: ND
  - K % kGroupSize == 0
  - view shape: 3-D `{G, K, N}`
  - storage shape: `{G, N, K}` (transposed, the last two dimensions are swapped)
  - stride: `[K*N, 1, K]` (dimensions 1 and 2 are transposed)
  - does not support null tensor

- **weightScale**
  - Data type: FLOAT8_E8M0
  - Format: ND/NCL/NCHW
  - view shape: 4-D `{G, ceildiv(K, 64), N, 2}`
  - storage shape: `{G, N, ceildiv(K, 64), 2}` (transposed, dimensions 2 and 3 are swapped)
  - stride: `[2*ceildiv(K,64)*N, 2, 2*ceildiv(K,64), 1]` (dimensions 2 and 3 are swapped)
  - Does not support null tensors.

- **weightOffsetOptional**
  - Currently not supported; must be nullptr.
  - outWeightOffsetOptional must also be nullptr.

- **biasOptional**
  - Data type: float16/bfloat16
  - Format: ND
  - Must be contiguous
  - Does not support null tensor (if provided)

- **kGroupSize**
  - Must be equal to 32

- **xDtype**
  - FLOAT8_E4M3FN

- **xScaleDtype**
  - FLOAT8_E8M0

- **outWeight**
  - Data type: same as weight
  - Format: FRACTAL_NZ_C0_32
  - view shape: same as weight view shape `{G, K, N}`
  - storage shape: 5-D `{G, ceildiv(K, 32), ceildiv(N, 16), 16, 32}`

- **outWeightScale**
  - Data type: same as weightScale
  - Format: ND
  - view shape: same as weightScale view shape
  - Storage shape: same as weightScale storage shape

- **outBiasOptional**
  - Data type: same as biasOptional
  - Format: ND
  - Must be contiguous
  - View shape: same as biasOptional
  - Storage shape: same as biasOptional

</details>

<!-- npu="950" id7 -->
<details>
<summary><strong>MM_A16S4 data flow</strong> (MM indicates Matmul; A16S4 indicates that the data type of x is FLOAT16/BFLOAT16, the data type of weight is INT4 (compact 4-bit, per-byte packing of 2 values), and pertensor/perchannel/pergroup quantization modes are supported)</summary>

- **weight**
  - Data type: INT4
  - Format: ND
  - View shape: 2-D `{K, N}` (K as reduction dimension, N as output dimension)
  - Supports contiguous layout or strict transpose of the last two dimensions (stride `[1, K]`)
  - The compact 4-bit packed dimension (contiguous dimension) must be even: N must be even when non-transposed, K must be even when transposed
  - Does not support null tensor

- **weightScale**
  - Data type: FLOAT16/BFLOAT16
  - Format: ND
  - View shape varies with quantization granularity:
    - pertensor: single element, `{1}` or `{1, 1}`
    - perchannel: `{N}` or `{1, N}`
    - pergroup: 2-D `{ceildiv(K, kGroupSize), N}`, and the number of groups must be greater than 1; when weight is a transposed input, weightScale must be consistent with the weight layout (both are transposed views)
  - does not support null tensor

- **weightOffsetOptional**
  - Optional; if provided, the data type and view shape must be the same as those of weightScale, the format is ND, and it does not support null tensor
  - outWeightOffsetOptional synchronization requirement: the view shape, storage shape, and data type must be the same as those of weightOffsetOptional

- **biasOptional**
  - Format: ND
  - view shape: `{N}` or `{1, N}`
  - Must be contiguous.
  - Does not support null tensor (if provided).

- **kGroupSize**
  - pertensor/perchannel: must be equal to 0.
  - pergroup: must be greater than 0, and dimension 0 of weightScale must be equal to ceildiv(K, kGroupSize).

- **xDtype**
  - FLOAT16/BFLOAT16

- **xScaleDtype**
  - Must be ACL_DT_UNDEFINED.

- **outWeight**
  - Data type: same as weight.
  - Format: ND direct copy (physical pass-through) for pertensor (transpose/non-transpose) and perchannel/pergroup transpose scenarios; FRACTAL_NZ_C0_16 (ND→NZ fractal transformation) for perchannel/pergroup non-transpose scenarios.
  - view shape: same as the weight view shape `{K, N}`.
  - storage shape: 4-D `{ceildiv(N, 16), ceildiv(K, 16), 16, 16}` for FRACTAL_NZ_C0_16 (N block-first, different from the K block-first of A8W4 NZ_C0_32); in the ND direct copy scenario, the view shape/strides remain the same as the input weight (pure pass-through).

- **outWeightScale**
  - Data type: same as weightScale.
  - Format: ND
  - view shape: same as the weightScale view shape.
  - storage shape: same as the weightScale storage shape.

- **outBiasOptional**
  - Data type: same as biasOptional
  - Format: ND
  - Must be contiguous
  - view shape: same as biasOptional
  - storage shape: same as biasOptional

</details>
<!-- end id7 -->

<!-- npu="950" id9 -->
<details>
<summary><strong>MM_A16F4 data flow</strong> (MM indicates Matmul; A16F4 indicates that the data type of x is FLOAT16/BFLOAT16, the data type of weight is FLOAT4_E2M1 (compact 4-bit, with per-byte packing of 2 values), and only the pergroup quantization mode is supported)</summary>

- **weight**
  - Data type: FLOAT4_E2M1
  - Format: ND
  - view shape: 2-D `{K, N}` (K as reduction dimension, N as output dimension)
  - Only contiguous (non-transpose, stride `[N, 1]`) is supported; transposed weight always returns ACLNN_ERR_PARAM_INVALID
  - The compact 4-bit packed dimension (contiguous dimension) must be even: when non-transposed, N must be even
  - Does not support null tensors

- **weightScale**
  - Data type: FLOAT16/BFLOAT16
  - Format: ND
  - view shape: pergroup 2-D `{ceildiv(K, kGroupSize), N}`, and the number of groups must be greater than 1
  - Does not support null tensors

- **weightOffsetOptional**
  - Currently does not support, must be nullptr
  - outWeightOffsetOptional must also be nullptr

- **biasOptional**
  - Format: ND
  - view shape: `{N}` or `{1, N}`
  - Must be contiguous
  - Does not support null tensor (if provided)

- **kGroupSize**
  - Must be greater than 0, and the 0th dimension of weightScale must be equal to ceildiv(K, kGroupSize)

- **xDtype**
  - FLOAT16/BFLOAT16

- **xScaleDtype**
  - Must be ACL_DT_UNDEFINED.

- **outWeight**
  - Data type: same as weight.
  - Format: FRACTAL_NZ_C0_16 (ND→NZ fractal transformation).
  - View shape: same as weight view shape `{K, N}`.
  - Storage shape: 4-D `{ceildiv(N, 16), ceildiv(K, 16), 16, 16}` (N block-first, different from the K block-first of A8W4 NZ_C0_32).

- **outWeightScale**
  - Data type: same as weightScale.
  - Format: ND
  - view shape: Same as the view shape of weightScale
  - storage shape: Same as the storage shape of weightScale

- **outBiasOptional**
  - Data type: Same as biasOptional
  - Format: ND
  - Must be contiguous
  - view shape: Same as biasOptional
  - storage shape: Same as biasOptional

</details>
<!-- end id9 -->

<!-- npu="950" id10 -->
<details>
<summary><strong>MM_A16MXF4 data flow</strong> (MM indicates Matmul; A16MXF4 indicates that the data type of x is FLOAT16/BFLOAT16, the data type of weight is FLOAT4_E2M1 (compact 4-bit, per-byte packing of 2 values), and the quantization mode is MX (MicroScaling))</summary>

- **weight**
  - Data type: FLOAT4_E2M1.
  - Format: ND.
  - View shape: 2-D `{K, N}` (with K as the reduction dimension and N as the output dimension).
  - Supports contiguous layout or strict transposition of the last two dimensions (stride `[1, K]`).
  - The packed dimension (contiguous dimension) of compact 4-bit data must be even: N must be even when non-transposed, and K must be even when transposed.
  - Does not support null tensors.

- **weightScale**
  - Data type: FLOAT8_E8M0.
  - Format: ND
  - view shape: 2-D `{ceildiv(K, 32), N}`; when the weight input is transposed, weightScale must be consistent with the weight layout (both transposed views, with stride `[1, ceildiv(K, 32)]`)
  - Does not support null tensors

- **weightOffsetOptional**
  - Currently does not support; must be nullptr
  - outWeightOffsetOptional must also be nullptr

- **biasOptional**
  - Format: ND
  - view shape: `{N}` or `{1, N}`
  - Must be contiguous
  - Does not support null tensors (if provided)

- **kGroupSize**
  - Must be equal to 32

- **xDtype**
  - FLOAT16/BFLOAT16

- **xScaleDtype**
  - Must be ACL_DT_UNDEFINED

- **outWeight**
  - Data type: same as weight
  - Format: FRACTAL_NZ_C0_16 for non-transpose scenario (ND→NZ fractal transformation); ND direct copy (physical pass-through) for transpose scenario
  - view shape: same as the weight view shape `{K, N}`
  - storage shape: 4-D `{ceildiv(N, 16), ceildiv(K, 16), 16, 16}` when the format is FRACTAL_NZ_C0_16 (N block-first, different from the K block-first of A8W4 NZ_C0_32); in the ND direct copy scenario, the view shape/strides remain the same as the input weight (pure pass-through)

- **outWeightScale**
  - data type: same as weightScale
  - format: ND
  - view shape: same as the weightScale view shape
  - storage shape: same as the weightScale storage shape

- **outBiasOptional**
  - data type: same as biasOptional
  - format: ND
  - Must be contiguous
  - view shape: same as biasOptional
  - storage shape: same as biasOptional

</details>
<!-- end id10 -->

- Data type and shape combinations other than the above data flows are reserved APIs. The current call returns ACLNN_ERR_PARAM_INVALID.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

**NOTE**: Calculate and construct the output tensor shape by yourself. For details, see the formulas in Constraints. The following example uses the MM_MX_A8W4 data flow:

- outWeight viewShape: same as weight viewShape
- outWeight storageShape: `{CeilDiv(K, 32), CeilDiv(N, 16), 16, 32}`
- outWeight format: `ACL_FORMAT_FRACTAL_NZ_C0_32`

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_weight_quant_preprocess.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto d : shape)
        size *= d;
    return size;
}

class AclRuntimeGuard {
public:
    explicit AclRuntimeGuard(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclRuntimeGuard()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init(aclrtStream* stream)
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        stream_ = *stream;
        return ACL_SUCCESS;
    }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    AclRuntimeGuard aclGuard(deviceId);
    auto ret = aclGuard.Init(&stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Init failed" << std::endl; return ret);

    // weight: FLOAT4_E2M1, transposed (MM_MX_A8W4)
    int64_t k = 64;
    int64_t n = 128;
    int64_t C0 = 32; // FLOAT4_E2M1 corresponds to C0=32.

    std::vector<int64_t> weightViewShape = {k, n};
    std::vector<int64_t> weightStorageShape = {n, k};
    std::vector<int64_t> weightStrides = {1, k};
    int64_t weightStorageSize = GetShapeSize(weightStorageShape);
    int64_t weightBytes = weightStorageSize / 2; // FP4: 4 bits = 0.5 bytes per element

    std::vector<int8_t> weightHostData(weightBytes, 0);
    void* weightDeviceAddr = nullptr;
    ret = aclrtMalloc(&weightDeviceAddr, weightBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc weight failed" << std::endl; return ret);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    ret = aclrtMemcpy(weightDeviceAddr, weightBytes, weightHostData.data(), weightBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Memcpy weight failed" << std::endl; return ret);
    aclTensor* weight = aclCreateTensor(weightViewShape.data(), weightViewShape.size(), ACL_FLOAT4_E2M1,
                                        weightStrides.data(), 0, ACL_FORMAT_ND, weightStorageShape.data(),
                                        weightStorageShape.size(), weightDeviceAddr);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightPtr(weight, aclDestroyTensor);
    CHECK_RET(weight != nullptr, std::cout << "Create weight tensor failed" << std::endl; return ACL_ERROR_FAILURE);

    // weightScale: FLOAT8_E8M0, 3-D transposed (MM_MX_A8W4)
    // viewShape: {ceildiv(K,64), N, 2} = {1, 128, 2}
    // storageShape: {N, ceildiv(K,64), 2} = {128, 1, 2}
    // transposed stride: {2, 2, 1} (dim0 <-> dim1)
    std::vector<int64_t> scaleViewShape = {k / 64, n, 2};
    std::vector<int64_t> scaleStorageShape = {n, k / 64, 2};
    std::vector<int64_t> scaleStrides = {2, 2, 1};
    int64_t scaleStorageSize = GetShapeSize(scaleStorageShape);
    int64_t scaleBytes = scaleStorageSize; // FP8: 1 byte per element

    std::vector<int8_t> scaleHostData(scaleBytes, 0);
    void* scaleDeviceAddr = nullptr;
    ret = aclrtMalloc(&scaleDeviceAddr, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc weightScale failed" << std::endl; return ret);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    ret = aclrtMemcpy(scaleDeviceAddr, scaleBytes, scaleHostData.data(), scaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Memcpy weightScale failed" << std::endl; return ret);
    aclTensor* weightScale = aclCreateTensor(scaleViewShape.data(), scaleViewShape.size(), ACL_FLOAT8_E8M0,
                                             scaleStrides.data(), 0, ACL_FORMAT_ND, scaleStorageShape.data(),
                                             scaleStorageShape.size(), scaleDeviceAddr);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightScalePtr(weightScale, aclDestroyTensor);
    CHECK_RET(weightScale != nullptr, std::cout << "Create weightScale tensor failed" << std::endl;
              return ACL_ERROR_FAILURE);

    // The user manually constructs outWeight (FRACTAL_NZ_C0_32).
    // viewShape is the same as weight viewShape, and storageShape is calculated using the formula.
    std::vector<int64_t> outWeightViewShape = {k, n};
    std::vector<int64_t> outWeightStorageShape = {CEIL_DIV(k, C0), CEIL_DIV(n, 16), 16, C0};
    int64_t outWeightStorageSize = GetShapeSize(outWeightStorageShape);
    int64_t outWeightBytes = outWeightStorageSize / 2; // FP4

    void* outWeightDeviceAddr = nullptr;
    ret = aclrtMalloc(&outWeightDeviceAddr, outWeightBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc outWeight failed" << std::endl; return ret);
    std::unique_ptr<void, aclError (*)(void*)> outWeightDeviceAddrPtr(outWeightDeviceAddr, aclrtFree);
    aclTensor* outWeight = aclCreateTensor(outWeightViewShape.data(), outWeightViewShape.size(), ACL_FLOAT4_E2M1,
                                           nullptr, 0, ACL_FORMAT_FRACTAL_NZ_C0_32, outWeightStorageShape.data(),
                                           outWeightStorageShape.size(), outWeightDeviceAddr);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outWeightPtr(outWeight, aclDestroyTensor);
    CHECK_RET(outWeight != nullptr, std::cout << "Create outWeight tensor failed" << std::endl;
              return ACL_ERROR_FAILURE);

    // Construct outWeightScale (both viewShape and storageShape are the same as weightScale).
    // Per implementation requirements: viewShape and storageShape of outWeightScale must both be the same as weightScale.
    void* outScaleDeviceAddr = nullptr;
    ret = aclrtMalloc(&outScaleDeviceAddr, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc outWeightScale failed" << std::endl; return ret);
    std::unique_ptr<void, aclError (*)(void*)> outScaleDeviceAddrPtr(outScaleDeviceAddr, aclrtFree);
    aclTensor* outWeightScale = aclCreateTensor(scaleViewShape.data(), scaleViewShape.size(), ACL_FLOAT8_E8M0,
                                                scaleStrides.data(), 0, ACL_FORMAT_ND, scaleStorageShape.data(),
                                                scaleStorageShape.size(), outScaleDeviceAddr);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outWeightScalePtr(outWeightScale, aclDestroyTensor);
    CHECK_RET(outWeightScale != nullptr, std::cout << "Create outWeightScale tensor failed" << std::endl;
              return ACL_ERROR_FAILURE);

    aclDataType xDtype = ACL_FLOAT8_E4M3FN;
    aclDataType xScaleDtype = ACL_FLOAT8_E8M0;
    int64_t kGroupSize = 32;

    // 1. Obtain the workspace and executor.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnWeightQuantPreprocessGetWorkspaceSize(
        weight, weightScale, nullptr, nullptr, // weightOffsetOptional, biasOptional
        xDtype, xScaleDtype, kGroupSize, outWeight, outWeightScale, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "GetWorkspaceSize failed" << std::endl; return ret);

    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc workspace failed" << std::endl; return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 2. Execute the computation.
    ret = aclnnWeightQuantPreprocess(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Preprocess failed" << std::endl; return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Synchronize failed" << std::endl; return ret);

    // 3. Release resources.
    workspaceAddrPtr.reset();
    outWeightScalePtr.reset();
    outWeightPtr.reset();
    weightScalePtr.reset();
    weightPtr.reset();
    outScaleDeviceAddrPtr.reset();
    outWeightDeviceAddrPtr.reset();
    scaleDeviceAddrPtr.reset();
    weightDeviceAddrPtr.reset();
    return 0;
}

```

<!-- npu="950" id8 -->
Call example for the MM_A16S4 data flow (x is FLOAT16, and weight is INT4 in compact layout; includes two use cases: perchannel non-transpose NZ_C0_16 output and pertensor ND direct copy):

- outWeight viewShape: same as weight viewShape.
- outWeight storageShape: `{CeilDiv(N, 16), CeilDiv(K, 16), 16, 16}` for the FRACTAL_NZ_C0_16 output use case (block-first in N, different from the `{CeilDiv(K, 32), CeilDiv(N, 16), 16, 32}` of MM_MX_A8W4); in the ND direct copy scenario, the view shape/strides remain the same as those of the input weight
- outWeight format: `ACL_FORMAT_FRACTAL_NZ_C0_16` for the perchannel/pergroup non-transpose scenario, and `ACL_FORMAT_ND` for the pertensor and perchannel/pergroup transpose scenario

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_weight_quant_preprocess.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto d : shape)
        size *= d;
    return size;
}

class AclRuntimeGuard {
public:
    explicit AclRuntimeGuard(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclRuntimeGuard()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init(aclrtStream* stream)
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        stream_ = *stream;
        return ACL_SUCCESS;
    }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

// Device-side Tensor holder: automatically releases tensors and device memory in reverse order of creation on destruction.
class DeviceTensor {
public:
    // bytesPerElem=0.5 indicates 4-bit compact packing (INT4/FP4, 2 values per byte), and memory is allocated as numel/2 bytes.
    DeviceTensor(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& storageShape,
                 const std::vector<int64_t>& strides, aclDataType dtype, aclFormat format, double bytesPerElem)
    {
        int64_t storageSize = GetShapeSize(storageShape);
        bytes_ = static_cast<int64_t>(storageSize * bytesPerElem);

        std::vector<int8_t> hostData(bytes_, 0);
        auto ret = aclrtMalloc(&deviceAddr_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc device memory failed" << std::endl; return);
        ret = aclrtMemcpy(deviceAddr_, bytes_, hostData.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Memcpy H2D failed" << std::endl; return);
        const int64_t* stridesData = strides.empty() ? nullptr : strides.data();
        tensor_ = aclCreateTensor(viewShape.data(), viewShape.size(), dtype, stridesData, 0, format,
                                  storageShape.data(), storageShape.size(), deviceAddr_);
        CHECK_RET(tensor_ != nullptr, std::cout << "Create tensor failed" << std::endl; return);
    }

    ~DeviceTensor()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
            tensor_ = nullptr;
        }
        if (deviceAddr_ != nullptr) {
            aclrtFree(deviceAddr_);
            deviceAddr_ = nullptr;
        }
    }

    aclTensor* Get() const { return tensor_; }
    bool IsValid() const { return tensor_ != nullptr && deviceAddr_ != nullptr; }

private:
    aclTensor* tensor_ = nullptr;
    void* deviceAddr_ = nullptr;
    int64_t bytes_ = 0;
};

int RunPreprocess(aclTensor* weight, aclTensor* weightScale, aclTensor* weightOffset, aclDataType xDtype,
                  int64_t kGroupSize, aclTensor* outWeight, aclTensor* outWeightScale, aclTensor* outWeightOffset,
                  aclrtStream stream)
{
    // The A16W4 data flow has no xScale semantics; always pass ACL_DT_UNDEFINED for xScaleDtype.
    aclDataType xScaleDtype = ACL_DT_UNDEFINED;

    // 1. Obtain the workspace and executor.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnWeightQuantPreprocessGetWorkspaceSize(weight, weightScale, weightOffset, nullptr, // biasOptional
                                                          xDtype, xScaleDtype, kGroupSize, outWeight, outWeightScale,
                                                          outWeightOffset, nullptr, // outBiasOptional
                                                          &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "GetWorkspaceSize failed, ret=" << ret << std::endl; return ret);

    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc workspace failed" << std::endl; return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 2. Execute the computation.
    ret = aclnnWeightQuantPreprocess(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Preprocess failed, ret=" << ret << std::endl; return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Synchronize failed" << std::endl; return ret);
    return ACL_SUCCESS;
}

// Use case 1: MM_A16S4_PERCHANNEL (INT4 perchannel, non-transposed weight -> FRACTAL_NZ_C0_16 output, offset direct copy pass-through)
int TestA16S4PerChannelNz(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 256;

    // weight: INT4, non-transposed contiguous [K, N]; the compact 4-bit packed dimension (N) must be even.
    DeviceTensor weight({k, n}, {k, n}, {n, 1}, ACL_INT4, ACL_FORMAT_ND, 0.5);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: perchannel {N}, FP16; kGroupSize must be 0 in the perchannel use case.
    DeviceTensor weightScale({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // weightOffsetOptional: A16S4 supports pass-through in all data flows and requires the same shape and dtype as weightScale
    DeviceTensor weightOffset({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightOffset.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: Constructed by the user. viewShape is the same as that of weight. The storageShape of NZ_C0_16 is
    // {ceildiv(N, 16), ceildiv(K, 16), 16, 16} (N block-first, different from the K block-first of A8W4 NZ_C0_32)
    std::vector<int64_t> outStorageShape = {CEIL_DIV(n, 16), CEIL_DIV(k, 16), 16, 16};
    DeviceTensor outWeight({k, n}, outStorageShape, {}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ_C0_16, 0.5);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    // outWeightScale/outWeightOffset: viewShape, storageShape, and dtype are all the same as those of the input (direct copy)
    DeviceTensor outWeightScale({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);
    DeviceTensor outWeightOffset({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightOffset.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), weightOffset.Get(), ACL_FLOAT16, 0, outWeight.Get(),
                             outWeightScale.Get(), outWeightOffset.Get(), stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16S4PerChannelNz failed" << std::endl; return ret);
    std::cout << "TestA16S4PerChannelNz success" << std::endl;
    return ACL_SUCCESS;
}

// Use case 2: MM_A16S4_PERTENSOR (INT4 pertensor, always ND direct copy, independent of transpose)
int TestA16S4PerTensorNd(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 256;

    // weight: INT4, contiguous [K, N] (pertensor performs a byte-wise direct copy for both transpose/non-transpose)
    DeviceTensor weight({k, n}, {k, n}, {n, 1}, ACL_INT4, ACL_FORMAT_ND, 0.5);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: single element {1} for pertensor, FP16; kGroupSize must be 0
    DeviceTensor weightScale({1}, {1}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);
    DeviceTensor weightOffset({1}, {1}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightOffset.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND direct copy, format is ND, storageShape is the same as viewShape
    DeviceTensor outWeight({k, n}, {k, n}, {}, ACL_INT4, ACL_FORMAT_ND, 0.5);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({1}, {1}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);
    DeviceTensor outWeightOffset({1}, {1}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightOffset.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), weightOffset.Get(), ACL_FLOAT16, 0, outWeight.Get(),
                             outWeightScale.Get(), outWeightOffset.Get(), stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16S4PerTensorNd failed" << std::endl; return ret);
    std::cout << "TestA16S4PerTensorNd success" << std::endl;
    return ACL_SUCCESS;
}

// WeightQuantPreprocess currently supports Ascend 950 only.
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    AclRuntimeGuard aclGuard(deviceId);
    auto ret = aclGuard.Init(&stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Init failed" << std::endl; return ret);

    ret = TestA16S4PerChannelNz(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = TestA16S4PerTensorNd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::cout << "All A16S4 examples run success" << std::endl;
    return 0;
}
```
<!-- end id8 -->

<!-- npu="950" id11 -->
The following is a call example for the MM_A16F4/MM_A16MXF4 data flows (x is FLOAT16 and weight is FLOAT4_E2M1 in compact layout; covering three use cases: A16F4 pergroup non-transpose NZ_C0_16 output, A16MXF4 non-transpose NZ_C0_16 output, and A16MXF4 transpose ND direct copy):

- outWeight viewShape: same as the weight viewShape
- outWeight storageShape: For the FRACTAL_NZ_C0_16 output use case, the shape is `{CeilDiv(N, 16), CeilDiv(K, 16), 16, 16}` (N is block-first, different from the `{CeilDiv(K, 32), CeilDiv(N, 16), 16, 32}` of MM_MX_A8W4); in the ND direct copy scenario, the view shape/strides remain the same as those of the input weight
- outWeight format: `ACL_FORMAT_FRACTAL_NZ_C0_16` in the non-transpose scenario, `ACL_FORMAT_ND` in the A16MXF4 transpose scenario

```cpp
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_weight_quant_preprocess.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto d : shape)
        size *= d;
    return size;
}

uint16_t FloatToFp16(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    return ((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
}

class AclRuntimeGuard {
public:
    explicit AclRuntimeGuard(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclRuntimeGuard()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init(aclrtStream* stream)
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        stream_ = *stream;
        return ACL_SUCCESS;
    }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

// Device-side Tensor holder: automatically releases tensors and device memory in reverse order of creation upon destruction
class DeviceTensor {
public:
    // bytesPerElem=0.5 indicates compact 4-bit packing (INT4/FP4, 2 values per byte), and memory is allocated in numel/2 bytes;
    // If hostData is not empty, initialize the device memory with its actual content; otherwise, zero-fill it.
    DeviceTensor(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& storageShape,
                 const std::vector<int64_t>& strides, aclDataType dtype, aclFormat format, double bytesPerElem,
                 const void* hostData = nullptr)
    {
        int64_t storageSize = GetShapeSize(storageShape);
        bytes_ = static_cast<int64_t>(storageSize * bytesPerElem);

        std::vector<uint8_t> zeroData;
        const void* src = hostData;
        if (src == nullptr) {
            zeroData.assign(bytes_, 0);
            src = zeroData.data();
        }
        auto ret = aclrtMalloc(&deviceAddr_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc device memory failed" << std::endl; return);
        ret = aclrtMemcpy(deviceAddr_, bytes_, src, bytes_, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Memcpy H2D failed" << std::endl; return);
        const int64_t* stridesData = strides.empty() ? nullptr : strides.data();
        tensor_ = aclCreateTensor(viewShape.data(), viewShape.size(), dtype, stridesData, 0, format,
                                  storageShape.data(), storageShape.size(), deviceAddr_);
        CHECK_RET(tensor_ != nullptr, std::cout << "Create tensor failed" << std::endl; return);
    }

    ~DeviceTensor()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
            tensor_ = nullptr;
        }
        if (deviceAddr_ != nullptr) {
            aclrtFree(deviceAddr_);
            deviceAddr_ = nullptr;
        }
    }

    aclTensor* Get() const { return tensor_; }
    bool IsValid() const { return tensor_ != nullptr && deviceAddr_ != nullptr; }
    int64_t Bytes() const { return bytes_; }

    // Copy the device memory back to the host buffer (for golden comparison).
    int CopyToHost(void* dst, int64_t bytes) const
    {
        auto ret = aclrtMemcpy(dst, bytes, deviceAddr_, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Memcpy D2H failed" << std::endl; return ret);
        return ACL_SUCCESS;
    }

private:
    aclTensor* tensor_ = nullptr;
    void* deviceAddr_ = nullptr;
    int64_t bytes_ = 0;
};

int RunPreprocess(aclTensor* weight, aclTensor* weightScale, aclDataType xDtype, int64_t kGroupSize,
                  aclTensor* outWeight, aclTensor* outWeightScale, aclrtStream stream)
{
    // The A16W4 data flow has no xScale semantics; always pass ACL_DT_UNDEFINED for xScaleDtype;
    // A16F4/A16MXF4 do not support offset; weightOffsetOptional/outWeightOffsetOptional must be nullptr.
    aclDataType xScaleDtype = ACL_DT_UNDEFINED;

    // 1. Obtain the workspace and executor.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnWeightQuantPreprocessGetWorkspaceSize(weight, weightScale, nullptr, // weightOffsetOptional
                                                          nullptr,                      // biasOptional
                                                          xDtype, xScaleDtype, kGroupSize, outWeight, outWeightScale,
                                                          nullptr, // outWeightOffsetOptional
                                                          nullptr, // outBiasOptional
                                                          &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "GetWorkspaceSize failed, ret=" << ret << std::endl; return ret);

    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc workspace failed" << std::endl; return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 2. Perform the computation.
    ret = aclnnWeightQuantPreprocess(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Preprocess failed, ret=" << ret << std::endl; return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Synchronize failed" << std::endl; return ret);
    return ACL_SUCCESS;
}

// Compare byte-by-byte with the input (scale/ND direct copy weight is a physical pass-through and requires bit-wise consistency)
bool VerifyBitWise(const char* tag, const DeviceTensor& out, const void* expected, int64_t bytes)
{
    std::vector<uint8_t> actual(bytes);
    auto ret = out.CopyToHost(actual.data(), bytes);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    if (std::memcmp(actual.data(), expected, bytes) != 0) {
        std::cout << tag << " bit-wise compare FAIL" << std::endl;
        return false;
    }
    std::cout << tag << " bit-wise compare PASS" << std::endl;
    return true;
}

// NZ_C0_16 compact 4-bit physical layout check: storage is {ceildiv(N,16), ceildiv(K,16), 16, 16} (N block-first),
// nibble linear index p = ((n1 * kBlocks + k1) * 16 + k0) * 16 + n0, low nibble is even p
bool VerifyNzC016(const char* tag, const DeviceTensor& out, const std::vector<uint8_t>& weightNibbles, int64_t k,
                  int64_t n)
{
    int64_t kBlocks = CEIL_DIV(k, 16);
    int64_t nBlocks = CEIL_DIV(n, 16);
    std::vector<uint8_t> expected(k * n / 2, 0);
    for (int64_t n1 = 0; n1 < nBlocks; n1++) {
        for (int64_t k1 = 0; k1 < kBlocks; k1++) {
            for (int64_t k0 = 0; k0 < 16; k0++) {
                for (int64_t n0 = 0; n0 < 16; n0++) {
                    int64_t kk = k1 * 16 + k0;
                    int64_t nn = n1 * 16 + n0;
                    uint8_t nib = (kk < k && nn < n) ? weightNibbles[kk * n + nn] : 0;
                    int64_t p = ((n1 * kBlocks + k1) * 16 + k0) * 16 + n0;
                    expected[p / 2] |= static_cast<uint8_t>(nib << ((p % 2) * 4));
                }
            }
        }
    }
    return VerifyBitWise(tag, out, expected.data(), k * n / 2);
}

// Common input: logical weight {K,N} nibbles, non-transpose physical packing (along N, low nibble = even n)
void MakeWeight(int64_t k, int64_t n, std::vector<uint8_t>& weightNibbles, std::vector<uint8_t>& weightPacked)
{
    weightNibbles.resize(k * n);
    for (int64_t i = 0; i < k * n; i++) {
        weightNibbles[i] = static_cast<uint8_t>((i * 7 + 3) % 15);
    }
    weightPacked.assign(k * n / 2, 0);
    for (int64_t kk = 0; kk < k; kk++) {
        for (int64_t nn = 0; nn + 1 < n; nn += 2) {
            weightPacked[(kk * n + nn) / 2] = (weightNibbles[kk * n + nn + 1] << 4) | weightNibbles[kk * n + nn];
        }
    }
}

// Use case 1: MM_A16F4_PERGROUP (FP4 pergroup, non-transpose weight -> FRACTAL_NZ_C0_16 output, scale direct copy pass-through)
int TestA16F4PerGroupNz(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 128;
    int64_t kGroupSize = 64; // pergroup: must be greater than 0, and dimension 0 of weightScale equals ceildiv(K, kGroupSize)
    int64_t g = CEIL_DIV(k, kGroupSize);

    // weight: FLOAT4_E2M1, non-transpose contiguous [K, N]; compact 4-bit packed dimension (N) must be even
    std::vector<uint8_t> weightNibbles, weightPacked;
    MakeWeight(k, n, weightNibbles, weightPacked);
    DeviceTensor weight({k, n}, {k, n}, {n, 1}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, 0.5, weightPacked.data());
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: pergroup 2-D {ceildiv(K, kGroupSize), N} (number of groups greater than 1), FP16
    std::vector<uint16_t> scale(g * n);
    for (int64_t i = 0; i < g * n; i++) {
        scale[i] = FloatToFp16(0.5f + 0.5f * static_cast<float>(i % 4));
    }
    DeviceTensor weightScale({g, n}, {g, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2, scale.data());
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: constructed by the user, viewShape is the same as weight; the storageShape of NZ_C0_16 is
    // {ceildiv(N, 16), ceildiv(K, 16), 16, 16} (N block-first, different from the K block-first of A8W4 NZ_C0_32)
    std::vector<int64_t> outStorageShape = {CEIL_DIV(n, 16), CEIL_DIV(k, 16), 16, 16};
    DeviceTensor outWeight({k, n}, outStorageShape, {}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_16, 0.5);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    // outWeightScale: viewShape, storageShape, and dtype all remain the same as input (direct copy)
    DeviceTensor outWeightScale({g, n}, {g, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), ACL_FLOAT16, kGroupSize, outWeight.Get(),
                             outWeightScale.Get(), stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16F4PerGroupNz failed" << std::endl; return ret);

    // golden comparison: the NZ_C0_16 conversion result is byte-by-byte identical to the expected host-side layout; scale is passed through via direct copy
    bool ok = VerifyNzC016("  outWeight(NZ_C0_16)", outWeight, weightNibbles, k, n);
    ok = VerifyBitWise("  outWeightScale", outWeightScale, scale.data(), g * n * 2) && ok;
    CHECK_RET(ok, std::cout << "TestA16F4PerGroupNz verify failed" << std::endl; return ACL_ERROR_FAILURE);
    std::cout << "TestA16F4PerGroupNz success" << std::endl;
    return ACL_SUCCESS;
}

// Use case 2: MM_A16MXF4 non-transpose (FP4 + E8M0 MX scale, non-transposed weight -> FRACTAL_NZ_C0_16 output)
int TestA16MXF4NonTransNz(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 128;
    int64_t kGroupSize = 32; // kGroupSize is fixed to 32 in the MX use case
    int64_t g = CEIL_DIV(k, kGroupSize);

    std::vector<uint8_t> weightNibbles, weightPacked;
    MakeWeight(k, n, weightNibbles, weightPacked);
    DeviceTensor weight({k, n}, {k, n}, {n, 1}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, 0.5, weightPacked.data());
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: MX 2-D {ceildiv(K, 32), N}, FLOAT8_E8M0 (1 byte per element)
    std::vector<uint8_t> scale(g * n);
    for (int64_t i = 0; i < g * n; i++) {
        scale[i] = static_cast<uint8_t>(0x7E + (i % 4)); // 0.5 / 1 / 2 / 4
    }
    DeviceTensor weightScale({g, n}, {g, n}, {n, 1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, 1, scale.data());
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    std::vector<int64_t> outStorageShape = {CEIL_DIV(n, 16), CEIL_DIV(k, 16), 16, 16};
    DeviceTensor outWeight({k, n}, outStorageShape, {}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_16, 0.5);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({g, n}, {g, n}, {n, 1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, 1);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), ACL_FLOAT16, kGroupSize, outWeight.Get(),
                             outWeightScale.Get(), stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16MXF4NonTransNz failed" << std::endl; return ret);

    bool ok = VerifyNzC016("  outWeight(NZ_C0_16)", outWeight, weightNibbles, k, n);
    ok = VerifyBitWise("  outWeightScale", outWeightScale, scale.data(), g * n) && ok;
    CHECK_RET(ok, std::cout << "TestA16MXF4NonTransNz verify failed" << std::endl; return ACL_ERROR_FAILURE);
    std::cout << "TestA16MXF4NonTransNz success" << std::endl;
    return ACL_SUCCESS;
}

// Use case 3: MM_A16MXF4 transpose (last two dimensions strictly transposed with stride [1, K] -> ND direct copy pass-through, out shares view shape/strides with input)
int TestA16MXF4TransNd(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 128;
    int64_t kGroupSize = 32;
    int64_t g = CEIL_DIV(k, kGroupSize);

    std::vector<uint8_t> weightNibbles, weightPackedNonTrans;
    MakeWeight(k, n, weightNibbles, weightPackedNonTrans);

    // Transposed weight: view {K, N} strides {1, K}, physically stored as {N, K/2} packed along K (low nibble = even k), K must be even
    std::vector<uint8_t> weightPacked(k * n / 2, 0);
    for (int64_t nn = 0; nn < n; nn++) {
        for (int64_t kk = 0; kk + 1 < k; kk += 2) {
            weightPacked[(nn * k + kk) / 2] = (weightNibbles[(kk + 1) * n + nn] << 4) | weightNibbles[kk * n + nn];
        }
    }
    DeviceTensor weight({k, n}, {k, n}, {1, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, 0.5, weightPacked.data());
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // Transposed view of weightScale: view {G, N} strides {1, G}, physically stored as {N, G}
    std::vector<uint8_t> scale(g * n);
    for (int64_t i = 0; i < g * n; i++) {
        scale[i] = static_cast<uint8_t>(0x7E + (i % 4));
    }
    std::vector<uint8_t> scalePhys(n * g);
    for (int64_t nn = 0; nn < n; nn++) {
        for (int64_t gg = 0; gg < g; gg++) {
            scalePhys[nn * g + gg] = scale[gg * n + nn];
        }
    }
    DeviceTensor weightScale({g, n}, {g, n}, {1, g}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, 1, scalePhys.data());
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND direct copy, format is ND, view shape/strides remain the same as input weight (pure pass-through)
    DeviceTensor outWeight({k, n}, {k, n}, {1, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, 0.5);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({g, n}, {g, n}, {1, g}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, 1);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), ACL_FLOAT16, kGroupSize, outWeight.Get(),
                             outWeightScale.Get(), stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16MXF4TransNd failed" << std::endl; return ret);

    // ND direct copy is physical pass-through: outWeight/outWeightScale is byte-by-byte identical with the input buffer
    bool ok = VerifyBitWise("  outWeight(ND direct copy)", outWeight, weightPacked.data(), k * n / 2);
    ok = VerifyBitWise("  outWeightScale", outWeightScale, scalePhys.data(), g * n) && ok;
    CHECK_RET(ok, std::cout << "TestA16MXF4TransNd verify failed" << std::endl; return ACL_ERROR_FAILURE);
    std::cout << "TestA16MXF4TransNd success" << std::endl;
    return ACL_SUCCESS;
}

// WeightQuantPreprocess currently supports Ascend 950 only.
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    AclRuntimeGuard aclGuard(deviceId);
    auto ret = aclGuard.Init(&stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Init failed" << std::endl; return ret);

    ret = TestA16F4PerGroupNz(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = TestA16MXF4NonTransNz(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = TestA16MXF4TransNd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::cout << "All A16F4 examples run success" << std::endl;
    return 0;
}
```
<!-- end id11 -->
