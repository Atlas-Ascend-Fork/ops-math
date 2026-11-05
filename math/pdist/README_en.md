# Pdist

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      √     |

## Function Description

- Operator function: the Pdist operator computes the p-norm distances (pairwise p-norm distances) between each pair of row vectors. The output is a one-dimensional vector in upper-triangular row-major order. It is equivalent to `torch.nn.functional.pdist`.

- Kernel implementation form: **PyPTO DSL** (`op_kernel/pdist.py`). Code generation is performed at the configure stage through `enable_pypto_kernel`, and the result is delivered as a pypto opc (MIX launch, the AIC stays idle, and all computation is performed on the Vector cores).

- Computation formula:

$$
y_{idx}=\left\|x_{i}-x_{j}\right\|_{p}=\left(\sum_{k=0}^{M-1}\left|x_{i,k}-x_{j,k}\right|^{p}\right)^{1/p},\quad idx=\frac{i\cdot(2N-i-1)}{2}+(j-i-1)
$$

$$
0\le i<j\le N-1
$$

p branch semantics:

| p Value | Semantics |
|--------|------|
| p = 0 | Counts the number of unequal dimensions (Hamming distance). NaN dimensions are counted as unequal |
| p = +inf | Maximum absolute difference (Chebyshev distance). NaN dimensions are ignored |
| 0 < p < +inf | The 1/p-th power of the sum of powers (Minkowski distance). p=2 is the Euclidean distance and p=1 is the Manhattan distance |

NaN semantics (aligned with GPU/CUDA): when p=0, NaN dimensions are counted as unequal; when p=+inf, NaN dimensions are ignored (the maximum absolute difference of non-NaN dimensions is output, and all-NaN row pairs output 0); for a general p (not 0/0.5/1/2), NaN dimensions are detected pair by pair and NaN is output; for p∈{0.5, 1, 2}, NaN propagates naturally through IEEE arithmetic.

fp16 input: on-chip intermediate accumulation always uses FP32, and the result is narrowed back to FP16 at the output.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
    <tbody>
     <tr>
       <td>x</td>
       <td>Input</td>
       <td>Input tensor, with the shape (N, M), where N≥2 and M≥1. Row i and row j (j>i) form a row pair for computation. Row-major contiguous layout is required.</td>
       <td>FLOAT16, FLOAT32</td>
       <td>ND</td>
      </tr>
     <tr>
       <td>p</td>
       <td>Attribute</td>
       <td>Norm order, which is p≥0 or +inf. The default value is 2.0 (Euclidean distance). p=0 counts the number of unequal dimensions, p=+inf takes the maximum absolute difference, and 0<p<inf is the 1/p-th power of the sum of powers.</td>
       <td>FLOAT</td>
       <td>-</td>
      </tr>
     <tr>
       <td>y</td>
       <td>Output</td>
       <td>Pairwise distance vector, with the shape (N*(N-1)/2,), arranged in upper-triangular row-major order. The dtype is the same as that of x.</td>
       <td>FLOAT16, FLOAT32</td>
       <td>ND</td>
      </tr>
     </tbody>
   </table>

## Constraint Description

  - x must be a two-dimensional (N, M) row-major contiguous tensor, and the dtype of y must be the same as that of x.
  - The shape of y must be (N*(N-1)/2,).
  - Intermediate accumulation of fp16 input uses FP32, and the result is narrowed back to FP16 at the output.
  - Platform differences:
    - When p=+inf and the input contains NaN: on Ascend 950PR/950DT, NaN dimensions are ignored (the maximum absolute difference of non-NaN dimensions is output, and all-NaN row pairs output 0), which is aligned with the GPU/CUDA semantics. On other platforms, NaN is propagated to the output.

  <table style="undefined;table-layout: fixed; width: 909px"><colgroup>
    <col style="width: 125px">
    <col style="width: 182px">
    <col style="width: 602px">
    </colgroup>
    <thead>
      <tr>
        <th>Specification Item</th>
        <th>Specification</th>
        <th>Specification Description</th>
      </tr>
    </thead>
    <tbody>
    <tr>
      <td>N</td>
      <td>2~65535</td>
      <td>Number of rows, at least 2 (at least one row pair).</td>
    </tr>
    <tr>
      <td>M</td>
      <td>1~65535</td>
      <td>Number of dimensions per row. When it exceeds the on-chip memory, automatic chunked reduction is performed.</td>
    </tr>
    <tr>
      <td>p</td>
      <td>[0, +inf)</td>
      <td>p≥0 or +inf; p=0 is the Hamming distance, p=+inf is the Chebyshev distance, and p<0 or NaN is rejected as a parameter error.</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>FLOAT16/FLOAT32</td>
      <td>The input and output have the same dtype; fp16 intermediate computation uses FP32 accumulation.</td>
    </tr>
    </tbody>
    </table>

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_pdist](examples/test_aclnn_pdist.cpp) | Invokes the pdist operator through the [aclnnPdist](docs/aclnnPdist_en.md) API, covering all p branches of fp32, multi-M chunking, and fp16. |
| aclnn invocation (Forward) | [test_aclnn_pdist_forward](examples/test_aclnn_pdist_forward.cpp) | Invokes the operator through the [aclnnPdistForward](docs/aclnnPdistForward_en.md) API (the composite scalar p entry). |
| GEIR invocation | [test_geir_pdist](examples/test_geir_pdist.cpp) | Invokes the pdist operator in GE IR graph mode (RunGraph). |
