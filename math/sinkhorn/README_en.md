# Sinkhorn

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: computes the Sinkhorn optimal transport plan, which can be used for expert routing in MoE models.

- Computation formula:

  $$
  p=Sinkhorn(cost, tol)
  $$

- Initialization:

  $$
  cost = exp(cost) \\
  d0 = ones(R) \\
  d1 = ones(C) \\
  eps = 0.00000001 \\
  error = 1e9 \\
  d1_old = d1 \\
  $$

- Repeated execution:

  $$
  d0 = \frac{1}{R \times (sum(d1 \cdot cost, 1) + eps)} \\
  d1 = \frac{1}{C \cdot \sum(d0.unsqueeze(1) \cdot cost, 0) + eps)} \\
  error = mean(|d1_old - d1|) \\
  d1_old = d1
  $$

  until the following condition is met:

  $$
  error \le tol
  $$

- Output:

  $$
  p = d1 \cdot cost \cdot d0.unsqueeze(1)
  $$

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|-----|-----------|----|---------|------|
|cost|Input|Denotes the cost tensor, which is `cost` in the formula. It is a two-dimensional matrix. The number of rows has no fixed upper limit, the number of columns must not be greater than 4168, and the total number of elements must not exceed UINT32_MAX. It is recommended that the value be in the range [0,1].|FLOAT, FLOAT16, BFLOAT16|ND|
|p|Output|Denotes the optimal transport tensor, which is `p` in the formula. The shape is the same as that of cost.|FLOAT, FLOAT16, BFLOAT16|ND|
|tol|Optional attribute|Denotes the error for computing Sinkhorn. The default value is 0.0001.|FLOAT|-|

- <term>Atlas A2 products</term>: Inf or NaN may be generated earlier when the FP16 input values are large.

- <term>Ascend 950PR&950DT products</term>: The numeric range of the FP16 input is the same as that of FP32.

## Constraint Description

- cost must be a two-dimensional matrix with non-negative dimensions. The number of columns must not be greater than 4168, the number of rows has no fixed upper limit, and the total number of elements must not exceed UINT32_MAX.
- It is recommended that the value of cost be in the range [0,1], but this range is not a rejection boundary. Values outside the range still participate in the computation and may produce Inf or NaN in the exponentiation, row-sum, or output stage.
- tol supports all FLOAT values. NaN, ±Inf, and negative values participate in the termination judgment according to floating-point comparison semantics.
- Empty tensors are valid.
- The maximum number of iterations is 10000. When the upper limit is reached, the state of the 10000th iteration is returned, and convergence is not guaranteed.
- The termination judgment strictly uses `!(error > tol)`: a comparison involving NaN terminates the iteration. When `+Inf` is compared with a finite tol, the iteration continues, and in the end it is still protected by the upper limit of 10000 iterations.

## Invocation Description

| Invocation Method | Sample Code | Description |
|---------|------|------|
| aclnn API | [test_aclnn_sinkhorn](examples/test_aclnn_sinkhorn.cpp) | Invokes the Sinkhorn operator through the [aclnnSinkhorn](docs/aclnnSinkhorn_en.md) API.|
| GEIR graph mode | - | Sinkhorn supports the GEIR graph mode.|
