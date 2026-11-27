# Operator List

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-17T12:41:29.868Z pushedAt=2026-11-18T07:58:22.392Z nl=2 -->

> **Instruction**:
>
> - **Operator directory**: The directory name is the lowercase underscore form of the operator name. Each directory contains all deliverables of the operator, including the code implementation, examples, and documents. For details about the directory, see [Project Directory](./install/dir_structure.md).
> - **Operator execution hardware unit**: Most operators run on the AI Core, and a few run on the AI CPU. By default, operators mentioned in the project refer to AI Core operators. For details about the AI Core and AI CPU, see "Concepts, Principles, and Terms > Hardware Architecture and Data Processing Principles" in [*Ascend C Operator Development*](https://hiascend.com/en/document/redirect/CannCommunityOpdevAscendC).
> - **Operator API list**: To facilitate operator calls, CANN provides a set of C APIs for executing operators, generally prefixed with aclnn. For the full API list, see [aclnn List](op_api_list.md).
> - **V version evolution**: Some operators have multiple V versions. Use the highest V version, which is compatible with all capabilities of lower V versions.

All operator categories and operators provided by the project are as follows:

<table><thead>
  <tr>
    <th rowspan="2">Category</th>
    <th rowspan="2">Directory</th>
    <th colspan="2">Implementation</th>
    <th>aclnn Call</th>
    <th>Graph Mode Call</th>
    <th rowspan="2">Operator Execution Hardware Unit</th>
    <th rowspan="2">Description</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
    <th>op_api</th>
    <th>op_graph</th>
  </tr></thead>
<tbody>
  <tr>
    <td>math</td>
    <td><a href="../../math/abs/README_en.md">abs</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the absolute value of each element of the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/accumulate_nv2/README_en.md">accumulate_nv2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs element-wise accumulation on multiple input tensors of the same shape and data type, and outputs the accumulated tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/acos/README_en.md">acos</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>The operator performs an arccosine operation on each element of the input and outputs the result.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/acosh/README_en.md">acosh</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Computes the inverse hyperbolic cosine. Supports the FLOAT, FLOAT16, and BF16 data types.</td>
  </tr>
    <tr>
    <td>math</td>
    <td><a href="../../math/acosh_grad/README_en.md">acosh_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CORE</td>
    <td>Computes the backward gradient of the Acosh operator. Formula: z = dy / sqrt(y² - 1).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/acos_grad/README_en.md">acos_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CORE</td>
    <td>Computes the backward gradient of the Acos operator. Formula: z = -dy / sqrt(1 - y²).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ada_cast/README_en.md">ada_cast</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CORE</td>
    <td>Converts uint16 to float16 and normalizes by pixel (HDRnet white-level scaling). Formula: y = SaturateCast_FP16(float32(x) × (1/pixel)).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/add/README_en.md">add</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CORE/AI CPU</td>
    <td>The operator performs addition.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/add_v2/README_en.md">add_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs element-wise addition on two input tensors. Compatible with TensorFlow AddV2 semantics. Supports broadcast and empty tensors.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/add_lora/README_en.md">add_lora</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Multiplies input x by the corresponding weightA and weightB according to input indices, then accumulates the results onto input y and outputs the result.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/add_mat_mat_elements/README_en.md">add_mat_mat_elements</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs element-wise multiplication on input tensors a and b, then performs a weighted sum with input tensor c using scalars alpha and beta, and outputs the result tensor cOut.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/add_n/README_en.md">add_n</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Performs element-wise addition and summation on the input.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/addcdiv/README_en.md">addcdiv</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>A tensor operation function that performs a combined multiply-divide-add operation, merging tensor division (with scaling) and tensor addition into a single operation.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/addcmul/README_en.md">addcmul</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Tensor operation function that performs a combined multiply-divide-add operation, merging tensor multiplication (with scaling) and tensor addition into a single operation.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/addr/README_en.md">addr</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the outer product of two one-dimensional vectors vec1 and vec2 to obtain a two-dimensional matrix, then multiplies the outer product result by a coefficient and adds it to the product of itself and a coefficient before outputting.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/adjacent_difference/README_en.md">adjacent_difference</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Compares the differences between adjacent elements of the input tensor. Returns 0 if adjacent elements are the same, otherwise returns 1.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/affine_grid">affine_grid</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Given a set of 3D affine parameter matrices (theta) and the output image size, generates a 2D or 3D grid.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/angle_v2/README_en.md">angle_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Takes the angle (in radians) of each element of the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/arg_max_v2/README_en.md">arg_max_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns the index of the maximum value of the tensor in the specified dimension (dim) and saves it to the out tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/arg_max_with_value/README_en.md">arg_max_with_value</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns the maximum value of the tensor in the specified dimension and its index. The maximum value is saved to out, and the index of the maximum value is saved to indices. If keepdim is false, the corresponding axis is not retained; if true, the dimension value of the specified axis is retained as 1.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/arg_min/README_en.md">arg_min</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns the index of the minimum value along the specified axis in the tensor and saves it to out. If keepdim is false, the corresponding axis is not retained. If keepdim is true, the dimension value of the specified axis is retained as 1.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/arg_min_with_value/README_en.md">arg_min_with_value</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns the minimum value and its index position along the specified dimension in self. The minimum value is saved to out, and the index of the minimum value is saved to indices. If keepdim is false, the corresponding axis is not retained. If keepdim is true, the dimension value of the specified axis is retained as 1.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/asin/README_en.md">asin</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU </td>
    <td>The operator performs an arcsine operation on each element of the input and outputs the result.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/asinh/README_en.md">asinh</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>The operator performs an inverse hyperbolic sine operation on each element of the input tensor and outputs the result.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/asin_grad/README_en.md">asin_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Backward computation of the Asin (arcsine) operator. Computes the input gradient dx according to the forward input x and the upstream gradient dy.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/asinh_grad/README_en.md">asinh_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Backward computation of the Asinh (inverse hyperbolic sine) operator. Computes the input gradient dx according to the forward input y and the upstream gradient dy.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/assign_add/README_en.md">assign_add</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>The operator performs addition on the original tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/assign_sub/README_en.md">assign_sub</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>The operator performs subtraction on the original tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/atan/README_en.md">atan</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>The operator performs an arctangent operation on each element of the input and outputs the result.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/atan2/README_en.md">atan2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the four-quadrant arctangent of each element in the input tensor self and the input scalar other.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/atanh/README_en.md">atanh</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Computes the inverse hyperbolic tangent. Supports FLOAT, FLOAT16, and BF16 data types.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/atan_grad/README_en.md">atan_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Backward computation of the Atan (arctangent) operator. Computes the input gradient dx from the forward input x and the upstream gradient dy, for use in neural network backward propagation.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/axpy/README_en.md">axpy</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Multiplies each element of the source operand by a scalar and adds the result to the corresponding element of the destination operand.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/axpy_v2/README_en.md">axpy_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Multiplies each element of the source operand by a tensor and adds the result to the corresponding element of the destination operand.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bessel_i0e/README_en.md">bessel_i0e</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the exponentially scaled modified Bessel function of the first kind of order zero, y = exp(-|x|) * I0(x). Supports graph mode calls only.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bessel_i1e/README_en.md">bessel_i1e</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the exponentially scaled modified Bessel function of the first kind, order 1.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bias/README_en.md">bias</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Derives the broadcast shape of bias based on axis, num_axes, and bias_from_blob, and computes y = x + broadcast(bias).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bias_add/README_en.md">bias_add</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Bias addition.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bias_add_grad/README_en.md">bias_add_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the gradient of the bias.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bincount/README_en.md">bincount</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the frequency of each number in a non-negative integer array.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/dense_bincount/README_en.md">dense_bincount</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Counts the occurrences or weight sum of each bin in one-dimensional or two-dimensional integer input.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bitwise_and/README_en.md">bitwise_and</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the bitwise AND of each element in the input tensor and the input scalar. The input tensor and other must be integer or BOOL variables. When the input is BOOL, computes the logical AND result.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bitwise_not/README_en.md">bitwise_not</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bitwise_or/README_en.md">bitwise_or</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the bitwise OR of each element in the input tensor self and the input scalar other. The inputs self and other must be integer or boolean types. For boolean types, logical OR is computed.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bitwise_xor/README_en.md">bitwise_xor</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cast/README_en.md">cast</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Converts the input tensor to the specified dtype.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cdist/README_en.md">cdist</a></td>
    <td> </td>
    <td> </td>
    <td>√</td>
    <td> </td>
    <td> </td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cdist_grad/README_en.md">cdist_grad</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ceil/README_en.md">ceil</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Returns the result of rounding up each element in the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cholesky/README_en.md">cholesky</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/compare_and_bit_pack/README_en.md">compare_and_bit_pack</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Compares the input with a threshold and packs the bits of the comparison result into a uint8 output.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/complex/README_en.md">complex</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Constructs a complex tensor from the real part real and the imaginary part imag.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/complex_abs/README_en.md">complex_abs</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/conj/README_en.md">conj</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Returns the conjugate of a complex tensor. Takes the conjugate of each complex element of the input (the real part remains unchanged and the imaginary part is negated).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cos/README_en.md">cos</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns the result of the cosine operation performed on the input tensor elements.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cosh/README_en.md">cosh</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Returns the result of the hyperbolic cosine operation performed on the input tensor elements.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cosine_similarity/README_en.md">cosine_similarity</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Computes the cosine similarity of two input tensors along the specified dimension.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cross/README_en.md">cross</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>The operator performs the linear_cross operation on the input tensor.</td>
  </tr>
    <tr>
    <td>math</td>
    <td><a href="../../math/confusion_matrix/README_en.md">confusion_matrix</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the confusion matrix of a classification task. Counts the occurrences or weighted sum of each (label, prediction) pair and outputs a num_classes×num_classes matrix.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cummax/README_en.md">cummax</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cummin/README_en.md">cummin</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cumprod/README_en.md">cumprod</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cumulative_logsumexp/README_en.md">cumulative_logsumexp</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the cumulative log-sum-exp on the input tensor along the specified dimension.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cumsum/README_en.md">cumsum</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>The operator accumulates the elements of the input tensor self along the specified dimension dim.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cumsum_cube/README_en.md">cumsum_cube</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/data_compare/README_en.md">data_compare</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Compares two input tensors of the same shape and dtype element-wise, and counts the total number of elements whose difference exceeds the tolerance range.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/dawsn/README_en.md">dawsn</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the Dawson integral F(x) = exp(-x^2) * integral_0^x exp(t^2) dt.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/diag_part/README_en.md">diag_part</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Obtains the diagonal elements of the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/digamma/README_en.md">digamma</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/div/README_en.md">div</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Performs tensor division.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/truncate_div/README_en.md">truncate_div</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs truncated division, with the result rounded toward zero.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/div_no_nan/README_en.md">div_no_nan</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs tensor division. When the denominator may be 0, it helps avoid NaN values caused by a zero denominator.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/dot/README_en.md">dot</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the dot product of two one-dimensional input tensors.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/eltwise/README_en.md">eltwise</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs element-wise operations on 1 to 32 input tensors of the same shape and dtype. Supports three compute modes: PRODUCT (element-wise product), SUM (element-wise weighted sum), and MAX (element-wise maximum).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/equal/README_en.md">equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Computes whether two tensors have the same size and elements.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/erf/README_en.md">erf</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Returns the value of the error function corresponding to each element in the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/erfc/README_en.md">erfc</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns the value of the complementary error function corresponding to each element in the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/exp/README_en.md">exp</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Returns a new tensor, where each element is the exponent of the corresponding element in the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/exp_segsum_grad/README_en.md">exp_segsum_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Backward computation of Segsum.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/expand/README_en.md">expand</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Broadcasts the input tensor to the specified shape.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/expint/README_en.md">expint</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the exponential integral function.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/expm1/README_en.md">expm1</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Computes the power of the natural constant e with the input tensor as the exponent, and subtracts 1 from the exponent result. For use cases where the input tensor values are small, this provides higher precision than directly computing the result with the formula.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/eye/README_en.md">eye</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns a two-dimensional tensor with a value of 1 on the diagonal and 0 elsewhere.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/floor/README_en.md">floor</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Rounds down each element in the input tensor and writes the result back to the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/floor_div/README_en.md">floor_div</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs division and rounds the result down.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/floor_mod/README_en.md">floor_mod</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the element-wise modulo operation of two input tensors.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/fresnel_sin/README_en.md">fresnel_sin</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the Fresnel sine integral.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/fresnel_cos/README_en.md">fresnel_cos</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the Fresnel cosine integral.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/fused_mul_add/README_en.md">fused_mul_add</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>A ternary element-wise fused operator that fuses Mul and Add into a single computation, with the formula y = x1 * x2 + x3.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/fused_mul_add_add/README_en.md">fused_mul_add_add</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>A quaternary element-wise fusion operator that fuses Mul, Add, and Add into a single computation. Formula: y = x1 * x2 + x3 + x4.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/fused_mul_add_n/README_en.md">fused_mul_add_n</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Fuses the mul and addn operators. Requires n of addn to be 2, and one input of mul to be a scalar or a tensor containing only one number.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/fused_mul_addn_l2loss/README_en.md">fused_mul_addn_l2loss</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>An internal graph fusion operator that merges the Mul+AddN+L2Loss subgraph. Computes y1 = x1*x3[0]+x2 element-wise, and reduces the full square of x1 to obtain the scalar y2 = Σx1²/2.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/gcd/README_en.md">gcd</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the greatest common divisor of two input tensors.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ger/README_en.md">ger</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the outer product of one-dimensional vectors vec1 and vec2 to obtain a two-dimensional matrix.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/get_shape/README_en.md">get_shape</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Obtains the shape information of one or more input tensors, and concatenates the size values of each dimension in input order into a one-dimensional int32 tensor as the output.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/greater/README_en.md">greater</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Determines whether each element of the input tensor is greater than the value of the other scalar.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/greater_equal/README_en.md">greater_equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Returns a BOOL tensor. Each value indicates whether the value at the corresponding position of the input tensor is greater than or equal to the value of another tensor or a scalar value.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/approximate_equal/README_en.md">approximate_equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Element-wise approximate equality check operator. Determines whether the corresponding elements of two tensors are approximately equal within the allowed error range. Outputs a boolean result.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/grouped_bias_add_grad/README_en.md">grouped_bias_add_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Backward computation of grouped bias addition (GroupedBiasAdd).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/hans_decode/README_en.md">hans_decode</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Decodes the compressed tensor based on PDF, and restores the tensor based on mantissa recombination.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/hans_encode/README_en.md">hans_encode</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs PDF statistics on the byte where the exponent of the input tensor is located. Performs lossless compression based on the PDF distribution statistics.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/histogram_fixed_width/README_en.md">histogram_fixed_width</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the histogram of a tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/histogram_v2/README_en.md">histogram_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the histogram of a tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/hypot/README_en.md">hypot</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the length of the hypotenuse of the right triangle formed by the corresponding elements of two input tensors element-wise.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/inplace_top_k_distance/README_en.md">inplace_top_k_distance</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Merges and reorders a set of sorted distance data with a new set of distance data, then takes the top k data. The result is updated in-place into the input.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/inv/README_en.md">inv</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the reciprocal of each element of the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/inv_grad/README_en.md">inv_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Backward gradient of the Inv (reciprocal) operator. Computes y = -grad * x * x.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/invert/README_en.md">invert</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Negates each element of the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_close/README_en.md">is_close</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns a new tensor with boolean elements. Determines whether the given self and other are close to each other. Returns True if the values are close, otherwise returns False.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_finite/README_en.md">is_finite</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Determines which elements of the input tensor are finite values, that is, not inf, -inf, or nan.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_inf/README_en.md">is_inf</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Determines which elements in the tensor are infinite, that is, inf or -inf.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_nan/README_en.md">is_nan</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Determines which elements in the tensor are NaN.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_neg_inf/README_en.md">is_neg_inf</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Determines whether the elements of the input tensor are negative infinity.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_pos_inf/README_en.md">is_pos_inf</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Determines whether the elements of the input tensor are positive infinity.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/kl_div/README_en.md">kl_div</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the KL divergence (a compatibility shell operator for legacy graph migration; the compute kernel forwards to KlDivV2).</td>
  </tr>
    <tr>
    <td>math</td>
    <td><a href="../../math/kl_div_v2/README_en.md">kl_div_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the KL divergence.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/left_shift/README_en.md">left_shift</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Computes the element-wise bitwise left shift of input tensors x and y.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/lerp/README_en.md">lerp</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs linear interpolation between the start and end tensors according to the given weight, and returns the interpolated tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/less/README_en.md">less</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>The operator determines whether each element in the input self is less than the value of the input other.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/less_equal/README_en.md">less_equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Determines whether the element value in the input self is less than or equal to the value of other, and writes the comparison result of each element of self with the value of other into out.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/lgamma/README_en.md">lgamma</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/lin_space/README_en.md">lin_space</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Generates an evenly spaced sequence of values. Creates a 1D vector of size steps, whose values are linearly and uniformly distributed from start to stop (inclusive).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/qr/README_en.md">linalg_qr</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/log/README_en.md">log</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>The operator performs element-wise logarithm computation on the elements of the input tensor x.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/log1p/README_en.md">log1p</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs the log1p operation on the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/log_add_exp/README_en.md">log_add_exp</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/logdet/README_en.md">logdet</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/logical_and/README_en.md">logical_and</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs a logical AND on the corresponding elements of two input tensors and returns a boolean tensor (True/False).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/logical_not/README_en.md">logical_not</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the element-wise logical NOT of the given input tensor. If the output type is not specified, the output tensor is of bool type. If the input tensor is not of bool type, zero is treated as False and non-zero as True.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/logical_or/README_en.md">logical_or</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs a logical OR on the elements of the given input tensors. When the two input tensors are not of bool type, 0 is treated as False and non-zero as True.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/logspace">logspace</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Creates a one-dimensional tensor of size steps, with values evenly spaced on a logarithmic scale from base^start to base^end.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/log_space_d/README_en.md">log_space_d</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Generates a one-dimensional logarithmically spaced sequence according to the start value, end value, number of steps, and base.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/masked_scale/README_en.md">masked_scale</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/maximum/README_en.md">maximum</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns a new tensor composed of the maximum values of the two input tensors.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/maximum_grad/README_en.md">maximum_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the backward gradient of the Maximum operator, y1 = grads * (x1 >= x2), y2 = grads * (x1 < x2).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/minimum/README_en.md">minimum</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns a new tensor composed of the minimum values of two input tensors.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/minimum_grad/README_en.md">minimum_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the backward gradient of the Minimum operator, y1 = grads * (x1 <= x2), y2 = grads * (x1 > x2).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/mod/README_en.md">mod</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the remainder of tensor elements.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/mul/README_en.md">mul</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Returns a new tensor composed of the products of the elements of two tensors.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/mul_addn/README_en.md">mul_addn</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Implements the fused computation of N>=2 mul and addn operations to reduce data movement time and memory usage.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/mul_no_nan/README_en.md">mul_no_nan</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs binary element-wise safe multiplication. When the multiplier x2 is 0, the result returns 0 to avoid NaN pollution caused by 0*inf and 0*nan.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/muls/README_en.md">muls</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns a new tensor composed of the product of each input tensor element and the specified scalar value.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/nan_to_num/README_en.md">nan_to_num</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Replaces NaN, positive infinity, and negative infinity values in the input with the values specified by nan, posinf, and neginf, respectively.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ndtri/README_en.md">ndtri</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Inverse standard normal cumulative distribution function (probit). Computes the standard normal quantile for each element of the input probability tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/neg/README_en.md">neg</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Computes the opposite number of each element.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/non_finite_check/README_en.md">non_finite_check</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Checks whether non-finite values (NaN, Inf, -Inf) exist in the input tensor_list.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/not_equal/README_en.md">not_equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>The operator compares two input tensors element-wise to determine whether they are not equal.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/one_hot/README_en.md">one_hot</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Converts discrete labels into binary encoding.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ones_like/README_en.md">ones_like</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Returns a tensor of the same shape and type with all elements set to 1.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/pdist/README_en.md">pdist</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the p-norm distance between each pair of row vectors in the input tensor, and outputs a one-dimensional vector arranged in upper-triangular row-major order.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/polar/README_en.md">polar</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Constructs a complex tensor whose elements are the Cartesian coordinates corresponding to polar coordinates, where the absolute value is abs and the angle is angle.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/population_count/README_en.md">population_count</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Extracts each bit of the binary representation and accumulates all bits that are 1.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/pow/README_en.md">pow</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Performs exponentiation according to the input exponent and power tensors.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/pows/README_en.md">pows</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Applies the power operation with exponent to each element of input.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/precision_compare">precision_compare</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Checks whether the elements of two tensors are the same, performs precision comparison, and returns the status code of the operator execution.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/prod_force_se_a/README_en.md">prod_force_se_a</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Backward force computation operator for the DeepPot-SE model. Multiplies the derivative of the neural network with respect to the descriptor by the derivative of the descriptor with respect to the coordinates element-wise, then reduces and accumulates the result to obtain the 3D force vector of each atom.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/prod_env_mat_a_calc_rij/README_en.md">prod_env_mat_a_calc_rij</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Neighbor computation of the DeepMD environment matrix. Outputs the neighbor index, squared distance, and the displacement vector from the central atom to the neighbor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/qr/README_en.md">q_r</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ragged_bin_count/README_en.md">ragged_bin_count</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>For ragged rows divided by splits, counts the occurrences or the sum of weights of each bin.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/range/README_en.md">range</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Generates a tensor of an arithmetic sequence from start to end with a step interval.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/rank/README_en.md">rank</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/Host CPU</td>
    <td>Returns the rank of the input tensor, that is, the number of dimensions of the tensor. The output is an INT32 scalar.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/real/README_en.md">real</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns the real part of a complex tensor. If the input is a real tensor, returns the original value.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/real_div/README_en.md">real_div</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>The operator returns the result of x1/x2 element by element.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reciprocal/README_en.md">reciprocal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns a new tensor with the reciprocal of each input element.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reciprocal_grad/README_en.md">reciprocal_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the backward gradient of the Reciprocal operator.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_all/README_en.md">reduce_all</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core/AI CPU</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_any/README_en.md">reduce_any</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_log_sum/README_en.md">reduce_log_sum</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_log_sum_exp/README_en.md">reduce_log_sum_exp</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_max/README_en.md">reduce_max</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_mean/README_en.md">reduce_mean</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>The operator computes the mean of a multidimensional vector along the specified dimension.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_min/README_en.md">reduce_min</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_nansum/README_en.md">reduce_nansum</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_prod/README_en.md">reduce_prod</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_std_v2/README_en.md">reduce_std_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the standard deviation along the specified dimension.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_std_v2_update/README_en.md">reduce_std_v2_update</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Receives the original input and the precomputed mean, and reduces along the specified dimension to compute the variance or standard deviation. Supports Bessel's correction and keepdim.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_std_with_mean/README_en.md">reduce_std_with_mean</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Given an externally precomputed mean, computes the standard deviation (invert=false) or its reciprocal (invert=true) along the specified dimension.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_sum/README_en.md">reduce_sum</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Returns the sum of each row of the input tensor in the given dimension.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_var/README_en.md">reduce_var</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns the variance of the specified dimension.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/rfft1_d/README_en.md">rfft1_d</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Performs RFFT (Fourier transform) on the input tensor self. The output is a complex tensor containing non-negative frequencies.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/right_shift/README_en.md">right_shift</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>The operator performs a bitwise right shift on the input tensors x and y element by element.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/round/README_en.md">round</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>The operator rounds the value of the input tensor to the nearest integer. If the value is equally distant from two integers, it rounds to the even integer.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/rsqrt/README_en.md">rsqrt</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Computes the reciprocal of the square root of each element of input (Tensor).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/rsqrt_grad/README_en.md">rsqrt_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Backward computation of rsqrt.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/scale/README_en.md">scale</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Scales the input x by scale and optionally adds bias: y = x * scale + bias. Supports channel broadcast controlled by axis/num_axes/scale_from_blob.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/search_sorted/README_en.md">search_sorted</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>The operator finds the positions where the given tensor values should be inserted in a sorted tensor sorted_sequence.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/segsum/README_en.md">segsum</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Performs segmentation and computation. Generates a semiseparable matrix with a diagonal of 1 and an upper triangle of 0.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/select/README_en.md">select</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Selects elements from self or other according to the condition and returns them (supports broadcast).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/select_v2/README_en.md">select_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Selects elements from self or other according to the condition and returns them (supports broadcast).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/get_dynamic_dims/README_en.md">get_dynamic_dims</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Extracts the actual dimension values from the dynamic input tensor according to the unknown dimensions marked in shape_info and outputs them.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/shape/README_en.md">shape</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/Host CPU</td>
    <td>Returns the shape of the input tensor, that is, a one-dimensional tensor composed of the dimension values of the input.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/shape_n/README_en.md">shape_n</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/Host CPU</td>
    <td>Returns the shapes of multiple input tensors. Each input corresponds to one output, which is a one-dimensional tensor composed of the dimension values of the corresponding input.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sign/README_en.md">sign</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs the Sign function element-wise on the input tensor and returns the result tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sign_bits_pack/README_en.md">sign_bits_pack</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Extracts the sign bits of floating-point numbers (float16/float32) and packs them into uint8 bytes. Every 8 sign bits are packed into 1 byte.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sign_bits_unpack/README_en.md">sign_bits_unpack</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Unpacks the 1-bit Adam sign bits of UINT8 type into a FLOAT32 or FLOAT16 tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/signbit/README_en.md">signbit</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Determines whether the sign bit of each element in the input self is 1, and returns a tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/silent_check/README_en.md">silent_check</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/silent_check_v2/README_en.md">silent_check_v2</a></td>
    <td> </td>
    <td> </td>
    <td>√</td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sin/README_en.md">sin</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs the sin operation on the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sparse_bincount/README_en.md">sparse_bincount</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Counts the occurrences or weighted sum of each value in a sparse tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sinc/README_en.md">sinc</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs the sinc operation on the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sinh/README_en.md">sinh</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>The operator performs a hyperbolic sine operation on each element of the input and outputs the result.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sinkhorn/README_en.md">sinkhorn</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Computes the Sinkhorn distance, which can be used for expert routing in MoE models.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/slogdet/README_en.md">slogdet</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Computes the sign and natural logarithm of the determinant of the input self.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sort/README_en.md">sort</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Sorts the elements in the input tensor in ascending or descending order along the specified dimension and returns the corresponding index values.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sort_with_index/README_en.md">sort_with_index</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Sorts the input tensor by element value, with the index values following the corresponding element values.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/spence/README_en.md">spence</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the Spence function (dilogarithm): S(x) = -integral_0^(1-x) ln(1-t)/t dt = Li2(1-x).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sqrt/README_en.md">sqrt</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Computes the square root of a non-negative number. Returns nan for negative numbers.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sqrt_grad/README_en.md">sqrt_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Computes the gradient of the square root.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/square/README_en.md">square</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Computes the square of each element of the input tensor element-wise.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/square_sum_all/README_en.md">square_sum_all</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the sum of squares of all elements for two input tensors of the same shape separately, and outputs two independent scalars.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/square_sum_v1/README_en.md">square_sum_v1</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Computes the sum of squares of the input tensor along the specified axis.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/squared_difference/README_en.md">squared_difference</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Subtracts the second input tensor from the first input tensor and computes the square of the result.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/stft/README_en.md">stft</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the Fourier transform of the input within a sliding window.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sub/README_en.md">sub</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Performs subtraction. The minuend is scaled by alpha.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sub_mul_concat/README_en.md">sub_mul_concat</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Internal fusion operator. It is generated by the existing CANN graph fusion process after converting the original Sub, Mul, and Concat/ConcatV2/ConcatV2D graphs. No public ACLNN or direct construction API is provided.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/svd/README_en.md">svd</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Computes the singular value decomposition of one or more matrices.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tan/README_en.md">tan</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Tangent trigonometric function.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tanh/README_en.md">tanh</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Activation function. Returns a tensor with the same shape as the input tensor and performs elementwise computation on the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tanh_grad/README_en.md">tanh_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Backward computation of Tanh.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tensor_equal/README_en.md">tensor_equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes whether two tensors have the same size and elements, and returns a Bool type.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tile/README_en.md">tile</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Copies the input tensor along each dimension by the number of times specified in repeats.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/top_k_pq_distance_v2/README_en.md">top_k_pq_distance_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Finds the k maximum or minimum elements and their indices on the last dimension using grouped extreme value pruning.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/top_k_v2/README_en.md">top_k_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns the k extreme values and their indices of the input tensor on the specified dimension. Supports specifying the k value through the input tensor.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/topk/README_en.md">topk</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Finds the k maximum or minimum elements and their indices in the input tensor along the specified dimension.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/top_k_v2_d/README_en.md">top_k_v2_d</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Finds the k maximum or minimum elements and their indices in the input tensor along the specified dimension. Compatible with TensorFlow TopKV2.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/top_k_pq_distance/README_en.md">top_k_pq_distance</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>TopK distance computation based on grouped extreme value pruning, used for vector retrieval tasks.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/trace/README_en.md">trace</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Computes the sum of the main diagonal elements of the matrix starting from the upper-left corner.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/transform_bias_rescale_qkv/README_en.md">transform_bias_rescale_qkv</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>An API for processing the Query, Key, and Value vectors in the multi-head attention mechanism. It adjusts the bias and rescale factors of these vectors.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/triangular_solve/README_en.md">triangular_solve</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Solves a system of equations with a square upper or lower triangular invertible matrix A and multiple right-hand sides b.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/trunc/README_en.md">trunc</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Truncates the input tensor element-wise toward zero.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/truncate_mod/README_en.md">truncate_mod</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes the remainder of tensors.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/xdivy/README_en.md">xdivy</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes x / y. The result is 0 when x == 0. Supports broadcast.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/xlogy/README_en.md">xlogy</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes x1 * log(x2). The result is 0 when x1 == 0. Supports broadcast.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/xlogy/README_en.md">x_log_y</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Computes x1 * log(x2) element-wise. Supports broadcast.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/xlog1py/README_en.md">xlog1py</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Computes x * log(1 + y). The result is 0 when x == 0. Supports broadcast.</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/zeta/README_en.md">zeta</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Computes the Hurwitz zeta function ζ(x, q).</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/zero_op/README_en.md">zero_op</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Fills the tensor with zeros.</td>
   </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/ball_query/README_en.md">ball_query</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Ball Query finds all points within the specified ball radius of point p1 (an upper limit of K is set in the implementation).</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/bitcast/README_en.md">bitcast</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Reinterprets the input tensor bitwise as the target data type. It does not change the underlying data, only the dataType and shape.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/calc_buckets_limit_and_offset/README_en.md">calc_buckets_limit_and_offset</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Computes the limits and offsets of each bucket according to ivf_counts and bucket_list, for vector retrieval use cases.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/chunk_cat/README_en.md">chunk_cat</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Splits all tensors in tensors into numChunks blocks along dimension dim, concatenates them along the dimension after dim, and finally converts the result to the data type of out.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/circular_pad/README_en.md">circular_pad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Fills the last two dimensions of the input tensor using circular padding of the input.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/circular_pad_grad/README_en.md">circular_pad_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Backward propagation of circular_pad.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/coalesce_sparse/README_en.md">coalesce_sparse</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Implements the coalesce() method optimized for Coo_Tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/coordinates_1d_to_2d/README_en.md">coordinates_1d_to_2d</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Converts 1D coordinates to 2D coordinates and computes row and column indices according to the shape information.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/depth_to_space/README_en.md">depth_to_space</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>The operator rearranges the depth (channel) dimension of the input tensor into spatial dimensions (height and width). Specifically, it divides the depth dimension of the input tensor by the specified block_size and rearranges these depth blocks into spatial dimensions, thereby increasing the spatial dimension size while reducing the channel dimension depth.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/diag/README_en.md">diag</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Creates a diagonal matrix with the input tensor of no more than 4 dimensions as the diagonal elements.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/diag_flat/README_en.md">diag_flat</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Creates a flattened diagonal matrix with the input array as the diagonal elements.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/diag_v2/README_en.md">diag_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Extracts the diagonal elements specified by diagonal from the input 2D tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/feeds_repeat/README_en.md">feeds_repeat</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>For the input feeds, copies the data on dimension 0 of the corresponding feeds the corresponding number of times according to the input feeds_repeat_times, and pads dimension 0 of the output y to the size of output_feeds_size.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/fill_diagonal_v2/README_en.md">fill_diagonal_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Fills the main diagonal of a matrix with a specified value.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/masked_select_v3/README_en.md">masked_select_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Selects the values at the corresponding positions of the input according to whether the mask is True. The input and mask satisfy the broadcast rule. The result is a one-dimensional tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/mem_set_v2/README_en.md">mem_set_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Assigns a specified value to the input tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v3_grad_replicate/README_en.md">pad_v3_grad_replicate</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Backward propagation of padv3 2D.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v3_grad_replication/README_en.md">pad_v3_grad_replication</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Backward propagation of padv3 3D.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v4_grad/README_en.md">pad_v4_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Backward propagation of the input after pad.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/reflection_pad3d_grad/README_en.md">reflection_pad3d_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Computes the backward propagation of the aclnnReflectionPad3d API.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/stack_ball_query/README_en.md">stack_ball_query</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Stack Ball Query is an alternative to KNN. It finds all points within a specified radius of point p1. An upper limit of K is set in the implementation.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/slice_last_dim/README_en.md">slice_last_dim</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Slices the last dimension of the input tensor. start and end specify the slice range, and stride specifies the step.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/slice_with_axes/README_en.md">slice_with_axes</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Extracts slices from the input tensor along the specified axes. For each axis, offsets specifies the start position and size specifies the slice length.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/slice_write/README_en.md">slice_write</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Writes the value tensor to the specified position of the x tensor (offset specified by begin). This is an in-place operation. The output shares the same memory as the input x.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/strided_slice_assign_v2/README_en.md">strided_slice_assign_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>StridedSliceAssign is a tensor slice assignment operation. It assigns the content of the inputValue tensor to the specified position in the target varRef tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/transpose_v2/README_en.md">transpose_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Implements the tensor dimension permutation operation. Rearranges the dimensions of the input tensor in the specified order.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/unfold_grad/README_en.md">unfold_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Implements the backward of the Unfold operator. Computes the corresponding gradient.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/concat_d/README_en.md">concat_d</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Concatenates all tensors in tensors along the dim dimension. Dimensions other than the corresponding dimension must be consistent.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/concat_dv2/README_en.md">concat_dv2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Concatenates all tensors in tensors along the dimension concat_dim. Dimensions other than the corresponding dimension must be consistent.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/concat_v2/README_en.md">concat_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>This operator concatenates multiple input tensors along the specified dimension.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/clip_by_norm_no_div_sum/README_en.md">clip_by_norm_no_div_sum</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Performs the ClipByNormNoDivSum operation on four input tensors that support broadcast.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/clip_by_value/README_en.md">clip_by_value</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>This operator limits all elements of the input to the range [clipValueMin, clipValueMax].</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/clip_by_value_v2/README_en.md">clip_by_value_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>This operator limits all elements of the input to the range [clipValueMin, clipValueMax].</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/invert_permutation/README_en.md">invert_permutation</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/conjugate_transpose/README_en.md">conjugate_transpose</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Transposes the input x along the dimension order specified by perm and takes the conjugate. For real types, this is equivalent to a normal Transpose.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/unpack/README_en.md">unpack</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Splits a tensor into multiple sub-tensors along a dimension.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/as_strided/README_en.md">as_strided</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Creates a tensor view that shares the same data memory as the original tensor by specifying a new shape and stride.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/assign/README_en.md">assign</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Fills the value of one input tensor into another input tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/broadcast_to/README_en.md">broadcast_to</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Broadcasts the input tensor to a specified shape.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/batch_to_space/README_en.md">batch_to_space</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Rearranges data of the batch dimension into spatial dimensions and crops the spatial dimensions.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/batch_to_space_nd/README_en.md">batch_to_space_nd</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Rearranges data of the batch dimension into spatial dimensions and crops the spatial dimensions.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/case_condition/README_en.md">case_condition</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Outputs a discrete case number according to the relationship between three input values and a threshold.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/concat/README_en.md">concat</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Concatenates multiple input tensors along the specified dimension and outputs a tensor containing all input data concatenated in order.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/confusion_transpose_d/README_en.md">confusion_transpose_d</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Fuses reshape and transpose operations.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/contiguous/README_en.md">contiguous</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/dynamic_partition/README_en.md">dynamic_partition</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Dynamically splits the input data into multiple tensors according to partition indices.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/expand_dims/README_en.md">expand_dims</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Inserts a dimension of size 1 at the specified position of the input tensor shape. It only changes the tensor shape and does not change the underlying data.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/fill/README_en.md">fill</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Fills the input tensor with the specified scalar value.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/fill_v2/README_en.md">fill_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Fills the input tensor with the specified scalar value, where the fill value is passed in as an attribute.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/flatten/README_en.md">flatten</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/flatten_v2/README_en.md">flatten_v2</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Flattens the dimensions of the input tensor in the range [axis, end_axis] into one dimension without changing the underlying data.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/gather_shapes/README_en.md">gather_shapes</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Collects the specified dimension values from the shapes of multiple input tensors and concatenates them into a one-dimensional tensor output in the order of axes.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/im2col/README_en.md">im2col</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Converts sliding local window data into column vectors and concatenates them into a large tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/masked_fill/README_en.md">masked_fill</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Fills the elements at positions where `mask` is `True` in the input tensor `x` with the specified value. `mask` must have the same shape as `x` or be broadcastable.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/masked_fill_range/README_en.md">masked_fill_range</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Fills the segment of the input tensor `x` within the interval windows given by `start`/`end` along `axis` with the per-round value from `value`.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/mirror_pad/README_en.md">mirror_pad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Fills the boundaries of the input tensor in a mirrored manner.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/npu_format_cast/README_en.md">npu_format_cast</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>This operator currently has no Ascend C code implementation. Developers are welcome to contribute. For the contribution method, see the <a href="../../CONTRIBUTING_en.md">contribution guide</a>.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pack/README_en.md">pack</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Packs the list of tensors in values into a tensor one dimension higher than each tensor in values by packing along the axis dimension.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad/README_en.md">pad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>According to the paddings configuration, fills each dimension of the input x with constant 0 before and after. Compatible with the TensorFlow Pad operator.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v2/README_en.md">pad_v2</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Fills the input tensor with a specified constant value.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v3/README_en.md">pad_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Fills the input tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v3_grad/README_en.md">pad_v3_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Backward propagation of the input after pad_v3.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/phony_concat/README_en.md">phony_concat</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Marks that GE allocates one contiguous memory block for all inputs and computes offsets. Performs no computation. The preceding custom operator writes by offset.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/phony_split/README_en.md">phony_split</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Marks that GE allocates one contiguous memory block for all outputs and computes offsets. Performs no computation. The following custom operator reads by offset.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/prevent_gradient/README_en.md">prevent_gradient</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>In the forward pass, the operator returns the input tensor unchanged. In backward propagation, the gradient cannot flow through. An error is reported immediately once it flows through. Used to explicitly block gradient backpropagation.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/reshape/README_en.md">reshape</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Reinterprets the input tensor to the target shape.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/roll/README_en.md">roll</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Rolls tensor elements along the specified dimension.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/size/README_en.md">size</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Returns the size of the input tensor, that is, an integer scalar of the number of tensor elements.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/slice/README_en.md">slice</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Extracts a sub-tensor from the input tensor according to the given range and stride.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/snapshot/README_en.md">snapshot</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>The operator returns a copy of the input tensor. The output has the same shape, data type, and content as the input.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/space_to_batch/README_en.md">space_to_batch</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Rearranges data of the spatial dimension into the batch dimension by blocks, and pads zeros to the spatial dimension.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/space_to_batch_nd/README_en.md">space_to_batch_nd</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Rearranges data of multiple spatial dimensions into the batch dimension by blocks, and pads zeros to the spatial dimensions. Supports 1-7 spatial dimensions (input tensor rank 2-8).</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/sparse_reshape/README_en.md">sparse_reshape</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Reshapes the indices of a sparse tensor from the input shape to the target shape.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/split/README_en.md">split</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Splits a tensor into num_split smaller tensors evenly along the specified dimension split_dim.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/split_v/README_en.md">split_v</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Splits the input tensor along the specified dimension by the specified lengths. The last block may be shorter.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/squeeze/README_en.md">squeeze</a></td>
    <td>×</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/Host CPU</td>
    <td>Removes dimensions of size 1 from the input tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/squeezev2/README_en.md">squeezev2</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Removes dimensions of size 1 from the tensor shape. When axis is empty, all dimensions of size 1 are removed. When axis is specified, only the specified dimension is removed (it must be 1 or an unknown dimension). The output data type is consistent with the input.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/squeezev3/README_en.md">squeezev3</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Removes dimensions with a dimension value of 1 from the tensor. The axes are provided as an optional input.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/stop_gradient/README_en.md">stop_gradient</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>The operator passes the input tensor to the output unchanged and prevents backward propagation from passing through this node.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/strided_slice/README_en.md">strided_slice</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Extracts a sub-tensor from the input tensor according to the specified start position, end position, and stride.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/strided_slice_grad/README_en.md">strided_slice_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Maps the gradient of the sub-tensor back to the corresponding positions of the original tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/strided_slice_v3/README_en.md">strided_slice_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Extracts a sub-tensor from the input tensor according to the specified start position, end position, and stride.</td>
   </tr>
   <tr>
    <td>conversion</td>
    <td><a href="../../conversion/tensor_move/README_en.md">tensor_move</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Moves the value of the input tensor to the output tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/tensor_redirect/README_en.md">tensor_redirect</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Copies the data of the input tensor to the output tensor bit by bit, as an identity mapping.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/tile_with_axis/README_en.md">tile_with_axis</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Copies the input tensor data along the specified dimension to expand the output tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/trans_data/README_en.md">trans_data</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Converts the data format of the tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/trans_shape/README_en.md">trans_shape</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Changes the output shape according to the outShape attribute and does not change the underlying data.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/transpose/README_en.md">transpose</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Completes tensor transposition.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/tril/README_en.md">tril</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Sets the upper-right part along the diagonal of the last two dimensions (counted from left to right by shape) of the input self tensor to zero.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/triu/README_en.md">triu</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Sets the lower-left part along the diagonal of the last two dimensions (counted from left to right by shape) of the input self tensor to zero. The diagonal parameter can be positive or negative and defaults to zero. A positive value moves the main diagonal toward the upper right, and a negative value moves it toward the lower left.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/unsqueeze/README_en.md">unsqueeze</a></td>
    <td>×</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/Host CPU</td>
    <td>Inserts a dimension of size 1 at the specified position.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/unsqueezev2/README_en.md">unsqueezev2</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Inserts a dimension whose dimension value is 1 into the tensor shape. When axis is empty, no dimension is inserted. When axis is specified, the dimension is inserted at the normalized position. The output data type is consistent with the input.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/unsqueezev3/README_en.md">unsqueezev3</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Inserts a dimension whose dimension value is 1 at the specified position of the input tensor. Only the shape is changed, and the data is not changed.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/view_copy/README_en.md">view_copy</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Copies the src tensor to the dst tensor according to the view parameters and returns dst.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/identity/README_en.md">identity</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/Host CPU</td>
    <td>The operator returns a tensor with the same shape and content as the input tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/identity_n/README_en.md">identity_n</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/Host CPU</td>
    <td>The operator returns a set of copies of the input tensors. The output has the same number, shape, data type, and content as the input.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/matrix_set_diag/README_en.md">matrix_set_diag</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Replaces the diagonal elements of the input tensor with the values of the diagonal tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/matrix_set_diag_v2/README_en.md">matrix_set_diag_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Replaces the diagonal elements of the input tensor with the values of the diagonal tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/matrix_diag/README_en.md">matrix_diag</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns a batched diagonal tensor from the given batched diagonal values.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/matrix_diag_part/README_en.md">matrix_diag_part</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns the main diagonal elements of the innermost matrices of the batched tensor.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/matrix_diag_v3/README_en.md">matrix_diag_v3</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Generates a matrix from the input single or multiple diagonal values. Positions outside the diagonal band are filled with padding_value.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/matrix_diag_part_v3/README_en.md">matrix_diag_part_v3</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Extracts one or more diagonals from the last two dimensions of the input matrix or batch of matrices.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/split_d/README_en.md">split_d</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>Splits the tensor into num_split smaller tensors evenly along the specified dimension split_dim. Unlike the Split operator, split_dim is provided as an attribute rather than an input.</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/strided_slice_assign/README_en.md">strided_slice_assign</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Assigns the content of the input tensor input_value to the slice position in the target tensor var specified by begin, end, and strides. The area outside the slice keeps the original value of var (in-place semantics).</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/parallel_concat/README_en.md">parallel_concat</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Concatenates N tensors of the same shape with a first-dimension size of 1 along the first dimension. The output shape is [N, d_1..d_k], and the data is copied bit by bit.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/drop_out_do_mask/README_en.md">drop_out_do_mask</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Sets the elements in the input to zero according to the value of the corresponding bit in mask, and scales them by scale or sets them to zero.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/drop_out_do_mask_v3/README_en.md">drop_out_do_mask_v3</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/drop_out_do_mask_v3_d/README_en.md">drop_out_do_mask_v3_d</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/drop_out_v3/README_en.md">drop_out_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Randomly sets the elements in the input to zero with probability p, and scales the output by 1/(1-p).</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/drop_out_v3_grad/README_en.md">drop_out_v3_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>DropOut backward operator. Computes the input gradient according to mask and scale.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/dsa_gen_bit_mask/README_en.md">dsa_gen_bit_mask</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/dsa_random_normal/README_en.md">dsa_random_normal</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/dsa_random_uniform/README_en.md">dsa_random_uniform</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/random_standard_normal_v2/README_en.md">random_standard_normal_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Generates a sequence of random numbers with standard normal distribution.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/random_uniform_int_v2/README_en.md">random_uniform_int_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Generates a sequence of random integers with uniform distribution.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/random_uniform_v2/README_en.md">random_uniform_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Generates a sequence of random floating-point numbers with uniform distribution.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/sim_thread_exponential/README_en.md">sim_thread_exponential</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_bernoulli/README_en.md">stateless_bernoulli</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Extracts binary random numbers from a Bernoulli distribution.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_drop_out_gen_mask/README_en.md">stateless_drop_out_gen_mask</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Generates a random mask according to the random seed, random offset, and keep probability.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_exponential/README_en.md">stateless_exponential</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Generates random numbers of exponential distribution.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_random/README_en.md">stateless_random</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Returns random numbers drawn from a discrete uniform distribution in the range [from, to - 1].</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_random_choice_with_mask/README_en.md">stateless_random_choice_with_mask</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Selects the corresponding index list based on the true values in input x. Then calls philox_random according to seed and offset to generate random numbers of the same length as the index list. Uses the random numbers to shuffle the index list, obtains the randomly swapped index list, and returns count elements.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_random_normal_v2/README_en.md">stateless_random_normal_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Generates normally distributed random numbers.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_random_normal_v3/README_en.md">stateless_random_normal_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Generates normally distributed random numbers.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_normal/README_en.md">stateless_normal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Generates normally distributed random numbers (bit-aligned with H20 GPU).</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_random_uniform_v2/README_en.md">stateless_random_uniform_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Generates uniformly distributed random numbers.</td>
  </tr>
  <tr>
      <td>math</td>
      <td><a href="../../math/amp_update_scale/README_en.md">amp_update_scale</a></td>
      <td>√</td>
      <td>√</td>
      <td>√</td>
      <td>×</td>
      <td>AI Core</td>
      <td>Implements dynamic scale update in AMP training. Dynamically adjusts the loss scale size according to the current scale value, the growth_tracker counter, and whether Inf/NaN is found.</td>
  </tr>
  <tr>
      <td>math</td>
      <td><a href="../../math/prod_virial_se_a/README_en.md">prod_virial_se_a</a></td>
      <td>√</td>
      <td>√</td>
      <td>×</td>
      <td>√</td>
      <td>AI Core</td>
      <td>Computes the type-A virial tensor based on the SE atom embedding descriptor in the deep potential energy model.</td>
  </tr>
  <tr>
      <td>math</td>
      <td><a href="../../math/prod_env_mat_a/README_en.md">prod_env_mat_a</a></td>
      <td>√</td>
      <td>√</td>
      <td>×</td>
      <td>√</td>
      <td>AI Core</td>
      <td>Computes the DeepMD-kit environment matrix descriptor (SeA), including two phases: neighbor list formatting and environment matrix computation.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_uniform/README_en.md">stateless_uniform</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Generates uniformly distributed random numbers.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_random_uniform_v3/README_en.md">stateless_random_uniform_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Generates uniformly distributed random numbers.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_randperm/README_en.md">stateless_randperm</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_truncated_normal_v2/README_en.md">stateless_truncated_normal_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td> </td>
    <td>√</td>
    <td>AI Core</td>
    <td>Generates random numbers of truncated normal distribution.</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/truncated_normal_v2/README_en.md">truncated_normal_v2</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/aipp_data/README_en.md">aipp_data</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>A data passthrough node for dynamic AIPP use cases. Passes the input tensor data directly to the output tensor out. The index attribute identifies the AIPP sequence number. The output shape and data type are consistent with the input.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/case/README_en.md">case</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Selects a subgraph branch to execute according to the value of `branch_index`. Passes the input tensor to the selected subgraph and returns its output tensor.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/const/README_en.md">const</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Creates a constant tensor from a tensor-like object for inference use cases. The Const operator has the same definition as the Constant operator. It derives the shape and data type of the output tensor y according to the tensor value and type stored in the value attribute. The output shape and data type are consistent with the tensor in the value attribute.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/const_place_holder/README_en.md">const_place_holder</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Creates a constant placeholder tensor. The operator processes inputs whose address is unchanged but whose value is uncertain. Converting the input to this type helps improve graph execution performance (avoids repeated refresh of the input address). It derives the shape and data type of the output tensor y according to the origin_shape and dtype attributes. The output shape is consistent with the origin_shape attribute, and the output data type is consistent with the dtype attribute.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/constant/README_en.md">constant</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Creates a constant tensor for training. The Constant operator derives the shape and data type of the output tensor y according to the tensor value and type stored in the value attribute. The output shape and data type are consistent with the tensor in the value attribute.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/control_trigger/README_en.md">control_trigger</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Used only as a placeholder for a control edge. Similar to NoOp, but always produces an active control output even if some control inputs are invalidated.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/data/README_en.md">data</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Provides input data for other operators. The Data operator acts as a data input node in the graph. It passes the input tensor x directly to the output tensor y. The index attribute identifies the sequence number of the data node in the network.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/destroy_temporary_variable/README_en.md">destroy_temporary_variable</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Destroys the temporary variable and returns its final value. All other uses of the temporary variable must be executed before this operator. This operator is used together with the TemporaryVariable operator: it matches the TemporaryVariable through the var_name attribute, reads the final value of the temporary variable, and triggers destruction. The output tensor y has the same shape and data type as the input tensor x.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/end/README_en.md">end</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Marks the output of a subgraph split by engine type. The End operator is an internal node of the graph engine and is used together with PlaceHolder. The output tensor y has the same shape and data type as the input tensor x.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/enter/README_en.md">enter</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Creates or finds a child frame and makes the input tensor x available within the child frame. The Enter operator is used together with the Exit operator to create a loop in the graph. The executor identifies the frame through the unique frame_name attribute. When is_constant is true, the output tensor y is a constant within the child frame. The output tensor y has the same shape and data type as the input tensor x.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/exit/README_en.md">exit</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Exits the current frame to its parent frame. The Exit operator is used together with the Enter operator to create a loop in the graph. It passes the input tensor x to the parent frame to make it available. The output tensor y has the same shape and data type as the input tensor x.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/file_constant/README_en.md">file_constant</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Creates a file constant tensor for processing very large weight data stored in a file. The shape and data type of the output tensor y are derived from the shape and dtype attributes. The output shape is consistent with the shape attribute, and the output data type is consistent with the dtype attribute.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/for/README_en.md">for</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Executes the `body` subgraph in a loop: the loop counter starts from `start`, steps by `delta`, and does not exceed `limit`. It passes the input tensor to the subgraph and returns its output tensor.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/guarantee_const/README_en.md">guarantee_const</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Guarantees to the runtime that the input tensor is a constant. This operator is a pass-through operator: the output tensor y is the input tensor x, without changing any shape, data type, or value. It only passes the semantic information that "the input is a constant" in the graph for graph optimization decisions. The output tensor y has the same shape and data type as the input tensor x.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/if/README_en.md">if</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Selects the `then_branch` or `else_branch` subgraph according to the boolean condition `cond`, passes the input tensor to the selected subgraph, and returns its output tensor.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/is_variable_initialized/README_en.md">is_variable_initialized</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Checks whether a tensor is initialized. The IsVariableInitialized operator receives the input tensor x and outputs a scalar boolean value y, indicating whether the input tensor is initialized. InferShape sets the output shape to a scalar (0-dimensional), and InferDataType sets the output data type to DT_BOOL.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/loop_cond/README_en.md">loop_cond</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Represents the loop termination condition. The LoopCond operator forwards the boolean scalar input tensor x to the output tensor y, used to pass the while-loop termination condition to the Switch operator. InferShape verifies that the input is a scalar and then copies the shape, and InferDataType verifies that the input is DT_BOOL and then copies the dtype.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/merge/README_en.md">merge</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Forwards the first available tensor in the input x to the output y, and gives the index of the tensor in the input in value_index. It is usually used with Switch to merge branches.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/next_iteration/README_en.md">next_iteration</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Makes the input available in the next iteration. The NextIteration operator passes the input tensor x directly to the output tensor y, used to pass the computation result of the current iteration to the next iteration in a while loop. InferShape copies the input shape to the output, and InferDataType copies the input dtype to the output.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/no_op/README_en.md">no_op</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Performs no operation and is used only as a placeholder for a control edge.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/partitioned_call/README_en.md">partitioned_call</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Passes the input tensor `args` to the subgraph `f` for execution and returns its output tensor.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/placeholder/README_en.md">placeholder</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Inserts a placeholder for a tensor that is always fed. The PlaceHolder operator receives the input tensor x. The shape and data type of the output y are consistent with the input x. InferShape copies the input shape to the output, and InferDataType copies the input dtype to the output.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/placeholder_withdefault/README_en.md">placeholder_withdefault</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Inserts a placeholder with a default value for a tensor. The PlaceholderWithDefault operator receives the input tensor x. The shape and data type of the output y are consistent with the input x. InferShape copies the input shape to the output, and InferDataType copies the input dtype to the output. The shape attribute specifies the shape information of the tensor.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/queue_data/README_en.md">queue_data</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Provides queue data for other operators. The QueueData operator acts as a queue data input node in the graph. It computes the length of the output tensor according to the output_types and output_shapes attributes (each item contains a 64-byte header, dimension description, and data bytes). The output is a one-dimensional tensor of the DT_UINT8 type.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/read_variable_op/README_en.md">read_variable_op</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Reads and returns the value of the input variable tensor. The shape and data type of the output tensor y are consistent with the input tensor x.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/ref_data/README_en.md">ref_data</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Provides input data that can be overwritten by reference for other operators. The RefData operator acts as a data input node in the graph. It directly passes the input tensor x to the output tensor y. The index attribute identifies the sequence number of the data node in the network. Unlike the Data operator, RefData can be overwritten by ref-type operators and acts as a variable.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/ref_enter/README_en.md">ref_enter</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Creates or finds a child frame and makes the input tensor x that can be overwritten by reference available in the child frame. The RefEnter operator is used together with the Exit operator to create a loop in the graph. The executor identifies the frame by the unique frame_name attribute. When is_constant is true, the output tensor y is a constant in the child frame. Unlike the Enter operator, the input of RefEnter can be overwritten by ref-type operators and acts as a variable.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/ref_exit/README_en.md">ref_exit</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Exits the current frame to its parent frame. The RefExit operator is used together with the Enter operator to create a loop in the graph. It passes the input tensor x that can be overwritten by reference to the parent frame to make it available. Unlike the Exit operator, RefExit acts on reference tensors that can be overwritten by ref-type operators. The output tensor y has the same shape and data type as the input tensor x.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/ref_identity/README_en.md">ref_identity</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>Host CPU</td>
    <td>The function is the same as Identity: the output tensor y has the same shape, data type, and content as the input tensor x. It is used for reference (ref) type data.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/ref_merge/README_en.md">ref_merge</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>The function is the same as Merge: it forwards the first available tensor in the input x to the output y, and gives the index of the tensor in the input in value_index. It is used for reference (ref) type data.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/ref_next_iteration/README_en.md">ref_next_iteration</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Makes the input tensor available in the next iteration of the loop. As a loop control flow operator, the RefNextIteration operator passes the input tensor x that can be overwritten by reference to the next iteration. It is used to pass variable state between iterations in a while loop. The output tensor y has the same shape and data type as the input tensor x.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/refswitch/README_en.md">refswitch</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>The function is the same as Switch: it forwards the input tensor data to output_true or output_false according to the boolean scalar input pred. The shape and data type of both outputs are the same as data. It is used for reference (ref) type data.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/switch/README_en.md">switch</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Forwards the input tensor data to the corresponding output port according to the boolean scalar input pred: forwards to output_true when pred is true, otherwise forwards to output_false. The shape and data type of both outputs are the same as data.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/switch_by_index/README_en.md">switch_by_index</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Writes the input data at the specified index to the specified register. The SwitchByIndex operator receives an int32 index tensor x1 (with shape (1,)) and a one-dimensional uint64 tensor x2, and has no output tensor. It is used for runtime label switching. InferShape/InferDataType is registered to return success directly.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/switch_n/README_en.md">switch_n</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Forwards the input tensor data to the corresponding dynamic output port according to the integer tensor input pred_value. The shape and data type of the selected output are consistent with data.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/temporary_variable/README_en.md">temporary_variable</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Creates a temporary variable tensor. After TemporaryVariable is used, pass the reference of the variable tensor to the paired DestroyTemporaryVariable operator for destruction. Derive the shape and data type of the output tensor y according to the shape and dtype attributes: the output shape is consistent with the shape attribute, and the output data type is consistent with the dtype attribute.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/var_handle_op/README_en.md">var_handle_op</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Creates a handle that points to a variable resource. The output tensor y is a scalar resource handle (of type DT_RESOURCE). The shape and dtype attributes describe the shape and data type of the variable pointed to by the handle (not the shape of the output itself), for use by downstream resource operators such as ReadVariableOp and AssignVariableOp. The container and shared_name attributes specify the container and shared name of the variable.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/var_is_initialized_op/README_en.md">var_is_initialized_op</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Checks whether a tensor (variable) has been initialized. The output is a bool scalar that indicates whether the input variable has been initialized. The input must be a Variable node. At compile time, the graph engine queries the initialization state of the Variable through VarIsInitializedOpPass and rewrites this operator node into a constant output.</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/variable/README_en.md">variable</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Creates a variable tensor. The Variable operator assigns a value through input x, records the value of the variable tensor internally, and outputs y as the created variable tensor. The InferShape function returns success but does not explicitly set the output description. The framework derives it by default (output follows input).</td>
  </tr>
  <tr>
    <td>ge</td>
    <td><a href="../../ge/while/README_en.md">while</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>Host CPU</td>
    <td>Executes the `body` subgraph in a loop until the `cond` subgraph returns false. When cond is true, body is executed repeatedly and the loop variable is passed to the next iteration. Otherwise, the current loop variable is returned as the output.</td>
  </tr>
</tbody>
</table>
