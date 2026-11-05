# AddMatMatElements

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: multiplies the input tensors `a` and `b` element-wise, and then computes the weighted sum of the product and the input tensor `c` by using `alpha` and `beta`. The result is written to the output tensor.
- Computation formula:

  $$
  out_i = c_i \times \beta + \alpha \times a_i \times b_i
  $$

- The output shape is fixed to `c.shape`. `a` and `b` are broadcast to `c.shape` respectively according to the tail-alignment rule. Expanding the shape of `c` through unified broadcasting is not supported.
- `alpha` and `beta` are tensors with the shape of `(1,)`, not attributes.

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>Parameter Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>c</td>
    <td>Input</td>
    <td>The anchor of the output shape. It participates in the accumulation after being scaled by beta. The rank ranges from 1 to 8.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>a</td>
    <td>Input</td>
    <td>Participates in the element-wise multiplication. The rank ranges from 0 to the rank of c, and the shape must be able to be broadcast to c.shape. A rank of 0 indicates a scalar tensor.</td>
    <td>Same as c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>b</td>
    <td>Input</td>
    <td>Participates in the element-wise multiplication. The rank ranges from 0 to the rank of c, and the shape must be able to be broadcast to c.shape. A rank of 0 indicates a scalar tensor.</td>
    <td>Same as c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>Input</td>
    <td>The coefficient tensor used to scale c, whose shape must be (1,).</td>
    <td>Same as c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>Input</td>
    <td>The coefficient tensor used to scale the product of a × b, whose shape must be (1,).</td>
    <td>Same as c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>c</td>
    <td>Output</td>
    <td>The computation result, whose shape equals the input c.shape.</td>
    <td>Same as c</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraint Description

- `c`, `a`, `b`, `beta`, `alpha`, and the output must have exactly the same data type. Only `FLOAT16` and `FLOAT` are supported, and data type promotion is not supported.
- The rank of `c` ranges from 1 to 8. The ranks of `a` and `b` range from 0 to the rank of `c`. When the rank is 0, the tensor is broadcast as a scalar tensor.
- `a` and `b` are tail-aligned to `c.shape` respectively. Each padded dimension must be 1 or equal to the corresponding dimension of `c`.
- The output shape is fixed to `c.shape`. For example, `c.shape=(1, 3)` and `a.shape=(2, 3)` are not supported, even if the three inputs can obtain the unified broadcast shape `(2, 3)`.
- The shapes of `alpha` and `beta` must be strictly `(1,)`.
- Empty tensors are supported. The empty tensor scenario must still satisfy the preceding rank and broadcast constraints.
- Dynamic shape and dynamic rank are supported. During shape inference, `-1` indicates an unknown dimension: if a specific dimension exists on the same tail-aligned axis, the specific dimension is used; if all dimensions are `-1`, `-1` is retained. If any input of `c`, `a`, or `b` is of unknown rank `[-2]`, the output is of unknown rank `[-2]`. The actual shape at runtime must satisfy the preceding constraints.
- The Kernel input and output support only the ND format. The GE graph can insert format conversion for some non-ND data tensors or the requested output layout, and converts them to ND before entering the Kernel. The format conversion capability depends on the rank and shape of the tensor, and it is not guaranteed that any format combination can be converted. `alpha` and `beta` must still be `(1,)` tensors in the ND format.

> Description: The current InferShape implementation performs shape inference based on `broadcast(a, b, c)`, while Host Tiling uses `c.shape` as the output anchor and checks the broadcast relationship from `a` and `b` to `c.shape`. The actual execution constraints are subject to Host Tiling.

### Exception Validation

| Violated Condition | Validation Stage | Error Behavior |
| :------- | :------- | :------- |
| A required input or output is missing | Graph building/compilation parameter validation | `E80001` |
| The rank of `c` is not within the range of 1 to 8 | Graph compilation parameter validation may reject it early; Host Tiling retains the full range validation | When the rank is greater than 8, the current integration path presents `E80012`. If it enters Host Tiling, `GRAPH_FAILED` is returned, the current integration path presents `EZ9999`, and a rank out-of-range log is recorded |
| The input dtype is not supported or the input dtypes are inconsistent | Host Tiling | `GRAPH_FAILED` is returned. The current integration path presents `EZ9999` externally and records a dtype validation log |
| The rank of `a` or `b` is greater than the rank of `c`, or they cannot be broadcast to `c.shape` | Host Tiling | `GRAPH_FAILED` is returned. The current integration path presents `EZ9999` externally and records a rank or broadcast validation log |
| The shape of `alpha` or `beta` is not `(1,)` | Host Tiling | `GRAPH_FAILED` is returned. The current integration path presents `EZ9999` externally and records a shape validation log |
| The graph input is declared as a non-ND format | GE format selection/TransData | Convertible data tensors are converted to ND by GE before execution. Non-convertible rank/shape/format combinations fail at the graph compilation stage and do not enter the Kernel of this operator |
| The output dtype or shape prefilled by the caller is inconsistent with the input | Graph inference | The output dtype follows the input `c`, and the output shape is inferred from the input shape. The values prefilled by the caller are overwritten by the inference results |
| The graph requests a non-ND output layout | GE format selection/TransData | GE can perform layout conversion after the ND output of the Kernel. The physical output buffer cannot be directly interpreted as a raw ND array |

> Description: Error codes are encapsulated by graph compilation or the runtime framework for validation failures. Different integration paths may return only `GRAPH_FAILED` and retain specific logs. The direct Kernel test path is not responsible for validating output descriptors and non-ND formats. Such scenarios should be validated at the graph building boundary, and the fact that the direct Kernel path does not reject them cannot be used to determine operator support or defects.

## Invocation Description

| Invocation Method | Sample Code | Description |
| :------- | :------- | :--- |
| Graph mode | [test_geir_add_mat_mat_elements.cpp](examples/test_geir_add_mat_mat_elements.cpp) | Invokes the AddMatMatElements operator by building a graph through the [operator IR](op_graph/add_mat_mat_elements_proto.h). |

### Graph-Mode Explicit Cast Description

- Connecting the output of an explicit Cast operator to AddMatMatElements is supported. When entering AddMatMatElements, `c`, `a`, `b`, `beta`, and `alpha` must have been converted to exactly the same `FLOAT16` or `FLOAT` data type, and the shapes of `beta` and `alpha` must still be `(1,)`.
- AddMatMatElements does not automatically insert Cast for mixed data type inputs, and implicit data type promotion is not supported. If the input data types are still inconsistent after explicit Cast, or the Cast target data type is not within the range of `FLOAT16` and `FLOAT`, graph compilation fails.
- Fusion of Cast and AddMatMatElements is not guaranteed, and Cast can be executed as an independent graph node and an independent kernel.
