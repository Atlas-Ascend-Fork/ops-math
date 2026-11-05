# MatrixDiagPart

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function Description

- Operator function: returns the main diagonal (k=0) of the innermost matrix in the input matrices.

## Parameter Description

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
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
      <td>Input parameter for computing the main diagonal. The last two dimensions are matrix dimensions.</td>
      <td>float16, float32, int32, int8, uint8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Main diagonal result, which has the same dtype as x.</td>
      <td>float16, float32, int32, int8, uint8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The number of dimensions of input x must be greater than or equal to 2 and less than or equal to 8. The last two dimensions represent matrices.
- The number of dimensions of output y is 1 less than that of input x, and the shape is x.shape[:-2]+[min(M, N)].
- Supported data types: float16, float32, int32, int8, uint8.
- Only the ND data format is supported.
- Only contiguous inputs are supported. Non-contiguous inputs are not supported.
- Empty tensors are supported (when M=0 or N=0, the output is an empty tensor).
- This operator is the MatrixDiagPart V1 version and extracts only the main diagonal (k=0).

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---- | ---- | ---- |
| Graph mode | [test_geir_matrix_diag_part](examples/test_geir_matrix_diag_part.cpp) | Invokes the MatrixDiagPart operator by building a graph through the [operator IR](op_graph/matrix_diag_part_proto.h). For compilation and verification, refer to [Operator Invocation](../../docs/zh/invocation/quick_op_invocation.md). |
