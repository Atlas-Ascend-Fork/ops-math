# BallQuery

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

## Function Description

- Operator function: ball query, a classic operator in PointNet++. For each query center point center_xyz[m,b], this operator searches xyz[b] in the order of k=0..N-1 for points that meet the distance criterion, and collects the indices of the first sample_num points.

- Computation formula:

Layout: xyz=(B,3,N), center_xyz=(M,B,3), and idx=(M,B,sample_num). For each (m,b), the center point is $(c_x, c_y, c_z) = center\_xyz[m,b,:]$. The operator computes the squared distance between the center point and each point $(x_k, y_k, z_k) = (xyz[b,0,k], xyz[b,1,k], xyz[b,2,k])$ in xyz[b]:

$$
d2 = (c_x - x_k)^2 + (c_y - y_k)^2 + (c_z - z_k)^2
$$

Distance criterion: $d2 == 0$ or $min\_radius^2 \leq d2 < max\_radius^2$. Hit points are written to idx in the order of k. The index of the first hit point is recorded as first_num. When the number of hit points is less than sample_num, the remaining positions are filled with first_num. If a query ball has 0 hit points, all sample_num output positions of that ball are filled with 0. This result is indistinguishable from hitting point 0 in the output.

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
      <td>xyz</td>
      <td>Input</td>
      <td>xyz coordinates of all points, with shape (B,3,N). The middle dimension must be 3 (coordinates are in the middle dimension).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>center_xyz</td>
      <td>Input</td>
      <td>Coordinates of query center points, with shape (M,B,3). The last dimension must be 3, and dimension 1 (B) must be the same as dimension 0 of xyz.</td>
      <td>Same as xyz</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>idx</td>
      <td>Output</td>
      <td>Indices of sampled points in xyz, with shape (M,B,sample_num).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min_radius</td>
      <td>Attribute</td>
      <td>Minimum query radius (inner boundary of the ring), which must be greater than or equal to 0.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_radius</td>
      <td>Attribute</td>
      <td>Maximum query radius (outer boundary of the ball), which must be greater than min_radius.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sample_num</td>
      <td>Attribute</td>
      <td>Maximum number of sampled points per ball, which must be greater than or equal to 1.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- xyz is a 3D tensor with shape (B,3,N), and the middle dimension must be 3.
- center_xyz is a 3D tensor with shape (M,B,3), and the last dimension must be 3.
- Dimension 0 (B) of xyz must be the same as dimension 1 (B) of center_xyz.
- Dimension 0 (B) and dimension 2 (N) of xyz, and dimension 0 (M (npoint)) of center_xyz must not exceed INT32_MAX (2147483647).
- The dtypes of xyz and center_xyz must be the same and must be FLOAT or FLOAT16.
- The dtype of the output idx is fixed to INT32, and the shape is (M,B,sample_num).
- Zero-length dimensions (empty tensors) are supported as input: when dimension 2 (N) of xyz is 0, each query ball has no candidate points and 0 hit points, and all idx output values are 0; when dimension 0 (M) of center_xyz is 0, the output shape is (0,B,sample_num); when B is 0, the output shape is (M,0,sample_num). All the preceding empty input scenarios are executed and returned normally.
- Dynamic shapes are supported: when building a graph, an input dimension can be marked as -1 (indicating that the dimension value is determined at runtime), and inputs with an unknown rank (shape marked as -2) are supported. Checks such as the coordinate dimension being 3, B consistency, and dimensions not exceeding INT32_MAX are deferred to the runtime tiling phase and performed based on the actual shape. When the rank is unknown, the output shape is inferred as (-1,-1,sample_num).
- min_radius must be a finite value (non-NaN and non-Inf) and must be greater than or equal to 0.
- max_radius must be a finite value (non-NaN and non-Inf), must be greater than 0, and must be greater than min_radius.
- sample_num must be greater than or equal to 1 and must not exceed INT32_MAX (2147483647).
- The input tensor must use a contiguous memory layout (contiguous): that is, data is closely arranged in row-major order in memory, without stride jumps or gaps (for details about the concept, see [Non-contiguous Tensor](../../docs/en/context/non_contiguous_tensor.md)). If a non-contiguous tensor (for example, a view generated by transpose or slicing) is passed in, the framework automatically performs contiguity processing, causing additional copy overhead.
- The operator supports only 3D inputs. Scalars, 1D, and 8D scenarios are not supported.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE graph mode</td>
    <td><a href="./examples/test_geir_ball_query.cpp">test_geir_ball_query</a></td>
    <td>Invokes the BallQuery operator by building a graph through the <a href="./op_graph/ball_query_proto.h">operator IR</a>. For operator compilation and verification, see <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a>.</td>
  </tr>
</tbody>
</table>
