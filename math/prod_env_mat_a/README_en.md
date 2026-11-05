# ProdEnvMatA

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: prod_env_mat_a is the environment matrix computation operator of DeepMD-kit. It computes the SeA (Smoothed Angular) environment matrix descriptor of atoms. The computation consists of two major phases. The first phase is neighbor list formatting: computes interatomic distances, sorts them by type, and fills the neighbor list nlist in segments based on sel_a. The second phase is environment matrix computation: uses the spline5_switch quintic spline smoothing function to compute the 4-component descriptor and 12-component derivatives, and normalizes them with davg/dstd.

- Computation formula:

For each central atom $i$ and its neighbor $j$, the relative distance $r_{ij}=\|\mathbf{r}_j-\mathbf{r}_i\|$ is computed. If $r_{ij} \leq r_{cut\_r}$, the original 4-component descriptor is computed through the quintic spline smoothing function $s=\text{spline5\_switch}(r_{ij}, r_{cut\_r\_smth}, r_{cut\_r})$:

$$
\mathbf{d}_{ij} = s \cdot \left(\frac{1}{r_{ij}},\ \frac{r_{ij,x}}{r_{ij}^2},\ \frac{r_{ij,y}}{r_{ij}^2},\ \frac{r_{ij,z}}{r_{ij}^2}\right)
$$

The normalized descriptor and derivatives are:

$$
\text{descrpt}_{ij} = \frac{\mathbf{d}_{ij} - \text{davg}_{t_i}}{\text{dstd}_{t_i}}, \quad \text{descrpt\_deriv}_{ij} = \frac{\mathbf{v}_{ij}}{\text{dstd}_{t_i}}
$$

where $t_i$ is the type of the central atom $i$, $\text{davg}_{t_i}$ and $\text{dstd}_{t_i}$ are the mean and standard deviation corresponding to that type, and $\mathbf{v}_{ij}$ is the derivative of the descriptor with respect to the coordinates of the central atom. If $r_{ij} > r_{cut\_r}$, $s=0$, and the descriptor is $-\text{davg}/\text{dstd}$.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
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
      <td>coord</td>
      <td>Input</td>
      <td>3D coordinates of all atoms. The shape is (nsample, nall*3), and the dimension must be 2.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>type</td>
      <td>Input</td>
      <td>Type indices of all atoms. The shape is (nsample, nall), the dimension must be 2, and shape[0] is the same as that of coord.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>natoms</td>
      <td>Input</td>
      <td>Atom count information. The shape is (2+ntypes,), the dimension must be 1, and shape[0] ≥ 3. The value is [nloc, nall, ntype_0_count, ...]. In the infershape phase, natoms[0] is read to obtain nloc.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>box</td>
      <td>Input</td>
      <td>Periodic boundary box (3×3 flattened). The shape is (nsample, 9).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mesh</td>
      <td>Input</td>
      <td>Mesh information (used for replicating neighbors). The shape is (nsample, 6).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>davg</td>
      <td>Input</td>
      <td>Descriptor mean (used for normalization). The shape is (ntypes, nnei*4), and the dimension must be 2.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dstd</td>
      <td>Input</td>
      <td>Descriptor standard deviation (used for normalization). The shape is (ntypes, nnei*4), the dimension must be 2, the dtype is the same as that of coord/davg, and elements cannot be 0.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>descrpt</td>
      <td>Output</td>
      <td>Normalized environment matrix descriptor. The shape is (nsample, nloc*nnei*4).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>descrpt_deriv</td>
      <td>Output</td>
      <td>Descriptor derivatives. The shape is (nsample, nloc*nnei*12).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rij</td>
      <td>Output</td>
      <td>Relative coordinates from the central atom to neighbors. The shape is (nsample, nloc*nnei*3).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nlist</td>
      <td>Output</td>
      <td>Formatted neighbor list indices. The shape is (nsample, nloc*nnei), and the element value is ≥ -1 (-1 indicates no neighbor).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rcut_a</td>
      <td>Attribute</td>
      <td>Angular cutoff radius (currently unused, retained for compatibility), ≥ 0.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rcut_r</td>
      <td>Attribute</td>
      <td>Radial cutoff radius, > 0 and ≥ rcut_r_smth.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rcut_r_smth</td>
      <td>Attribute</td>
      <td>Smoothing start radius, > 0 and ≤ rcut_r.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sel_a</td>
      <td>Attribute</td>
      <td>Number of angular neighbors selected for each type. It is a non-empty list, and elements are ≥ 0. nnei=sum(sel_a).</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sel_r</td>
      <td>Attribute</td>
      <td>Number of radial neighbors selected for each type (not used by this operator). Elements are ≥ 0.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- coord is a 2-dimension tensor. The shape is (nsample, nall*3), and the dtype is float32.
- type is a 2-dimension tensor. The shape is (nsample, nall), the dtype is int32, shape[0] is the same as coord.shape[0], and the element value is ≥ 0 (virtual atoms with type < 0 are skipped).
- natoms is a 1-dimension tensor. The shape is (2+ntypes,), the dtype is int32, and shape[0] ≥ 3. natoms[0]=nloc (number of local atoms), natoms[1]=nall (total number of atoms), and the remaining elements are atom counts of each type.
- box is a 2-dimension tensor. The shape is (nsample, 9), and the dtype is float32.
- mesh is a 2-dimension tensor. The shape is (nsample, 6), and the dtype is int32.
- davg is a 2-dimension tensor. The shape is (ntypes, nnei*4), the dtype is float32, and the dtype is the same as that of coord/dstd.
- dstd is a 2-dimension tensor. The shape is (ntypes, nnei*4), the dtype is float32, the dtype is the same as that of coord/davg, and elements cannot be 0 (division denominator).
- rcut_a must be ≥ 0.
- rcut_r must be > 0 and ≥ rcut_r_smth.
- rcut_r_smth must be > 0 and ≤ rcut_r.
- sel_a must be a non-empty list, and all elements are ≥ 0. nnei=sum(sel_a) determines the second dimension of the output shape.
- Elements of sel_r must be ≥ 0 (sel_r is not used by this operator and is retained for compatibility).
- natoms[0] (nloc) determines the second dimension of the output shape. In the infershape phase, the natoms input is read through value dependency.
- Output element values of nlist are ≥ -1, and -1 indicates no neighbor at the position (positions that are not filled are padded with -1).
- Input tensors must have a contiguous memory layout (contiguous): that is, data is densely arranged in row-major order in memory, without stride hops or gaps (for the concept, see [Non-contiguous Tensor](../../docs/en/context/non_contiguous_tensor.md)). If a non-contiguous tensor (such as a view obtained by transpose or slicing) is passed in, the framework automatically makes it contiguous, which incurs additional copy overhead.
- The operator supports only 2D inputs. Scalars, 1D, or 8D scenarios are not supported.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/test_geir_prod_env_mat_a.cpp">test_geir_prod_env_mat_a</a></td>
    <td>For operator compilation and verification, refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a>.</td>
  </tr>
</tbody>
</table>
