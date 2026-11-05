# ProdForceSeA

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: ProdForceSeA (Product Force for Smooth Edition Angular) is the backward force computation operator of the DeepPot-SE model in DeepMD-kit. It multiplies the derivatives of the neural network with respect to the descriptor (net_deriv) and the derivatives of the descriptor with respect to the coordinates (in_deriv) element-wise, then performs reduction accumulation to obtain the 3D force vector of each atom.

- Computation formula:

Force of the central atom (negative sign):

$$
force[k][i][d] = -\sum_{a=0}^{ndescrpt-1} net\_deriv[k][i][a] \times in\_deriv[k][i][a][d]
$$

Force of the neighbor atom (positive sign, accumulated by the nlist index):

$$
force[k][j][d] += \sum_{a=aa\_start}^{aa\_end-1} net\_deriv[k][i][a] \times in\_deriv[k][i][a][d]
$$

where $ndescrpt = 4 \times nnei$, $nnei = n\_a\_sel + n\_r\_sel$, $j = nlist[k][i][j]$, and $[aa\_start, aa\_end) = [j \times 4, j \times 4 + 4)$.

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
      <td>net_deriv</td>
      <td>Input</td>
      <td>Derivatives of the network with respect to the descriptor, shape=(nframes, nloc*nnei*4).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>in_deriv</td>
      <td>Input</td>
      <td>Derivatives of the descriptor with respect to the coordinates, shape=(nframes, nloc*nnei*4*3).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nlist</td>
      <td>Input</td>
      <td>Neighbor list, shape=(nframes, nloc*nnei). -1 indicates a virtual neighbor.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>natoms</td>
      <td>Input</td>
      <td>Atom count information, shape=(ntypes+2,). natoms[0]=nloc, natoms[1]=nall.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>n_a_sel</td>
      <td>Attribute</td>
      <td>Number of selected angular neighbors. The value range is [0, +∞), required.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>n_r_sel</td>
      <td>Attribute</td>
      <td>Number of selected radial neighbors. The value range is [0, +∞), required. n_a_sel + n_r_sel > 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>atom_force</td>
      <td>Output</td>
      <td>Forces on atoms, shape=(nframes, nall, 3).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- Only the float32 data type is supported.
- The complete set of dtypes declared by the REG_OP in the IR prototype [op_graph/prod_force_se_a_proto.h](./op_graph/prod_force_se_a_proto.h) is FLOAT16, FLOAT, and FLOAT64 (this is the registration scope on the graph side, which is wider than the delivered implementation). The current operator implementation and OpDef deliver only FLOAT. The preceding FLOAT16 and FLOAT64 have no corresponding kernels. If a graph is built in graph mode with either of these two dtypes, no available implementation exists and the graph is rejected.
- The dtypes of net_deriv and in_deriv must be the same.
- In nlist, an element value of -1 indicates a virtual neighbor, which is skipped.
- n_a_sel>=0 and n_r_sel>=0, and n_a_sel+n_r_sel>0.
- The number of elements of natoms is >= 3, natoms[0]=nloc>=0, and natoms[1]=nall>=nloc.
- natoms is a value-dependent input. Dimension 1 (nall) of the output atom_force shape is determined by the runtime value of natoms[1].
- Neighbor force accumulation uses intra-frame serial read-modify-write without atomic operations, and the output is deterministic.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_prod_force_se_a](./examples/arch35/test_geir_prod_force_se_a.cpp)   | Invokes the ProdForceSeA operator by building a graph through the [operator IR](./op_graph/prod_force_se_a_proto.h). |
