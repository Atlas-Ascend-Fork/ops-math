# ProdEnvMatACalcRij

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes neighbors for the DeepMD environment matrix. Based on the neighbor table given by mesh, finds the neighbors within the radial cutoff radius for each central atom in coord, groups them by atom type, sorts them by distance from near to far, and takes the first several ones. It outputs the neighbor indices nlist, the squared distances distance, the displacement vectors rij from the central atom to neighbors, and the three components rij_x, rij_y, and rij_z of rij. box does not participate in the computation currently.

- Computation formula: Let the coordinates of the central atom $i$ be $\mathbf{r}_i=(x_i,y_i,z_i)$ and the coordinates of the candidate neighbor $j$ be $\mathbf{r}_j=(x_j,y_j,z_j)$. The displacement and squared distance are:

  $$
  \Delta\mathbf{r}_{ij}=\mathbf{r}_j-\mathbf{r}_i,\quad
  d_{ij}=\lVert\mathbf{r}_j-\mathbf{r}_i\rVert^2=(x_j-x_i)^2+(y_j-y_i)^2+(z_j-z_i)^2
  $$

  When $d_{ij}<\mathrm{rcut\_r}^2$, $j$ is regarded as a neighbor of $i$. Neighbors of each atom type are sorted in ascending order of $d_{ij}$, and at most $\mathrm{sel\_a}[t]$ neighbors are taken for type $t$ and written to the outputs in order: $\mathrm{nlist}$ stores the neighbor indices, $\mathrm{distance}$ stores $d_{ij}$, $\mathrm{rij}$ is arranged alternately in the order of $(\Delta x,\Delta y,\Delta z)$, and $\mathrm{rij\_x}$, $\mathrm{rij\_y}$, and $\mathrm{rij\_z}$ store $\Delta x$, $\Delta y$, and $\Delta z$ respectively. Unfilled neighbor slots are set to -1 in nlist, $\mathrm{rcut\_r}^2+1$ in distance, and 0 in the displacements.

- Let nsample be the number of samples (dimension 0 of coord), nloc the number of central atoms (the first element of natoms), and nnei the total number of neighbor slots of a single central atom (equal to the sum of elements of sel_a). Then each sample produces nloc×nnei neighbor slots. The correspondence between parameters and formula variables: coord corresponds to $\mathbf{r}$, rcut_r corresponds to the cutoff radius, and sel_a corresponds to the maximum number of neighbors selected for each type.

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 160px">
  <col style="width: 150px">
  <col style="width: 420px">
  <col style="width: 190px">
  <col style="width: 100px">
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
      <td>Atom coordinates, 2-dimension, with the shape `[nsample, nall * 3]`. The coordinates of each atom are arranged consecutively in the order of x, y, and z.</td>
      <td>FLOAT, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>type</td>
      <td>Input</td>
      <td>Atom types, 2-dimension, with the shape `[nsample, nall]`. The value range is `[0, number of sel_a elements)`.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>natoms</td>
      <td>Input</td>
      <td>Atom count information, 1-dimension. The number of elements is not less than 3. The first element is the number of central atoms nloc, and the second element is the total number of atoms nall.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>box</td>
      <td>Input</td>
      <td>Box matrix. This input is not used by the current operator. The data type must be the same as that of coord.</td>
      <td>FLOAT, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mesh</td>
      <td>Input</td>
      <td>Neighbor list, 1-dimension. It sequentially stores the number of central atoms, ilist, numneigh, and the neighbor index table of each central atom.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rcut_a</td>
      <td>Optional Attribute</td>
      <td>Cutoff radius of the angular information. The default value is 1.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rcut_r</td>
      <td>Optional Attribute</td>
      <td>Cutoff radius of the radial information. Only atoms whose squared distance is smaller than its square are regarded as neighbors. The default value is 1.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rcut_r_smth</td>
      <td>Optional Attribute</td>
      <td>Smoothing cutoff radius of the radial information. The default value is 1.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sel_a</td>
      <td>Optional Attribute</td>
      <td>Maximum number of neighbors selected for each atom type in the angular information. The sum of its elements is the total number of neighbor slots nnei of a single central atom. The default value is empty.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sel_r</td>
      <td>Optional Attribute</td>
      <td>Maximum number of neighbors selected for each atom type in the radial information. The default value is empty.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rij</td>
      <td>Output</td>
      <td>Displacement vectors from the central atom to neighbors, 2-dimension, with the shape `[nsample, nloc * nnei * 3]`. The values of each neighbor are arranged consecutively in the order of Δx, Δy, and Δz. The data type is the same as that of coord.</td>
      <td>FLOAT, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nlist</td>
      <td>Output</td>
      <td>Neighbor indices, 2-dimension, with the shape `[nsample, nloc * nnei]`. Unfilled slots are -1.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>distance</td>
      <td>Output</td>
      <td>Squared distances between the central atom and neighbors, 2-dimension, with the shape `[nsample, nloc * nnei]`. The data type is the same as that of coord.</td>
      <td>FLOAT, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rij_x</td>
      <td>Output</td>
      <td>x component of the displacement vectors, 2-dimension, with the shape `[nsample, nloc * nnei]`. The data type is the same as that of coord.</td>
      <td>FLOAT, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rij_y</td>
      <td>Output</td>
      <td>y component of the displacement vectors, 2-dimension, with the shape `[nsample, nloc * nnei]`. The data type is the same as that of coord.</td>
      <td>FLOAT, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rij_z</td>
      <td>Output</td>
      <td>z component of the displacement vectors, 2-dimension, with the shape `[nsample, nloc * nnei]`. The data type is the same as that of coord.</td>
      <td>FLOAT, DOUBLE</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The number of samples (dimension 0) of coord and type must be the same. Dimension 1 of coord must equal nall×3, and dimension 1 of type must equal nall.
- When neighbors with equal squared distances exist, they are sorted by atom index in ascending order.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_prod_env_mat_a_calc_rij](./examples/test_geir_prod_env_mat_a_calc_rij.cpp)   | Invokes the ProdEnvMatACalcRij operator by building a graph through the [operator IR](./op_graph/prod_env_mat_a_calc_rij_proto.h). |
