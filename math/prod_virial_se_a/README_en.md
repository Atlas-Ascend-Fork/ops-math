# ProdVirialSeA

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

- Operator function: ProdVirialSeA is the virial tensor computation operator in the deep potential model of DeepMD-kit. It computes the type A (angular) virial tensor based on the SE (Smooth Edition) atomic embedding descriptor. The operator receives the network gradient, internal derivatives, relative positions, and the neighbor list, and obtains the total virial tensor of the system and the atomic-level virial tensor through triple tensor contraction accumulation. It uses a deterministic implementation: inter-frame multi-core parallelism, idz 9-thread parallelism, and serial accumulation, without atomic operations.

- Computation formula:

$$
\text{virial}[\text{dd0}][\text{dd1}] = \sum_{i} \sum_{j} \sum_{\mu} \text{net\_deriv}[i][j][\mu] \times \text{rij}[i][j][\text{dd1}] \times \text{in\_deriv}[i][j][\mu][\text{dd0}]
$$

$$
\text{atom\_virial}[\text{j\_idx}][\text{dd0}][\text{dd1}] \mathrel{+}= \text{tmp}
$$

where `nnei = n_a_sel + n_r_sel`, `ndescrpt = nnei * 4`, and `tmp` is the intermediate result of the preceding triple contraction, which is scatter-accumulated to the virial components of the corresponding atom by the neighbor index `j_idx`. `j_idx` is taken from `nlist`, and negative or out-of-range values are skipped.

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
      <td>Network gradient tensor, with the shape (nframes, nloc * ndescrpt), which is net_deriv in the formula.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>in_deriv</td>
      <td>Input</td>
      <td>Internal derivative tensor, with the shape (nframes, nloc * ndescrpt * 3), which is in_deriv in the formula.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rij</td>
      <td>Input</td>
      <td>Neighbor relative position tensor, with the shape (nframes, nloc * nnei * 3), which is rij in the formula.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nlist</td>
      <td>Input</td>
      <td>Neighbor list tensor, with the shape (nframes, nloc * nnei). It stores the global indices of the neighbors of each local atom. A negative or out-of-range value indicates an invalid neighbor.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>natoms</td>
      <td>Input</td>
      <td>Atom count tensor, with the shape (2 + ntypes,). natoms[0]=nloc (number of local atoms), and natoms[1]=nall (total number of atoms, including ghost atoms). It is used to deduce the output shape of atom_virial and is value-dependent.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>virial</td>
      <td>Output</td>
      <td>Total virial tensor of the system, with the shape (nframes, 9). The 3x3 matrix is flattened into 9 elements.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>atom_virial</td>
      <td>Output</td>
      <td>Atomic-level virial tensor, with the shape (nframes, nall * 9). Each atom has 9 elements.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>n_a_sel</td>
      <td>Attribute</td>
      <td>Number of selected type A (angular) neighbors. It determines nnei = n_a_sel + n_r_sel together with n_r_sel.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>n_r_sel</td>
      <td>Attribute</td>
      <td>Number of selected type R (radial) neighbors. It determines nnei = n_a_sel + n_r_sel together with n_a_sel.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- Deterministic computation: deterministic implementation by default, with inter-frame multi-core parallelism + idz 9-thread parallelism + serial (i,j,μ) accumulation, without atomic operations. Results of multiple runs are bitwise identical.
- The dtypes of the three floating-point inputs net_deriv, in_deriv, and rij must be the same.
- Dimension 0 (nframes) of net_deriv, in_deriv, rij, and nlist must be the same.
- natoms is a value-dependent input and must be a const tensor (Const operator). Otherwise, shape[1] of atom_virial is set to -1 (dynamic shape).
- The shape size of natoms must be greater than or equal to 3, that is, it contains at least three elements [nloc, nall, ...].
- nall (natoms[1]) must be greater than 0.
- The shape of virial must be [nframes, 9].
- shape[1] of atom_virial must be a multiple of 9.
- n_a_sel and n_r_sel must be greater than or equal to 0, and n_a_sel + n_r_sel must be greater than 0. In addition, `nloc * (n_a_sel + n_r_sel) * 4 == net_deriv.shape[1]` must be satisfied.
- If a neighbor index in nlist is negative or greater than or equal to nall, the neighbor is skipped (not included in the accumulation).
- Inputs support only 2 dimensions (ND format).
- The operator prototype declares support for the three data types FLOAT, FLOAT16, and DOUBLE. However, this operator supports only FLOAT and FLOAT16 on Ascend 950 and does not support DOUBLE (the AI Core computing units of the Ascend 950 hardware do not provide float64 computation instructions).
- This operator is supported only on the Ascend 950 series products.

## Invocation Description

| Invocation Method   | Sample Code | Description  |
| ------------ | ------------ | ------------ |
| Graph-mode invocation | [test_geir_prod_virial_se_a](./examples/test_geir_prod_virial_se_a.cpp) | Invokes the ProdVirialSeA operator by building a graph through the [operator IR](./op_graph/prod_virial_se_a_proto.h) |
