# CdistGrad

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

- Operator function: performs the backpropagation of Cdist.
- Computation formula:

  $$
  \begin{aligned}
  out&=grad \cdot y' \\
  &= grad \cdot \left( \sqrt[p]{\sum (x_1 - x_2)^p} \right)' \\
  &= grad \cdot \frac{1}{p} \times \left( \sum (x_1 - x_2)^p \right)^{\frac{1}{p}-1} \times p \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \left( \sum (x_1 - x_2)^p \right)^{\frac{-(p-1)}{p}} \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \left( \sum (x_1 - x_2)^p \right)^{\frac{1}{p} \times (-(p-1))} \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \frac{diff^{p-1}}{cdist^{p-1}} \\
  &= grad \cdot \frac{diff \times |diff|^{p-2}}{cdist^{p-1}}
  \end{aligned}
  $$

  - $\mathrm{diff} = x_1 - x_2$: variable difference
  - $\mathrm{cdist} = \sqrt[p]{\sum (x_1 - x_2)^p}$: $p$-norm distance

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
      <td>grad</td>
      <td>Input</td>
      <td>Input grad in the formula.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>Input x1 in the formula.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input x2 in the formula.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cdist</td>
      <td>Input</td>
      <td>Input cdist in the formula.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>Attribute</td>
      <td>Input p in the formula.</td>
      <td>float</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output out in the formula.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                                                         |
| ---------------- | --------------------------- |----------------------------------------------------------------------------|
| aclnn API  | [test_aclnn_cdist_backward](examples/test_aclnn_cdist_backward.cpp) | Invokes the CdistGrad operator through the [aclnnCdistBackward](docs/aclnnCdistBackward_en.md) API. |
