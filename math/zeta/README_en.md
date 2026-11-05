# Zeta

English | [简体中文](./README.md)

## Product Support Status

| Product                                        | Supported |
| :------------------------------------------ | :------: |
| Ascend 950PR&950DT products                   |    √     |
| Atlas A3 products |    √     |
| Atlas A2 products |    √     |
| Atlas 200I/500 A2 inference products                  |    √     |
| Atlas inference products                          |    √     |
| Atlas training products                          |    √     |

## Function Description

- Operator function: computes the Hurwitz zeta function, that is, ζ(x, q) = Σ(n=0 to ∞) 1/(q+n)^x.

- Computation formula:

$$z = \zeta(x, q) = \sum_{n=0}^{\infty} \frac{1}{(q+n)^x}$$

## Parameter Description

<table style="table-layout: fixed; width: 1005px"><colgroup>
<col style="width: 140px">
<col style="width: 140px">
<col style="width: 180px">
<col style="width: 213px">
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
    <td>x</td>
    <td>Input</td>
    <td>Input tensor, which is the exponent parameter of the Hurwitz zeta function.</td>
    <td>FLOAT, DOUBLE</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>q</td>
    <td>Input</td>
    <td>Input tensor, which is the shift parameter of the Hurwitz zeta function. It must have the same data type as x.</td>
    <td>FLOAT, DOUBLE</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>z</td>
    <td>Output</td>
    <td>Output tensor, which is the computation result of the Hurwitz zeta function and has the same type and shape as x.</td>
    <td>FLOAT, DOUBLE</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraint Description

- The inputs x and q must have the same data type.
- The inputs x and q must have the same data size (broadcast is not supported).

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph-mode invocation | [test_geir_zeta](examples/test_geir_zeta.cpp) | Invokes the Zeta operator through the [operator IR](op_graph/zeta_proto.h) API. |
