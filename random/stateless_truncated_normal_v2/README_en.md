# StatelessTruncatedNormalV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR&950DT products                         |    √     |
| Atlas A3 products     |    ×     |
| Atlas A2 products |    ×     |
| Atlas 200I/500 A2 inference products                      |    ×     |
| Atlas inference products                              |    ×     |
| Atlas training products                              |    ×     |

## Function Description

- Operator function: stateless truncated normal distribution random number generator. It uses the Philox4x32 algorithm and takes key and counter as inputs to generate uint32 random numbers that follow the uniform distribution. It converts the uniform distribution into the standard normal distribution through the Box-Muller transform, generating random numbers that follow the standard normal distribution N(0,1). Samples whose absolute values are greater than 2.0 are truncated, discarded, and resampled.
- Stateless feature: random number generation is fully determined by the externally passed key/counter. The same input is guaranteed to produce the same output (deterministic).
- Benchmark: TensorFlow `tf.raw_ops.StatelessTruncatedNormalV2`

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
      <td>shape</td>
      <td>Input</td>
      <td>Shape of the output tensor.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>Key of the Philox random number generator, shape=[1].</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counter</td>
      <td>Input</td>
      <td>Counter of the Philox random number generator, shape=[2].</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alg</td>
      <td>Input</td>
      <td>Random number algorithm ID (1=Philox; currently only Philox is supported).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Specifies the data type of the output (0=float32, 1=float16, 27=bfloat16 ,11=float64).</td>
      <td>INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Generated truncated normal distribution random number sequence.</td>
      <td>FLOAT16, FLOAT, BF16, FLOAT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The alg parameter currently supports only the Philox algorithm (alg=1).
- The shape of key must be [1].
- The shape of counter must be [2] (required by the Philox algorithm).
- When the number of elements in the output shape is 0, an empty tensor is directly returned.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph-mode invocation | [test_geir_stateless_truncated_normal_v2](./examples/test_geir_stateless_truncated_normal_v2.cpp) | Invokes the StatelessTruncatedNormalV2 operator by building a graph through the [operator IR](./op_graph/stateless_truncated_normal_v2_proto.h). |
