# AdaCast

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: an input preprocessing operator for the HDRnet (High Dynamic Range Network) real-time image enhancement network. It converts the uint16 HDR image data output by the sensor into a float16 tensor and completes normalization based on the white level (pixel), which is equivalent to fusing the two steps of integer-to-float type conversion and scaling by the divisor into a single element-wise operation.
- Computation formula:

$$
out_{i} = \mathrm{Cast\_FP16}^{sat+round}\bigl(\mathrm{Cast\_FP32}(self_{i}) \times (1 / pixel)\bigr)
$$

In the final step, saturation (clipping to the fp16 value range [-65504, 65504]) is performed first, and then the value is rounded and converted to float16 in round-to-nearest-even mode.

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
      <td>The HDR uint16 image data to be normalized, which is self_i in the formula. The shape supports 1D to 4D.</td>
      <td>UINT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The normalized float16 tensor, which is out_i in the formula. Its shape is exactly the same as that of x.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pixel</td>
      <td>Optional attribute</td>
      <td><ul><li>The white level normalization base, and the scaling factor is 1/pixel.</li><li>The default value is 65535.</li><li>The value must be a positive integer (pixel &gt; 0). If the value is not positive, operator compilation reports an error.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- Only the ND data format is supported.
- The rank of the input x ranges from [1, 4] (0-D scalars and rank greater than 4 are both rejected). The shape of the output y must be exactly the same as that of x (element-wise semantics, broadcast is not supported).
- The type combination is fixed to UINT16 → FLOAT16, and other dtype combinations are not supported.
- pixel is an optional attribute. When it is not passed in, the default value 65535 is used. The value must be a positive integer (pixel > 0). If the value is not positive, operator compilation reports an error.
- Empty tensors (with 0 elements) are supported. The host side returns early and does not dispatch the kernel.
- A deterministic implementation is used by default, and the same input always produces the same output.

## Invocation Description

| Invocation Method | Sample Code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| GE graph mode | [test_geir_ada_cast.cpp](./examples/test_geir_ada_cast.cpp) | Invokes the AdaCast operator by building a graph through the [operator IR](./op_graph/ada_cast_proto.h). |

> Note: This operator provides only the GE graph mode API. In `CMakeLists.txt`, `ACLNNTYPE aclnn_exclude` is configured and no `op_api/` directory exists. Therefore, the ACLNN and PyTorch/TensorFlow APIs are not provided.
