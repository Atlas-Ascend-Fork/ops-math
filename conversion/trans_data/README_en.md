# TransData

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    √     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |

## Function Description

- Operator function: performs format conversion for various data formats.

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
      <td>src</td>
      <td>Input</td>
      <td>Tensor to be converted.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT32, INT8, BOOL, INT16, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src_format</td>
      <td>Attribute</td>
      <td>String of the source data format.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_format</td>
      <td>Attribute</td>
      <td>String of the destination data format.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>src_subformat</td>
      <td>Attribute</td>
      <td>Source sub data format.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_subformat</td>
      <td>Attribute</td>
      <td>Destination sub data format.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groups</td>
      <td>Attribute</td>
      <td>Name of the groups column in the Constraint Description.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst</td>
      <td>Output</td>
      <td>Output result.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT32, INT8, BOOL, INT16, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

* For branches with padding: INT16, INT64, UINT8, UINT16, UINT32, and UINT64 are not supported.

## Constraint Description

* The following value ranges must be met:
'<====>' indicates that the format is supported in both directions, whether as input or output.
'====>' indicates that this format conversion is not supported, and the input and output data types must correspond to each other.
[Introduction to Data Formats](../../docs/en/context/data_format.md)

  | src_format <====> dst_format | dtype                              | C0    | groups |
  | :-------------------------: | :--------------------------------: |:-----:| :----: |
  | NCHW <====> NC1HWC0         | float32, int32,uint32              | 8,16  | 1      |
  | NCHW <====> NC1HWC0         | bfloat16, float16                  | 16    | 1      |
  | NCHW <====> NC1HWC0         | int8, uint8                        | 32    | 1      |
  | NHWC <====> NC1HWC0         | float32, int32,uint32              | 8,16  | 1      |
  | NHWC <====> NC1HWC0         | bfloat16, float16                  | 16    | 1      |
  | NHWC <====> NC1HWC0         | int8,  uint8                       | 32    | 1      |
  | ND <====> FRACTAL_NZ        | float32, int32,uint32              | 16    | 1      |
  | ND <====> FRACTAL_NZ        | bfloat16, float16                  | 16    | 1      |
  | ND <====> FRACTAL_NZ        | int8, uint8                        | 32    | 1      |
  | NCHW <====> FRACTAL_Z       | float32, int32,uint32              | 8,16  | 1      |
  | NCHW <====> FRACTAL_Z       | bfloat16, float16                  | 16    | 1      |
  | NCHW <====> FRACTAL_Z       | int8,  uint8                       | 32    | 1      |
  | HWCN <====> FRACTAL_Z       | float32, int32,uint32              | 8,16  | 1      |
  | HWCN <====> FRACTAL_Z       | bfloat16, float16                  | 16    | 1      |
  | HWCN <====> FRACTAL_Z       | int8, uint8                        | 32    | 1      |
  | NCDHW <====> NDC1HWC0       | float32, int32,uint32              | 8,16  | 1      |
  | NCDHW <====> NDC1HWC0       | bfloat16, float16                  | 16    | 1      |
  | NCDHW <====> NDC1HWC0       | int8, uint8                        | 32    | 1      |
  | NDHWC <====> NDC1HWC0       | float32, int32,uint32              | 8,16  | 1      |
  | NDHWC <====> NDC1HWC0       | bfloat16, float16                  | 16    | 1      |
  | NDHWC <====> NDC1HWC0       | int8, uint8                        | 32    | 1      |
  | NCDHW <====> FRACTAL_Z_3D   | float32, int32,uint32              | 8,16  | 1      |
  | NCDHW <====> FRACTAL_Z_3D   | bfloat16, float16                  | 16    | 1      |
  | NCDHW <====> FRACTAL_Z_3D   | int8, uint8                        | 32    | 1      |
  | DHWCN <====> FRACTAL_Z_3D   | float32, int32,uint32              | 16    | 1      |
  | DHWCN <====> FRACTAL_Z_3D   | bfloat16, float16                  | 16    | 1      |
  | DHWCN <====> FRACTAL_Z_3D   | int8, uint8                        | 32    | 1      |
  | NCHW <====> FRACTAL_Z       | float32, uint32, int32             | 8     | >1     |
  | NCHW <====> FRACTAL_Z       | float16, bfloat16, uint16, int16   | 16    | >1     |
  | HWCN ====> FRACTAL_Z        | float16, bfloat16, uint16, int16   | 16    | >1     |
  | NCDHW <====> FRACTAL_Z_3D   | float32, uint32, int32             | 8     | >1     |
  | NCDHW <====> FRACTAL_Z_3D   | float16, bfloat16, uint16, int16   | 16    | >1     |
  | FRACTAL_Z_3D ====> DHWCN    | float32, uint32, int32             | 8     | >1     |
  | FRACTAL_Z_3D ====> DHWCN    | float16, bfloat16, uint16, int16   | 16    | >1     |
  | NCHW ====> FRACTAL_Z_C04    | float16, bfloat16                  | 16    | 1      |
  | FRACTAL_Z_C04 ====> NCHW    | float32                            | 16    | 1      |
  | ND ====> FRACTAL_NZ_C0_16   | float32, uint32, int32             | 16    | 1      |

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_trans_matmul_weight](examples/arch35/test_aclnn_trans_matmul_weight.cpp) | Invokes the Pack operator through the [aclnnTransMatmulWeight](docs/aclnnTransMatmulWeight_en.md) API. |
