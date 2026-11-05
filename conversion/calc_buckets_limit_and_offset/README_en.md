# CalcBucketsLimitAndOffset

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

- Operator function: obtains the count and offset of the corresponding bucket from ivf_counts and ivf_offset based on bucket_list indexes, limits the total count of all buckets based on total_limit, and outputs the limited count and the corresponding offsets.

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
    <td>bucket_list</td>
    <td>Input</td>
    <td>1-D tensor, a list of bucket indexes, whose values are the subscripts of ivf_counts and ivf_offset.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>ivf_counts</td>
    <td>Input</td>
    <td>1-D tensor, the count value of each bucket.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>ivf_offset</td>
    <td>Input</td>
    <td>1-D tensor, the offset of each bucket.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>total_limit</td>
    <td>Attribute</td>
    <td>Upper limit of the total count of all buckets.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>buckets_limit</td>
    <td>Output</td>
    <td>1-D tensor, the limited count value of each bucket, whose total does not exceed total_limit.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>buckets_offset</td>
    <td>Output</td>
    <td>1-D tensor, the offsets corresponding to bucket_list.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraint Description

- The values in bucket_list must be within the range [0, number of elements in ivf_counts) and [0, number of elements in ivf_offset)
- The numbers of elements in ivf_counts and ivf_offset must be greater than or equal to the maximum index value in bucket_list
- total_limit must be a non-negative integer

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph-mode API | [test_geir_calc_buckets_limit_and_offset](examples/test_geir_calc_buckets_limit_and_offset.cpp) | Invokes the CalcBucketsLimitAndOffset operator through the [operator IR](op_graph/calc_buckets_limit_and_offset_proto.h) API. |
