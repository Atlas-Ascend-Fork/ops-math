# TopKPQDistance

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR&950DT products                                    |    √     |
| Atlas A3 products                    |    √     |
| Atlas A2 products                    |    √     |
| Atlas 200I/500 A2 inference products                                   |    √     |
| Atlas inference products                                            |    √     |
| Atlas training products                                            |    √     |

## Function Description

- Operator function: computes TopK distances based on grouped extreme-value pruning and is used in vector retrieval scenarios. It finds the k largest or smallest distances from multiple groups of PQ distance data, together with the corresponding ivf and index.

- Computation process:
  1. Select the TopK extreme values from grouped_extreme_distance (grouped pruning).
  2. Expand the selected groups and compute the exact TopK distances.
  3. Output the TopK distances, ivf, and index.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| actual_count | Input (dynamic) | Number of actual elements in each group | INT32 | ND |
| pq_distance | Input (dynamic) | PQ distance data | FLOAT16, FLOAT | ND |
| grouped_extreme_distance | Input (dynamic) | Extreme distance of each group | FLOAT16, FLOAT | ND |
| pq_ivf | Input (dynamic) | IVF index | INT32 | ND |
| pq_index | Input (dynamic) | PQ index | INT32 | ND |
| order | Attribute (Optional) | Sorting method. "ASC" for ascending order or "DES" for descending order. The default value is "ASC". | STRING | - |
| k | Attribute (Required) | The value of k for TopK | INT | - |
| group_size | Attribute (Required) | Group size | INT | - |
| topk_distance | Output | TopK distance values | FLOAT16, FLOAT | ND |
| topk_ivf | Output | ivf corresponding to TopK | INT32 | ND |
| topk_index | Output | index corresponding to TopK | INT32 | ND |

## Constraint Description

- k cannot be 0 and cannot be greater than the sum of actual_count.
- group_size cannot be 0.
- actual_count must be integer-divisible by group_size.
- Multiple groups of dynamic inputs (data_batch) are supported, and each group contains 5 inputs.

## Invocation Description

| Invocation Method   | Sample Code | Description |
|------------|----------|------|
| Graph-mode invocation | [test_geir_top_k_pq_distance](./examples/test_geir_top_k_pq_distance.cpp) | Invokes the TopKPQDistance operator by building a graph through the [operator IR](./op_graph/top_k_pq_distance_proto.h). |
