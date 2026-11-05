# StackBallQuery

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :-------: |
| <term>Ascend 950PR&950DT products</term>                             | √        |
| <term>Atlas A3 products</term>     | √        |
| <term>Atlas A2 products</term> | √        |
| <term>Atlas 200I/500 A2 inference products</term>                      |     ×     |
| <term>Atlas inference products</term>                             | ×        |
| <term>Atlas training products</term>                              | √        |
| Kirin X90 processor series products|√|
| Kirin 9030 processor series products|√|

## Function Description

- Operator function: Stack Ball Query is an alternative to KNN, which is used to find all points within a specified radius of point p1 (an upper limit of K is set in the implementation).
The returned points are not necessarily the points closest to p1, but the first K points within the radius.
Compared with the Ball Query operator, Stack Ball Query stacks the inputs of the Ball Query operator, and the dimensions of center_xyz and xyz change from 3 to 2.
Advantage: when a large number of points exist within the radius, this algorithm is faster than the KNN algorithm, and it also guarantees a fixed region scale, making local region features more general.
Models in use: this operator is proposed in the PointNet++ model, and is used in this model and its derivative models.
The specific operations of Stack Ball Query are as follows:
Compute the distance between xyz and each point in the same batch based on the input center_xyz.
If the distance is less than max_radius, save the index of the xyz point.
Exit the loop after sample_num indexes that meet the requirements are found.
Output the saved indexes.

## Parameter Description

| Parameter Name               | Input/Output/Attribute | Description                                         | Data Type         | Data Format |
| -------------------- | -------------- | -------------------------------------------- | ---------------- | -------- |
| xyz                  | Input           | 2D tensor, which contains the stacked coordinate values of xyz           | FLOAT16, FLOAT32 | ND       |
| center_xyz           | Input           | 2D tensor, which contains the stacked coordinate values of center_xyz    | FLOAT16, FLOAT32 | ND       |
| xyz_batch_cnt        | Input           | 1D tensor, which indicates the number of xyz points in each batch        | INT32, INT64     | ND        |
| center_xyz_batch_cnt | Input           | 1D tensor, which indicates the number of center_xyz points in each batch | INT32, INT64     | ND       |
| idx                  | Output           | Index values obtained after stack ball query               | INT32            | ND       |
| max_radius           | Input attribute       | Maximum radius value                                   | FLOAT            | -       |
| sample_num           | Input attribute       | Maximum number of samples                                   | INT              | -       |

## Constraint Description

- xyz is a 2D tensor with the shape (3,N). The 0th dimension must be 3 (corresponding to the x/y/z coordinate axes, with coordinates stored by row), and the 1st dimension N is the total number of points after stacking.
- center_xyz is a 2D tensor with the shape (M,3). The 1st dimension must be 3 (each center contains the three coordinates cx/cy/cz).
- xyz_batch_cnt is a 1D tensor with the shape (B,), where B is the number of batches.
- center_xyz_batch_cnt is a 1D tensor with the shape (B,), and dim[0] must be the same as dim[0] of xyz_batch_cnt.
- The dtypes of xyz and center_xyz must be the same, and must be FLOAT or FLOAT16.
- The dtypes of xyz_batch_cnt and center_xyz_batch_cnt must be the same, and must be INT32 or INT64.
- The dtype of the output idx is fixed to INT32, and the shape is (M,sample_num), where M is center_xyz.shape[0].
- max_radius must be greater than 0.
- sample_num must be greater than 0.
- The operator supports only 2D inputs (xyz and center_xyz) and 1D inputs (batch_cnt), and does not support scalar or other dimension scenarios.

## Invocation Description

| Invocation Method  | Sample Code                                                              | Description                                                                        |
| --------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| aclnn API | [test_geir_stack_ball_query](./examples/test_geir_stack_ball_query.cpp) | Invokes the StackBallQuery operator by building a graph through the [operator IR](./op_graph/stack_ball_query_proto.h). |
