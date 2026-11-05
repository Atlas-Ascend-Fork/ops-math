# HistogramFixedWidth

English | [简体中文](./README.md)

## Product Support Status

| Product                                                       | Supported |
| :---------------------------------------------------------| :--------:|
| <term>Ascend 950PR&950DT products</term>       | √        |
| <term>Atlas A3 products</term> | √        |
| <term>Atlas A2 products</term> | √        |
| <term>Atlas 200I/500 A2 inference products</term>      | ×        |
| <term>Atlas inference products</term>      | √        |
| <term>Atlas training products</term>                      |    ×     |

## Function Description

- Operator function: computes the histogram of a tensor. This operator is compatible with the histogram_fixed_width operator of TensorFlow.
- Computation formula: uses range=[min,max] as the upper and lower bounds of the statistics, divides the space between min and max into nbins intervals of equal width, and counts the number of elements in tensor x in each interval. Elements smaller than min are counted into the first interval, and elements greater than max are counted into the last interval.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description                        | Data Type              | Data Format |
| --------| ----------------| --------------------------------------| ------------------------------| ----------|
| x       | Input            | Input tensor, which is the data to be counted.             | FLOAT, FLOAT16, INT32, INT64 | ND       |
| range   | Input            | Tensor with the shape [2], containing [min,max]. | FLOAT, FLOAT16, INT32, INT64 | ND       |
| nbins   | Input            | Scalar tensor, which is the number of histogram intervals.       | INT32                | ND       |
| y       | Output           | Histogram result, with the shape [nbins].             | INT32                | ND       |
|dtype   | Attribute           |Reserved attribute, and the default value is 3                 | INT64                         |-|

## Constraint Description

The max of the input tensor range must be greater than min. Otherwise, an error is reported.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|--------|----------------------------------------------------------------|
| Graph-mode invocation | [test_geir_histogram_fixed_width](./examples/test_geir_histogram_fixed_width.cpp) | Invokes the HistogramFixedWidth operator by building a graph through the [operator IR](./op_graph/histogram_fixed_width_proto.h). |
