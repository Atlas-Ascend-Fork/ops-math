# ConfusionMatrix

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the confusion matrix of a classification task. Given the ground-truth labels and the predicted labels predictions, the operator counts the number of occurrences of each (label, prediction) pair and generates a num_classes×num_classes matrix. When weights is not empty, the contribution of each match is the corresponding weight value instead of 1.

- Computation formula:

  If weights is specified:

  $$
  y[labels_i, predictions_i] = y[labels_i, predictions_i] + weights_i
  $$

  Otherwise:

  $$
  y[labels_i, predictions_i] = y[labels_i, predictions_i] + 1
  $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
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
      <td>labels</td>
      <td>Input</td>
      <td>Ground-truth labels, a 1-dimensional tensor, with the value range [0, num_classes).</td>
      <td>INT8, INT32, UINT8, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>predictions</td>
      <td>Input</td>
      <td>Predicted labels, a 1-dimensional tensor, whose shape and dtype are the same as those of labels.</td>
      <td>INT8, INT32, UINT8, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>Input</td>
      <td>Weight of each sample, which can be empty. The shape is the same as that of labels; when it is not empty, the dtype must be the same as the dtype of the output y (that is, the same as the dtype attribute).</td>
      <td>FLOAT, FLOAT16, INT32, INT8, UINT8</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>num_classes</td>
      <td>Attribute</td>
      <td>Number of classes, with the value range [1, 4096].</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Data type of the output tensor.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Confusion matrix, whose shape is [num_classes, num_classes].</td>
      <td>FLOAT, FLOAT16, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The shapes of labels and predictions must be the same.
- The dtypes of labels and predictions must be the same.
- When weights is not empty, its shape must be the same as that of labels, and its dtype must be the same as the dtype of the output y (that is, the dtype attribute).
- The values in labels and predictions must be in the range [0, num_classes).
- The value range of num_classes is [1, 4096].
- labels and predictions are 1-dimensional tensors, and the supported data types are FLOAT, FLOAT16, INT32, INT8, and UINT8.
- Accuracy constraint: when the output is FLOAT16, the count is accurate within the exact integer representation range of fp16 (≤2048); if the count of a single bucket exceeds this range, precision loss occurs (accumulation is performed at the output precision). For scenarios with large counts, it is recommended to use float32 or int32 for the output.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|--------|------|
| Graph-mode invocation | [test_geir_confusion_matrix](./examples/test_geir_confusion_matrix.cpp) | Invokes the ConfusionMatrix operator by building a graph through the [operator IR](./op_graph/confusion_matrix_proto.h). |
