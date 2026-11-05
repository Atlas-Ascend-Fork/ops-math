# MinimumGrad

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term> |√|
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: MinimumGrad is the backward gradient computation operator of the `minimum` operator. It computes the gradients of the forward propagation output with respect to the two inputs x1 and x2.
- Computation formula:

$$
y1 = grads \times (x1 \leq x2)
$$

$$
y2 = grads \times (x1 > x2)
$$

## Parameter Description

|Parameter Name|Input/Output/Attribute|Description|Data Type|Data Format|
|-----|-----------|----|---------|------|
|grads|Input|Gradient input, which indicates the gradient of the forward output y of minimum. It is `grads` in the formula.|FLOAT16, FLOAT, BF16, INT32|ND|
|x1|Input|The first input of forward propagation, which is `x1` in the formula.|FLOAT16, FLOAT, BF16, INT32|ND|
|x2|Input|The second input of forward propagation, which is `x2` in the formula.|FLOAT16, FLOAT, BF16, INT32|ND|
|y1|Output|Gradient of x1, which is `y1` in the formula.|FLOAT16, FLOAT, BF16, INT32|ND|
|y2|Output|Gradient of x2, which is `y2` in the formula.|FLOAT16, FLOAT, BF16, INT32|ND|
|grad_x|Optional attribute|Whether to compute and output y1. The default value is `true`.|BOOL|-|
|grad_y|Optional attribute|Whether to compute and output y2. The default value is `true`.|BOOL|-|

## Constraint Description

- The shapes of the inputs `grads`, `x1`, and `x2` must satisfy the broadcast rule.
- The shapes of the outputs `y1` and `y2` are the shapes of `grads`, `x1`, and `x2` after broadcasting.
- The BF16 data type is supported only on Ascend 950PR&950DT products.
- The attributes `grad_x` and `grad_y` cannot both be `false`.

## Invocation Description

| Invocation Method | Invocation Sample                                                      | Description                                                      |
|--------------|-----------------------------------------------------------|---------------------------------------------------------|
| Graph-mode invocation | [test_geir_minimum_grad](./examples/arch35/test_geir_minimum_grad.cpp) | Invokes the MinimumGrad operator by building a graph through the [operator IR](./op_graph/minimum_grad_proto.h). |
