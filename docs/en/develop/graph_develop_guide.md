# Graph Mode Adaptation Guide

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-17T12:34:14.571Z pushedAt=2026-11-18T07:58:22.354Z nl=2 -->

## Overview

This document introduces the graph mode adaptation method for custom operators. The overall process is consistent with the operator development guide ([AI Core Operator Development Guide](aicore_develop_guide.md)/[AI CPU Operator Development Guide](aicpu_develop_guide.md)). Notably, **aclnn adaptation is not required**. Only the following deliverable adaptation is needed to call the operator in graph mode.

```text
${op_name}                              # Replace with the lowercase underscore form of the actual operator name
├── op_host                             # Host-side implementation
│   └── ${op_name}_infershape.cpp       # InferShape implementation, implements operator shape inference and infers output shape at runtime
├── op_graph                            # Graph fusion related implementation
│   ├── CMakeLists.txt                  # cmakelist file on the op_graph side
│   ├── ${op_name}_graph_infer.cpp      # InferDataType file, implements operator type inference and infers output dataType at runtime
└── └── ${op_name}_proto.h              # Operator prototype definition, used to identify the operator in the graph optimization and fusion phases
```

This document uses the `AddExample` operator (assumed to be an AI Core operator) as an example to introduce the implementation of the graph entry deliverable. The graph entry implementation for AI CPU operators is basically similar. For the complete code, see `add_example` and `add_example_aicpu` in the `examples` directory.

## Shape and DataType Inference

Graph mode requires two deliverables: ```${op_name}_infershape.cpp``` and ```${op_name}_graph_infer.cpp```

**Deliverable 1: ${op_name}_infershape.cpp**

The InferShape function infers the output shape based on the input shape.

The following is an example. For the complete code of the `AddExample` operator, see [add_example_infershape.cpp](../../../examples/add_example/op_host/add_example_infershape.cpp) under `examples/add_example/op_host`.

```cpp
// The AddExample operator logic is to add two numbers, therefore the output shape is consistent with the input shape
static ge::graphStatus InferShapeAddExample(gert::InferShapeContext* context)
{
    ....
    // Obtain the input shape
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    // Obtain the output shape
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    // Obtain the input DimNum
    auto xShapeSize = xShape->GetDimNum();
    // Set the output DimNum
    yShape->SetDimNum(xShapeSize);
    // Set the input Dim values to the output in sequence
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }
    ....
}
// inferShape registration
IMPL_OP_INFERSHAPE(AddExample).InferShape(InferShapeAddExample);
```

**Deliverable 2: ${op_name}_graph_infer.cpp**

The InferDataType function infers the output DataType based on the input DataType.

The following is an example. For the complete code of the `AddExample` operator, see [add_example_graph_infer.cpp](../../../examples/add_example/op_graph/add_example_graph_infer.cpp) under `examples/add_example/op_graph`.

```cpp
// The AddExample operator logic is to add two numbers, therefore the output dataType is consistent with the input dataType
static ge::graphStatus InferDataTypeAddExample(gert::InferDataTypeContext* context)
{
    ....
    // Obtain the input dataType
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    // Set the input dataType to the output
    context->SetOutputDataType(IDX_0, sizeDtype);
    ....
}

// Register InferDataType
IMPL_OP(AddExample).InferDataType(InferDataTypeAddExample);
```

## Operator Prototype Configuration

**Deliverable: ${op_name}_proto.h**

To invoke graph mode, register the operator prototype with [Graph Engine](https://www.hiascend.com/eng/cann/graph-engine) (GE) so that GE can identify the input, output, and attribute information of this type of operator. Registration is completed through the `REG_OP` API. Developers need to define basic information such as the operator input and output tensor types and quantities.

### Common Data Types

Common tensor/attribute data type examples are as follows:

| Tensor Type | Attribute Type | Example |
|-----|------|-----|
| int64 | / | DT_INT64 |
| int32 | / | DT_INT32 |
| int16 | / | DT_INT16 |
| int8 | / | DT_INT8 |
| double | / | DT_DOUBLE |
| float32 | / | DT_FLOAT |
| float16 | / | DT_FLOAT16 |
| bfloat16 | / | DT_BF16 |
| complex128 | / | DT_COMPLEX128 |
| complex64 | / | DT_COMPLEX64 |
| complex32 | / | DT_COMPLEX32 |
| / | int | Int |
| / | bool | Bool |
| / | string | String |
| / | float | Float |
| / | list | ListInt |

### Input and Output Definition

| Input/Output | Keyword | Example |
|-----|------|-----|
| required input | INPUT | .INPUT(${name}, TensorType({input_dtype})) |
| Optional input | OPTIONAL_INPUT | .OPTIONAL_INPUT(${name}, TensorType({optional_input_dtype})) |
| Required attribute | REQUIRED_ATTR | .REQUIRED_ATTR(${name}, ${dtype}) |
| Optional attribute | ATTR | .ATTR(${name}, ${dtype}, ${default_value}) |
| Output | OUTPUT | .OUTPUT(${name}, TensorType({output_dtype})) |

### Code Example

The following example code shows how to register the `AddExample` operator:

```cpp
REG_OP(AddExample)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AddExample)
```

For the complete code, see [add_example_proto.h](../../../examples/add_example/op_graph/add_example_proto.h) in the `examples/add_example/op_graph` directory.
