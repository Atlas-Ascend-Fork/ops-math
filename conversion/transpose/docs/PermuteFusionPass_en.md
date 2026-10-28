# PermuteFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T17:03:49.766Z pushedAt=2026-10-13T01:47:29.754Z -->

## Fusion Pattern

This fusion replaces the Permute operator in the graph with the TransposeD or Transpose operator. The replacement method depends on the running platform:

<!-- npu="A3,910b,910,310p,310b" id16 -->
**Use case 1**:

<!-- npu="A3,910b,910,310p,310b" id14 -->
Permute is replaced with the TransposeD operator. The dimension permutation attribute of Permute is passed as the `perm` attribute of TransposeD.
The supported platforms are as follows:
<!-- end id14 -->
<!-- npu="A3" id7 -->
<term>Atlas A3 products</term>
<!-- end id7 -->
<!-- npu="910b" id8 -->
<term>Atlas A2 products</term>
<!-- end id8 -->
<!-- npu="310b" id9 -->
<term>Atlas 200I/500 A2 inference products</term>
<!-- end id9 -->
<!-- npu="310p" id10 -->
<term>Atlas inference products</term>
<!-- end id10 -->
<!-- npu="910" id11 -->
<term>Atlas training products</term>
<!-- end id11 -->

The graph structures before and after fusion are as follows:

![](../../../docs/en/figures/PermuteFusionPass_1.png)
<!-- end id16 -->

<!-- npu="950" id15 -->
**Use case 2**:

<!-- npu="950" id13 -->
Replace Permute with the Transpose operator. The dimension permutation attribute of Permute is passed as a Const node to the `perm` input of Transpose.
Supported products are as follows:
<!-- end id13 -->

<!-- npu="950" id12 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id12 -->

The graph structures before and after fusion are as follows:

![](../../../docs/en/figures/PermuteFusionPass_2.png)
<!-- end id15 -->

The dimension permutation order of the Permute operator is preferentially read from the `perm` attribute. If the attribute does not exist, the `order` attribute is read. The shape and dtype of the output after replacement are the same as those of the original Permute output.

## Constraints

- Structural constraints:
  - The graph contains a Permute operator, with input x and output y.
  - The Permute operator must carry the `perm` or `order` attribute to specify the dimension permutation order. The attribute value must be a valid full permutation of the dimensions of x, without repetition or omission.
- Version constraint: The fusion takes effect when the GE compiler version is 9.1.0 or later.

## Applicable Products

<!-- npu="310b" id1 -->
<term>Atlas 200I/500 A2 inference products</term>
<!-- end id1 -->

<!-- npu="310p" id2 -->
<term>Atlas inference products</term>
<!-- end id2 -->

<!-- npu="910" id3 -->
<term>Atlas training products</term>
<!-- end id3 -->

<!-- npu="910b" id4 -->
<term>Atlas A2 products</term>
<!-- end id4 -->

<!-- npu="A3" id5 -->
<term>Atlas A3 products</term>
<!-- end id5 -->

<!-- npu="950" id6 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id6 -->
