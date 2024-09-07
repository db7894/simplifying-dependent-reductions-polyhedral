# simplifying-dependent-reductions-polyhedral

A basic implementation of an MLIR pass based on the paper ["Simplfying Dependent Reductions in the Polyhedral Model"](https://arxiv.org/abs/2007.11203) by Yang et al. This isn't a fully general/robust pass but written for pedagogical purposes. I've only tested it on two examples: a simple prefix sum and a depth-3 loopnest reduction. 

```
bazel build //:optimizer
bazel-bin/optimizer prefix_sum_test.mlir -p simplify-dependent-reduce -o output.mlir
```
