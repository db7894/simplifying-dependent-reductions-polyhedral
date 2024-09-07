# simplifying-dependent-reductions-polyhedral

This paper: https://arxiv.org/abs/2007.11203

```
bazel build //:optimizer
bazel-bin/optimizer prefix_sum_test.mlir -p simplify-dependent-reduce -o output.mlir
```
