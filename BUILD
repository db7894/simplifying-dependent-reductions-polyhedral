cc_library(
    name = "simplify_dependent_reductions_pass",
    srcs = ["SimplifyDependentReductions.cpp"],
    hdrs = ["SimplifyDependentReductions.h"],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "optimizer",
    srcs = ["main.cpp"],
    deps = [
        ":simplify_dependent_reductions_pass",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:MlirOptLib",
    ],
)
