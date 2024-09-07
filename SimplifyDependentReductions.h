#ifndef SIMPLIFY_DEPENDENT_REDUCTIONS_PASS_H
#define SIMPLIFY_DEPENDENT_REDUCTIONS_PASS_H

#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createSimplifyDependentReductionsPass();

#endif // SIMPLIFY_DEPENDENT_REDUCTIONS_PASS_H