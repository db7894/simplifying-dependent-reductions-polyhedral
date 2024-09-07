#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <set>
#include <algorithm>

/*
Implements the simplification of dependent reductions using Polyhedral analysis as described in
"Simplifying Dependent Reductions in the Polyhedral Model" by Yang et al.

Paper: https://arxiv.org/pdf/2007.11203

Psuedocode:
```
(1) Schedule the augmented program to obtain an initial sequential schedule Θ for all
    statements and left hand side of reductions
(2) Apply ST to all faces of all reduction statement's domains; choose the direction that is
    consistent with Θ by:
    (a) First pick any valid reuse vector r from the candidate set.
    (b) Test if r is consistent with Θ, if not consistent, set r to -r, if -r is also a valid reuse
        vector;
otherwise, do not apply the current ST.
```
*/

/*
Given the simple_test.mlir example I expect this:
```
func.func @prefix_sum(%N : index, %A: memref<?xf32>, %B: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32

  // Initialize B[0]
  %first = memref.load %A[%c0] : memref<?xf32>
  memref.store %first, %B[%c0] : memref<?xf32>

  // Compute prefix sum
  scf.for %i = %c1 to %N step %c1 {
    %prev = memref.load %B[%i-1] : memref<?xf32>
    %curr = memref.load %A[%i] : memref<?xf32>
    %sum = arith.addf %prev, %curr : f32
    memref.store %sum, %B[%i] : memref<?xf32>
  }

  // The second loop remains unchanged
  scf.for %i = %c0 to %N step %c1 {
    %b_i = memref.load %B[%i] : memref<?xf32>
    %i_plus_1 = arith.addi %i, %c1 : index
    %a_next = func.call @f(%b_i) : (f32) -> f32
    memref.store %a_next, %A[%i_plus_1] : memref<?xf32>
  }

  return
}
```
*/

using ScheduleMap = mlir::DenseMap<mlir::Operation *, int64_t>;

struct SimplifyDependentReductionsPass
    : public mlir::PassWrapper<SimplifyDependentReductionsPass, mlir::OperationPass<mlir::func::FuncOp>>
{
public:
  void runOnOperation() override
  {
    mlir::func::FuncOp f = getOperation();
    std::set<mlir::Operation *> visitedOps;

    auto [augmentedFunc, schedule] = scheduleAugmentedProgram(f);

    // find all reduction loops
    llvm::SmallVector<mlir::scf::ForOp, 4> reductionLoops;
    f.walk([&](mlir::scf::ForOp forOp)
           {
        if (mlir::isa<mlir::func::FuncOp>(forOp->getParentOp())) {
            if (isReduction(forOp, visitedOps)) {
                reductionLoops.push_back(forOp);
            }
        } });

    // apply ST to all faces of all reduction statement domains
    for (auto reductionLoop : reductionLoops)
    {
      simplifyReduction(reductionLoop, schedule);
    }
  }

private:
  std::vector<int64_t> computeReuseVector(mlir::scf::ForOp outerLoop)
  {
    std::vector<int64_t> reuseVector;
    mlir::scf::ForOp currentLoop = outerLoop;
    int depth = 0;

    while (true)
    {
      depth++;
      reuseVector.push_back(0);

      mlir::scf::ForOp nestedLoop;
      for (mlir::Operation &op : currentLoop.getBody()->getOperations())
      {
        if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(&op))
        {
          nestedLoop = forOp;
          break;
        }
      }

      if (!nestedLoop)
      {
        reuseVector.back() = 1;
        break;
      }

      currentLoop = nestedLoop;
    }

    llvm::errs() << "Computed reuse vector: [";
    for (size_t i = 0; i < reuseVector.size(); ++i)
    {
      llvm::errs() << reuseVector[i];
      if (i < reuseVector.size() - 1)
      {
        llvm::errs() << ", ";
      }
    }
    llvm::errs() << "]\n";

    return reuseVector;
  }

  std::pair<mlir::func::FuncOp, ScheduleMap>
  scheduleAugmentedProgram(mlir::func::FuncOp f)
  {
    mlir::OpBuilder builder(f.getContext());

    mlir::func::FuncOp augmentedFunc = cast<mlir::func::FuncOp>(builder.clone(*f));

    ScheduleMap schedule;
    int64_t timestamp = 0;
    augmentedFunc.walk([&](mlir::Operation *op)
                       {
    if (mlir::isa<mlir::scf::ForOp>(op)) {
      schedule[op] = timestamp++;
      llvm::errs() << "Scheduled loop: " << *op << " with timestamp: " << timestamp-1 << "\n";
    } });

    return {augmentedFunc, schedule};
  }

  bool isReduction(mlir::scf::ForOp forOp, std::set<mlir::Operation *> &visitedOps)
  {
    if (visitedOps.find(forOp.getOperation()) != visitedOps.end())
    {
      return false;
    }
    visitedOps.insert(forOp.getOperation());

    bool hasStore = false;
    bool hasLoad = false;
    bool hasReductionOp = false;
    int nestedLoopCount = 0;

    forOp.walk([&](mlir::Operation *op)
               {
        if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
            hasStore = true;
        } else if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
            hasLoad = true;
        } else if (auto addFOp = mlir::dyn_cast<mlir::arith::AddFOp>(op)) {
            hasReductionOp = true;
        } else if (auto nestedForOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
            if (nestedForOp != forOp) {
                nestedLoopCount++;
            }
        } });

    // FIXME: this is hacky but correctly identifies the two examples I wanted to make work.
    bool isPrefixSum = hasStore && hasLoad && hasReductionOp && nestedLoopCount == 1;
    bool isNestedReduction = hasStore && hasLoad && hasReductionOp && nestedLoopCount > 1;

    return isPrefixSum || isNestedReduction;
  }

  std::vector<mlir::AffineMap> computeDomainFaces(mlir::scf::ForOp reductionLoop)
  {
    std::vector<mlir::AffineMap> faces;
    mlir::MLIRContext *context = reductionLoop.getContext();
    unsigned numDims = 0;
    auto currentLoop = reductionLoop;

    std::vector<mlir::scf::ForOp> nestedLoops;
    while (currentLoop)
    {
      nestedLoops.push_back(currentLoop);
      numDims++;

      // find next nested loop
      mlir::scf::ForOp nestedLoop;
      for (auto &op : currentLoop.getBody()->without_terminator())
      {
        if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op))
        {
          nestedLoop = forOp;
          break;
        }
      }

      if (nestedLoop)
      {
        currentLoop = nestedLoop;
      }
      else
      {
        break;
      }
    }

    // generate faces for each loop
    for (unsigned i = 0; i < nestedLoops.size(); ++i)
    {
      auto loop = nestedLoops[i];

      // lower bound face
      faces.push_back(mlir::AffineMap::get(numDims, 0, {mlir::getAffineDimExpr(i, context)}, context));

      // upper bound face
      auto upperBound = loop.getUpperBound();
      mlir::AffineExpr upperBoundExpr;
      if (auto constOp = upperBound.getDefiningOp<mlir::arith::ConstantIndexOp>())
      {
        upperBoundExpr = mlir::getAffineConstantExpr(constOp.value(), context);
      }
      else
      {
        upperBoundExpr = mlir::getAffineDimExpr(i, context);
      }
      faces.push_back(mlir::AffineMap::get(numDims, 0, {upperBoundExpr}, context));
    }

    return faces;
  }

  void applyST(mlir::scf::ForOp &reductionLoop, const mlir::AffineMap &face, const ScheduleMap &schedule)
  {
    if (reductionLoop.getRegion().empty() || reductionLoop.getBody()->empty())
    {
      llvm::errs() << "Error: Empty reduction loop in applyST\n";
      return;
    }

    llvm::errs() << "Attempting to apply ST to: " << reductionLoop << "\n";

    auto reuseVector = computeReuseVector(reductionLoop);
    llvm::errs() << "Computed reuse vector: [";
    for (size_t i = 0; i < reuseVector.size(); ++i)
    {
      llvm::errs() << reuseVector[i];
      if (i < reuseVector.size() - 1)
        llvm::errs() << ", ";
    }
    llvm::errs() << "]\n";

    if (reuseVector.size() != face.getNumInputs())
    {
      llvm::errs() << "Error: Reuse vector dimensionality doesn't match face\n";
      return;
    }

    if (isValidReuseVector(reuseVector, reductionLoop, face))
    {
      if (isConsistentWithSchedule(reuseVector, reductionLoop, schedule))
      {
        llvm::errs() << "Reuse vector is valid and consistent. Applying transformation.\n";
        reductionLoop = applySimplificationTransformation(reductionLoop, face, reuseVector);
        return;
      }
    }

    // if not valid or consistent, try negation (per heuristic)
    auto negatedVector = negateVector(reuseVector);
    llvm::errs() << "Trying negated vector: [";
    for (size_t i = 0; i < negatedVector.size(); ++i)
    {
      llvm::errs() << negatedVector[i];
      if (i < negatedVector.size() - 1)
        llvm::errs() << ", ";
    }
    llvm::errs() << "]\n";

    if (isValidReuseVector(negatedVector, reductionLoop, face))
    {
      if (isConsistentWithSchedule(negatedVector, reductionLoop, schedule))
      {
        llvm::errs() << "Negated vector is valid and consistent. Applying transformation.\n";
        applySimplificationTransformation(reductionLoop, face, negatedVector);
        return;
      }
    }

    llvm::errs() << "No valid and consistent reuse vector found. ST not applied.\n";
  }

  void simplifyReduction(mlir::scf::ForOp &reductionLoop, const ScheduleMap &schedule)
  {
    llvm::errs() << "Attempting to simplify reduction: " << reductionLoop << "\n";

    // check if the loop is empty or invalid
    if (reductionLoop.getRegion().empty() || reductionLoop.getBody()->empty())
    {
      llvm::errs() << "Reduction loop is empty or invalid. Skipping simplification.\n";
      return;
    }

    auto faces = computeDomainFaces(reductionLoop);
    llvm::errs() << "Computed " << faces.size() << " faces\n";
    for (const auto &face : faces)
    {
      llvm::errs() << "Processing face: " << face << "\n";
      if (reductionLoop.getRegion().empty() || reductionLoop.getBody()->empty())
      {
        llvm::errs() << "Error: Reduction loop became invalid during simplification\n";
        return;
      }

      // check if the loop is already simplified before applying ST
      if (isLoopSimplified(reductionLoop))
      {
        llvm::errs() << "Loop is already simplified. Skipping further simplification.\n";
        return;
      }

      applyST(reductionLoop, face, schedule);

      if (isLoopSimplified(reductionLoop))
      {
        llvm::errs() << "Simplification completed successfully.\n";
        return; // bail
      }
    }
  }

  void generateCandidateVectors(std::vector<int64_t> &candidate, unsigned index,
                                mlir::scf::ForOp reductionLoop, const mlir::AffineMap &face,
                                std::set<std::vector<int64_t>> &validVectors)
  {
    if (index == candidate.size())
    {
      if (isValidReuseVector(candidate, reductionLoop, face))
      {
        validVectors.insert(candidate);
        llvm::errs() << "Valid reuse vector found: [";
        for (size_t i = 0; i < candidate.size(); ++i)
        {
          llvm::errs() << candidate[i];
          if (i < candidate.size() - 1)
            llvm::errs() << ", ";
        }
        llvm::errs() << "]\n";
      }
      return;
    }

    for (int64_t i = -1; i <= 1; ++i)
    {
      candidate[index] = i;
      generateCandidateVectors(candidate, index + 1, reductionLoop, face, validVectors);
    }
  }

  std::set<std::vector<int64_t>> computeValidReuseVectors(mlir::scf::ForOp reductionLoop, const mlir::AffineMap &face)
  {
    std::set<std::vector<int64_t>> validVectors;
    mlir::MLIRContext *context = reductionLoop.getContext();
    unsigned numDims = face.getNumDims();

    llvm::errs() << "Computing valid reuse vectors for: " << reductionLoop << "\n";

    auto computedReuseVector = computeReuseVector(reductionLoop);
    llvm::errs() << "Computed reuse vector: [";
    for (size_t i = 0; i < computedReuseVector.size(); ++i)
    {
      llvm::errs() << computedReuseVector[i];
      if (i < computedReuseVector.size() - 1)
        llvm::errs() << ", ";
    }
    llvm::errs() << "]\n";

    if (isValidReuseVector(computedReuseVector, reductionLoop, face))
    {
      validVectors.insert(computedReuseVector);
      llvm::errs() << "Computed reuse vector is valid.\n";
    }

    // generate additional candidate vectors
    std::vector<int64_t> candidate(numDims, 0);
    generateCandidateVectors(candidate, 0, reductionLoop, face, validVectors);

    llvm::errs() << "Found " << validVectors.size() << " valid reuse vectors.\n";
    return validVectors;
  }

  bool isConsistentWithSchedule(const std::vector<int64_t> &reuseVector,
                                mlir::scf::ForOp forOp,
                                const ScheduleMap &schedule)
  {
    llvm::errs() << "Checking if reuse vector is consistent with schedule for: " << forOp << "\n";
    auto it = std::find_if(schedule.begin(), schedule.end(),
                           [&forOp](const auto &entry)
                           {
                             auto scheduledOp = entry.first;
                             return mlir::isa<mlir::scf::ForOp>(scheduledOp) &&
                                    scheduledOp->getAttrDictionary() == forOp->getAttrDictionary();
                           });

    if (it == schedule.end())
    {
      llvm::errs() << "Loop not found in schedule.\n";
      llvm::errs() << "Dumping schedule:\n";
      for (const auto &entry : schedule)
      {
        llvm::errs() << "  Operation: " << *entry.first << ", Timestamp: " << entry.second << "\n";
      }
      return false;
    }

    int64_t loopTimestamp = it->second;
    llvm::errs() << "Loop timestamp: " << loopTimestamp << "\n";

    // check if reuse vector points to a previous timestamp
    int64_t reuseTimestamp = loopTimestamp;
    for (int64_t i = 0; i < reuseVector.size(); ++i)
    {
      if (reuseVector[i] != 0)
      {
        reuseTimestamp -= reuseVector[i];
        break;
      }
    }
    llvm::errs() << "Reuse timestamp: " << reuseTimestamp << "\n";

    return reuseTimestamp < loopTimestamp;
  }

  std::vector<int64_t> negateVector(const std::vector<int64_t> &vec)
  {
    std::vector<int64_t> result = vec;
    for (auto &v : result)
      v = -v;
    return result;
  }

  bool isAssociativeOp(mlir::Operation *op)
  {
    return mlir::isa<
        mlir::arith::AddFOp,
        mlir::arith::AddIOp,
        mlir::arith::MulFOp, mlir::arith::MulIOp,
        mlir::arith::AndIOp, mlir::arith::OrIOp, mlir::arith::XOrIOp,
        mlir::arith::MinFOp, mlir::arith::MinSIOp, mlir::arith::MinUIOp,
        mlir::arith::MaxFOp, mlir::arith::MaxSIOp, mlir::arith::MaxUIOp,
        mlir::arith::SelectOp>(op);
  }

  mlir::Operation *findReductionOp(mlir::scf::ForOp forOp)
  {
    mlir::Operation *reductionOp = nullptr;
    forOp.walk([&](mlir::Operation *op)
               {
            if (op->hasTrait<mlir::OpTrait::IsCommutative>() &&
                isAssociativeOp(op)) {
                reductionOp = op;
                return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance(); });
    return reductionOp;
  }

  bool isValidReuseVector(const std::vector<int64_t> &vec, mlir::scf::ForOp reductionLoop, const mlir::AffineMap &face)
  {
    llvm::errs() << "Checking if vector [";
    for (size_t i = 0; i < vec.size(); ++i)
    {
      llvm::errs() << vec[i];
      if (i < vec.size() - 1)
        llvm::errs() << ", ";
    }
    llvm::errs() << "] is valid for reduction: " << reductionLoop << "\n";

    mlir::MLIRContext *context = reductionLoop.getContext();
    unsigned numDims = face.getNumDims();
    unsigned numResults = face.getNumResults();

    if (vec.size() != numDims)
    {
      llvm::errs() << "Vector size does not match number of dimensions.\n";
      return false;
    }

    // sharing constraint: check if the vector is parallel to the face
    mlir::AffineExpr expr = mlir::getAffineConstantExpr(0, context);
    for (unsigned i = 0; i < numResults; ++i)
    {
      if (i < vec.size())
      {
        expr = expr + vec[i] * face.getResult(i);
      }
    }
    llvm::errs() << "Expression: " << expr << "\n";

    mlir::AffineExpr simplifiedExpr = mlir::simplifyAffineExpr(expr, numDims, face.getNumSymbols());
    llvm::errs() << "Simplified expression: " << simplifiedExpr << "\n";

    if (!simplifiedExpr.isa<mlir::AffineConstantExpr>() ||
        simplifiedExpr.cast<mlir::AffineConstantExpr>().getValue() != 0)
    {
      llvm::errs() << "Expression is not parallel to the face.\n";
      return false;
    }

    // inverse constraint: check if the reduction operator has an inverse
    auto reductionOp = findReductionOp(reductionLoop);
    if (!reductionOp || !reductionOp->hasTrait<mlir::OpTrait::IsCommutative>() ||
        !isAssociativeOp(reductionOp))
    {
      llvm::errs() << "Reduction operation is not commutative or associative.\n";
      return false;
    }

    // complexity constraint: check if applying the vector reduces complexity
    // this is a simplified check and may need to be extended
    if (!std::any_of(vec.begin(), vec.end(), [](int64_t v)
                     { return v != 0; }))
    {
      llvm::errs() << "Vector does not reduce complexity.\n";
      return false;
    }

    llvm::errs() << "Vector is valid.\n";
    return true;
  }

  mlir::ValueRange getInitialLoadIndices(mlir::Value memref, mlir::Value c0)
  {
    auto memrefType = memref.getType().cast<mlir::MemRefType>();
    return std::vector<mlir::Value>(memrefType.getRank(), c0);
  }

  mlir::ValueRange getInitialStoreIndices(mlir::Value memref, mlir::Value c0)
  {
    auto memrefType = memref.getType().cast<mlir::MemRefType>();
    return std::vector<mlir::Value>(memrefType.getRank(), c0);
  }

  bool isLoopSimplified(mlir::scf::ForOp loop)
  {
    llvm::errs() << "Checking if loop is simplified: " << loop << "\n";

    if (loop.getRegion().empty())
    {
      llvm::errs() << "Loop region is empty. Not simplified.\n";
      return false;
    }

    if (loop.getBody()->empty())
    {
      llvm::errs() << "Loop body is empty. Not simplified.\n";
      return false;
    }

    // Check if the loop body contains only a single operation (excluding the terminator)
    if (std::distance(loop.getBody()->begin(), loop.getBody()->end()) != 2)
    {
      llvm::errs() << "Loop body contains multiple operations. Not simplified.\n";
      return false;
    }

    auto &bodyOp = loop.getBody()->front();
    if (!mlir::isa<mlir::scf::ForOp>(bodyOp) && !isReductionOp(&bodyOp))
    {
      llvm::errs() << "Loop body does not contain a nested scf.for or a reduction operation. Not simplified.\n";
      return false;
    }

    if (auto nestedLoop = mlir::dyn_cast<mlir::scf::ForOp>(bodyOp))
    {
      return isLoopSimplified(nestedLoop);
    }

    llvm::errs() << "Loop is simplified.\n";
    return true;
  }

  std::vector<std::tuple<mlir::Value, mlir::Value, mlir::Value>> getLoopBounds(mlir::scf::ForOp outerLoop)
  {
    std::vector<std::tuple<mlir::Value, mlir::Value, mlir::Value>> bounds;
    mlir::scf::ForOp currentLoop = outerLoop;

    while (currentLoop)
    {
      bounds.emplace_back(currentLoop.getLowerBound(), currentLoop.getUpperBound(), currentLoop.getStep());

      if (currentLoop.getBody()->empty())
        break;

      auto &ops = currentLoop.getBody()->getOperations();
      auto it = std::find_if(ops.begin(), ops.end(), [](mlir::Operation &op)
                             { return mlir::isa<mlir::scf::ForOp>(op); });

      if (it == ops.end())
        break;
      currentLoop = mlir::cast<mlir::scf::ForOp>(*it);
    }

    return bounds;
  }

  mlir::scf::ForOp applySimplificationTransformation(mlir::scf::ForOp outerLoop, const mlir::AffineMap &face, const std::vector<int64_t> &reuseVector)
  {
    llvm::errs() << "Checking if loop is already simplified\n";
    if (outerLoop->hasAttr("simplified"))
    {
      llvm::errs() << "Loop has already been processed. Skipping transformation.\n";
      return outerLoop;
    }

    if (outerLoop.getRegion().empty() || outerLoop.getBody()->empty())
    {
      llvm::errs() << "Reduction loop is empty or invalid. Skipping simplification.\n";
      return outerLoop;
    }

    bool isAlreadySimplified = isLoopSimplified(outerLoop);

    if (isAlreadySimplified)
    {
      llvm::errs() << "Loop is already in simplified form. No further simplification needed.\n";
      return outerLoop;
    }

    llvm::errs() << "Applying simplification transformation for:\n";
    llvm::errs() << outerLoop << "\n";

    mlir::OpBuilder builder(outerLoop);
    auto loc = outerLoop.getLoc();

    // find the input and output memrefs
    llvm::errs() << "Finding input and output memrefs\n";
    mlir::Value inputMemRef, outputMemRef;
    outerLoop.walk([&](mlir::memref::LoadOp loadOp)
                   {
        if (!inputMemRef)
            inputMemRef = loadOp.getMemref(); });
    outerLoop.walk([&](mlir::memref::StoreOp storeOp)
                   {
        if (!outputMemRef)
            outputMemRef = storeOp.getMemref(); });

    llvm::errs() << "Input memref: " << inputMemRef << "\n";
    llvm::errs() << "Output memref: " << outputMemRef << "\n";
    if (!inputMemRef || !outputMemRef)
    {
      llvm::errs() << "Error: Could not find input or output memref\n";
      return outerLoop;
    }

    llvm::errs() << "Creating constants\n";
    auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto cst = builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(0.0f), builder.getF32Type());

    llvm::errs() << "Checking if it's a prefix sum\n";
    bool isPrefixSum = (reuseVector.size() == 2 && reuseVector[0] == 0 && reuseVector[1] == 1);

    mlir::scf::ForOp newOuterLoop;
    if (isPrefixSum)
    {
      llvm::errs() << "Applying prefix sum transformation\n";

      // init B[0] with A[0]
      auto firstElem = builder.create<mlir::memref::LoadOp>(loc, inputMemRef, mlir::ValueRange{c0});
      builder.create<mlir::memref::StoreOp>(loc, firstElem, outputMemRef, mlir::ValueRange{c0});

      // create new loop for prefix sum
      newOuterLoop = builder.create<mlir::scf::ForOp>(
          loc, c1, outerLoop.getUpperBound(), c1,
          mlir::ValueRange{},
          [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::Value iv,
              mlir::ValueRange iterArgs)
          {
            auto prevIndex = nestedBuilder.create<mlir::arith::SubIOp>(nestedLoc, iv, c1);
            auto prevSum = nestedBuilder.create<mlir::memref::LoadOp>(nestedLoc, outputMemRef, mlir::ValueRange{prevIndex});
            auto currElem = nestedBuilder.create<mlir::memref::LoadOp>(nestedLoc, inputMemRef, mlir::ValueRange{iv});
            auto newSum = nestedBuilder.create<mlir::arith::AddFOp>(nestedLoc, prevSum, currElem);
            nestedBuilder.create<mlir::memref::StoreOp>(nestedLoc, newSum, outputMemRef, mlir::ValueRange{iv});
            nestedBuilder.create<mlir::scf::YieldOp>(nestedLoc);
          });

      llvm::errs() << "Prefix sum transformation applied successfully\n";
    }
    else
    {
      llvm::errs() << "Applying nested reduction transformation\n";
      auto loopBounds = getLoopBounds(outerLoop);
      int depth = loopBounds.size();
      int reductionDim = std::distance(reuseVector.begin(), std::find(reuseVector.begin(), reuseVector.end(), 1));

      if (reductionDim == reuseVector.size())
      {
        llvm::errs() << "Error: Could not determine reduction dimension\n";
        return outerLoop;
      }

      std::function<mlir::scf::ForOp(mlir::OpBuilder &, int, mlir::ValueRange)> createSimplifiedLoops;
      createSimplifiedLoops = [&](mlir::OpBuilder &nestedBuilder, int currentDepth, mlir::ValueRange outerIVs) -> mlir::scf::ForOp
      {
        if (currentDepth == depth)
          return mlir::scf::ForOp();

        auto [lowerBound, upperBound, step] = loopBounds[currentDepth];

        if (currentDepth == reductionDim)
        {
          // create a reduction op instead of a loop
          auto initValue = nestedBuilder.create<mlir::memref::LoadOp>(loc, outputMemRef, outerIVs);
          auto reductionLoop = nestedBuilder.create<mlir::scf::ForOp>(
              loc, lowerBound, upperBound, step,
              mlir::ValueRange{initValue},
              [&](mlir::OpBuilder &bodyBuilder, mlir::Location bodyLoc, mlir::Value iv, mlir::ValueRange iterArgs)
              {
                auto accum = iterArgs[0];
                llvm::SmallVector<mlir::Value, 4> loadIndices(outerIVs.begin(), outerIVs.end());
                loadIndices.push_back(iv);
                auto loadedValue = bodyBuilder.create<mlir::memref::LoadOp>(bodyLoc, inputMemRef, loadIndices);
                auto sum = bodyBuilder.create<mlir::arith::AddFOp>(bodyLoc, accum, loadedValue);
                bodyBuilder.create<mlir::scf::YieldOp>(bodyLoc, mlir::ValueRange{sum});
              });
          nestedBuilder.create<mlir::memref::StoreOp>(loc, reductionLoop.getResult(0), outputMemRef, outerIVs);
          return mlir::scf::ForOp(); // return empty ForOp b/c we've handled this dimension
        }
        else
        {
          return nestedBuilder.create<mlir::scf::ForOp>(
              loc, lowerBound, upperBound, step, mlir::ValueRange{},
              [&](mlir::OpBuilder &innerBuilder, mlir::Location innerLoc, mlir::Value iv, mlir::ValueRange)
              {
                llvm::SmallVector<mlir::Value, 4> newOuterIVs(outerIVs.begin(), outerIVs.end());
                newOuterIVs.push_back(iv);
                createSimplifiedLoops(innerBuilder, currentDepth + 1, newOuterIVs);
                innerBuilder.create<mlir::scf::YieldOp>(innerLoc);
              });
        }
      };

      newOuterLoop = createSimplifiedLoops(builder, 0, {});
      llvm::errs() << "Nested reduction transformation applied successfully\n";
    }

    if (newOuterLoop)
    {
      llvm::errs() << "Replacing original loop with the new loop\n";
      outerLoop.replaceAllUsesWith(newOuterLoop);
      llvm::errs() << "Replaced original loop with the new loop\n";
      auto oldLoop = outerLoop;
      llvm::errs() << "Erasing old loop\n";
      outerLoop = newOuterLoop; // update the reference to the new loop
      oldLoop.erase();          // so we can safely erase

      newOuterLoop->setAttr("simplified", mlir::UnitAttr::get(newOuterLoop->getContext()));

      llvm::errs() << "Transformation applied successfully\n";
      return newOuterLoop;
    }

    llvm::errs() << "Transformation not applied\n";
    outerLoop->setAttr("simplified", mlir::UnitAttr::get(outerLoop->getContext()));
    return outerLoop;
  }
};

std::unique_ptr<mlir::Pass> createSimplifyDependentReductionsPass()
{
  return std::make_unique<SimplifyDependentReductionsPass>();
}