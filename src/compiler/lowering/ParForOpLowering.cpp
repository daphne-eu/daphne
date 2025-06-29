#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <mlir/Transforms/RegionUtils.h>
#include "compiler/utils/CompilerUtils.h"

using namespace mlir;

class ParForOpLoweringPattern : public RewritePattern {
  
public:
    ParForOpLoweringPattern(MLIRContext *context) : RewritePattern("daphne.parfor", 1, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!isa<daphne::ParForOp>(op))
            return failure();

        auto parForOp = cast<daphne::ParForOp>(op);
        rewriter.startRootUpdate(parForOp);
        // ********************************************************************
        // Loop dependency analysis: 1. Determinate dependency candidates 
        // ********************************************************************
        Location loc = op->getLoc();
        //auto candidates = parForOp.getOutputs();
        
        rewriter.finalizeRootUpdate(parForOp);
        return success();
    }
};

namespace {
struct ParForLoweringPass : public PassWrapper<ParForLoweringPass, OperationPass<func::FuncOp>> {
    void runOnOperation() final {
        func::FuncOp func = getOperation();
        RewritePatternSet patterns(&getContext());
        patterns.add<ParForOpLoweringPattern>(&getContext());

        ConversionTarget target(getContext());

        target.addLegalDialect<daphne::DaphneDialect>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // end anonymous namespace

std::unique_ptr<Pass> daphne::createParForOpLoweringPass() { return std::make_unique<ParForLoweringPass>(); }