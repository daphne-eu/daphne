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

    Value dctx;
  
public:
    ParForOpLoweringPattern(MLIRContext *context, Value dctx) : RewritePattern("daphne.parfor", 1, context), dctx(dctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!isa<daphne::ParForOp>(op))
            return failure();

        auto parForOp = cast<daphne::ParForOp>(op);
        rewriter.startRootUpdate(parForOp);
        
        Location loc = op->getLoc();

        // todo: move and remove  
        static int idx = 0;
        std::string funcName = "parfor_body_" + std::to_string(idx++);
        auto symbolRef = SymbolRefAttr::get(rewriter.getContext(), funcName);
        parForOp.setFuncNameAttr(symbolRef);
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

        target.addLegalDialect<func::FuncDialect, daphne::DaphneDialect, memref::MemRefDialect, arith::ArithDialect>();
        target.addDynamicallyLegalOp<daphne::ParForOp>(
            [](daphne::ParForOp op) { return op.getFuncName().has_value(); });

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // end anonymous namespace

std::unique_ptr<Pass> daphne::createParForOpLoweringPass() { return std::make_unique<ParForLoweringPass>(); }