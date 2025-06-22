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
        // Parfor may have dependencies to daphne context.
        // Since the block of this op is IsolatedFromAbove, we need to add daphne context as block argument 
        // and replace usages of context pointer with block argument respectivly. 
        // The context pointer becomes just a part of parfor body function arguments.   
        mlir::Block &entryBlock = parForOp.getRegion().front();
        mlir::Value dctxArg = entryBlock.addArgument(dctx.getType(), dctx.getLoc());
       auto args = llvm::SmallVector<Value>(parForOp.getArgs());
        args.push_back(dctx);
        parForOp.getArgsMutable().assign(args);
        //parForOp.getArgsMutable().assign(dctx);
        static int idx = 0;
        std::string funcName = "parfor_body_" + std::to_string(idx++);
        Location loc = op->getLoc();

        // Create function pointer and assign it to parfor's func attribute
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
        auto dctx = CompilerUtils::getDaphneContext(func);
        patterns.add<ParForOpLoweringPattern>(&getContext(), dctx);

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