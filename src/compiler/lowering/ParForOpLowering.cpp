#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

class ParForOpLoweringPattern : public RewritePattern {
  public:
    ParForOpLoweringPattern(MLIRContext *context) : RewritePattern("daphne.parfor", 1, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!isa<daphne::ParForOp>(op))
            return failure();

        auto parForOp = cast<daphne::ParForOp>(op);
        auto module = op->getParentOfType<ModuleOp>();
        if (!module) {
            op->emitError("Expected to find a parent module");
            return failure();
        }
        rewriter.startRootUpdate(parForOp);
        static int idx = 0;
        std::string funcName = "parfor_body_" + std::to_string(idx++);
        Location loc = op->getLoc();

        // Save insertion point and move to module start
        auto ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToStart(module.getBody());

        // Create new function type (void())
        auto funcType = rewriter.getFunctionType({}, {});
        auto funcOp = rewriter.create<func::FuncOp>(loc, funcName, funcType);
        
        // Move loop body into new function
        auto &funcBlock = funcOp.getBody().emplaceBlock();
        for (auto arg : parForOp.getBodyStmt().front().getArguments()) {
            funcBlock.addArgument(arg.getType(), arg.getLoc());
        }

        // Move operations into the function
        funcBlock.getOperations().splice(funcBlock.end(), parForOp.getBodyStmt().front().getOperations());

        // add block terminator for validity of parfor op
        rewriter.setInsertionPointToStart(&parForOp.getBodyStmt().front());
        rewriter.create<daphne::ReturnOp>(loc);

        // Ensure there's a terminator
        // rewriter.setInsertionPointToEnd(&funcBlock);
        // rewriter.create<func::ReturnOp>(loc);
        // Restore insertion point
        rewriter.restoreInsertionPoint(ip);
        
        // Create function pointer and assign it to parfor's func attribute
        auto symbolRef = SymbolRefAttr::get(rewriter.getContext(), funcOp.getName());
        parForOp.setFuncNameAttr(symbolRef);
        funcOp.setVisibility(SymbolTable::Visibility::Public);
        rewriter.finalizeRootUpdate(parForOp);
        return success();
    }
};

namespace {
struct ParForLoweringPass : public PassWrapper<ParForLoweringPass, OperationPass<ModuleOp>> {
    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        patterns.add<ParForOpLoweringPattern>(&getContext());

        ConversionTarget target(getContext());
       
        target.addLegalDialect<func::FuncDialect, daphne::DaphneDialect>();
        target.addDynamicallyLegalOp<daphne::ParForOp>([](daphne::ParForOp op) { return op.getFuncName().has_value(); });

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // end anonymous namespace

std::unique_ptr<Pass> daphne::createParForOpLoweringPass() { return std::make_unique<ParForLoweringPass>(); }