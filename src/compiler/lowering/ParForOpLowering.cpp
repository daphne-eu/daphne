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
#include <mlir/Dialect/SCF/IR/SCF.h>

using namespace mlir;

// TODO: plz rename me 
// TODO: Probably RewritePattern is not a good fit here, but idk 
class ParForOpLoweringPattern : public RewritePattern {

public:
    ParForOpLoweringPattern(MLIRContext *context) : RewritePattern("daphne.parfor", 1, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!isa<daphne::ParForOp>(op))
            return failure();

        auto parForOp = cast<daphne::ParForOp>(op);
        Location loc = op->getLoc();
        
        auto candidates = parForOp.getResults();
        for(auto c : candidates) {
            // always true if scalar
            if (!c.getType().isa<mlir::ShapedType>() || c.getType().isa<daphne::UnknownType>())
                throw ErrorHandler::compilerError(
                    c.getDefiningOp(), "ParForOpLoweringPattern",
                    "Boom !!! dependency analysis for ParForOp failed. ");                    
            
            // always true if complex object mutated completely, in instance matrix multiplication. 
            if(c.getType().isa<daphne::MatrixType>()) 
                throw ErrorHandler::compilerError(
                    c.getDefiningOp(), "ParForOpLoweringPattern",
                    "Boom !!! dependency analysis for ParForOp failed.");
            // TODO: add other checks based only on type of candidate. 
            
            checkCandidateAgainstBlock(&parForOp.getBodyStmt().getBlocks().front(), c);
        }
        return success();
    }

private:
    /**
     * @brief Checks if the given candidates are valid for the block. 
     * Recursively checks all blocks of ParForOp. 
     */
    void checkCandidateAgainstBlock(Block *b, mlir::OpResult candidate) const {
        for(Operation &op : b->getOperations()) {
            // if operand is nested go into it 
            // otherwise check the if there is dependency
            if(auto ifOp = dyn_cast<mlir::scf::IfOp>(op)) {
                for(auto &thenBlock : ifOp.getThenRegion().getBlocks()) {
                    checkCandidateAgainstBlock(&thenBlock, candidate);
                }
                for(auto &elseBlock : ifOp.getElseRegion().getBlocks()) {
                    checkCandidateAgainstBlock(&elseBlock, candidate);
                }
            } else if(auto whileOp = dyn_cast<mlir::scf::WhileOp>(op)) {
                for(auto &bodyBlock : whileOp.getBefore().getBlocks()) {
                    checkCandidateAgainstBlock(&bodyBlock, candidate);
                }
                for(auto &bodyBlock : whileOp.getAfter().getBlocks()) {
                    checkCandidateAgainstBlock(&bodyBlock, candidate);
                }
            } else if(auto forOp = dyn_cast<mlir::scf::ForOp>(op)) {
                for(auto &bodyBlock : forOp.getLoopBody()) {
                    checkCandidateAgainstBlock(&bodyBlock, candidate);
                }
            } else if (auto parForOp = dyn_cast<daphne::ParForOp>(op)) {
                for(auto &bodyBlock : parForOp.getBodyStmt().getBlocks()) {
                    checkCandidateAgainstBlock(&bodyBlock, candidate);
                }
            } else {
                // TODO: check candidate against operation for inter-iterational dependencies
                // 1) check output dependency GCD/Banjee or other method 
                // 2) check true/anti dependencies 
            }
        }
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