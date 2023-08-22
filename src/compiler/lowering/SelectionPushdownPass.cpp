#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneInferFrameLabelsOpInterface.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <utility>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace mlir;

namespace
{
    struct SelectionPushdown : public RewritePattern{

        SelectionPushdown(MLIRContext * context, PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context)
        {}

        LogicalResult matchAndRewrite(
            Operation *op,
            PatternRewriter &rewriter
        ) const override
        {
            if(llvm::dyn_cast<mlir::daphne::FilterRowOp>(op)){
                //iterate through the second operand of the filterrowop
                std::vector<Operation *> comparisons;
                Operation * currentBitmap = op->getOperand(1).getDefiningOp();
                while (llvm::dyn_cast<mlir::daphne::EwAndOp>(currentBitmap) || llvm::dyn_cast<mlir::daphne::EwOrOp>(currentBitmap)) {
                    Operation * comparison = currentBitmap->getOperand(1).getDefiningOp();
                    comparisons.push_back(comparison);
                    currentBitmap = currentBitmap->getOperand(0).getDefiningOp();
                }
                comparisons.push_back(currentBitmap);
                
            }
            return success();
        }
    };

    struct SelectionPushdownPass : public PassWrapper<SelectionPushdownPass, OperationPass<ModuleOp>> {
    
    void runOnOperation() final;
    
    };
}

void SelectionPushdownPass::runOnOperation() {
    auto module = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();

    target.addDynamicallyLegalOp<mlir::daphne::FilterRowOp>([&](Operation * op) {
        Operation * dataSource = op->getOperand(0).getDefiningOp();
        if (llvm::dyn_cast<mlir::daphne::InnerJoinOp>(dataSource)) {
            //Currently not checking whether all comparisons beforehand are possible on the available frames without join
            return false;
        }
        
        return true;
    });

    patterns.add<SelectionPushdown>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
    
}

std::unique_ptr<Pass> daphne::createSelectionPushdownPass()
{
    return std::make_unique<SelectionPushdownPass>();
}