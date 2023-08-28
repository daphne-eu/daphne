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
    std::string getFrameName(Operation * constantOp) {
        std::string frameName = "";
        if (llvm::dyn_cast<mlir::daphne::ConstantOp>(constantOp)) {
            frameName = constantOp->getAttrOfType<StringAttr>("value").getValue().str();
        } else {
            throw std::runtime_error("getFrameName: constantOp is not a FrameOp or ConstantOp");
        }
        return frameName.substr(0, frameName.find("."));
    }

    bool checkIfBitmapOp(Operation * op) {
        if (llvm::dyn_cast<mlir::daphne::EwLtOp>(op) 
        || llvm::dyn_cast<mlir::daphne::EwGtOp>(op)
        || llvm::dyn_cast<mlir::daphne::EwEqOp>(op)
        || llvm::dyn_cast<mlir::daphne::EwNeqOp>(op)
        || llvm::dyn_cast<mlir::daphne::EwLeOp>(op)
        || llvm::dyn_cast<mlir::daphne::EwGeOp>(op)) {
            return true;
        }
        return false;
    }
    Operation * getCompareFunctionExtractColOp(Operation * op) {
        Operation * possibleCmpOp = op;
        if (llvm::dyn_cast<mlir::daphne::CastOp>(op)) {
            possibleCmpOp = op->getOperand(0).getDefiningOp()       // CreateFrameOp
                            ->getOperand(0).getDefiningOp();        // Possible BitmapOp
        }
        if (checkIfBitmapOp(possibleCmpOp)) {
            return possibleCmpOp->getOperand(0).getDefiningOp()    // CastOp
                    ->getOperand(0).getDefiningOp();    // ExtractColOp
        }
        return nullptr;
    };
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
                Operation * joinSourceOp = op->getOperand(0).getDefiningOp();
                Operation * inputFrameOpLhs = joinSourceOp->getOperand(0).getDefiningOp();
                Operation * inputFrameOpRhs = joinSourceOp->getOperand(1).getDefiningOp();
                std::string inputColumnNameLhs = inputFrameOpLhs->getResult(0).getType().cast<mlir::daphne::FrameType>().getLabels()->at(0);
                std::string inputColumnNameRhs = inputFrameOpRhs->getResult(0).getType().cast<mlir::daphne::FrameType>().getLabels()->at(0);
                std::string inputFrameNameLhs = inputColumnNameLhs.substr(0, inputColumnNameLhs.find("."));
                std::string inputFrameNameRhs = inputColumnNameRhs.substr(0, inputColumnNameRhs.find("."));

                OpResult lhsFrameJoin = inputFrameOpLhs->getResult(0);
                OpResult rhsFrameJoin = inputFrameOpRhs->getResult(0);

                Operation * currentBitmap = op->getOperand(1).getDefiningOp();
                while (llvm::dyn_cast<mlir::daphne::EwAndOp>(currentBitmap)) {
                    Operation * comparison = currentBitmap->getOperand(1).getDefiningOp();
                    comparisons.push_back(comparison);
                    currentBitmap = currentBitmap->getOperand(0).getDefiningOp();
                }
                comparisons.push_back(currentBitmap); 

                //update sources of the comparisons
                for (Operation * comparison : comparisons) {
                    Operation * compareFunctionExtractColOp = getCompareFunctionExtractColOp(comparison);
                    if (compareFunctionExtractColOp) {
                        std::string currentFrameName = getFrameName(compareFunctionExtractColOp->getOperand(1).getDefiningOp());
                        if (currentFrameName == inputFrameNameLhs) {
                            compareFunctionExtractColOp->setOperand(0, inputFrameOpLhs->getResult(0));
                            op->setOperand(0, inputFrameOpLhs->getResult(0));
                            lhsFrameJoin = op->getResult(0);
                        } else if (currentFrameName == inputFrameNameRhs) {
                            compareFunctionExtractColOp->setOperand(0, inputFrameOpRhs->getResult(0));
                            op->setOperand(0, inputFrameOpRhs->getResult(0));
                            rhsFrameJoin = op->getResult(0);
                        } else {
                            throw std::runtime_error("SelectionPushdown: frame name not found");
                        }
                    }
                }
                //rewire filterRowOp and InnerJoinOp
                rewriter.startRootUpdate(op);
                op->getResult(0).replaceAllUsesWith(joinSourceOp->getResult(0));
                joinSourceOp->setOperand(0, lhsFrameJoin);
                joinSourceOp->setOperand(1, rhsFrameJoin);
                joinSourceOp->moveAfter(op);
                rewriter.finalizeRootUpdate(op);
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