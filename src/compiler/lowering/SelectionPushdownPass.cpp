#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneInferFrameLabelsOpInterface.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"
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
    std::string getColumnName(Operation * constantOp) {
        std::string columnName = "";
        if (llvm::dyn_cast<mlir::daphne::ConstantOp>(constantOp)) {
            columnName = constantOp->getAttrOfType<StringAttr>("value").getValue().str();
        } else {
            throw std::runtime_error("getFrameName: constantOp is not a FrameOp or ConstantOp");
        }
        return columnName;
    }

    mlir::daphne::FilterRowOp restructureComparisonsBeforeJoin(PatternRewriter &rewriter, std::vector<Operation *> comparisons, Operation * join, OpResult joinData) {
        Operation * tempOp = nullptr;
        for (Operation * comparison : comparisons) {
            if (tempOp == nullptr) {
                tempOp = comparison;
            } else {
                auto newEwAndOp = rewriter.create<mlir::daphne::EwAndOp>(comparison->getLoc(), comparison->getResult(0).getType(), tempOp->getResult(0), comparison->getResult(0));
                tempOp = newEwAndOp;
            }
        }
        auto newFilterRowOp = rewriter.create<mlir::daphne::FilterRowOp>(tempOp->getLoc(), joinData.getType(), joinData, tempOp->getResult(0));
        join->replaceUsesOfWith(joinData, newFilterRowOp->getResult(0));

        return newFilterRowOp;
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
    Operation * getCompareFunctionExtractColOp(Operation * op, size_t sourceOpIndex) {
        Operation * possibleCmpOp = op;
        if (llvm::dyn_cast<mlir::daphne::CastOp>(op)) {
            possibleCmpOp = op->getOperand(0).getDefiningOp()                   // CreateFrameOp
                            ->getOperand(0).getDefiningOp();                    // Possible BitmapOp
        }
        if (checkIfBitmapOp(possibleCmpOp)) {
            if (llvm::dyn_cast<mlir::daphne::ConstantOp>(possibleCmpOp->getOperand(sourceOpIndex).getDefiningOp())) {
                return nullptr;
            }
            return possibleCmpOp->getOperand(sourceOpIndex).getDefiningOp()     // CastOp
                    ->getOperand(0).getDefiningOp();                            // ExtractColOp
        }
        return nullptr;
    };

    bool comparisonRhsSameFrame(Operation * comparison) {
        auto lhsExtract = getCompareFunctionExtractColOp(comparison, 0);
        auto rhsExtract = getCompareFunctionExtractColOp(comparison, 1);
        auto lhsExtractColumnName = getColumnName(lhsExtract->getOperand(1).getDefiningOp());
        auto rhsExtractColumnName = getColumnName(rhsExtract->getOperand(1).getDefiningOp());
        // We support the pushdown of a comparison between columns if they are on the same frame
        if (lhsExtractColumnName.substr(0, lhsExtractColumnName.find(".")) == rhsExtractColumnName.substr(0, rhsExtractColumnName.find("."))) {
            return true;
        }
        return false;
    }

    mlir::LogicalResult moveComparisonsBeforeJoin(PatternRewriter &rewriter, Operation * filterRowOp, std::vector<Operation *> comparisons, Operation * firstFilterRowOp) {
        Operation * joinSourceOp = filterRowOp->getOperand(0).getDefiningOp();
        Operation * inputFrameOpLhs = joinSourceOp->getOperand(0).getDefiningOp();
        Operation * inputFrameOpRhs = joinSourceOp->getOperand(1).getDefiningOp();
        std::vector<std::string> * inputColumnNamesLhs = inputFrameOpLhs->getResult(0).getType().cast<mlir::daphne::FrameType>().getLabels();
        std::vector<std::string> * inputColumnNamesRhs = inputFrameOpRhs->getResult(0).getType().cast<mlir::daphne::FrameType>().getLabels();
        //std::string inputFrameNameLhs = inputColumnNameLhs.substr(0, inputColumnNameLhs.find("."));
        //std::string inputFrameNameRhs = inputColumnNameRhs.substr(0, inputColumnNameRhs.find("."));

        OpResult lhsFrameJoin = inputFrameOpLhs->getResult(0);
        OpResult rhsFrameJoin = inputFrameOpRhs->getResult(0);

        //update sources of the comparisons
        std::vector<Operation *> lhsComparisons;
        std::vector<Operation *> rhsComparisons;
        for (Operation * comparison : comparisons) {
            Operation * compareFunctionExtractColOpLhs = getCompareFunctionExtractColOp(comparison, 0);
            // Update both operands of the comparison if they are on the same frame
            Operation * compareFunctionExtractColOpRhs = getCompareFunctionExtractColOp(comparison, 1);
            if (compareFunctionExtractColOpLhs) {
                std::string currentName = getColumnName(compareFunctionExtractColOpLhs->getOperand(1).getDefiningOp());
                if ((std::find(inputColumnNamesLhs->begin(), inputColumnNamesLhs->end(), currentName) != inputColumnNamesLhs->end())) {
                    compareFunctionExtractColOpLhs->setOperand(0, inputFrameOpLhs->getResult(0));
                    if (compareFunctionExtractColOpRhs) {
                        compareFunctionExtractColOpRhs->setOperand(0, inputFrameOpLhs->getResult(0));
                    }
                    lhsComparisons.push_back(comparison);
                } else if ((std::find(inputColumnNamesRhs->begin(), inputColumnNamesRhs->end(), currentName) != inputColumnNamesRhs->end())) {
                    compareFunctionExtractColOpLhs->setOperand(0, inputFrameOpRhs->getResult(0));
                    if (compareFunctionExtractColOpRhs) {
                        compareFunctionExtractColOpRhs->setOperand(0, inputFrameOpRhs->getResult(0));
                    }
                    rhsComparisons.push_back(comparison);
                } else {
                    throw std::runtime_error("SelectionPushdown: frame name not found");
                }
            }
        }
        mlir::daphne::FilterRowOp lhsFilterRowOp = nullptr;
        if (!lhsComparisons.empty()) {
            lhsFilterRowOp = restructureComparisonsBeforeJoin(rewriter, lhsComparisons, joinSourceOp, lhsFrameJoin);         
        }
        mlir::daphne::FilterRowOp rhsFilterRowOp = nullptr;
        if (!rhsComparisons.empty()) {
            rhsFilterRowOp = restructureComparisonsBeforeJoin(rewriter, rhsComparisons, joinSourceOp, rhsFrameJoin);
        }

        //rewire filterRowOp and InnerJoinOp
        rewriter.startRootUpdate(filterRowOp);
        filterRowOp->getResult(0).replaceAllUsesWith(joinSourceOp->getResult(0));
        //filterRowOp->moveAfter(firstFilterRowOp);
        joinSourceOp->moveAfter(firstFilterRowOp);
        rewriter.finalizeRootUpdate(filterRowOp);

        if(llvm::dyn_cast<mlir::daphne::InnerJoinOp>(inputFrameOpLhs) && lhsFilterRowOp) {
            moveComparisonsBeforeJoin(rewriter, lhsFilterRowOp, lhsComparisons, firstFilterRowOp);
        }
        if(llvm::dyn_cast<mlir::daphne::InnerJoinOp>(inputFrameOpRhs) && rhsFilterRowOp) {
            moveComparisonsBeforeJoin(rewriter, rhsFilterRowOp, rhsComparisons, firstFilterRowOp);
        }

        rewriter.eraseOp(filterRowOp);

        return success();
    }

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
                while (llvm::dyn_cast<mlir::daphne::EwAndOp>(currentBitmap)) {
                    Operation * comparison = currentBitmap->getOperand(1).getDefiningOp();
                    comparisons.push_back(comparison);
                    currentBitmap = currentBitmap->getOperand(0).getDefiningOp();
                }
                comparisons.push_back(currentBitmap); 
                
                return moveComparisonsBeforeJoin(rewriter, op, comparisons, op);
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
            // Check if all operations beforehand are ConstantOps (Pushdown of Comparisons between Columns not supported yet)
            std::vector<Operation *> comparisons;

            Operation * currentBitmap = op->getOperand(1).getDefiningOp();       
            while (llvm::dyn_cast<mlir::daphne::EwAndOp>(currentBitmap)) {
                Operation * comparison = currentBitmap->getOperand(1).getDefiningOp()       // CastOp
                                        ->getOperand(0).getDefiningOp()                     // CreateFrameOp
                                        ->getOperand(0).getDefiningOp();                    // BitmapOp;
                if (!llvm::dyn_cast<mlir::daphne::ConstantOp>(comparison->getOperand(1).getDefiningOp())) {
                    if (!comparisonRhsSameFrame(comparison)) {
                        return true;
                    }
                }
                comparisons.push_back(comparison);
                currentBitmap = currentBitmap->getOperand(0).getDefiningOp();
            }
            // Pushdown of OR not supported yet
            if (llvm::dyn_cast<mlir::daphne::EwOrOp>(currentBitmap)) {
                return true;
            }
            auto comparison = currentBitmap->getOperand(0).getDefiningOp()       // CreateFrameOp
                                        ->getOperand(0).getDefiningOp();                      // BitmapOp;
            if (!llvm::dyn_cast<mlir::daphne::ConstantOp>(comparison->getOperand(1).getDefiningOp())) {
                if (!comparisonRhsSameFrame(comparison)) {
                    return true;
                }
            }
            comparisons.push_back(currentBitmap); 
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