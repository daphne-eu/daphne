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
#include <vector>

using namespace mlir;

namespace
{
    struct ColumnarOpOptimization : public RewritePattern{

        ColumnarOpOptimization(MLIRContext * context, PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context)
        {}

        LogicalResult matchAndRewrite(
            Operation *op,
            PatternRewriter &rewriter
        ) const override
        {
            if (llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(op)) {
                Operation * prevOp = op->getOperand(0).getDefiningOp();
                std::vector<mlir::Value> posLists{op->getOperand(1)};
                for (mlir::Value posList : prevOp->getOperands().drop_front(1)) {
                    posLists.push_back(posList);
                }
                auto projectionPathOp = rewriter.replaceOpWithNewOp<daphne::ColumnProjectionPathOp>(op, op->getResult(0).getType(), prevOp->getOperand(0), posLists);
                op->getResult(0).replaceAllUsesWith(projectionPathOp->getResult(0));
                return success();
            } else if (llvm::dyn_cast<mlir::daphne::ColumnIntersectOp>(op)) {
                Operation * rightOp = op->getOperand(1).getDefiningOp();
                mlir::Value rightOpSource = rightOp->getOperand(0);
                Operation * geOp = nullptr;
                Operation * leOp = nullptr;
                Operation * leftPrevIntersectOp = op;
                
                Operation * leftOp = op->getOperand(0).getDefiningOp();
                while (llvm::dyn_cast<mlir::daphne::ColumnIntersectOp>(leftOp)) {
                    leftPrevIntersectOp = leftOp;
                    Operation * rightTempOp = leftOp->getOperand(1).getDefiningOp();
                    if (llvm::dyn_cast<mlir::daphne::ColumnGeOp>(rightTempOp) && llvm::dyn_cast<mlir::daphne::ColumnLeOp>(rightOp) && rightOpSource == rightTempOp->getOperand(0)) {
                        geOp = rightTempOp;
                        leOp = rightOp;
                        break;
                    } else if (llvm::dyn_cast<mlir::daphne::ColumnLeOp>(rightTempOp) && llvm::dyn_cast<mlir::daphne::ColumnGeOp>(rightOp) && rightOpSource == rightTempOp->getOperand(0)) {
                        leOp = rightTempOp;
                        geOp = rightOp;
                        break;
                    }
                    leftOp = leftOp->getOperand(0).getDefiningOp();
                }
                if (llvm::dyn_cast<mlir::daphne::ColumnGeOp>(leftOp) && llvm::dyn_cast<mlir::daphne::ColumnLeOp>(rightOp) && rightOpSource == leftOp->getOperand(0)) {
                    geOp = leftOp;
                    leOp = rightOp;
                } else if (llvm::dyn_cast<mlir::daphne::ColumnLeOp>(leftOp) && llvm::dyn_cast<mlir::daphne::ColumnGeOp>(rightOp) && rightOpSource == leftOp->getOperand(0)) {
                    leOp = leftOp;
                    geOp = rightOp;
                }

                if (geOp == nullptr || leOp == nullptr) {
                    return failure();
                }
                rewriter.setInsertionPointAfter(rightOp);
                auto betweenOp = rewriter.create<daphne::ColumnBetweenOp>(rightOp->getLoc(), rightOp->getResult(0).getType(), leOp->getOperand(0), geOp->getOperand(1), leOp->getOperand(1));
                
                rewriter.startRootUpdate(op);
                //Both operations share the same intersect
                if (op == leftPrevIntersectOp) {
                    op->getResult(0).replaceAllUsesWith(betweenOp->getResult(0));
                } else {
                    op->replaceUsesOfWith(rightOp->getResult(0), betweenOp->getResult(0));
                    for (mlir::Value intersectOperands: leftPrevIntersectOp->getOperands()) {
                        auto intersectOperandsOp = intersectOperands.getDefiningOp();
                        if (intersectOperandsOp != geOp && intersectOperandsOp != leOp) {
                            leftPrevIntersectOp->getResult(0).replaceAllUsesWith(intersectOperandsOp->getResult(0));
                        } 
                    }
                }
                rewriter.finalizeRootUpdate(op);

                return success();
            }           
        }
    };

    struct OptimizeColumnarOpPass : public PassWrapper<OptimizeColumnarOpPass, OperationPass<func::FuncOp>> {

        OptimizeColumnarOpPass() = default;
    
        void runOnOperation() final;
    
    };
}

void OptimizeColumnarOpPass::runOnOperation() {
    auto module = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();
    target.addDynamicallyLegalOp<mlir::daphne::ColumnProjectOp>([&](Operation *op) {
        Operation * prevOp = op->getOperand(0).getDefiningOp();
        if (llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(prevOp)) {
            return false;
        } else if (llvm::dyn_cast<mlir::daphne::ColumnProjectionPathOp>(op)) {
            return false;
        }
        return true;
    });

    target.addDynamicallyLegalOp<mlir::daphne::ColumnIntersectOp>([](Operation *op) {
        if (op->use_empty()) {
            return true;
        }
        Operation * rightOp = op->getOperand(1).getDefiningOp();
        mlir::Value rightOpSource = rightOp->getOperand(0);
        Operation * partnerOp = nullptr;
        if (llvm::dyn_cast<mlir::daphne::ColumnLeOp>(rightOp) || llvm::dyn_cast<mlir::daphne::ColumnGeOp>(rightOp)) {
            Operation * leftOp = op->getOperand(0).getDefiningOp();
            while (llvm::dyn_cast<mlir::daphne::ColumnIntersectOp>(leftOp)) {
                Operation * rightTempOp = leftOp->getOperand(1).getDefiningOp();
                if (llvm::dyn_cast<mlir::daphne::ColumnGeOp>(rightTempOp) && llvm::dyn_cast<mlir::daphne::ColumnLeOp>(rightOp) && rightOpSource == rightTempOp->getOperand(0)) {
                    partnerOp = rightTempOp;
                    break;
                } else if (llvm::dyn_cast<mlir::daphne::ColumnLeOp>(rightTempOp) && llvm::dyn_cast<mlir::daphne::ColumnGeOp>(rightOp) && rightOpSource == rightTempOp->getOperand(0)) {
                    partnerOp = rightTempOp;
                    break;
                }
                leftOp = leftOp->getOperand(0).getDefiningOp();
            }
            if (llvm::dyn_cast<mlir::daphne::ColumnGeOp>(leftOp) && llvm::dyn_cast<mlir::daphne::ColumnLeOp>(rightOp) && rightOpSource == leftOp->getOperand(0)) {
                partnerOp = leftOp;
            } else if (llvm::dyn_cast<mlir::daphne::ColumnLeOp>(leftOp) && llvm::dyn_cast<mlir::daphne::ColumnGeOp>(rightOp) && rightOpSource == leftOp->getOperand(0)) {
                partnerOp = leftOp;
            }
        }

        if (partnerOp != nullptr) {
            return false;
        } else {
            return true;
        }
    }); 

    patterns.add<ColumnarOpOptimization>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();  
}

std::unique_ptr<Pass> daphne::createOptimizeColumnarOpPass()
{
    return std::make_unique<OptimizeColumnarOpPass>();
}

