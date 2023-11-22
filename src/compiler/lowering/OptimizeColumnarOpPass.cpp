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
    Operation * getCmpSourceOp(Operation * op) {
        auto dataOp = op->getOperand(0).getDefiningOp();
        if(llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(dataOp)) {
            return dataOp;
        } else {
            return dataOp                                     //CastOp
                ->getOperand(0).getDefiningOp();         //ExtractColOp
        }
    }

    template<typename FirstOpType, typename SecondOpType>
    void checkMatchingSourceAndCorrectTypes(
        Operation *& firstOp,
        Operation *& secondOp,
        Operation * leftOp,
        Operation * rightOp,
        mlir::OpResult leftOpSource = nullptr,
        mlir::OpResult rightOpSource = nullptr

    ) {
        if (llvm::dyn_cast<FirstOpType>(leftOp) && llvm::dyn_cast<SecondOpType>(rightOp) && leftOpSource == rightOpSource) {
            firstOp = leftOp;
            secondOp = rightOp;
        }
    }

    // Get the ge and le operations and their corresponding intersect operation
    void getLeAndGeOps(
        Operation * intersectOp,
        Operation *& geOp,
        Operation *& leOp,
        Operation *& leftPrevIntersectOp
    ) {
        Operation * leftOp = intersectOp->getOperand(0).getDefiningOp();
        Operation * rightOp = intersectOp->getOperand(1).getDefiningOp();
        Operation * rightOpSource = getCmpSourceOp(rightOp);
        while (llvm::dyn_cast<mlir::daphne::ColumnIntersectOp>(leftOp)) {
            leftPrevIntersectOp = leftOp;
            Operation * rightTempOp = leftOp->getOperand(1).getDefiningOp();
            Operation * rightTempOpSource = getCmpSourceOp(rightTempOp);
            if (llvm::dyn_cast<mlir::daphne::ExtractColOp>(rightTempOpSource) && llvm::dyn_cast<mlir::daphne::ExtractColOp>(rightOpSource) ) {
                if (rightTempOpSource->getOperand(0) == rightOpSource->getOperand(0) && rightTempOpSource->getOperand(1) == rightOpSource->getOperand(1)) {
                    checkMatchingSourceAndCorrectTypes<mlir::daphne::ColumnGeOp, mlir::daphne::ColumnLeOp>(geOp, leOp, rightTempOp, rightOp);
                    checkMatchingSourceAndCorrectTypes<mlir::daphne::ColumnLeOp, mlir::daphne::ColumnGeOp>(leOp, geOp, rightTempOp, rightOp);
                }
            } else {
                checkMatchingSourceAndCorrectTypes<mlir::daphne::ColumnGeOp, mlir::daphne::ColumnLeOp>(geOp, leOp, rightTempOp, rightOp, rightTempOpSource->getResult(0), rightOpSource->getResult(0));
                checkMatchingSourceAndCorrectTypes<mlir::daphne::ColumnLeOp, mlir::daphne::ColumnGeOp>(leOp, geOp, rightTempOp, rightOp, rightTempOpSource->getResult(0), rightOpSource->getResult(0));
            }

            // If we have found our operations we can stop searching
            if (leOp != nullptr && geOp != nullptr) {
                break;
            }

            leftOp = leftOp->getOperand(0).getDefiningOp();
        }
        auto leftOpSource = getCmpSourceOp(leftOp);
        if (llvm::dyn_cast<mlir::daphne::ExtractColOp>(leftOpSource) && llvm::dyn_cast<mlir::daphne::ExtractColOp>(rightOpSource) ) {
            if (leftOpSource->getOperand(0) == rightOpSource->getOperand(0) && leftOpSource->getOperand(1) == rightOpSource->getOperand(1)) {
                checkMatchingSourceAndCorrectTypes<mlir::daphne::ColumnGeOp, mlir::daphne::ColumnLeOp>(geOp, leOp, leftOp, rightOp);
                checkMatchingSourceAndCorrectTypes<mlir::daphne::ColumnLeOp, mlir::daphne::ColumnGeOp>(leOp, geOp, leftOp, rightOp);
            }
        } else {
            checkMatchingSourceAndCorrectTypes<mlir::daphne::ColumnGeOp, mlir::daphne::ColumnLeOp>(geOp, leOp, leftOp, rightOp, leftOpSource->getResult(0), rightOpSource->getResult(0));
            checkMatchingSourceAndCorrectTypes<mlir::daphne::ColumnLeOp, mlir::daphne::ColumnGeOp>(leOp, geOp, leftOp, rightOp, leftOpSource->getResult(0), rightOpSource->getResult(0));
        }

    }

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
                // Gather all position lists for the projection path
                std::vector<mlir::Value> posLists{op->getOperand(1)};
                for (mlir::Value posList : prevOp->getOperands().drop_front(1)) {
                    posLists.push_back(posList);
                }
                // Replae the source operation with a new projection path operation with the position lists and the data input of the previous operation
                auto projectionPathOp = rewriter.replaceOpWithNewOp<daphne::ColumnProjectionPathOp>(op, op->getResult(0).getType(), prevOp->getOperand(0), posLists);
                op->getResult(0).replaceAllUsesWith(projectionPathOp->getResult(0));

                return success();
            } else if (llvm::dyn_cast<mlir::daphne::ColumnIntersectOp>(op)) {
                Operation * rightOp = op->getOperand(1).getDefiningOp();
                Operation * geOp = nullptr;
                Operation * leOp = nullptr;
                Operation * leftPrevIntersectOp = op;
                
                getLeAndGeOps(op, geOp, leOp, leftPrevIntersectOp);

                // Sanity check, should never occur due to the definition if the dynamic legality check
                if (geOp == nullptr || leOp == nullptr) {
                    return failure();
                }

                auto betweenOp = rewriter.create<daphne::ColumnBetweenOp>(rightOp->getLoc(), rightOp->getResult(0).getType(), leOp->getOperand(0), geOp->getOperand(1), leOp->getOperand(1));
                betweenOp->moveBefore(op);
                
                rewriter.startRootUpdate(op);
                if (op == leftPrevIntersectOp) {
                    // Both operations share the same intersect and the intersect will be deleted in later passes
                    op->getResult(0).replaceAllUsesWith(betweenOp->getResult(0));
                } else {
                    // The operations do not share a intersect. We have to rewire one operation from a previous intersect that will be deleted later
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
        // According to benchmark results we only introduce a ProjectionPathOp if the result of the first projection 
        // is not used by anything other than a projection
        if (llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(prevOp)) {
            for (Operation * prevOpUser : prevOp->getResult(0).getUsers()) {
                if (!llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(prevOpUser)) {
                    return true;
                }
            }
            return false;
        } else if (llvm::dyn_cast<mlir::daphne::ColumnProjectionPathOp>(prevOp)) {
            for (Operation * prevOpUser : prevOp->getResult(0).getUsers()) {
                if (!llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(prevOpUser)) {
                    return true;
                }
            }
            return false;
        }
        return true;
    });

    target.addDynamicallyLegalOp<mlir::daphne::ColumnIntersectOp>([](Operation *op) {
        // Between was introduced but operation will get deleted in later passes
        if (op->use_empty()) {
            return true;
        }

        Operation * rightOp = op->getOperand(1).getDefiningOp();
        Operation * geOp = nullptr;
        Operation * leOp = nullptr;
        Operation * leftPrevIntersectOp = op;
        if (llvm::dyn_cast<mlir::daphne::ColumnLeOp>(rightOp) || llvm::dyn_cast<mlir::daphne::ColumnGeOp>(rightOp)) {
            getLeAndGeOps(op, geOp, leOp, leftPrevIntersectOp);
        }

        if (leOp != nullptr && geOp != nullptr) {
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

