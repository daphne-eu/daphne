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
    Operation * getCmpWOBitmap(Operation * op) {
        if (mlir::dyn_cast<mlir::daphne::CastOp>(op)) {
            if (op->getOperand(0).getType().isa<mlir::daphne::FrameType>() && op->getResult(0).getType().isa<mlir::daphne::ColumnType>()) {
                auto frameOp = op->getOperand(0).getDefiningOp();
                if (mlir::dyn_cast<mlir::daphne::CreateFrameOp>(frameOp)) {
                    // We only check one of the operands because we know that the other ones should also be columns
                    auto bitmapOp = frameOp->getOperand(0).getDefiningOp();
                    if (mlir::dyn_cast<mlir::daphne::PositionListBitmapConverterOp>(bitmapOp)) {
                        auto castOp2 = bitmapOp->getOperand(0).getDefiningOp();
                        if (mlir::dyn_cast<mlir::daphne::CastOp>(castOp2)) {
                            if (castOp2->getOperand(0).getType().isa<mlir::daphne::ColumnType>()) {
                                return castOp2->getOperand(0).getDefiningOp();
                            }
                        }
                    }
                }
            }
        } else if (mlir::dyn_cast<mlir::daphne::ColumnIntersectOp>(op)) {
            return op;
        }
        return nullptr;
    }


    struct ColumnarReduceReplacement : public RewritePattern{

        ColumnarReduceReplacement(MLIRContext * context, PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context)
        {}

        LogicalResult matchAndRewrite(
            Operation *op,
            PatternRewriter &rewriter
        ) const override
        {
            if (llvm::dyn_cast<mlir::daphne::CastOp>(op)) {
                auto columnName = op->getOperand(0).getDefiningOp()         // Get the extract op
                                                ->getOperand(1).getDefiningOp()          // Get the string op
                                                ->getAttr("value").cast<StringAttr>().getValue().str();
                auto frameOp = op->getOperand(0).getDefiningOp()->getOperand(0).getDefiningOp();
                auto numInputColMatrix = frameOp->getNumOperands()/2;
                Operation *predecessor = nullptr;
                // Check which input operation into our frame is the wanted column
                for (int i = 0; i < numInputColMatrix; i++) {
                    auto inputName = frameOp->getOperand(i+numInputColMatrix).getDefiningOp()->getAttr("value").cast<StringAttr>().getValue().str();
                    auto inputMatrixOp = frameOp->getOperand(i).getDefiningOp();
                    if (columnName == inputName) {
                        predecessor = inputMatrixOp->getOperand(0).getDefiningOp();
                    }
                }
                auto castResult = op->getResult(0);

                rewriter.startRootUpdate(op);
                castResult.replaceAllUsesWith(predecessor->getResult(0));
                rewriter.finalizeRootUpdate(op);

                return success();
            } else if (llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(op)) {
                Operation * posListOp = op->getOperand(1).getDefiningOp()               // Get the cast op
                                            ->getOperand(0).getDefiningOp()             // Get the bitmappositionlistconverter op
                                            ->getOperand(0).getDefiningOp()             // Get the positionlistbitmapconverter op
                                            ->getOperand(0).getDefiningOp()             // Get the cast op
                                            ->getOperand(0).getDefiningOp();            // Get the position list source op

                rewriter.startRootUpdate(op);
                op->replaceUsesOfWith(op->getOperand(1), posListOp->getResult(0));
                rewriter.finalizeRootUpdate(op);

                return success();
            } else if (llvm::dyn_cast<mlir::daphne::ColumnAndOp>(op)) {
                auto firstOp = getCmpWOBitmap(op->getOperand(0).getDefiningOp());
                auto secondOp = getCmpWOBitmap(op->getOperand(1).getDefiningOp());

                rewriter.startRootUpdate(op);
                // Due to the dynamically legal function both operands should come from a column operation
                op->replaceUsesOfWith(op->getOperand(0), firstOp->getResult(0));
                op->replaceUsesOfWith(op->getOperand(1), secondOp->getResult(0));
                auto intersect = rewriter.replaceOpWithNewOp<daphne::ColumnIntersectOp>(op, op->getResult(0).getType(), op->getOperands());
                op->getResult(0).replaceAllUsesWith(intersect->getResult(0));
                // To keep compatibility with the rest of the pipeline we need to convert the result of the intersect op back to a position list if we convert it to a matrix
                for (mlir::Operation * followOp : intersect->getResult(0).getUsers()) {
                    if (mlir::dyn_cast<mlir::daphne::CastOp>(followOp) && followOp->getResult(0).getType().isa<mlir::daphne::MatrixType>()) {
                        rewriter.setInsertionPointAfter(followOp);
                        auto numRows = rewriter.create<mlir::daphne::ColumnNumRowsOp>(followOp->getLoc(), rewriter.getIndexType(), intersect->getOperand(1).getDefiningOp()->getOperand(0));
                        auto users = followOp->getResult(0).getUsers();
                        auto converter = rewriter.create<mlir::daphne::PositionListBitmapConverterOp>(followOp->getLoc(), followOp->getResult(0).getType(), followOp->getResult(0), numRows->getResult(0));
                        for (mlir::Operation * user : users) {
                            user->replaceUsesOfWith(followOp->getResult(0), converter->getResult(0));
                        }
                    }
                }
                rewriter.finalizeRootUpdate(op);

                return success();
            }
        }
    };

    struct ReduceColumnarOpPass : public PassWrapper<ReduceColumnarOpPass, OperationPass<func::FuncOp>> {

        ReduceColumnarOpPass() = default;
    
        void runOnOperation() final;
    
    };
}

void ReduceColumnarOpPass::runOnOperation() {
    auto module = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();
    target.addDynamicallyLegalOp<mlir::daphne::CastOp>([&](Operation *op) {
        if (op->getResult(0).use_empty()) {
            return true;
        }
        if (op->getResult(0).getType().isa<mlir::daphne::ColumnType>()) {
            if(op->getOperand(0).getType().isa<mlir::daphne::FrameType>()) {
                auto extractOp = op->getOperand(0).getDefiningOp();
                if (llvm::dyn_cast<mlir::daphne::ExtractColOp>(extractOp)) {
                    auto frameOp = extractOp->getOperand(0).getDefiningOp();
                    if (llvm::dyn_cast<mlir::daphne::CreateFrameOp>(frameOp)) {
                        // We only check one of the operands because we know that the other ones should also be columns
                        auto castOp = frameOp->getOperand(0).getDefiningOp();
                        if (llvm::dyn_cast<mlir::daphne::CastOp>(castOp)) {
                            if (castOp->getOperand(0).getType().isa<mlir::daphne::ColumnType>()) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        return true;
    });

    target.addDynamicallyLegalOp<mlir::daphne::ColumnProjectOp>([&](Operation *op) {
        Operation * castOp = op->getOperand(1).getDefiningOp();
        if (mlir::dyn_cast<mlir::daphne::CastOp>(castOp)) {
            auto bitmapOp = castOp->getOperand(0).getDefiningOp();
            if (mlir::dyn_cast<mlir::daphne::BitmapPositionListConverterOp>(bitmapOp)) {
                auto posOp = bitmapOp->getOperand(0).getDefiningOp();
                if (mlir::dyn_cast<mlir::daphne::PositionListBitmapConverterOp>(posOp)) {
                    auto castOp2 = posOp->getOperand(0).getDefiningOp();
                    if (mlir::dyn_cast<mlir::daphne::CastOp>(castOp2)) {
                        if (castOp2->getOperand(0).getType().isa<mlir::daphne::ColumnType>()) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    });

    target.addDynamicallyLegalOp<mlir::daphne::ColumnAndOp>([&](Operation *op) {
        Operation * firstOp = nullptr;
        Operation * secondOp = nullptr;
        auto op1 = op->getOperand(0).getDefiningOp();
        firstOp = getCmpWOBitmap(op1);

        auto op2 = op->getOperand(1).getDefiningOp();
        secondOp = getCmpWOBitmap(op2);

        // Only apply optimisation if source comparisons are on columns with position list outputs,
        // which should be the case after our first pass
        if (firstOp == nullptr || secondOp == nullptr) {
            return true;
        } else {
            return false;
        }
    });

    patterns.add<ColumnarReduceReplacement>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
    
}

std::unique_ptr<Pass> daphne::createReduceColumnarOpPass()
{
    return std::make_unique<ReduceColumnarOpPass>();
}

