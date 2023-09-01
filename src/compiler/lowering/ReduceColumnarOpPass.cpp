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
    Operation * getCmpWOBitmap(Operation * castOp) {
        if (castOp->getOperand(0).getType().isa<mlir::daphne::FrameType>() && castOp->getResult(0).getType().isa<mlir::daphne::ColumnType>()) {
            auto frameOp = castOp->getOperand(0).getDefiningOp();
            if (mlir::dyn_cast<mlir::daphne::CreateFrameOp>(frameOp)) {
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
                auto columnName = op->getOperand(0).getDefiningOp()->getOperand(1).getDefiningOp()->getAttr("value").cast<StringAttr>().getValue().str();
                auto frameOp = op->getOperand(0).getDefiningOp()->getOperand(0).getDefiningOp();
                auto numInputColMatrix = frameOp->getNumOperands()/2;
                Operation *predecessor = nullptr;
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
                Operation * posListOp = op->getOperand(1).getDefiningOp()
                                            ->getOperand(0).getDefiningOp()
                                            ->getOperand(0).getDefiningOp()
                                            ->getOperand(0).getDefiningOp();
                if (mlir::dyn_cast<mlir::daphne::CastOp>(posListOp)) {
                    posListOp = posListOp->getOperand(0).getDefiningOp();
                }
                rewriter.startRootUpdate(op);
                op->replaceUsesOfWith(op->getOperand(1), posListOp->getResult(0));
                rewriter.finalizeRootUpdate(op);
                return success();
            } else if (llvm::dyn_cast<mlir::daphne::ColumnIntersectOp>(op)) {
                auto firstOp = getCmpWOBitmap(op->getOperand(0).getDefiningOp());
                auto secondOp = getCmpWOBitmap(op->getOperand(1).getDefiningOp());
                rewriter.startRootUpdate(op);
                if (firstOp != nullptr) {
                    op->replaceUsesOfWith(op->getOperand(0), firstOp->getResult(0));
                }
                if (secondOp != nullptr) {
                    op->replaceUsesOfWith(op->getOperand(1), secondOp->getResult(0));
                }
                rewriter.finalizeRootUpdate(op);
                return success();
            } else if (llvm::dyn_cast<mlir::daphne::ColumnAndOp>(op)) {
                auto firstOp = getCmpWOBitmap(op->getOperand(0).getDefiningOp());
                auto secondOp = getCmpWOBitmap(op->getOperand(1).getDefiningOp());
                rewriter.startRootUpdate(op);
                if (firstOp != nullptr) {
                    op->replaceUsesOfWith(op->getOperand(0), firstOp->getResult(0));
                }
                if (secondOp != nullptr) {
                    op->replaceUsesOfWith(op->getOperand(1), secondOp->getResult(0));
                }
                auto intersect = rewriter.replaceOpWithNewOp<daphne::ColumnIntersectOp>(op, op->getResult(0).getType(), op->getOperands());
                op->getResult(0).replaceAllUsesWith(intersect->getResult(0));
                for (mlir::Operation * followOp : intersect->getResult(0).getUsers()) {
                    if (mlir::dyn_cast<mlir::daphne::CastOp>(followOp)) {
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
                auto castOrPosOp = bitmapOp->getOperand(0).getDefiningOp();
                if (mlir::dyn_cast<mlir::daphne::CastOp>(castOrPosOp)) {
                    if (castOrPosOp->getOperand(0).getType().isa<mlir::daphne::ColumnType>()) {
                        return false;
                    }
                }
                if (mlir::dyn_cast<mlir::daphne::PositionListBitmapConverterOp>(castOrPosOp)) {
                    auto castOp2 = castOrPosOp->getOperand(0).getDefiningOp();
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

    target.addDynamicallyLegalOp<mlir::daphne::ColumnIntersectOp, mlir::daphne::ColumnAndOp>([&](Operation *op) {
        Operation * firstOp = nullptr;
        Operation * secondOp = nullptr;
        auto castOp = op->getOperand(0).getDefiningOp();
        if (mlir::dyn_cast<mlir::daphne::CastOp>(castOp)) {
            firstOp = getCmpWOBitmap(castOp);
        }
        auto castOp2 = op->getOperand(1).getDefiningOp();
        if (mlir::dyn_cast<mlir::daphne::CastOp>(castOp2)) {
            firstOp = getCmpWOBitmap(castOp2);
        }
        if (firstOp == nullptr && secondOp == nullptr) {
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
