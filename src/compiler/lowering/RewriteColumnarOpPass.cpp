#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

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
    template <class DaphneCmp, class ColumnCmp>
    mlir::LogicalResult compareOp(PatternRewriter &rewriter, Operation *op) {
        DaphneCmp geOp = llvm::dyn_cast<DaphneCmp>(op);
        mlir::Value geInout = op->getOperand(0);
        auto prevOp = geInout.getDefiningOp();

        if(!geOp){
            return failure();
        }

        mlir::Type vt = mlir::daphne::UnknownType::get(rewriter.getContext());
        mlir::Type resType = mlir::daphne::ColumnType::get(
            rewriter.getContext(), vt
        );
        mlir::Type resTypeCast = mlir::daphne::MatrixType::get(
            rewriter.getContext(), vt
        );

        mlir::daphne::CastOp cast;
        if(llvm::dyn_cast<mlir::daphne::CastOp>(prevOp)) {
            cast = rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(prevOp, resType, prevOp->getOperand(0));
        } else {
            cast = rewriter.create<mlir::daphne::CastOp>(prevOp->getLoc(), resType, prevOp->getResult(0));
        }
        auto columnGe = rewriter.create<ColumnCmp>(prevOp->getLoc(), cast, geOp->getOperand(1));
        auto finalCast = rewriter.create<mlir::daphne::CastOp>(prevOp->getLoc(), resTypeCast, columnGe->getResult(0));
        auto numRows = rewriter.create<mlir::daphne::NumRowsOp>(prevOp->getLoc(), rewriter.getIndexType(), cast->getOperand(0));
        rewriter.replaceOpWithNewOp<mlir::daphne::PositionListBitmapConverterOp>(geOp, resTypeCast, finalCast->getResult(0), numRows);
        return success();
    }

    struct ColumnarOpReplacement : public RewritePattern{

        ColumnarOpReplacement(MLIRContext * context, PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context)
        {}

        LogicalResult matchAndRewrite(
            Operation *op,
            PatternRewriter &rewriter
        ) const override
        {
            if(llvm::dyn_cast<mlir::daphne::EwGeOp>(op)){
                return compareOp<mlir::daphne::EwGeOp, mlir::daphne::ColumnGeOp>(rewriter, op);
            } else if(llvm::dyn_cast<mlir::daphne::EwGtOp>(op)) {
                return compareOp<mlir::daphne::EwGtOp, mlir::daphne::ColumnGtOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwLeOp>(op)) {
                return compareOp<mlir::daphne::EwLeOp, mlir::daphne::ColumnLeOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwLtOp>(op)) {
                return compareOp<mlir::daphne::EwLtOp, mlir::daphne::ColumnLtOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwEqOp>(op)) {
                return compareOp<mlir::daphne::EwEqOp, mlir::daphne::ColumnEqOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwNeqOp>(op)) {
                return compareOp<mlir::daphne::EwNeqOp, mlir::daphne::ColumnNeqOp>(rewriter, op);
            }
        }
    };

    struct RewriteColumnarOpPass : public PassWrapper<RewriteColumnarOpPass, OperationPass<ModuleOp>> {
    
    void runOnOperation() final;
    
    };
}

void RewriteColumnarOpPass::runOnOperation() {
    auto module = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();
    target.addIllegalOp<mlir::daphne::EwGeOp, mlir::daphne::EwGtOp, mlir::daphne::EwLeOp, mlir::daphne::EwLtOp, mlir::daphne::EwEqOp, mlir::daphne::EwNeqOp>();

    patterns.add<ColumnarOpReplacement>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createRewriteColumnarOpPass()
{
    return std::make_unique<RewriteColumnarOpPass>();
}

