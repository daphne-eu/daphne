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
    void deletePriorCast(PatternRewriter &rewriter, Operation *op){
        mlir::Value opInout = op->getOperand(0);
        mlir::daphne::CastOp castOp = llvm::dyn_cast<mlir::daphne::CastOp>(opInout.getDefiningOp());

        if(!castOp)
            return;
        rewriter.eraseOp(castOp);
        return;
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
                mlir::daphne::EwGeOp geOp = llvm::dyn_cast<mlir::daphne::EwGeOp>(op);
                mlir::Value geInout = op->getOperand(0);
                mlir::daphne::CastOp castOp = llvm::dyn_cast<mlir::daphne::CastOp>(geInout.getDefiningOp());

                if(!geOp || !castOp){
                    return failure();
                }

                mlir::Type vt = mlir::daphne::UnknownType::get(rewriter.getContext());
                mlir::Type resType = mlir::daphne::ColumnType::get(
                    rewriter.getContext(), vt
                );
                mlir::Type resTypeCast = mlir::daphne::MatrixType::get(
                    rewriter.getContext(), vt
                );
                auto cast = rewriter.create<mlir::daphne::CastOp>(castOp->getLoc(), resType, castOp->getOperand(0));
                auto columnGe = rewriter.create<mlir::daphne::ColumnGeOp>(castOp->getLoc(), cast, geOp->getOperand(1));
                deletePriorCast(rewriter, geOp);
                rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(geOp, resTypeCast, columnGe->getResult(0));
                return success();
            } else if(llvm::dyn_cast<mlir::daphne::EwGtOp>(op)) {
                mlir::daphne::EwGtOp gtOp = llvm::dyn_cast<mlir::daphne::EwGtOp>(op);
                mlir::Value gtInout = op->getOperand(0);
                mlir::daphne::CastOp castOp = llvm::dyn_cast<mlir::daphne::CastOp>(gtInout.getDefiningOp());

                if(!gtOp || !castOp){
                    return failure();
                }

                mlir::Type vt = mlir::daphne::UnknownType::get(rewriter.getContext());
                mlir::Type resType = mlir::daphne::ColumnType::get(
                    rewriter.getContext(), vt
                );
                mlir::Type resTypeCast = mlir::daphne::MatrixType::get(
                    rewriter.getContext(), vt
                );
                auto cast = rewriter.create<mlir::daphne::CastOp>(castOp->getLoc(), resType, castOp->getOperand(0));
                auto columnGt = rewriter.create<mlir::daphne::ColumnGtOp>(castOp->getLoc(), cast, gtOp->getOperand(1));
                deletePriorCast(rewriter, gtOp);
                rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(gtOp, resTypeCast, columnGt->getResult(0));
                return success();
            }else if(llvm::dyn_cast<mlir::daphne::EwLeOp>(op)) {
                mlir::daphne::EwLeOp leOp = llvm::dyn_cast<mlir::daphne::EwLeOp>(op);
                mlir::Value leInout = op->getOperand(0);
                mlir::daphne::CastOp castOp = llvm::dyn_cast<mlir::daphne::CastOp>(leInout.getDefiningOp());

                if(!leOp || !castOp){
                    return failure();
                }

                mlir::Type vt = mlir::daphne::UnknownType::get(rewriter.getContext());
                mlir::Type resType = mlir::daphne::ColumnType::get(
                    rewriter.getContext(), vt
                );
                mlir::Type resTypeCast = mlir::daphne::MatrixType::get(
                    rewriter.getContext(), vt
                );
                auto cast = rewriter.create<mlir::daphne::CastOp>(castOp->getLoc(), resType, castOp->getOperand(0));
                auto columnLe = rewriter.create<mlir::daphne::ColumnLeOp>(castOp->getLoc(), cast, leOp->getOperand(1));
                deletePriorCast(rewriter, leOp);
                rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(leOp, resTypeCast, columnLe->getResult(0));
                return success();
            }else if(llvm::dyn_cast<mlir::daphne::EwLtOp>(op)) {
                mlir::daphne::EwLtOp ltOp = llvm::dyn_cast<mlir::daphne::EwLtOp>(op);
                mlir::Value ltInout = op->getOperand(0);
                mlir::daphne::CastOp castOp = llvm::dyn_cast<mlir::daphne::CastOp>(ltInout.getDefiningOp());

                if(!ltOp || !castOp){
                    return failure();
                }

                mlir::Type vt = mlir::daphne::UnknownType::get(rewriter.getContext());
                mlir::Type resType = mlir::daphne::ColumnType::get(
                    rewriter.getContext(), vt
                );
                mlir::Type resTypeCast = mlir::daphne::MatrixType::get(
                    rewriter.getContext(), vt
                );
                auto cast = rewriter.create<mlir::daphne::CastOp>(castOp->getLoc(), resType, castOp->getOperand(0));
                auto columnLt = rewriter.create<mlir::daphne::ColumnLtOp>(castOp->getLoc(), cast, ltOp->getOperand(1));
                deletePriorCast(rewriter, ltOp);
                rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(ltOp, resTypeCast, columnLt->getResult(0));
                return success();
            }else if(llvm::dyn_cast<mlir::daphne::EwEqOp>(op)) {
                mlir::daphne::EwEqOp eqOp = llvm::dyn_cast<mlir::daphne::EwEqOp>(op);
                mlir::Value eqInout = op->getOperand(0);
                mlir::daphne::CastOp castOp = llvm::dyn_cast<mlir::daphne::CastOp>(eqInout.getDefiningOp());

                if(!eqOp || !castOp){
                    return failure();
                }

                mlir::Type vt = mlir::daphne::UnknownType::get(rewriter.getContext());
                mlir::Type resType = mlir::daphne::ColumnType::get(
                    rewriter.getContext(), vt
                );
                mlir::Type resTypeCast = mlir::daphne::MatrixType::get(
                    rewriter.getContext(), vt
                );
                auto cast = rewriter.create<mlir::daphne::CastOp>(castOp->getLoc(), resType, castOp->getOperand(0));
                auto columnEq = rewriter.create<mlir::daphne::ColumnEqOp>(castOp->getLoc(), cast, eqOp->getOperand(1));
                deletePriorCast(rewriter, eqOp);
                rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(eqOp, resTypeCast, columnEq->getResult(0));
                return success();
            }else if(llvm::dyn_cast<mlir::daphne::EwNeqOp>(op)) {
                mlir::daphne::EwNeqOp neqOp = llvm::dyn_cast<mlir::daphne::EwNeqOp>(op);
                mlir::Value neqInout = op->getOperand(0);
                mlir::daphne::CastOp castOp = llvm::dyn_cast<mlir::daphne::CastOp>(neqInout.getDefiningOp());

                if(!neqOp || !castOp){
                    return failure();
                }

                mlir::Type vt = mlir::daphne::UnknownType::get(rewriter.getContext());
                mlir::Type resType = mlir::daphne::ColumnType::get(
                    rewriter.getContext(), vt
                );
                mlir::Type resTypeCast = mlir::daphne::MatrixType::get(
                    rewriter.getContext(), vt
                );
                auto cast = rewriter.create<mlir::daphne::CastOp>(castOp->getLoc(), resType, castOp->getOperand(0));
                auto columnNeq = rewriter.create<mlir::daphne::ColumnNeqOp>(castOp->getLoc(), cast, neqOp->getOperand(1));
                deletePriorCast(rewriter, neqOp);
                rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(neqOp, resTypeCast, columnNeq->getResult(0));
                return success();
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

