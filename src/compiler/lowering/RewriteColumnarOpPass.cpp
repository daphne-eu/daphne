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

    struct ColumnarOpReplacement : public RewritePattern{

        ColumnarOpReplacement(MLIRContext * context, PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context)
        {}

        LogicalResult matchAndRewrite(
            Operation *op,
            PatternRewriter &rewriter
        ) const override
        {
            mlir::daphne::EwGeOp geOp = llvm::dyn_cast<mlir::daphne::EwGeOp>(op);
            mlir::Value gtInout = op->getOperand(0);
            mlir::daphne::CastOp castOp = llvm::dyn_cast<mlir::daphne::CastOp>(gtInout.getDefiningOp());

            if(!geOp || !castOp){
                return failure();
            }

            //rewriter.create<mlir::daphne::ConstantOp>(castOp->getLoc(), static_cast<int64_t>(1));
            //rewriter.create<mlir::daphne::CastOp>(ltOp->getLoc(), castOp->getOperand(0), castOp->getResult(0).getType());

            rewriter.replaceOpWithNewOp<mlir::daphne::ConstantOp>(castOp, static_cast<int64_t>(1));
            auto cast = rewriter.create<mlir::daphne::CastOp>(castOp->getLoc(), castOp->getResult(0).getType(), castOp->getOperand(0));
            mlir::Type vt = mlir::daphne::UnknownType::get(rewriter.getContext());
            mlir::Type resType = mlir::daphne::ColumnType::get(
                rewriter.getContext(), vt
            );
            auto cast2 = rewriter.create<mlir::daphne::CastOp>(castOp->getLoc(), resType, castOp->getOperand(0));
            auto gt = rewriter.replaceOpWithNewOp<mlir::daphne::EwLeOp>(geOp, cast, geOp->getOperand(1));
            //gt->getUsers();

            //rewriter.create<mlir::daphne::EwGtOp>(ltOp->getLoc(), cast, ltOp->getOperand(1));

            return success();
            
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
    target.addIllegalOp<mlir::daphne::EwGeOp>();

    patterns.add<ColumnarOpReplacement>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createRewriteColumnarOpPass()
{
    return std::make_unique<RewriteColumnarOpPass>();
}

