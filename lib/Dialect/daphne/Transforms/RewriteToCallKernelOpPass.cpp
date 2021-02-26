#include "mlir/Dialect/daphne/Daphne.h"
#include "mlir/Dialect/daphne/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>
#include <iostream>

using namespace mlir;

namespace
{

    struct KernelReplacement : public RewritePattern
    {

        KernelReplacement(PatternBenefit benefit = 1)
        : RewritePattern(benefit, Pattern::MatchAnyOpTypeTag())
        {
        }

        LogicalResult matchAndRewrite(Operation *op,
                                      PatternRewriter &rewriter) const override
        {
            // The name of the kernel function to call.
            StringRef callee;

            // Determine the name of the kernel function to call based on the
            // operation, types, etc., or return failure() on error.
            if (llvm::dyn_cast<daphne::PrintOp>(op)) {
                Type t = llvm::dyn_cast<daphne::PrintOp>(op).input().getType();
                if (t.isSignedInteger(64))
                    callee = StringRef("printScaI64");
                else if (t.isF64())
                    callee = StringRef("printScaF64");
                else if (t.isa<daphne::MatrixType>()) {
                    Type et = t.dyn_cast<daphne::MatrixType>().getElementType();
                    if (et.isSignedInteger(64))
                        callee = StringRef("printDenI64");
                    else if (et.isF64())
                        callee = StringRef("printDenF64");
                    else
                        return failure();
                }
                else
                    return failure();
            }
            else if (llvm::dyn_cast<daphne::RandOp>(op)) {
                // Derive the element type of the matrix to be generated from
                // the type of the "min" argument. (Note that the "max"
                // argument is guaranteed to have the same type.)
                Type et = llvm::dyn_cast<daphne::RandOp>(op).min().getType();

                if (et.isSignedInteger(64))
                    callee = StringRef("randDenI64");
                else if (et.isF64())
                    callee = StringRef("randDenF64");
                else
                    return failure();
            }
            else if (llvm::dyn_cast<daphne::TransposeOp>(op)) {
                Type et = op->getOperand(0).getType().dyn_cast<daphne::MatrixType>().getElementType();
                if (et.isSignedInteger(64))
                    callee = "transposeDenI64";
                else if (et.isF64())
                    callee = "transposeDenF64";
                else
                    return failure();
            }
            else if (llvm::dyn_cast<daphne::AddOp>(op)) {
                if (op->getOperand(0).getType().isa<daphne::MatrixType>() &&
                    op->getOperand(1).getType().isa<daphne::MatrixType>()) {
                    Type et = op->getOperand(0).getType().dyn_cast<daphne::MatrixType>().getElementType();
                    if (et.isSignedInteger(64))
                        callee = "addDenDenDenI64";
                    else if (et.isF64())
                        callee = "addDenDenDenF64";
                    else
                        return failure();
                }
                else
                    return failure();
            }
            else if (llvm::dyn_cast<daphne::SetCellOp>(op)) {
                Type et = llvm::dyn_cast<daphne::SetCellOp>(op).mat().getType().dyn_cast<daphne::MatrixType>().getElementType();
                if (et.isSignedInteger(64))
                    callee = StringRef("setCellDenI64");
                else if (et.isF64())
                    callee = StringRef("setCellDenF64");
            }
            else
                return failure();

            // Create a CallKernelOp for the kernel function to call and return
            // success().
            auto kernel = rewriter.create<daphne::CallKernelOp>(
                    op->getLoc(),
                    callee,
                    op->getOperands(),
                    op->getResultTypes()
                    );
            rewriter.replaceOp(op, kernel.getResults());
            return success();
        }
    };

    struct RewriteToCallKernelOpPass
    : public PassWrapper<RewriteToCallKernelOpPass, OperationPass<ModuleOp>>
    {
        void runOnOperation() final;
    };
}

void RewriteToCallKernelOpPass::runOnOperation()
{
    auto module = getOperation();

    OwningRewritePatternList patterns;

    // convert other operations
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect, LLVM::LLVMDialect,
            daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, ModuleTerminatorOp, FuncOp>();
    target.addIllegalOp<
            daphne::PrintOp, daphne::RandOp, daphne::TransposeOp, daphne::SetCellOp
    >();
    target.addDynamicallyLegalOp<daphne::AddOp>([](daphne::AddOp op)
    {
        const bool legal = (
                !op->getOperand(0).getType().isa<daphne::MatrixType>() &&
                !op->getOperand(1).getType().isa<daphne::MatrixType>()
                );
        return legal;
    });

    patterns.insert<KernelReplacement>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();

}

std::unique_ptr<Pass> daphne::createRewriteToCallKernelOpPass()
{
    return std::make_unique<RewriteToCallKernelOpPass>();
}
