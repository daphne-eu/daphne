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
            if (op->getName().getStringRef().equals(daphne::PrintOp::getOperationName())) {
                auto operands = op->getOperands();
                auto kernel = rewriter.create<daphne::CallKernelOp>(
                        op->getLoc(),
                        op->getOperand(0).getType().isa<IntegerType>()
                        ? "printInt"
                        : (
                        op->getOperand(0).getType().isa<FloatType>() ? "printDouble" : "printMatrix"
                        ),
                        operands,
                        op->getResultTypes()
                        );
                rewriter.replaceOp(op, kernel.getResults());
                return success();
            }
            else if (op->getName().getStringRef().equals(daphne::RandOp::getOperationName())) {
                auto operands = op->getOperands();
                auto kernel = rewriter.create<daphne::CallKernelOp>(
                        op->getLoc(),
                        "randMatF64",
                        operands,
                        op->getResultTypes()
                        );
                rewriter.replaceOp(op, kernel.getResults());
                return success();
            }
            else if (op->getName().getStringRef().equals(daphne::TransposeOp::getOperationName())) {
                auto operands = op->getOperands();
                auto kernel = rewriter.create<daphne::CallKernelOp>(
                        op->getLoc(),
                        "transpose",
                        operands,
                        op->getResultTypes()
                        );
                rewriter.replaceOp(op, kernel.getResults());
                return success();
            }
            else if (op->getName().getStringRef().equals(daphne::AddOp::getOperationName())) {
                auto operands = op->getOperands();
                if (op->getOperand(0).getType().isa<daphne::MatrixType>() &&
                    op->getOperand(1).getType().isa<daphne::MatrixType>()) {
                    auto kernel = rewriter.create<daphne::CallKernelOp>(
                            op->getLoc(),
                            "addMM",
                            operands,
                            op->getResultTypes()
                            );
                    rewriter.replaceOp(op, kernel.getResults());
                    return success();
                }
                else
                    return failure();
            }
            return failure();
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
            daphne::PrintOp, daphne::RandOp, daphne::TransposeOp
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