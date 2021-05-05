/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

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

        KernelReplacement(MLIRContext * context, PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context)
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
                Type t = llvm::dyn_cast<daphne::PrintOp>(op).arg().getType();
                if (t.isSignedInteger(64))
                    callee = StringRef("printSca__int64_t");
                else if (t.isF64())
                    callee = StringRef("printSca__double");
                else if (t.isa<daphne::MatrixType>()) {
                    Type et = t.dyn_cast<daphne::MatrixType>().getElementType();
                    if (et.isSignedInteger(64))
                        callee = StringRef("printObj__DenseMatrix_int64_t");
                    else if (et.isF64())
                        callee = StringRef("printObj__DenseMatrix_double");
                    else
                        return failure();
                }
                else
                    return failure();
            }
            else if (llvm::dyn_cast<daphne::RandMatrixOp>(op)) {
                // Derive the element type of the matrix to be generated from
                // the type of the "min" argument. (Note that the "max"
                // argument is guaranteed to have the same type.)
                Type et = llvm::dyn_cast<daphne::RandMatrixOp>(op).min().getType();

                if (et.isSignedInteger(64))
                    callee = StringRef("randMatrix__DenseMatrix_int64_t__int64_t");
                else if (et.isF64())
                    callee = StringRef("randMatrix__DenseMatrix_double__double");
                else
                    return failure();
            }
            else if (llvm::dyn_cast<daphne::TransposeOp>(op)) {
                Type et = op->getOperand(0).getType().dyn_cast<daphne::MatrixType>().getElementType();
                if (et.isSignedInteger(64))
                    callee = "transpose__DenseMatrix_int64_t__DenseMatrix_int64_t";
                else if (et.isF64())
                    callee = "transpose__DenseMatrix_double__DenseMatrix_double";
                else
                    return failure();
            }
#if 0
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
#endif
//            else if (llvm::dyn_cast<daphne::SumOp>(op)) {
//                Type et = llvm::dyn_cast<daphne::SumOp>(op).in().getType().dyn_cast<daphne::MatrixType>().getElementType();
//                if (et.isSignedInteger(64))
//                    callee = "sumDenScaI64";
//                else if (et.isF64())
//                    callee = "sumDenScaF64";
//                else
//                    return failure();
//            }
#if 0
            else if (llvm::dyn_cast<daphne::SetCellOp>(op)) {
                Type et = llvm::dyn_cast<daphne::SetCellOp>(op).mat().getType().dyn_cast<daphne::MatrixType>().getElementType();
                if (et.isSignedInteger(64))
                    callee = StringRef("setCellDenI64");
                else if (et.isF64())
                    callee = StringRef("setCellDenF64");
            }
#endif
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

    OwningRewritePatternList patterns(&getContext());

    // convert other operations
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect, LLVM::LLVMDialect,
            daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, FuncOp>();
    target.addIllegalOp<
            daphne::PrintOp, daphne::RandMatrixOp, daphne::TransposeOp, daphne::SetCellOp/*, daphne::SumOp*/
    >();
    target.addDynamicallyLegalOp<daphne::AddOp>([](daphne::AddOp op)
    {
        const bool legal = (
                !op->getOperand(0).getType().isa<daphne::MatrixType>() &&
                !op->getOperand(1).getType().isa<daphne::MatrixType>()
                );
        return legal;
    });

    patterns.insert<KernelReplacement>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();

}

std::unique_ptr<Pass> daphne::createRewriteToCallKernelOpPass()
{
    return std::make_unique<RewriteToCallKernelOpPass>();
}
