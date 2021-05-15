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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace mlir;

namespace
{
    
    // TODO We might want to merge this with ValueTypeUtils, and maybe place it
    // somewhere central.
    std::string mlirTypeToCppTypeName(Type t) {
        if(t.isF64())
            return "double";
        else if(t.isF32())
            return "float";
        else if(t.isSignedInteger(8))
            return "int8_t";
        else if(t.isSignedInteger(32))
            return "int32_t";
        else if(t.isSignedInteger(64))
            return "int64_t";
        else if(t.isUnsignedInteger(8))
            return "uint8_t";
        else if(t.isUnsignedInteger(32))
            return "uint32_t";
        else if(t.isUnsignedInteger(64))
            return "uint64_t";
        else if(t.isSignlessInteger(1))
            return "bool";
        else if(t.isa<daphne::MatrixType>())
            return "DenseMatrix_" + mlirTypeToCppTypeName(
                    t.dyn_cast<daphne::MatrixType>().getElementType()
            );
        throw std::runtime_error(
                "no C++ type name known for the given MLIR type"
        );
    }

    struct KernelReplacement : public RewritePattern
    {

        KernelReplacement(MLIRContext * context, PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context)
        {
        }

        LogicalResult matchAndRewrite(Operation *op,
                                      PatternRewriter &rewriter) const override
        {
            // Determine the name of the kernel function to call by convention
            // based on the DaphneIR operation and the types of its results and
            // arguments.

            std::stringstream callee;
            callee << op->getName().stripDialect().str();

            Operation::result_type_range resultTypes = op->getResultTypes();
            for(size_t i = 0; i < resultTypes.size(); i++)
                callee << "__" << mlirTypeToCppTypeName(resultTypes[i]);
            
            Operation::operand_type_range operandTypes = op->getOperandTypes();
            for(size_t i = 0; i < operandTypes.size(); i++)
                callee << "__" << mlirTypeToCppTypeName(operandTypes[i]);

            // Create a CallKernelOp for the kernel function to call and return
            // success().
            auto kernel = rewriter.create<daphne::CallKernelOp>(
                    op->getLoc(),
                    callee.str(),
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
    target.addLegalDialect<StandardOpsDialect, LLVM::LLVMDialect, scf::SCFDialect>();
    target.addLegalOp<ModuleOp, FuncOp>();
    target.addIllegalDialect<daphne::DaphneDialect>();
    target.addLegalOp<daphne::ConstantOp, daphne::ReturnOp, daphne::CallKernelOp>();

    patterns.insert<KernelReplacement>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();

}

std::unique_ptr<Pass> daphne::createRewriteToCallKernelOpPass()
{
    return std::make_unique<RewriteToCallKernelOpPass>();
}
