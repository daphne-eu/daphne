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
#include <string_view>

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
        else if(t.isIndex())
            return "size_t";
        else if(t.isa<daphne::MatrixType>())
            return "DenseMatrix_" + mlirTypeToCppTypeName(
                    t.dyn_cast<daphne::MatrixType>().getElementType()
            );
        else if(t.isa<daphne::FrameType>())
            return "Frame";
        throw std::runtime_error(
                "no C++ type name known for the given MLIR type"
        );
    }

    struct KernelReplacement : public RewritePattern
    {
		bool use_cuda;
        KernelReplacement(MLIRContext * context, bool use_cuda = false, PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context), use_cuda(use_cuda)
        {
        }

        LogicalResult matchAndRewrite(Operation *op,
                                      PatternRewriter &rewriter) const override
        {
            // Determine the name of the kernel function to call by convention
            // based on the DaphneIR operation and the types of its results and
            // arguments.

            std::stringstream callee;
            if(use_cuda) {
            	std::string_view op_name{op->getName().stripDialect().data()};
            	std::string_view arg_sv("matMul");
            	if(op_name.compare(arg_sv)) {
            		std::cout << "need to insert cuda context init/destroy ops" << std::endl;
					callee << "CUDA_" << op_name;
				}
            }
            else
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
    	bool use_cuda;
    	RewriteToCallKernelOpPass(bool use_cuda) : use_cuda(use_cuda){}
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

    patterns.insert<KernelReplacement>(&getContext(), use_cuda);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();

}

std::unique_ptr<Pass> daphne::createRewriteToCallKernelOpPass(bool use_cuda)
{
    return std::make_unique<RewriteToCallKernelOpPass>(use_cuda);
}
