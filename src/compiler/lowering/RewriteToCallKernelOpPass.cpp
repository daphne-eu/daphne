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
#include <tuple>

using namespace mlir;

namespace
{
    
    // TODO We might want to merge this with ValueTypeUtils, and maybe place it
    // somewhere central.
    std::string mlirTypeToCppTypeName(Type t, bool generalizeToStructure) {
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
            if(generalizeToStructure)
                return "Structure";
            else
                return "DenseMatrix_" + mlirTypeToCppTypeName(
                        t.dyn_cast<daphne::MatrixType>().getElementType(),
                        false
                );
        else if(t.isa<daphne::FrameType>())
            if(generalizeToStructure)
                return "Structure";
            else
                return "Frame";
        else if(t.isa<daphne::StringType>())
            // This becomes "const char *" (which makes perfect sense for
            // strings) when inserted into the typical "const DT *" template of
            // kernel input parameters.
            return "char";
        else if(t.isa<daphne::DaphneContextType>())
            return "DaphneContext";
        throw std::runtime_error(
                "no C++ type name known for the given MLIR type"
        );
    }

    class KernelReplacement : public RewritePattern
    {
        // TODO This method is only required since MLIR does not seem to
        // provide a means to get this information.
        static size_t getNumODSOperands(Operation * op) {
            if(llvm::isa<daphne::CreateFrameOp>(op))
                return 2;
            if(llvm::isa<daphne::SetColLabelsOp>(op))
                return 2;
            throw std::runtime_error(
                    "unsupported operation: " + op->getName().getStringRef().str()
            );
        }
        
        // TODO This method is only required since MLIR does not seem to
        // provide a means to get this information. But, for instance, the
        // isVariadic boolean array is automatically generated *within* the
        // getODSOperandIndexAndLength method.
        static std::tuple<unsigned, unsigned, bool> getODSOperandInfo(Operation * op, unsigned index) {
            if(auto concreteOp = llvm::dyn_cast<daphne::CreateFrameOp>(op)) {
                auto idxAndLen = concreteOp.getODSOperandIndexAndLength(index);
                static bool isVariadic[] = {true, true};
                return std::make_tuple(
                        idxAndLen.first,
                        idxAndLen.second,
                        isVariadic[index]
                );
            }
            if(auto concreteOp = llvm::dyn_cast<daphne::SetColLabelsOp>(op)) {
                auto idxAndLen = concreteOp.getODSOperandIndexAndLength(index);
                static bool isVariadic[] = {false, true};
                return std::make_tuple(
                        idxAndLen.first,
                        idxAndLen.second,
                        isVariadic[index]
                );
            }
            throw std::runtime_error(
                    "unsupported operation: " + op->getName().getStringRef().str()
            );
        }
        
        /**
         * @brief The value of type `DaphneContext` to insert as the first
         * argument to all kernel calls.
         */
        Value dctx;

    public:
        /**
         * Creates a new KernelReplacement rewrite pattern.
         * 
         * @param mctx The MLIR context.
         * @param dctx The DaphneContext to pass to the kernels.
         * @param benefit
         */
        KernelReplacement(MLIRContext * mctx, Value dctx, PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, mctx), dctx(dctx)
        {
        }

        LogicalResult matchAndRewrite(Operation *op,
                                      PatternRewriter &rewriter) const override
        {
            Location loc = op->getLoc();
            
            // Determine the name of the kernel function to call by convention
            // based on the DaphneIR operation and the types of its results and
            // arguments.

            std::stringstream callee;
            callee << op->getName().stripDialect().str();
            
            // TODO Don't enumerate all ops, decide based on a trait.
            const bool generalizeInputTypes =
                llvm::isa<daphne::CreateFrameOp>(op) |
                llvm::isa<daphne::NumCellsOp>(op) |
                llvm::isa<daphne::NumColsOp>(op) |
                llvm::isa<daphne::NumRowsOp>(op);
            
            // Append names of result types to the kernel name.
            Operation::result_type_range resultTypes = op->getResultTypes();
            for(size_t i = 0; i < resultTypes.size(); i++)
                callee << "__" << mlirTypeToCppTypeName(resultTypes[i], false);
            
            // Append names of operand types to the kernel name. Variadic
            // operands, which can have an arbitrary number of occurrences, are
            // treated specially.
            Operation::operand_type_range operandTypes = op->getOperandTypes();
            // The operands of the CallKernelOp may differ from the operands
            // of the given operation, if it has a variadic operand.
            std::vector<Value> newOperands;
            
            if(
                // TODO Unfortunately, one needs to know the exact N for
                // AtLeastNOperands... There seems to be no simple way to
                // detect if an operation has variadic ODS operands.
                op->hasTrait<OpTrait::VariadicOperands>() ||
                op->hasTrait<OpTrait::AtLeastNOperands<1>::Impl>()
            ) {
                // For operations with variadic operands, we replace all
                // occurrences of a variadic operand by a single operand of
                // type VariadicPack as well as an operand for the number of
                // occurrences. All occurrences of the variadic operand are
                // stored in the VariadicPack.
                const size_t numODSOperands = getNumODSOperands(op);
                for(size_t i = 0; i < numODSOperands; i++) {
                    auto odsOpInfo = getODSOperandInfo(op, i);
                    const unsigned idx = std::get<0>(odsOpInfo);
                    const unsigned len = std::get<1>(odsOpInfo);
                    const bool isVariadic = std::get<2>(odsOpInfo);
                    
                    callee << "__" << mlirTypeToCppTypeName(operandTypes[idx], generalizeInputTypes);
                    if(isVariadic) {
                        // Variadic operand.
                        callee << "_variadic__size_t";
                        auto cvpOp = rewriter.create<daphne::CreateVariadicPackOp>(
                                loc,
                                daphne::VariadicPackType::get(
                                        rewriter.getContext(),
                                        op->getOperand(idx).getType()
                                ),
                                rewriter.getIndexAttr(len)
                        );
                        for(size_t k = 0; k < len; k++)
                            rewriter.create<daphne::StoreVariadicPackOp>(
                                    loc,
                                    cvpOp,
                                    op->getOperand(idx + k),
                                    rewriter.getIndexAttr(k)
                            );
                        newOperands.push_back(cvpOp);
                        newOperands.push_back(rewriter.create<daphne::ConstantOp>(
                                loc, rewriter.getIndexAttr(len))
                        );
                    }
                    else
                        // Non-variadic operand.
                        newOperands.push_back(op->getOperand(i));
                }
            }
            else
                // For operations without variadic operands, we simply append
                // the name of the type of each operand and pass all operands
                // to the CallKernelOp as-is.
                for(size_t i = 0; i < operandTypes.size(); i++) {
                    callee << "__" << mlirTypeToCppTypeName(operandTypes[i], generalizeInputTypes);
                    newOperands.push_back(op->getOperand(i));
                }
            
            // Inject the current DaphneContext as the last input parameter to
            // (almost) all kernel calls.
            if(!llvm::isa<daphne::CreateDaphneContextOp, daphne::DestroyDaphneContextOp>(op))
                newOperands.push_back(dctx);

            // Create a CallKernelOp for the kernel function to call and return
            // success().
            auto kernel = rewriter.create<daphne::CallKernelOp>(
                    loc,
                    callee.str(),
                    newOperands,
                    op->getResultTypes()
                    );
            rewriter.replaceOp(op, kernel.getResults());
            return success();
        }
    };

    struct RewriteToCallKernelOpPass
    : public PassWrapper<RewriteToCallKernelOpPass, FunctionPass>
    {
        void runOnFunction() final;
    };
}

void RewriteToCallKernelOpPass::runOnFunction()
{
    FuncOp func = getFunction();

    OwningRewritePatternList patterns(&getContext());

    // Specification of (il)legal dialects/operations. All DaphneIR operations
    // but those explicitly marked as legal will be replaced by CallKernelOp.
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect, LLVM::LLVMDialect, scf::SCFDialect>();
    target.addLegalOp<ModuleOp, FuncOp>();
    target.addIllegalDialect<daphne::DaphneDialect>();
    target.addLegalOp<
            daphne::ConstantOp,
            daphne::ReturnOp,
            daphne::CallKernelOp,
            daphne::CreateVariadicPackOp,
            daphne::StoreVariadicPackOp
    >();
    
    // Determine the DaphneContext valid in the MLIR function being rewritten.
    mlir::Value dctx = nullptr;
    auto ops = func.body().front().getOps<daphne::CreateDaphneContextOp>();
    for(auto op : ops) {
        if(!dctx)
            dctx = op.getResult();
        else
            throw std::runtime_error(
                    "function body block contains more than one CreateDaphneContextOp"
            );
    }
    if(!dctx)
        throw std::runtime_error(
                "function body block contains no CreateDaphneContextOp"
        );

    // Apply conversion to CallKernelOps.
    patterns.insert<KernelReplacement>(&getContext(), dctx);
    if (failed(applyPartialConversion(func, target, std::move(patterns))))
        signalPassFailure();

}

std::unique_ptr<Pass> daphne::createRewriteToCallKernelOpPass()
{
    return std::make_unique<RewriteToCallKernelOpPass>();
}
