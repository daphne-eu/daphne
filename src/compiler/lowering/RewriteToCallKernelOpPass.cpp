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

#include "compiler/CompilerUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include <memory>
#include <utility>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <tuple>

using namespace mlir;

namespace
{
    class KernelReplacement : public RewritePattern
    {
        // TODO This method is only required since MLIR does not seem to
        // provide a means to get this information.
        static size_t getNumODSOperands(Operation * op) {
            if(llvm::isa<daphne::OrderOp>(op))
                return 4;
            if(llvm::isa<daphne::GroupOp>(op))
                return 3;
            if(llvm::isa<daphne::CreateFrameOp, daphne::SetColLabelsOp>(op))
                return 2;
            if(llvm::isa<daphne::DistributedComputeOp>(op))
                return 1;
            throw std::runtime_error(
                    "lowering to kernel call not yet supported for this variadic operation: "
                    + op->getName().getStringRef().str()
            );
        }

        // TODO This method is only required since MLIR does not seem to
        // provide a means to get this information. But, for instance, the
        // isVariadic boolean array is automatically generated *within* the
        // getODSOperandIndexAndLength method.
        static std::tuple<unsigned, unsigned, bool> getODSOperandInfo(Operation * op, unsigned index) {
            // TODO Simplify those by a macro.
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
            if(auto concreteOp = llvm::dyn_cast<daphne::DistributedComputeOp>(op)) {
                auto idxAndLen = concreteOp.getODSOperandIndexAndLength(index);
                static bool isVariadic[] = {true};
                return std::make_tuple(
                    idxAndLen.first,
                    idxAndLen.second,
                    isVariadic[index]
                );
            }
            if(auto concreteOp = llvm::dyn_cast<daphne::GroupOp>(op)) {
                auto idxAndLen = concreteOp.getODSOperandIndexAndLength(index);
                static bool isVariadic[] = {false, true, true, true};
                return std::make_tuple(
                        idxAndLen.first,
                        idxAndLen.second,
                        isVariadic[index]
                );
            }
            if(auto concreteOp = llvm::dyn_cast<daphne::OrderOp>(op)) {
                auto idxAndLen = concreteOp.getODSOperandIndexAndLength(index);
                static bool isVariadic[] = {false, true, true, false};
                return std::make_tuple(
                        idxAndLen.first,
                        idxAndLen.second,
                        isVariadic[index]
                );
            }
            throw std::runtime_error(
                    "lowering to kernel call not yet supported for this variadic operation: "
                    + op->getName().getStringRef().str()
            );
        }

        /**
         * @brief The value of type `DaphneContext` to insert as the last
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

            // check CUDA support and valid device ID
//            auto attr = op->getAttr("cuda_device");
//            if(attr && attr.dyn_cast<IntegerAttr>().getInt() > -1) {
            if(op->hasAttr("cuda_device")) {
//                op->hasTrait<mlir::OpTrait::CUDASupport>() &&
//                auto attr = op->getAttr("cuda_device");
//                if(attr && attr.dyn_cast<IntegerAttr>().getInt() > -1) {
//                if(attr.dyn_cast<IntegerAttr>().getInt() > -1)
                    callee << "CUDA";
//                else
//                    std::cout << "attr = null: " << op->getName().getStringRef().str() << std::endl;
            }

            callee << '_' << op->getName().stripDialect().data();


            // TODO Don't enumerate all ops, decide based on a trait.
            const bool generalizeInputTypes =
                llvm::isa<daphne::CreateFrameOp>(op) ||
                llvm::isa<daphne::DistributedComputeOp>(op) ||
                llvm::isa<daphne::NumCellsOp>(op) ||
                llvm::isa<daphne::NumColsOp>(op) ||
                llvm::isa<daphne::NumRowsOp>(op) ||
                llvm::isa<daphne::IncRefOp>(op) ||
                llvm::isa<daphne::DecRefOp>(op);

            // Append names of result types to the kernel name.
            Operation::result_type_range resultTypes = op->getResultTypes();
            for(size_t i = 0; i < resultTypes.size(); i++)
                callee << "__" << CompilerUtils::mlirTypeToCppTypeName(resultTypes[i], false);

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
                // detect if an operation has variadic ODS operands with any N.
                op->hasTrait<OpTrait::VariadicOperands>() ||
                op->hasTrait<OpTrait::AtLeastNOperands<1>::Impl>() ||
                op->hasTrait<OpTrait::AtLeastNOperands<2>::Impl>()
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

                    callee << "__" << CompilerUtils::mlirTypeToCppTypeName(operandTypes[idx], generalizeInputTypes);
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
                        for(int64_t k = 0; k < len; k++)
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
                    callee << "__" << CompilerUtils::mlirTypeToCppTypeName(operandTypes[i], generalizeInputTypes);
                    newOperands.push_back(op->getOperand(i));
                }
            
            if(auto groupOp = llvm::dyn_cast<daphne::GroupOp>(op)) {
                // GroupOp carries the aggregation functions to apply as an
                // attribute. Since attributes to not automatically become
                // inputs to the kernel call, we need to add them explicitly
                // here.
                
                callee << "__GroupEnum_variadic__size_t";
                
                ArrayAttr aggFuncs = groupOp.aggFuncs();
                const size_t numAggFuncs = aggFuncs.size();
                const Type t = rewriter.getIntegerType(32, false);
                auto cvpOp = rewriter.create<daphne::CreateVariadicPackOp>(
                        loc,
                        daphne::VariadicPackType::get(rewriter.getContext(), t),
                        rewriter.getIndexAttr(numAggFuncs)
                );
                size_t k = 0;
                for(Attribute aggFunc : aggFuncs.getValue())
                    rewriter.create<daphne::StoreVariadicPackOp>(
                            loc,
                            cvpOp,
                            rewriter.create<daphne::ConstantOp>(
                                    loc,
                                    rewriter.getIntegerAttr(
                                            t,
                                            static_cast<uint32_t>(
                                                    aggFunc.dyn_cast<daphne::GroupEnumAttr>().getValue()
                                            )
                                    )
                            ),
                            rewriter.getIndexAttr(k++)
                    );
                newOperands.push_back(cvpOp);
                newOperands.push_back(rewriter.create<daphne::ConstantOp>(
                        loc, rewriter.getIndexAttr(numAggFuncs))
                );
            }

            if(auto distCompOp = llvm::dyn_cast<daphne::DistributedComputeOp>(op)) {
                MLIRContext newContext;
                OpBuilder tempBuilder(&newContext);
                std::string funcName = "dist";

                auto &bodyBlock = distCompOp.body().front();
                auto funcType = tempBuilder.getFunctionType(
                    bodyBlock.getArgumentTypes(), bodyBlock.getTerminator()->getOperandTypes());
                auto funcOp = tempBuilder.create<FuncOp>(loc, funcName, funcType);

                BlockAndValueMapping mapper;
                distCompOp.body().cloneInto(&funcOp.getRegion(), mapper);

                // write recompile region as string constant
                std::string s;
                llvm::raw_string_ostream stream(s);
                funcOp.print(stream);

                auto strTy = daphne::StringType::get(rewriter.getContext());
                Value
                    rewriteStr = rewriter.create<daphne::ConstantOp>(loc, strTy, rewriter.getStringAttr(stream.str()));
                callee << "__" << CompilerUtils::mlirTypeToCppTypeName(strTy, false);
                newOperands.push_back(rewriteStr);
            }

            // Inject the current DaphneContext as the last input parameter to
            // all kernel calls, unless it's a CreateDaphneContextOp.
            if(!llvm::isa<daphne::CreateDaphneContextOp>(op))
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
        RewriteToCallKernelOpPass() = default;
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
            daphne::StoreVariadicPackOp,
            daphne::VectorizedPipelineOp,
            daphne::GenericCallOp
    >();
    target.addDynamicallyLegalOp<daphne::CastOp>([](daphne::CastOp op) {
        return op.isTrivialCast() || op.isMatrixPropertyCast();
    });

    // Determine the DaphneContext valid in the MLIR function being rewritten.
    mlir::Value dctx = CompilerUtils::getDaphneContext(func);
    func->walk([&](daphne::VectorizedPipelineOp vpo)
    {
      vpo.ctxMutable().assign(dctx);
    });

    // Apply conversion to CallKernelOps.
    patterns.insert<KernelReplacement>(&getContext(), dctx);
    if (failed(applyPartialConversion(func, target, std::move(patterns))))
        signalPassFailure();

}

std::unique_ptr<Pass> daphne::createRewriteToCallKernelOpPass()
{
    return std::make_unique<RewriteToCallKernelOpPass>();
}
