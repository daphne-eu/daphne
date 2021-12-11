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
#include "compiler/CompilerUtils.h"

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
            if(llvm::isa<daphne::CreateFrameOp, daphne::SetColLabelsOp>(op))
                return 2;
            if(llvm::isa<daphne::DistributedComputeOp>(op))
                return 1;
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
            if(auto concreteOp = llvm::dyn_cast<daphne::DistributedComputeOp>(op)) {
                auto idxAndLen = concreteOp.getODSOperandIndexAndLength(index);
                static bool isVariadic[] = {true};
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

        /**
         * @brief User configuration influencing the rewrite pass
         */
        const DaphneUserConfig& cfg;
    public:
        /**
         * Creates a new KernelReplacement rewrite pattern.
         *
         * @param mctx The MLIR context.
         * @param dctx The DaphneContext to pass to the kernels.
         * @param benefit
         */
        KernelReplacement(MLIRContext * mctx, Value dctx, const DaphneUserConfig& cfg,
                PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, mctx), dctx(dctx), cfg(cfg)
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
            std::string_view op_name{op->getName().stripDialect().data()};
#ifdef USE_CUDA
            //ToDo: this will go away with a gpu ops rewrite pass
            std::array<std::string_view, 9> gpu_ops({ "affineForward", "avgPoolForward", "batchNorm2DTestForward",
                    "biasAddForward", "conv2DForward", "matMul", "maxPoolForward", "reluForward", "softmaxForward"});
//            std::cout << op_name << std::endl;
            if(cfg.use_cuda) {
                if(std::find(gpu_ops.begin(), gpu_ops.end(), op_name) != gpu_ops.end()) {
                    callee << '_' << op_name << "_CUDA";
                }
                else
                    callee << '_' << op_name;
            }
            else
#endif
                callee << '_' << op_name;
//            std::cout << "callee: " << callee.str() << std::endl;
            // TODO Don't enumerate all ops, decide based on a trait.
            const bool generalizeInputTypes =
                llvm::isa<daphne::CreateFrameOp>(op) ||
                llvm::isa<daphne::DistributedComputeOp>(op) ||
                llvm::isa<daphne::NumCellsOp>(op) ||
                llvm::isa<daphne::NumColsOp>(op) ||
                llvm::isa<daphne::NumRowsOp>(op);
            
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
        const DaphneUserConfig& cfg;
        explicit RewriteToCallKernelOpPass(const DaphneUserConfig& cfg) : cfg(cfg) { }
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
    func->walk([&](daphne::VectorizedPipelineOp vpo)
    {
      vpo.ctxMutable().assign(dctx);
    });

    // Apply conversion to CallKernelOps.
    patterns.insert<KernelReplacement>(&getContext(), dctx, cfg);
    if (failed(applyPartialConversion(func, target, std::move(patterns))))
        signalPassFailure();

}

std::unique_ptr<Pass> daphne::createRewriteToCallKernelOpPass(const DaphneUserConfig& cfg)
{
    return std::make_unique<RewriteToCallKernelOpPass>(cfg);
}
