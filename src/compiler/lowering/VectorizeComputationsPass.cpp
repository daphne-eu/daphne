/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

using namespace mlir;

namespace
{
struct Vectorize : public OpInterfaceConversionPattern<daphne::Vectorizable>
{
    using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

    LogicalResult
    matchAndRewrite(daphne::Vectorizable op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
        auto vSplits = op.getVectorSplits();
        auto vCombines = op.getVectorCombines();
        // TODO: although we do create enum attributes, it might make sense/make it easier to
        //  just directly use an I64ArrayAttribute
        std::vector<Attribute> vSplitAttrs;
        vSplitAttrs.reserve(vSplits.size());
        for (auto vSplit : vSplits) {
            vSplitAttrs.push_back(daphne::VectorSplitAttr::get(getContext(), vSplit));
        }
        std::vector<Attribute> vCombineAttrs;
        vCombineAttrs.reserve(vCombines.size());
        for (auto vCombine : vCombines) {
            vCombineAttrs.push_back(daphne::VectorCombineAttr::get(getContext(), vCombine));
        }
        auto pipeline = rewriter.create<daphne::VectorizedPipelineOp>(op->getLoc(),
            op->getResultTypes(),
            operands,
            rewriter.getArrayAttr(vSplitAttrs),
            rewriter.getArrayAttr(vCombineAttrs));
        Block *bodyBlock = rewriter.createBlock(&pipeline.body());

        for(auto argTy : ValueRange(operands).getTypes()) {
            bodyBlock->addArgument(argTy);
        }
        auto *cloned = op->clone();
        cloned->setOperands(bodyBlock->getArguments());
        rewriter.setInsertionPointToStart(bodyBlock);
        rewriter.insert(cloned);
        auto retOp = rewriter.create<daphne::ReturnOp>(cloned->getLoc(), cloned->getResults());

        rewriter.replaceOp(op, pipeline->getResults());
        return success();
    }
};

struct VectorizeComputationsPass
    : public PassWrapper<VectorizeComputationsPass, OperationPass<ModuleOp>>
{
    void runOnOperation() final;
};
}

void VectorizeComputationsPass::runOnOperation()
{
    auto module = getOperation();

    OwningRewritePatternList patterns(&getContext());

    // convert other operations
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect, LLVM::LLVMDialect, scf::SCFDialect>();
    target.addLegalOp<ModuleOp, FuncOp>();
    target.addDynamicallyLegalDialect<daphne::DaphneDialect>([](Operation *op)
    {
      // TODO: support scalars
      return !llvm::isa<daphne::Vectorizable>(op) || op->getParentOfType<daphne::VectorizedPipelineOp>()
          || llvm::any_of(op->getOperandTypes(), [&](Type ty)
          { return !ty.isa<daphne::MatrixType>(); });
    });

    patterns.insert<Vectorize>(&getContext());

    if(failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
    //module.dump();
}

std::unique_ptr<Pass> daphne::createVectorizeComputationsPass()
{
    return std::make_unique<VectorizeComputationsPass>();
}
