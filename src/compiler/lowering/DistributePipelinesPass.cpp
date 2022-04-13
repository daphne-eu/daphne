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
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/**
 * @brief Replaces vectorized pipelines by distributed pipelines.
 */
struct DistributePipelines : public OpConversionPattern<daphne::VectorizedPipelineOp>
{
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(daphne::VectorizedPipelineOp op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext newContext;
        OpBuilder tempBuilder(&newContext);
        std::string funcName = "dist";

        auto &bodyBlock = op.body().front();
        auto funcType = tempBuilder.getFunctionType(
            bodyBlock.getArgumentTypes(), bodyBlock.getTerminator()->getOperandTypes());
        auto funcOp = tempBuilder.create<FuncOp>(op.getLoc(), funcName, funcType);

        BlockAndValueMapping mapper;
        op.body().cloneInto(&funcOp.getRegion(), mapper);

        std::string s;
        llvm::raw_string_ostream stream(s);
        funcOp.print(stream);
        Value irStr = rewriter.create<daphne::ConstantOp>(op.getLoc(), stream.str());
        
        rewriter.replaceOpWithNewOp<daphne::DistributedPipelineOp>(
                op.getOperation(),
                op.outputs().getTypes(), irStr, op.inputs(),
                op.out_rows(), op.out_cols(), op.splits(), op.combines()
        );
        
        return success();
    }
};

struct DistributePipelinesPass
    : public PassWrapper<DistributePipelinesPass, OperationPass<ModuleOp>>
{
    void runOnOperation() final;
};

void DistributePipelinesPass::runOnOperation()
{
    auto module = getOperation();

    OwningRewritePatternList patterns(&getContext());

    // convert other operations
    ConversionTarget target(getContext());
    // TODO do we need all these?
    target.addLegalDialect<StandardOpsDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, FuncOp>();
    target.addDynamicallyLegalOp<daphne::VectorizedPipelineOp>([](daphne::VectorizedPipelineOp op)
    {
        // TODO Carefully decide if this pipeline shall be distributed, e.g.,
        // based on physical input size. For now, all pipelines are distributed
        // (false means this pipeline is illegal and must be rewritten).
        return false;
    });

    patterns.insert<DistributePipelines>(&getContext());

    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createDistributePipelinesPass()
{
    return std::make_unique<DistributePipelinesPass>();
}
