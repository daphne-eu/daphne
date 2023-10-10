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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/**
 * @brief Replaces vectorized pipelines by distributed pipelines.
 */
struct DistributePipelines : public OpConversionPattern<daphne::VectorizedPipelineOp>
{
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(daphne::VectorizedPipelineOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext newContext;
        OpBuilder tempBuilder(&newContext);
        std::string funcName = "dist";

        auto &bodyBlock = op.getBody().front();
        auto funcType = tempBuilder.getFunctionType(
            bodyBlock.getArgumentTypes(), bodyBlock.getTerminator()->getOperandTypes());
        auto funcOp = tempBuilder.create<func::FuncOp>(op.getLoc(), funcName, funcType);

        IRMapping mapper;
        op.getBody().cloneInto(&funcOp.getRegion(), mapper);

        tempBuilder.setInsertionPointToStart(&funcOp.getBody().front());

        // Copy constant operations into IR fragment.
        std::vector<Value> newInputs;
        std::vector<Attribute> newSplits;
        llvm::BitVector eraseVector(funcOp.getNumArguments());

        for (size_t idx = 0; idx < op.getBody().getNumArguments(); ++idx) {
            // Find operand from argument
            auto vecOperand = op.getInputs()[idx];

            if (auto constantOp = vecOperand.getDefiningOp<daphne::ConstantOp>()) {
                // Add constant operation and remove argument
                auto newConOp = constantOp.clone();
                tempBuilder.insert(newConOp);
                funcOp.getArgument(idx).replaceAllUsesWith(newConOp);
                // Erase vector
                eraseVector[idx] = true;

            } else {
                // Else add to input/splits array.
                newInputs.push_back(op.getInputs()[idx]);
                newSplits.push_back(op.getSplits()[idx]);
            }            
        }
        funcOp.eraseArguments(eraseVector);
        
        std::string s;
        llvm::raw_string_ostream stream(s);
        funcOp.print(stream);
        Value irStr = rewriter.create<daphne::ConstantOp>(op.getLoc(), stream.str());

        rewriter.replaceOpWithNewOp<daphne::DistributedPipelineOp>(
                op.getOperation(),
                op.getOutputs().getTypes(), irStr, newInputs,
                op.getOutRows(), op.getOutCols(), rewriter.getArrayAttr(newSplits), op.getCombines()
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

    RewritePatternSet patterns(&getContext());

    // convert other operations
    ConversionTarget target(getContext());
    // TODO do we need all these?
    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();
    target.addDynamicallyLegalOp<daphne::VectorizedPipelineOp>([](daphne::VectorizedPipelineOp op)
    {
        // TODO Carefully decide if this pipeline shall be distributed, e.g.,
        // based on physical input size. For now, all pipelines are distributed
        // (false means this pipeline is illegal and must be rewritten).
        return false;
    });

    patterns.add<DistributePipelines>(&getContext());

    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createDistributePipelinesPass()
{
    return std::make_unique<DistributePipelinesPass>();
}
