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
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>

using namespace mlir;

namespace
{
struct Distribute : public OpInterfaceConversionPattern<daphne::Distributable>
{
    using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

    LogicalResult
    matchAndRewrite(daphne::Distributable op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
        std::vector<Value> distributedInputs;
        for (auto zipIt : llvm::zip(operands, op.getOperandDistrPrimitives())) {
            Value operand = std::get<0>(zipIt);
            bool isBroadcast = std::get<1>(zipIt);
            if (operand.getType().isa<daphne::HandleType>())
                // The operand is already distributed/broadcasted, we can
                // directly use it.
                // TODO Check if it is distributed the way we need it here
                // (distributed/broadcasted), but so far, this is not tracked
                // at compile-time.
                distributedInputs.push_back(operand);
            else if (auto co = dyn_cast_or_null<daphne::DistributedCollectOp>(operand.getDefiningOp()))
                // The operand has just been collected from a distributed data
                // object, so we should reuse the original distributed data
                // object.
                distributedInputs.push_back(co.getArg());
            else {
                // The operands need to be distributed/broadcasted first.
                Type t = daphne::HandleType::get(getContext(), operand.getType());
                if(isBroadcast)
                    distributedInputs.push_back(rewriter.create<daphne::BroadcastOp>(op->getLoc(), t, operand));
                else
                    distributedInputs.push_back(rewriter.create<daphne::DistributeOp>(op->getLoc(), t, operand));
            }
        }
        auto results = op.createEquivalentDistributedDAG(rewriter, distributedInputs);

        rewriter.replaceOp(op, results);
        return success();
    }
};

struct DistributeComputationsPass
    : public PassWrapper<DistributeComputationsPass, OperationPass<ModuleOp>>
{
    void runOnOperation() final;

    StringRef getArgument() const final { return "distribute-computation"; }
    StringRef getDescription() const final { return "TODO"; }
};
}

bool onlyMatrixOperands(Operation * op) {
    return llvm::all_of(op->getOperandTypes(), [](Type t) {
        return t.isa<daphne::MatrixType>();
    });
}

void DistributeComputationsPass::runOnOperation()
{
    auto module = getOperation();

    RewritePatternSet patterns(&getContext());

    // convert other operations
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, scf::SCFDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();
    target.addDynamicallyLegalDialect<daphne::DaphneDialect>([](Operation *op)
    {
        // An operation is legal (does not need to be replaced), if ...
        return
                // ... it is not distributable
                !llvm::isa<daphne::Distributable>(op) ||
                // ... it is inside some distributed computation already
                op->getParentOfType<daphne::DistributedComputeOp>() ||
                // ... not all of its operands are matrices
                // TODO Support distributing frames and scalars.
                !onlyMatrixOperands(op);
    });

    patterns.add<Distribute>(&getContext());

    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createDistributeComputationsPass()
{
    return std::make_unique<DistributeComputationsPass>();
}
