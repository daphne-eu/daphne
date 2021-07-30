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
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>

namespace mlir
{
namespace daphne
{
#include "ir/daphneir/DaphneDistributableOpInterface.cpp.inc"
}
}

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
        for (auto operand : operands) {
            if (operand.getType().isa<daphne::HandleType>()) {
                distributedInputs.push_back(operand);
            }
            else {
                // TODO: if DistributedCollectOp, just use handle input of it
                distributedInputs.push_back(rewriter.create<daphne::DistributeOp>(op->getLoc(),
                    daphne::HandleType::get(getContext(), operand.getType()),
                    operand));
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
};
}

void DistributeComputationsPass::runOnOperation()
{
    auto module = getOperation();

    OwningRewritePatternList patterns(&getContext());

    // convert other operations
    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, FuncOp>();
    target.addDynamicallyLegalDialect<daphne::DaphneDialect>([](Operation *op)
    {
      return !llvm::isa<daphne::Distributable>(op) || op->getParentOfType<daphne::DistributedComputeOp>();
    });

    patterns.insert<Distribute>(&getContext());

    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
    //module.dump();
}

std::unique_ptr<Pass> daphne::createDistributeComputationsPass()
{
    return std::make_unique<DistributeComputationsPass>();
}
