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
#include <set>

using namespace mlir;

namespace
{
struct VectorizeComputationsPass
    : public PassWrapper<VectorizeComputationsPass, OperationPass<FuncOp>>
{
    void runOnOperation() final;
};
}

void VectorizeComputationsPass::runOnOperation()
{
    auto func = getOperation();

    auto isMatrixComputation = [](Operation *v)
    {
      return llvm::any_of(v->getOperandTypes(), [&](Type ty)
      {
        return ty.isa<daphne::MatrixType>();
      });
    };

    // Find vectorizable operations and their inputs of vectorizable operations
    std::vector<daphne::Vectorizable> vectOps;
    func->walk([&](daphne::Vectorizable op)
    {
      if(isMatrixComputation(op))
          vectOps.emplace_back(op);
    });
    std::vector<daphne::Vectorizable> vectorizables(vectOps.begin(), vectOps.end());
    std::multimap<daphne::Vectorizable, daphne::Vectorizable> possibleMerges;
    for(auto v : vectorizables) {
        for(auto e : llvm::zip(v->getOperands(), v.getVectorSplits())) {
            auto operand = std::get<0>(e);
            auto defOp = operand.getDefiningOp<daphne::Vectorizable>();
            if(defOp && isMatrixComputation(defOp)) {
                auto split = std::get<1>(e);
                // find the corresponding `OpResult` to figure out combine
                auto opResult = *llvm::find(defOp->getResults(), operand);
                auto combine = defOp.getVectorCombines()[opResult.getResultNumber()];

                if(split == daphne::VectorSplit::ROWS) {
                    if(combine == daphne::VectorCombine::ROWS)
                        possibleMerges.insert({v, defOp});
                }
                else if (split == daphne::VectorSplit::NONE) {
                    // can't be merged
                }
                else {
                    throw std::runtime_error("VectorSplit case `" + stringifyEnum(split).str() + "` not handled");
                }
            }
        }
    }

    // Collect vectorizable operations that can be computed together in pipelines
    std::map<daphne::Vectorizable, size_t> operationToPipelineIx;
    std::vector<std::vector<daphne::Vectorizable>> pipelines;
    for(auto vIt = vectorizables.rbegin(); vIt != vectorizables.rend(); ++vIt) {
        auto v = *vIt;
        size_t pipelineIx;
        auto pipelineIt = operationToPipelineIx.find(v);
        if(pipelineIt != operationToPipelineIx.end()) {
            pipelineIx = pipelineIt->second;
        }
        else {
            pipelineIx = pipelines.size();
            std::vector<daphne::Vectorizable> pipeline;
            pipeline.push_back(v);
            pipelines.push_back(pipeline);
        }

        // iterate all operands that could be combined into the pipeline
        auto itRange = possibleMerges.equal_range(v);
        for(auto it = itRange.first; it != itRange.second; ++it) {
            auto operandVectorizable = it->second;
            // TODO: more in depth check for if merging is possible, currently this is very conservative
            if(operandVectorizable->hasOneUse()) {
                pipelines[pipelineIx].push_back(operandVectorizable);
                operationToPipelineIx[operandVectorizable] = pipelineIx;
            }
        }
    }

    OpBuilder builder(func);
    // Create the `VectorizedPipelineOp`s
    for(auto pipeline : pipelines) {
        auto valueIsPartOfPipeline = [&](Value operand)
        {
          return llvm::any_of(pipeline, [&](daphne::Vectorizable lv)
          { return lv == operand.getDefiningOp(); });
        };
        std::vector<Attribute> vSplitAttrs;
        std::vector<Attribute> vCombineAttrs;
        std::vector<Location> locations;
        std::vector<Value> results;
        std::vector<Value> operands;
        std::vector<Value> outRows;
        std::vector<Value> outCols;

        // first op in pipeline is last in DAG
        // TODO: more complex behaviour will be necessary here once `hasOneUse` check above gets more complex
        builder.setInsertionPoint(pipeline.front());
        for(auto vIt = pipeline.rbegin(); vIt != pipeline.rend(); ++vIt) {
            auto v = *vIt;
            auto vSplits = v.getVectorSplits();
            auto vCombines = v.getVectorCombines();
            // TODO: although we do create enum attributes, it might make sense/make it easier to
            //  just directly use an I64ArrayAttribute
            for(auto i = 0u; i < v->getNumOperands(); ++i) {
                auto operand = v->getOperand(i);
                if(!valueIsPartOfPipeline(operand)) {
                    vSplitAttrs.push_back(daphne::VectorSplitAttr::get(&getContext(), vSplits[i]));
                    operands.push_back(operand);
                }
            }
            for(auto vCombine : vCombines) {
                vCombineAttrs.push_back(daphne::VectorCombineAttr::get(&getContext(), vCombine));
            }
            locations.push_back(v->getLoc());
            for(auto result : v->getResults()) {
                results.push_back(result);
            }
            for (auto outSize : v.createOpsOutputSizes(builder)) {
                outRows.push_back(outSize.first);
                outCols.push_back(outSize.second);
            }
        }
        auto loc = pipeline.front().getLoc();
        auto pipelineOp = builder.create<daphne::VectorizedPipelineOp>(loc,
            ValueRange(results).getTypes(),
            operands,
            outRows,
            outCols,
            builder.getArrayAttr(vSplitAttrs),
            builder.getArrayAttr(vCombineAttrs),
            nullptr);
        Block *bodyBlock = builder.createBlock(&pipelineOp.body());

        for(auto argTy : ValueRange(operands).getTypes()) {
            bodyBlock->addArgument(argTy);
        }

        auto argsIx = 0u;
        auto resultsIx = 0u;
        for(auto vIt = pipeline.rbegin(); vIt != pipeline.rend(); ++vIt) {
            auto v = *vIt;
            auto numOperands = v->getNumOperands();
            auto numResults = v->getNumResults();

            v->moveBefore(bodyBlock, bodyBlock->end());

            for(auto i = 0u; i < numOperands; ++i) {
                if(!valueIsPartOfPipeline(v->getOperand(i))) {
                    v->setOperand(i, bodyBlock->getArgument(argsIx++));
                }
            }

            auto pipelineReplaceResults = pipelineOp->getResults().drop_front(resultsIx).take_front(numResults);
            resultsIx += numResults;
            for(auto z : llvm::zip(v->getResults(), pipelineReplaceResults)) {
                auto old = std::get<0>(z);
                auto replacement = std::get<1>(z);

                // TODO: switch to type based size inference instead
                // FIXME: if output is dynamic sized, we can't do this
                // replace `NumRowOp` and `NumColOp`s for output size inference
                for (auto &use : old.getUses()) {
                    auto *op = use.getOwner();
                    if (auto nrowOp = llvm::dyn_cast<daphne::NumRowsOp>(op)) {
                        nrowOp.replaceAllUsesWith(pipelineOp.out_rows()[replacement.getResultNumber()]);
                        nrowOp.erase();
                    }
                    if (auto ncolOp = llvm::dyn_cast<daphne::NumColsOp>(op)) {
                        ncolOp.replaceAllUsesWith(pipelineOp.out_cols()[replacement.getResultNumber()]);
                        ncolOp.erase();
                    }
                }
                // Replace only if not used by pipeline op
                old.replaceUsesWithIf(replacement, [&](OpOperand &opOperand)
                {
                  return llvm::count(pipeline, opOperand.getOwner()) == 0;
                });
            }
        }
        // TODO: remove size information in bodyBlock
        builder.setInsertionPointToEnd(bodyBlock);
        builder.create<daphne::ReturnOp>(loc, results);
    }
}

std::unique_ptr<Pass> daphne::createVectorizeComputationsPass()
{
    return std::make_unique<VectorizeComputationsPass>();
}
