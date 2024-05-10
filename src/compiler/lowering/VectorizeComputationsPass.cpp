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


#include "compiler/utils/CompilerUtils.h"
#include <util/ErrorHandler.h>
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <set>
#include <iostream>

using namespace mlir;

namespace
{
    /**
     * @brief Recursive function checking if the given value is transitively dependant on the operation `op`.
     * @param value The value to check
     * @param op The operation to check
     * @return true if there is a dependency, false otherwise
     */
    bool valueDependsOnResultOf(Value value, Operation *op) {
        if (auto defOp = value.getDefiningOp()) {
            if (defOp == op)
                return true;
#if 1
            // TODO This crashes if defOp and op are not in the same block.
            // At the same time, it does not seem to be strictly required.
//            if (defOp->isBeforeInBlock(op))
            // Nevertheless, this modified line seems to be a good soft-filter;
            // without that, the vectorization pass may take very long on
            // programs with 100s of operations.
            if (defOp->getBlock() == op->getBlock() && defOp->isBeforeInBlock(op))
                // can't have results of `op` as inputs, as it is defined before
                return false;
#endif
            for (auto operand : defOp->getOperands()) {
                if (valueDependsOnResultOf(operand, op))
                    return true;
            }
        }
        return false;
    }

    /**
     * @brief Check if the vectorizable operation can directly be fused into the pipeline, without requiring any other
     * operation to be fused first.
     * @param opBefore The vectorizable operation to check
     * @param pipeline The pipeline
     * @return true if it can be directly fused, false otherwise
     */
    bool isDirectlyFusible(daphne::Vectorizable opBefore, const std::vector<daphne::Vectorizable>& pipeline) {
        for (auto pipeOp : pipeline) {
            for (auto operand : pipeOp->getOperands()) {
                if (std::find(pipeline.begin(), pipeline.end(), operand.getDefiningOp()) != pipeline.end()) {
                    // transitive dependencies inside the pipeline are of course fine.
                    continue;
                }
                if (operand.getDefiningOp() != opBefore && valueDependsOnResultOf(operand, opBefore)) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * @brief Greedily fuses the operation into the pipeline if possible.
     * @param operationToPipelineIx A map of operations to their index in the pipelines collection
     * @param pipelines The collection of pipelines
     * @param currentPipelineIx The index of the current pipeline into which we want to possibly fuse the operation
     * @param operationToCheck The operation we possibly want to fuse into the current pipeline
     */
    void greedyPipelineFusion(std::map<daphne::Vectorizable, size_t> &operationToPipelineIx,
                              std::vector<std::vector<daphne::Vectorizable>> &pipelines,
                              size_t currentPipelineIx, daphne::Vectorizable operationToCheck) {
        auto &currentPipeline = pipelines[currentPipelineIx];
        auto existingPipelineIt = operationToPipelineIx.find(operationToCheck);
        if(existingPipelineIt != operationToPipelineIx.end()) {
            // existing pipeline is sure to be after the current pipeline (due to reverse iteration order)
            auto existingPipelineIx = existingPipelineIt->second;
            auto &existingPipeline = pipelines[existingPipelineIx];
            for (auto op : currentPipeline) {
                if (!isDirectlyFusible(op, existingPipeline)) {
                    continue;
                }
            }
            // append existing to current
            currentPipeline.insert(currentPipeline.end(), existingPipeline.begin(), existingPipeline.end());
            for (auto vectorizable : existingPipeline) {
                operationToPipelineIx[vectorizable] = currentPipelineIx;
            }
            // just make it empty, it will be skipped later. Ixs changes and reshuffling is therefore not necessary.
            existingPipeline.clear();
        }
        else if(isDirectlyFusible(operationToCheck, currentPipeline)) {
            currentPipeline.push_back(operationToCheck);
            operationToPipelineIx[operationToCheck] = currentPipelineIx;
        }
    }

    /**
     * @brief Moves operation which are between the operations, which should be fused into a single pipeline, before
     * or after the position where the pipeline will be placed.
     * @param pipelinePosition The position where the pipeline will be
     * @param pipeline The pipeline for which this function should be executed
     */
    void movePipelineInterleavedOperations(Block::iterator pipelinePosition, const std::vector<daphne::Vectorizable> &pipeline) {
        // first operation in pipeline vector is last in IR, and the last is the first
        auto startPos = pipeline.back()->getIterator();
        auto endPos = pipeline.front()->getIterator();
        auto currSkip = pipeline.rbegin();
        std::vector<Operation*> moveBeforeOps;
        std::vector<Operation*> moveAfterOps;
        for(auto it = startPos; it != endPos; ++it) {
            if (it == (*currSkip)->getIterator()) {
                ++currSkip;
                continue;
            }

            bool dependsOnPipeline = false;
            auto pipelineOpsBeforeIt = currSkip;
            while (--pipelineOpsBeforeIt != pipeline.rbegin()) {
                for (auto operand : it->getOperands()) {
                    if(valueDependsOnResultOf(operand, *pipelineOpsBeforeIt)) {
                        dependsOnPipeline = true;
                        break;
                    }
                }
                if (dependsOnPipeline) {
                    break;
                }
            }
            // check first pipeline op
            for (auto operand : it->getOperands()) {
                if(valueDependsOnResultOf(operand, *pipelineOpsBeforeIt)) {
                    dependsOnPipeline = true;
                    break;
                }
            }
            if (dependsOnPipeline) {
                moveAfterOps.push_back(&(*it));
            }
            else {
                moveBeforeOps.push_back(&(*it));
            }
        }

        for(auto moveBeforeOp: moveBeforeOps) {
            moveBeforeOp->moveBefore(pipelinePosition->getBlock(), pipelinePosition);
        }
        for(auto moveAfterOp: moveAfterOps) {
            moveAfterOp->moveAfter(pipelinePosition->getBlock(), pipelinePosition);
            pipelinePosition = moveAfterOp->getIterator();
        }
    }

    struct VectorizeComputationsPass : public PassWrapper<VectorizeComputationsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;
    };
}

void VectorizeComputationsPass::runOnOperation()
{
    auto func = getOperation();
    // TODO: fuse pipelines that have the matching inputs, even if no output of the one pipeline is used by the other.
    //  This requires multi-returns in way more cases, which is not implemented yet.

    // Find vectorizable operations and their inputs of vectorizable operations
    std::vector<daphne::Vectorizable> vectOps;
    func->walk([&](daphne::Vectorizable op)
    {
      if(CompilerUtils::isMatrixComputation(op))
          vectOps.emplace_back(op);
    });
    std::vector<daphne::Vectorizable> vectorizables(vectOps.begin(), vectOps.end());
    std::multimap<daphne::Vectorizable, daphne::Vectorizable> possibleMerges;
    for(auto v : vectorizables) {
        for(auto e : llvm::zip(v->getOperands(), v.getVectorSplits())) {
            auto operand = std::get<0>(e);
            auto defOp = operand.getDefiningOp<daphne::Vectorizable>();
            if(defOp && v->getBlock() == defOp->getBlock() && CompilerUtils::isMatrixComputation(defOp)) {
                // defOp is not a candidate for fusion with v, if the
                // result/operand along which we would fuse is used within a
                // nested block (e.g., control structure) between defOp and v.
                // In that case, we cannot, in general, move the using
                // operation before or after the pipeline.
                // TODO This is actually too restrictive. There are situations
                // when it would be safe (also taking NoSideEffect into
                // account).
                bool qualified = true;
                for(OpOperand & use : operand.getUses()) {
                    Operation * user = use.getOwner();
                    if(user->getBlock() != v->getBlock()) {
                        // user must be in a child block of the block in which
                        // v resides, because we have already checked that v
                        // and defOp are in the same block.
                        while(user->getBlock() != v->getBlock())
                            user = user->getParentOp();
                        if(user->isBeforeInBlock(v)) {
                            qualified = false;
                            break;
                        }
                    }
                }

                if(qualified){
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
                        throw ErrorHandler::compilerError(
                            v, "VectorizeComputationsPass",
                            "VectorSplit case `" + stringifyEnum(split).str() +
                                "` not handled");
                    }
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
            // TODO: this fuses greedily, the first pipeline we can fuse this operation into, we do. improve
            greedyPipelineFusion(operationToPipelineIx, pipelines, pipelineIx, operandVectorizable);
        }
    }

    OpBuilder builder(func);
    // Create the `VectorizedPipelineOp`s
    for(auto pipeline : pipelines) {
        if(pipeline.empty()) {
            continue;
        }
        auto valueIsPartOfPipeline = [&](Value operand) {
            return llvm::any_of(pipeline, [&](daphne::Vectorizable lv) { return lv == operand.getDefiningOp(); });
        };
        std::vector<Attribute> vSplitAttrs;
        std::vector<Attribute> vCombineAttrs;
        std::vector<Location> locations;
        std::vector<Value> results;
        std::vector<Value> operands;
        std::vector<Value> outRows;
        std::vector<Value> outCols;

        // first op in pipeline is last in IR
        builder.setInsertionPoint(pipeline.front());
        // move all operations, between the operations that will be part of the pipeline, before or after the
        // completed pipeline
        movePipelineInterleavedOperations(builder.getInsertionPoint(), pipeline);
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
            for(auto result: v->getResults()) {
                results.push_back(result);
            }
            for(auto outSize: v.createOpsOutputSizes(builder)) {
                outRows.push_back(outSize.first);
                outCols.push_back(outSize.second);
            }
        }
        std::vector<Location> locs;
        locs.reserve(pipeline.size());
        for(auto op: pipeline) {
            locs.push_back(op->getLoc());
        }
        auto loc = builder.getFusedLoc(locs);
        auto pipelineOp = builder.create<daphne::VectorizedPipelineOp>(loc,
            ValueRange(results).getTypes(),
            operands,
            outRows,
            outCols,
            builder.getArrayAttr(vSplitAttrs),
            builder.getArrayAttr(vCombineAttrs),
            nullptr);
        Block *bodyBlock = builder.createBlock(&pipelineOp.getBody());

        for(size_t i = 0u; i < operands.size(); ++i) {
            auto argTy = operands[i].getType();
            switch (vSplitAttrs[i].cast<daphne::VectorSplitAttr>().getValue()) {
                case daphne::VectorSplit::ROWS: {
                    auto matTy = argTy.cast<daphne::MatrixType>();
                    // only remove row information
                    argTy = matTy.withShape(-1, matTy.getNumCols());
                    break;
                }
                case daphne::VectorSplit::NONE:
                    // keep any size information
                    break;
            }
            bodyBlock->addArgument(argTy, builder.getUnknownLoc());
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
            for(auto z: llvm::zip(v->getResults(), pipelineReplaceResults)) {
                auto old = std::get<0>(z);
                auto replacement = std::get<1>(z);

                // TODO: switch to type based size inference instead
                // FIXME: if output is dynamic sized, we can't do this
                // replace `NumRowOp` and `NumColOp`s for output size inference
                for(auto& use: old.getUses()) {
                    auto* op = use.getOwner();
                    if(auto nrowOp = llvm::dyn_cast<daphne::NumRowsOp>(op)) {
                        nrowOp.replaceAllUsesWith(pipelineOp.getOutRows()[replacement.getResultNumber()]);
                        nrowOp.erase();
                    }
                    if(auto ncolOp = llvm::dyn_cast<daphne::NumColsOp>(op)) {
                        ncolOp.replaceAllUsesWith(pipelineOp.getOutCols()[replacement.getResultNumber()]);
                        ncolOp.erase();
                    }
                }
                // Replace only if not used by pipeline op
                old.replaceUsesWithIf(replacement, [&](OpOperand& opOperand) {
                    return llvm::count(pipeline, opOperand.getOwner()) == 0;
                });
            }
        }
        bodyBlock->walk([](Operation* op) {
            for(auto resVal: op->getResults()) {
                if(auto ty = resVal.getType().dyn_cast<daphne::MatrixType>()) {
                    resVal.setType(ty.withShape(-1, -1));
                }
            }
        });
        builder.setInsertionPointToEnd(bodyBlock);
        builder.create<daphne::ReturnOp>(loc, results);
    }
}

std::unique_ptr<Pass> daphne::createVectorizeComputationsPass() {
    return std::make_unique<VectorizeComputationsPass>();
}
