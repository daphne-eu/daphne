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
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include <compiler/utils/VectorizationUtils.h>
#include <util/ErrorHandler.h>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>
#include <memory>
#include <set>

using namespace mlir;

namespace {

/**
 * @brief Check if the vectorizable operation can directly be fused into the
 * pipeline, without requiring any other operation to be fused first.
 * @param opBefore The vectorizable operation to check
 * @param pipeline The pipeline
 * @return true if it can be directly fused, false otherwise
 */
bool isDirectlyFusible(daphne::Vectorizable opBefore, const std::vector<daphne::Vectorizable> &pipeline) {
    for (auto pipeOp : pipeline) {
        for (auto operand : pipeOp->getOperands()) {
            if (std::find(pipeline.begin(), pipeline.end(), operand.getDefiningOp()) != pipeline.end()) {
                // transitive dependencies inside the pipeline are of course
                // fine.
                continue;
            }
            if (operand.getDefiningOp() != opBefore && VectorizationUtils::valueDependsOnResultOf(operand, opBefore)) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Greedily fuses the operation into the pipeline if possible.
 * @param operationToPipelineIx A map of operations to their index in the
 * pipelines collection
 * @param pipelines The collection of pipelines
 * @param currentPipelineIx The index of the current pipeline into which we want
 * to possibly fuse the operation
 * @param operationToCheck The operation we possibly want to fuse into the
 * current pipeline
 */
void greedyPipelineFusion(std::map<daphne::Vectorizable, size_t> &operationToPipelineIx,
                          std::vector<std::vector<daphne::Vectorizable>> &pipelines, size_t currentPipelineIx,
                          daphne::Vectorizable operationToCheck) {
    auto &currentPipeline = pipelines[currentPipelineIx];
    auto existingPipelineIt = operationToPipelineIx.find(operationToCheck);
    if (existingPipelineIt != operationToPipelineIx.end()) {
        // existing pipeline is sure to be after the current pipeline (due to
        // reverse iteration order)
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
        // just make it empty, it will be skipped later. Ixs changes and
        // reshuffling is therefore not necessary.
        existingPipeline.clear();
    } else if (isDirectlyFusible(operationToCheck, currentPipeline)) {
        currentPipeline.push_back(operationToCheck);
        operationToPipelineIx[operationToCheck] = currentPipelineIx;
    }
}

struct VectorizeComputationsPass : public PassWrapper<VectorizeComputationsPass, OperationPass<func::FuncOp>> {
    /**
     * @brief If set to `false`, this pass will consider all vectorizable ops (those that have the MLIR interface
     * `Vectorizable`) with at least one matrix argument or result for vectorization; if set to `true` only ops that
     * additionally have a specific attribute (`CompilerUtils::ATTR_VEC`) set to `true` will be considered.
     */
    const bool isRestricted;

    VectorizeComputationsPass(bool isRestricted) : isRestricted(isRestricted){};

    void runOnOperation() final;
};
} // namespace

void VectorizeComputationsPass::runOnOperation() {
    auto func = getOperation();
    // TODO: fuse pipelines that have the matching inputs, even if no output of
    // the one pipeline is used by the other.
    //  This requires multi-returns in way more cases, which is not implemented
    //  yet.

    // Find vectorizable operations and their inputs of vectorizable operations.
    // If this->isRestricted is false (default), all vectorizable operations with at least one matrix argument or result
    // are considered; if this->isRestricted is true, only ops that additionally have a specific attribute
    // (`CompilerUtils::ATTR_VEC`) set to `true` will be considered.
    std::vector<daphne::Vectorizable> vectOps;
    func->walk([&](daphne::Vectorizable op) {
        if (CompilerUtils::isMatrixComputation(op) &&
            (!isRestricted || CompilerUtils::isAttrTrue(op, CompilerUtils::ATTR_VEC)))
            vectOps.emplace_back(op);
    });
    std::vector<daphne::Vectorizable> vectorizables(vectOps.begin(), vectOps.end());
    std::multimap<daphne::Vectorizable, daphne::Vectorizable> possibleMerges;
    for (auto v : vectorizables) {
        for (auto e : llvm::zip(v->getOperands(), v.getVectorSplits())) {
            auto operand = std::get<0>(e);
            auto defOp = operand.getDefiningOp<daphne::Vectorizable>();
            if (defOp && v->getBlock() == defOp->getBlock() && CompilerUtils::isMatrixComputation(defOp) &&
                (!isRestricted || CompilerUtils::isAttrTrue(defOp, CompilerUtils::ATTR_VEC))) {
                // defOp is not a candidate for fusion with v, if the
                // result/operand along which we would fuse is used within a
                // nested block (e.g., control structure) between defOp and v.
                // In that case, we cannot, in general, move the using
                // operation before or after the pipeline.
                // TODO This is actually too restrictive. There are situations
                // when it would be safe (also taking NoSideEffect into
                // account).
                bool qualified = true;
                for (OpOperand &use : operand.getUses()) {
                    Operation *user = use.getOwner();
                    if (user->getBlock() != v->getBlock()) {
                        // user must be in a child block of the block in which
                        // v resides, because we have already checked that v
                        // and defOp are in the same block.
                        while (user->getBlock() != v->getBlock())
                            user = user->getParentOp();
                        if (user->isBeforeInBlock(v)) {
                            qualified = false;
                            break;
                        }
                    }
                }

                if (qualified) {
                    auto split = std::get<1>(e);
                    // find the corresponding `OpResult` to figure out combine
                    auto opResult = *llvm::find(defOp->getResults(), operand);
                    auto combine = defOp.getVectorCombines()[opResult.getResultNumber()];

                    if (split == daphne::VectorSplit::ROWS) {
                        if (combine == daphne::VectorCombine::ROWS)
                            possibleMerges.insert({v, defOp});
                    } else if (split == daphne::VectorSplit::NONE) {
                        // can't be merged
                    } else {
                        throw ErrorHandler::compilerError(v, "VectorizeComputationsPass",
                                                          "VectorSplit case `" + stringifyEnum(split).str() +
                                                              "` not handled");
                    }
                }
            }
        }
    }

    // Collect vectorizable operations that can be computed together in
    // pipelines
    std::map<daphne::Vectorizable, size_t> operationToPipelineIx;
    std::vector<std::vector<daphne::Vectorizable>> pipelines;
    for (auto vIt = vectorizables.rbegin(); vIt != vectorizables.rend(); ++vIt) {
        auto v = *vIt;
        size_t pipelineIx;
        auto pipelineIt = operationToPipelineIx.find(v);
        if (pipelineIt != operationToPipelineIx.end()) {
            pipelineIx = pipelineIt->second;
        } else {
            pipelineIx = pipelines.size();
            std::vector<daphne::Vectorizable> pipeline;
            pipeline.push_back(v);
            pipelines.push_back(pipeline);
        }

        // iterate all operands that could be combined into the pipeline
        auto itRange = possibleMerges.equal_range(v);
        for (auto it = itRange.first; it != itRange.second; ++it) {
            auto operandVectorizable = it->second;
            // TODO: this fuses greedily, the first pipeline we can fuse this
            // operation into, we do. improve
            greedyPipelineFusion(operationToPipelineIx, pipelines, pipelineIx, operandVectorizable);
        }
    }

    OpBuilder builder(func);
    // Create the `VectorizedPipelineOp`s
    for (auto pipeline : pipelines) {
        if (pipeline.empty()) {
            continue;
        }
        auto valueIsPartOfPipeline = [&](Value operand) {
            return llvm::any_of(pipeline, [&](daphne::Vectorizable lv) { return lv == operand.getDefiningOp(); });
        };

        // Find out the necessary information before creating the VectorizedPipelineOp.
        std::vector<Attribute> vSplitAttrs;
        std::vector<Attribute> vCombineAttrs;
        std::vector<Location> locations;
        std::vector<Value> results;
        std::vector<Value> operands;
        std::vector<Value> outRows;
        std::vector<Value> outCols;

        // The first op in the pipeline is the last op in the IR.
        builder.setInsertionPoint(pipeline.front());
        // In between the ops to fuse there may be other ops in the block. Move those other ops before (if they don't
        // depend on the pipeline's results) or after (if they depend on the pipeline's results) the point where the
        // VectorizedPipelineOp will be created.
        std::vector<Operation *> pipelineOps;
        for (auto vo : pipeline)
            pipelineOps.push_back(vo.getOperation());
        VectorizationUtils::movePipelineInterleavedOperations(builder.getInsertionPoint(), pipelineOps);
        for (auto vIt = pipeline.rbegin(); vIt != pipeline.rend(); ++vIt) {
            auto v = *vIt;
            auto vSplits = v.getVectorSplits();
            auto vCombines = v.getVectorCombines();
            // TODO: although we do create enum attributes, it might make
            // sense/make it easier to
            //  just directly use an I64ArrayAttribute
            for (auto i = 0u; i < v->getNumOperands(); ++i) {
                auto operand = v->getOperand(i);
                if (!valueIsPartOfPipeline(operand)) {
                    vSplitAttrs.push_back(daphne::VectorSplitAttr::get(&getContext(), vSplits[i]));
                    operands.push_back(operand);
                }
            }
            for (auto vCombine : vCombines) {
                vCombineAttrs.push_back(daphne::VectorCombineAttr::get(&getContext(), vCombine));
            }
            locations.push_back(v->getLoc());
            for (auto result : v->getResults()) {
                results.push_back(result);
            }
            for (auto outSize : v.createOpsOutputSizes(builder)) {
                outRows.push_back(outSize.first);
                outCols.push_back(outSize.second);
            }
        }
        auto loc = builder.getFusedLoc(locations);
        // Determine the result types of the VectorizedPipelineOp.
        // Map scalar result types to matrix types, because we will rewrite the pipeline to return scalar results as 1x1
        // matrices further down.
        std::vector<Type> resTys;
        for (Value res : results) {
            Type resTy = res.getType();
            if (CompilerUtils::isScaType(resTy))
                resTys.push_back(daphne::MatrixType::get(&getContext(), resTy).withShape(1, 1));
            else
                resTys.push_back(resTy);
        }

        // Create the VectorizedPipelineOp.
        auto pipelineOp = builder.create<daphne::VectorizedPipelineOp>(loc, resTys, operands, outRows, outCols,
                                                                       builder.getArrayAttr(vSplitAttrs),
                                                                       builder.getArrayAttr(vCombineAttrs), nullptr);
        Block *bodyBlock = builder.createBlock(&pipelineOp.getBody());

        // Remove information on certain data characteristics from the operands inside the pipeline.
        for (size_t i = 0u; i < operands.size(); ++i) {
            auto argTy = operands[i].getType();
            switch (vSplitAttrs[i].cast<daphne::VectorSplitAttr>().getValue()) {
            case daphne::VectorSplit::ROWS: {
                // Remove information on the number of rows, because the number of rows of individual chunks is subject
                // to run-time scheduling.
                // TODO Other information (e.g., symmetry) could also become invalid on individual chunks.
                auto matTy = argTy.cast<daphne::MatrixType>();
                argTy = matTy.withShape(-1, matTy.getNumCols());
                break;
            }
            case daphne::VectorSplit::NONE:
                // All information on data characteristics stays valid, because this operand is broadcasted.
                break;
            }
            bodyBlock->addArgument(argTy, builder.getUnknownLoc());
        }

        // Move the ops to fuse into the newly created VectorizedPipelineOp.
        std::vector<Value> resultsWithScaAsMat; // same as results, but scalars casted to 1x1 matrices
        auto argsIx = 0u;
        auto resultsIx = 0u;
        for (auto vIt = pipeline.rbegin(); vIt != pipeline.rend(); ++vIt) {
            auto v = *vIt;
            auto numOperands = v->getNumOperands();
            auto numResults = v->getNumResults();

            // Move the op into the VectorizedPipelineOp.
            v->moveBefore(bodyBlock, bodyBlock->end());

            // Rewire the op's operands to the block arguments of the VectorizedPipelineOp.
            for (auto i = 0u; i < numOperands; ++i) {
                if (!valueIsPartOfPipeline(v->getOperand(i))) {
                    v->setOperand(i, bodyBlock->getArgument(argsIx++));
                }
            }

            // Rewire uses of the op's results to the corresponding results of the VectorizedPipelineOp.
            auto pipelineReplaceResults = pipelineOp->getResults().drop_front(resultsIx).take_front(numResults);
            resultsIx += numResults;
            for (auto [old, replacement] : llvm::zip(v->getResults(), pipelineReplaceResults)) {
                Value replacementVal;
                Type oldTy = old.getType();
                if (CompilerUtils::isScaType(oldTy)) {
                    // If this result of v is a scalar, cast this result to a 1x1 matrix inside the VectorizedPipelineOp
                    // and cast the corresponding pipeline result back to a scalar after the VectorizedPipelineOp.

                    builder.setInsertionPointAfter(v);
                    Value oldMat = builder.create<daphne::CastOp>(old.getLoc(), replacement.getType(), old);
                    builder.setInsertionPointAfter(pipelineOp);
                    Value replacementSca = builder.create<daphne::CastOp>(replacement.getLoc(), oldTy, replacement);

                    resultsWithScaAsMat.push_back(oldMat);
                    replacementVal = replacementSca;
                } else {
                    resultsWithScaAsMat.push_back(old);
                    replacementVal = replacement;
                }

                // TODO: switch to type based size inference instead
                // FIXME: if output is dynamic sized, we can't do this
                // replace `NumRowOp` and `NumColOp`s for output size inference
                // FIXME: there are cases where old.getUses returns a null value, which should not happen.
                // This indicates that we messed up somewhere in this pass and at this point, we have
                // an invalid IR. See #881, using the safer llvm::make_early_inc_range iterator circumvents the segfault
                // at this point, but dose not resolve the root cause (invalid IR).
                for (auto &use : llvm::make_early_inc_range(old.getUses())) {
                    auto *op = use.getOwner();
                    if (!op)
                        continue;

                    if (auto nrowOp = llvm::dyn_cast<daphne::NumRowsOp>(op)) {
                        nrowOp.replaceAllUsesWith(pipelineOp.getOutRows()[replacement.getResultNumber()]);
                        nrowOp.erase();
                    }
                    if (auto ncolOp = llvm::dyn_cast<daphne::NumColsOp>(op)) {
                        ncolOp.replaceAllUsesWith(pipelineOp.getOutCols()[replacement.getResultNumber()]);
                        ncolOp.erase();
                    }
                }
                // Replace only if not used by pipeline op
                old.replaceUsesWithIf(replacementVal, [&](OpOperand &opOperand) {
                    return llvm::count(pipeline, opOperand.getOwner()) == 0 &&
                           !pipelineOp->isProperAncestor(opOperand.getOwner());
                });
            }
        }

        // Set the shapes of matrix intermediates inside the pipeline to unknown.
        // TODO Also do that for other data characteristics that are not valid on the individual chunks (e.g.,
        // symmetry).
        // TODO Maybe we could benefit from another InferencePass inside the pipeline.
        bodyBlock->walk([](Operation *op) {
            for (auto resVal : op->getResults()) {
                if (auto ty = resVal.getType().dyn_cast<daphne::MatrixType>()) {
                    resVal.setType(ty.withShape(-1, -1));
                }
            }
        });

        // Create the VectorizedPipelineOp's terminating ReturnOp.
        builder.setInsertionPointToEnd(bodyBlock);
        builder.create<daphne::ReturnOp>(loc, resultsWithScaAsMat);
    }
}

std::unique_ptr<Pass> daphne::createVectorizeComputationsPass(bool isRestricted) {
    return std::make_unique<VectorizeComputationsPass>(isRestricted);
}
