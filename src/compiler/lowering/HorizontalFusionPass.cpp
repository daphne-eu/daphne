/*
 *  Copyright 2025 The DAPHNE Consortium
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

#include <compiler/utils/VectorizationUtils.h>
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <vector>

#include <cstddef>

using namespace mlir;

namespace {

/**
 * @brief Performs horizontal fusion (sibling fusion) of existing `VectorizedPipelineOp`s in order to enable scan
 * sharing on common inputs.
 *
 * More precisely, this pass identifies all pairs of pipelines that have at least one non-broadcasted input in common
 * (in terms of both the value and the split). Broadcasted inputs are not relevant, since they cannot be used for scan
 * sharing. Then, it successively merges these pairs of pipelines by (1) creating a new `VectorizedPipelineOp` with the
 * concatenation of the inputs, splits, results, combines, and pipeline bodies of the two original pipelines, (2)
 * rewiring the results of the new pipeline with the uses of the results of the two original pipelines, and (3) erasing
 * the two original pipelines.
 *
 * After this pass, the fused pipelines may have redundant inputs, but these will be eliminated later through the
 * canonicalization of `VectorizedPipelineOp`.
 *
 * The following DaphneDSL script shows a simple example where horizontal fusion is useful:
 *
 * ```
 * X = rand(10^6, 10^3, 0.0, 1.0, 1, -1);
 * print(aggMin(X));
 * print(aggMax(X));
 * ```
 *
 * Without horizontal fusion, the two full aggregations would end up in separate pipelines, each of which needs to scan
 * the entire matrix `X`. With horizontal fusion, both aggregations end up in a single pipeline sharing the scan over
 * `X`, which is more efficient.
 */
struct HorizontalFusionPass : public PassWrapper<HorizontalFusionPass, OperationPass<func::FuncOp>> {
    void runOnOperation() final;
};

} // namespace

/**
 * @brief Moves all operations in the given source block to the given destination block (except for the terminating
 * `daphne::ReturnOp`), adds all used arguments of the given source block to the given destination block, and appends
 * the operands of the source block's `daphne::ReturnOp` to the given vector of return values.
 *
 * @param src The source block.
 * @param dst The destination block.
 * @param returnVals The return values (return parameter).
 */
void moveOperationsToBlock(Block *src, Block *dst, std::vector<Value> &returnVals) {
    while (!src->empty()) {
        auto op = src->begin();
        for (size_t i = 0; i < op->getNumOperands(); i++) {
            Value operand = op->getOperand(i);
            if (llvm::isa<BlockArgument>(operand)) {
                BlockArgument blockArg = dst->addArgument(operand.getType(), operand.getLoc());
                op->setOperand(i, blockArg);
            }
        }
        if (!llvm::isa<daphne::ReturnOp>(op))
            op->moveBefore(dst, dst->end());
        else {
            returnVals.insert(returnVals.end(), op->operand_begin(), op->operand_end());
            op->erase();
        }
    }
}

void HorizontalFusionPass::runOnOperation() {
    func::FuncOp func = getOperation();

    // -----------------------------------------------------------------
    // Step 1: Identify horizontal fusion opportunities
    // -----------------------------------------------------------------

    // A pair of pipelines is an opportunity for horizontal fusion if the two pipelines have at least one
    // non-broadcasted input in common (in terms of both the value and the split). Broadcasted inputs are not relevant,
    // since they cannot be used for scan sharing.

    // Create a vector of all VectorizedPipelineOps in the function.
    std::vector<daphne::VectorizedPipelineOp> vpOps;
    func->walk([&](daphne::VectorizedPipelineOp op) { vpOps.push_back(op); });

    // Find all pairs of pipelines for horizontal fusion.
    std::vector<std::pair<daphne::VectorizedPipelineOp, daphne::VectorizedPipelineOp>> horFusionPairs;
    for (auto it1 = vpOps.begin(); it1 != vpOps.end(); it1++) {
        daphne::VectorizedPipelineOp vpOp1 = *it1;
        size_t numInputs1 = vpOp1.getSplits().size();
        for (auto it2 = next(it1); it2 != vpOps.end(); it2++) {
            daphne::VectorizedPipelineOp vpOp2 = *it2;
            size_t numInputs2 = vpOp2.getSplits().size();

            // The two VectorizedPipelineOps must be in the same block.
            if (vpOp1->getBlock() != vpOp2->getBlock())
                continue;

            // The two VectorizedPipelineOps must not (transitively) depend on each other.
            // We know that vpOp1 is before vpOp2 in the IR block.
            if (VectorizationUtils::operationDependsOnResultOf(vpOp2, vpOp1))
                continue;

            // Check for a qualifying common input of the two pipelines.
            for (size_t i1 = 0; i1 < numInputs1; i1++)
                for (size_t i2 = 0; i2 < numInputs2; i2++)
                    if (
                        // same value
                        vpOp1.getInputs()[i1] == vpOp2.getInputs()[i2] &&
                        // same split
                        vpOp1.getSplits()[i1] == vpOp2.getSplits()[i2] &&
                        // not broadcasted
                        vpOp1.getSplits()[i1].cast<daphne::VectorSplitAttr>().getValue() != daphne::VectorSplit::NONE
                        //
                    ) {
                        horFusionPairs.push_back({vpOp1, vpOp2});
                        break; // we only need to find one common input
                    }
        }
    }

    // -----------------------------------------------------------------
    // Step 2: Merge VectorizedPipelineOps
    // -----------------------------------------------------------------

    // For each pair of pipelines to fuse, we create a new VectorizedPipelineOp with the concatenation of the inputs,
    // splits, results, combines, and pipeline bodies of the two original pipelines. Note that any duplicate inputs will
    // be eliminated later during the canonicalization of VectorizedPipelineOp. We rewire the results of the new
    // pipeline with the uses of the results of the original pipelines. Finally, we erase the two original pipelines.

    OpBuilder builder(func);

    while (!horFusionPairs.empty()) {
        // Extract the first pair of pipelines to fuse.
        auto [vpOp1, vpOp2] = horFusionPairs[0];

        // Move any ops that are in-between the two original pipelines w.r.t. the linear order inside the IR block out
        // of the way (either before or after the point where we will create the merged pipeline).
        builder.setInsertionPointAfter(vpOp1);
        VectorizationUtils::movePipelineInterleavedOperations(builder.getInsertionPoint(), {vpOp2, vpOp1});

        // Merge/concat the attributes and values of the two original pipelines.
        std::vector<Value> inputs(vpOp1.getInputs().begin(), vpOp1.getInputs().end());
        inputs.insert(inputs.end(), vpOp2.getInputs().begin(), vpOp2.getInputs().end());
        std::vector<Value> outRows(vpOp1.getOutRows().begin(), vpOp1.getOutRows().end());
        outRows.insert(outRows.end(), vpOp2.getOutRows().begin(), vpOp2.getOutRows().end());
        std::vector<Value> outCols(vpOp1.getOutCols().begin(), vpOp1.getOutCols().end());
        outCols.insert(outCols.end(), vpOp2.getOutCols().begin(), vpOp2.getOutCols().end());
        std::vector<Attribute> splits(vpOp1.getSplits().begin(), vpOp1.getSplits().end());
        splits.insert(splits.end(), vpOp2.getSplits().begin(), vpOp2.getSplits().end());
        std::vector<Value> results(vpOp1->getResults().begin(), vpOp1->getResults().end());
        results.insert(results.end(), vpOp2->getResults().begin(), vpOp2->getResults().end());
        std::vector<Attribute> combines(vpOp1.getCombines().begin(), vpOp1.getCombines().end());
        combines.insert(combines.end(), vpOp2.getCombines().begin(), vpOp2.getCombines().end());

        // Create the new VectorizedPipelineOp.
        Location loc = builder.getFusedLoc({vpOp1.getLoc(), vpOp2->getLoc()});
        auto vpOpNew = builder.create<daphne::VectorizedPipelineOp>(loc, ValueRange(results).getTypes(), inputs,
                                                                    outRows, outCols, builder.getArrayAttr(splits),
                                                                    builder.getArrayAttr(combines), nullptr);

        // Move the operations inside the two original pipelines into the new pipeline.
        std::vector<Value> returnValsNew;
        Block *b1 = &vpOp1.getBody().getBlocks().front();
        Block *b2 = &vpOp2.getBody().getBlocks().front();
        Block *bNew = builder.createBlock(&vpOpNew.getBody());
        moveOperationsToBlock(b1, bNew, returnValsNew);
        moveOperationsToBlock(b2, bNew, returnValsNew);

        // Create the new ReturnOp.
        builder.setInsertionPointToEnd(bNew);
        builder.create<daphne::ReturnOp>(loc, returnValsNew);

        // Rewire all uses of the results of the two original pipelines to corresponding results of the new pipeline.
        for (size_t i = 0; i < results.size(); ++i)
            results[i].replaceAllUsesWith(vpOpNew.getResult(i));

        // Update the remaining previously found fusion opportunities by replacing all occurrences of the two original
        // pipelines by the new pipeline.
        std::vector<std::pair<daphne::VectorizedPipelineOp, daphne::VectorizedPipelineOp>> newHorFusionPairs;
        for (size_t i = 1; i < horFusionPairs.size(); i++) {
            auto [vpOpAOld, vpOpBOld] = horFusionPairs[i];
            daphne::VectorizedPipelineOp vpOpANew = (vpOpAOld == vpOp1 || vpOpAOld == vpOp2) ? vpOpNew : vpOpAOld;
            daphne::VectorizedPipelineOp vpOpBNew = (vpOpBOld == vpOp1 || vpOpBOld == vpOp2) ? vpOpNew : vpOpBOld;
            if (vpOpANew != vpOpBNew)
                newHorFusionPairs.push_back({vpOpANew, vpOpBNew});
        }
        horFusionPairs = newHorFusionPairs;

        // Remove the two original VectorizedPipelineOps.
        vpOp1->erase();
        vpOp2->erase();
    }
}

std::unique_ptr<Pass> daphne::createHorizontalFusionPass() { return std::make_unique<HorizontalFusionPass>(); }