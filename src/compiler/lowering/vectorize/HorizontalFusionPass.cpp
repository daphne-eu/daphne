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

#include "compiler/lowering/vectorize/VectorUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cstddef>
#include <iterator>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <memory>
#include <vector>

#include <llvm/ADT/STLExtras.h>
#include "llvm/Support/Casting.h"

#include <spdlog/spdlog.h>
#include <util/ErrorHandler.h>

using namespace mlir;

namespace
{

    //-----------------------------------------------------------------
    // Class
    //-----------------------------------------------------------------

    struct HorizontalFusionPass : public PassWrapper<HorizontalFusionPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;
    };

    //-----------------------------------------------------------------
    // Helper function
    //-----------------------------------------------------------------

    void moveOperationToBlock(mlir::Builder &builder, mlir::Block *src, mlir::Block *dest, std::vector<mlir::Value> &newResults) {

        // Iterate over all operations in src block and move them to dest block.
        // Rewrite block arguments of operations to the dest block arguments 
        // and store values for the results for overriding of the old values.
        while(!src->empty()) {
            auto op = src->begin();

            for(size_t i = 0; i < op->getNumOperands(); ++i) {
                auto operand = op->getOperand(i);
                if (llvm::isa<mlir::BlockArgument>(operand)) {
                    auto blockArgument = dest->addArgument(operand.getType(), builder.getUnknownLoc());
                    op->setOperand(i, blockArgument);
                }
            }
            if (!llvm::isa<daphne::ReturnOp>(op)) {
                op->moveBefore(dest, dest->end());
            }
            else {
                newResults.insert(newResults.end(), op->operand_begin(), op->operand_end());
                op->erase();
                return;
            }
        }
    }

}

//-----------------------------------------------------------------
// Horizontal Fusion / Sibling Fusion (Scan-sharing of inputs)
//-----------------------------------------------------------------
//
// Two operations share a single operand from the same producer.
//
//          producer
//         /        \
//   consumer1   consumer2 
//
// => (consumer1, consumer2)
//
// consumer1 and consumer2 cannot have a producer-consumer relationship directly or transitively,
// as if a merge of these operations where possible it would happen in Greedy1/Greedy2.

void HorizontalFusionPass::runOnOperation()
{   
    auto func = getOperation();

    // After merging of pipelines, we need to rerun the pass
    // to check for additional (changed) fusion possiblities.
    bool changed = true;
    while(changed) {
        changed = false;

        std::vector<daphne::VectorizedPipelineOp> pipelineOps;
        func->walk([&](daphne::VectorizedPipelineOp op) {
            pipelineOps.emplace_back(op);
        });
        std::reverse(pipelineOps.begin(), pipelineOps.end()); 

        //-----------------------------------------------------------------
        // Identify horizontal fusion possibilities
        //-----------------------------------------------------------------

        // Check for overlapping/intersection of operands between pipeline arguments.
        // They need to be compatible according to the corresponding split of an argument.
        std::vector<PipelineOpPair> horizontalRelationships;
        for (auto it1 = pipelineOps.begin(); it1 != pipelineOps.end(); ++it1) {
            auto pipeOp1 = *it1;

            // Store defOps for the corresponding arguments of pipeOp1.
            llvm::SmallVector<mlir::Operation*> defOpsArgs;
            // Running over the split size for consideration of relevant args (excl. OutCols, OutRows).
            for(size_t operandIx1 = 0; operandIx1 < pipeOp1.getSplits().size(); ++operandIx1) {
                auto operand1 = pipeOp1->getOperand(operandIx1);
                if (auto defOp = operand1.getDefiningOp()) {
                    defOpsArgs.push_back(defOp);
                }
            }

            for (auto it2 = next(it1); it2 != pipelineOps.end(); ++it2) {
                auto pipeOp2 = *it2;

                // PipelineOps need to be in the same block.
                if (pipeOp1->getBlock() != pipeOp2->getBlock())
                    continue;

                // PipelineOps cannot (transitively) depend on each other.
                if (VectorUtils::arePipelineOpsDependent(pipeOp1, pipeOp2))
                    continue;

                // Checking for overlapping arguments.
                for(size_t operandIx2 = 0; operandIx2 < pipeOp2.getSplits().size(); ++operandIx2) {
                    auto operand2 = pipeOp2->getOperand(operandIx2);

                    if (auto defOp = operand2.getDefiningOp()) {

                        // Check if defOp is also in the defOps for the pipeOp1 arguments.
                        auto fIt = std::find(defOpsArgs.begin(), defOpsArgs.end(), defOp);
                        if (fIt != defOpsArgs.end()) {
                            
                            size_t operandIx1 = std::distance(defOpsArgs.begin(), fIt);

                            if (pipeOp1.getSplits()[operandIx1] == pipeOp2.getSplits()[operandIx2] && 
                                pipeOp1.getSplits()[operandIx1].cast<daphne::VectorSplitAttr>().getValue() != daphne::VectorSplit::NONE) {
                                horizontalRelationships.push_back({pipeOp1, pipeOp2});
                                break; // We only need one case of arguments matching.
                            }
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------
        // Merge VectorizedPipelineOps
        //----------------------------------------------------------------- 

        for(auto pipeOpPair : horizontalRelationships) {
        
            auto [pipeOp1, pipeOp2] = pipeOpPair;

            mlir::Block* b1 = &pipeOp1.getBody().getBlocks().front();
            mlir::Block* b2 = &pipeOp2.getBody().getBlocks().front();

            // Merge attributes and values
            auto vSplitAttrs = std::vector<mlir::Attribute>(pipeOp1.getSplits().begin(), pipeOp1.getSplits().end());
            vSplitAttrs.insert(vSplitAttrs.end(), pipeOp2.getSplits().begin(), pipeOp2.getSplits().end());

            auto vCombineAttrs = std::vector<mlir::Attribute>(pipeOp1.getCombines().begin(), pipeOp1.getCombines().end());
            vCombineAttrs.insert(vCombineAttrs.end(), pipeOp2.getCombines().begin(), pipeOp2.getCombines().end());

            auto oldResults = std::vector<mlir::Value>(pipeOp1->getResults().begin(), pipeOp1->getResults().end());
            oldResults.insert(oldResults.end(), pipeOp2->getResults().begin(), pipeOp2->getResults().end());

            auto operands = std::vector<mlir::Value>(pipeOp1->getOperands().begin(), pipeOp1->getOperands().begin() + pipeOp1.getSplits().size());
            operands.insert(operands.end(), pipeOp2->getOperands().begin(), pipeOp2->getOperands().begin() + pipeOp2.getSplits().size());

            auto outRows = std::vector<mlir::Value>(pipeOp1.getOutRows().begin(), pipeOp1.getOutRows().end());
            outRows.insert(outRows.end(), pipeOp2.getOutRows().begin(), pipeOp2.getOutRows().end());

            auto outCols = std::vector<mlir::Value>(pipeOp1.getOutCols().begin(), pipeOp1.getOutCols().end());
            outCols.insert(outCols.end(), pipeOp2.getOutCols().begin(), pipeOp2.getOutCols().end());

            // Create new PipelineOp 
            mlir::OpBuilder builder(func);
            auto loc = builder.getFusedLoc({pipeOp1.getLoc(), pipeOp2->getLoc()});
            auto pipelineOp = builder.create<mlir::daphne::VectorizedPipelineOp>(loc,
                mlir::ValueRange(oldResults).getTypes(),
                operands,
                outRows,
                outCols,
                builder.getArrayAttr(vSplitAttrs),
                builder.getArrayAttr(vCombineAttrs),
                nullptr);
            mlir::Block *bodyBlock = builder.createBlock(&pipelineOp.getBody()); 

            //Move operations to new PipelineOp block.
            auto newResults = std::vector<mlir::Value>();
            moveOperationToBlock(builder, b1, bodyBlock, newResults);
            moveOperationToBlock(builder, b2, bodyBlock, newResults);

            // Create new ReturnOp.
            builder.setInsertionPointToEnd(bodyBlock);
            builder.create<mlir::daphne::ReturnOp>(loc, newResults);

            // Rewrite all uses to new ReturnOp.
            for (size_t i = 0; i < oldResults.size(); ++i) {
                oldResults.at(i).replaceAllUsesWith(pipelineOp.getResult(i));
            }

            // Place to the location after the last PipelineOp of this pair.
            // Is this sufficient?
            pipelineOp->moveAfter(pipeOp1);

            // Clean up
            pipeOp1->erase();
            pipeOp2->erase();

            //suboptimal
            changed = true;
            break;
        }
    }

    return;
}


std::unique_ptr<Pass> daphne::createHorizontalFusionPass() {
    return std::make_unique<HorizontalFusionPass>();
}