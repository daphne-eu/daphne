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

#include <mlir/IR/Operation.h>

bool VectorizationUtils::valueDependsOnResultOf(mlir::Value value, mlir::Operation *op) {
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

bool VectorizationUtils::operationDependsOnResultOf(mlir::Operation *src, mlir::Operation *op) {
    for (auto operand : src->getOperands())
        if (valueDependsOnResultOf(operand, op))
            return true;
    return false;
}

void VectorizationUtils::movePipelineInterleavedOperations(mlir::Block::iterator pipelinePosition,
                                                           const std::vector<mlir::Operation *> &pipeline) {
    // first operation in pipeline vector is last in IR, and the last is the
    // first
    auto startPos = pipeline.back()->getIterator();
    auto endPos = pipeline.front()->getIterator();
    auto currSkip = pipeline.rbegin();
    std::vector<mlir::Operation *> moveBeforeOps;
    std::vector<mlir::Operation *> moveAfterOps;
    for (auto it = startPos; it != endPos; ++it) {
        if (it == (*currSkip)->getIterator()) {
            ++currSkip;
            continue;
        }

        auto pipelineOpsBeforeIt = currSkip;
        while (--pipelineOpsBeforeIt != pipeline.rbegin())
            if (operationDependsOnResultOf(&(*it), *pipelineOpsBeforeIt))
                break;
        // check first pipeline op
        if (operationDependsOnResultOf(&(*it), *pipelineOpsBeforeIt))
            moveAfterOps.push_back(&(*it));
        else
            moveBeforeOps.push_back(&(*it));
    }

    // TODO We need to do just one of these (before/after), since the others can just remain where they were.
    for (auto moveBeforeOp : moveBeforeOps) {
        moveBeforeOp->moveBefore(pipelinePosition->getBlock(), pipelinePosition);
    }
    for (auto moveAfterOp : moveAfterOps) {
        moveAfterOp->moveAfter(pipelinePosition->getBlock(), pipelinePosition);
        // TODO Could be avoided by reverse-iterating over moveAfterOps.
        pipelinePosition = moveAfterOp->getIterator();
    }
}