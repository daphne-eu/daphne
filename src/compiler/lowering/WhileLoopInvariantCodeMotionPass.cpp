/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <memory>

using namespace mlir;

/**
 * @brief This is a very limited variant of loop invariant code motion (LICM),
 * tailored just to WhileOp.
 * 
 * We need this because MLIR does not seem to support LICM for while loops.
 * Nevertheless, we should clarify this (see #175).
 * 
 * This pass is strongly inspired by MLIR's LoopInvariantCodeMotion.cpp, but
 * significantly simplified.
 */
struct WhileLoopInvariantCodeMotionPass
: public PassWrapper <WhileLoopInvariantCodeMotionPass, OperationPass<func::FuncOp>> {
    void runOnOperation() final;

    StringRef getArgument() const final { return "while-loop-invariant-code-motion"; }
    StringRef getDescription() const final { return "TODO"; }
};

void WhileLoopInvariantCodeMotionPass::runOnOperation() {
    getOperation()->walk([&](scf::WhileOp whileOp) {
        Region & loopBody = whileOp.getAfter();

        SmallPtrSet<Operation *, 8> willBeMovedSet;
        SmallVector<Operation *, 8> opsToMove;

        auto isDefinedOutsideOfBody = [&](Value value) {
            auto definingOp = value.getDefiningOp();
            return (definingOp && !!willBeMovedSet.count(definingOp)) ||
                    !loopBody.isAncestor(value.getParentRegion());
        };

        for(auto & block : loopBody)
            for(auto & op : block.without_terminator()) {
                auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);
                if(
                    llvm::all_of(op.getOperands(), isDefinedOutsideOfBody) &&
                    op.hasTrait<OpTrait::ZeroRegions>() && // such that we don't need to recurse
                    memInterface && memInterface.hasNoEffect()
                ) {
                    opsToMove.push_back(&op);
                    willBeMovedSet.insert(&op);
                }
            }

        for(auto op : opsToMove)
            op->moveBefore(whileOp);
    });
}

std::unique_ptr<Pass> daphne::createWhileLoopInvariantCodeMotionPass() {
    return std::make_unique<WhileLoopInvariantCodeMotionPass>();
}
