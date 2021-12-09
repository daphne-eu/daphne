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

#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;

/**
 * @brief Inserts DaphneIR's `FreeOp` for all intermediate results that must
 * be released.
 */
struct InsertFreeOpPass : public PassWrapper<InsertFreeOpPass, FunctionPass>
{
    explicit InsertFreeOpPass() {}
    void runOnFunction() final;
};

/**
 * @brief Inserts a `FreeOp` to release the given SSA value, if necessary.
 * @param builder
 * @param v
 */
void processValue(OpBuilder builder, Value v) {
    // TODO Address handles from the distributed runtime.
    // We only need to free DAPHNE data objects like matrices and frames.
    if(!v.getType().isa<daphne::MatrixType, daphne::FrameType>())
        return;
    
    Operation * defOp = v.getDefiningOp();
    bool ascended = false;

    Operation * freeAfterOp = nullptr;
    if(v.use_empty()) {
        // If the given SSA value has no uses, we want to free it directly
        // after the op that defined it (nullptr for block args).
        // Note that ideally, there should be no unused SSA values.
        if(defOp)
            freeAfterOp = defOp;
    }
    else {
        // If the given SSA value has uses, we need to find the last of them.
        // Note that the iterator over the uses provided by the value does not
        // seem to follow any useful order, in general, so we need to find out
        // which use is the last one.
        // Furthermore, we want to free an SSA value in the block where it was
        // defined, to simplify things. So if the user of the value is in a
        // descendant block, we need to find its parent op in the block where
        // the given value is defined.
        Operation * lastUseOp = nullptr;
        for(OpOperand & use : v.getUses()) {
            bool thisAscended = false;
            Operation * thisUseOp = use.getOwner();
            // Find parent op in the block where v is defined.
            while(thisUseOp->getBlock() != v.getParentBlock()) {
                thisUseOp = thisUseOp->getParentOp();
                thisAscended = true;
            }
            // Determine if this is a later use.
            if(!lastUseOp || lastUseOp->isBeforeInBlock(thisUseOp)) {
                lastUseOp = thisUseOp;
                ascended = thisAscended;
            }
        }
        freeAfterOp = lastUseOp;
    }

    if(freeAfterOp) {
        // The given value is used within its block.
        
        // Don't insert a FreeOp in certain cases.
        if(freeAfterOp->hasTrait<OpTrait::IsTerminator>())
            // The value is handed out of its block (e.g., return, yield, ...).
            // It will be freed somewhere outside.
            return;
        if(isa<scf::WhileOp, scf::ForOp>(freeAfterOp) && defOp != freeAfterOp && !ascended)
            // The last use of the value is a loop. So the value will be freed
            // within the loop (via a block argument).
            // Note that, if a loop is a user of a value, this implies that the
            // corresponding DaphneDSL variable is changed inside the loop. So
            // the loop should always be the last use of the variable.
            // TODO However, there are ways to trick that, e.g.
            // `x = ...; y = x; while(...) {x = ...};` The original SSA value
            // of x will still be available via y, which would lead to a double
            // free. But eventually, we want to have copy-on-write semantics
            // anyway.
            return;
        if(auto co = dyn_cast<daphne::CastOp>(freeAfterOp))
            if(co.isTrivialCast() || co.isMatrixPropertyCast())
                // The last use is a trivial cast, which does nothing at 
                // runtime. We need to avoid a double free.
                return;

        builder.setInsertionPointAfter(freeAfterOp);
    }
    else {
        // The given value is an unused block arg. Free it at the beginning of
        // the block.
        builder.setInsertionPointToStart(v.getParentBlock());
    }
    builder.create<daphne::FreeOp>(builder.getUnknownLoc(), v);
}

/**
 * Inserts `FreeOp`s for all values defined in the given block, if necessary.
 * @param builder
 * @param b
 */
void processBlock(OpBuilder builder, Block * b) {
    // Free block arguments. Note that arguments of functions and vectorized
    // pipelines must be freed at call sites.
    if(!isa<FuncOp, daphne::VectorizedPipelineOp>(b->getParentOp()))
        for(BlockArgument& arg : b->getArguments())
            processValue(builder, arg);
    for(Operation& op : b->getOperations()) {
        // Free op results.
        for(Value v : op.getResults())
            processValue(builder, v);
        // Recurse into the op, but leave DistributedComputeOp alone, for now.
        if(isa<daphne::DistributedComputeOp>(op))
            continue;
        for(Region& r : op.getRegions())
            for(Block& b2 : r.getBlocks())
                processBlock(builder, &b2);
    }
}

void InsertFreeOpPass::runOnFunction()
{
    FuncOp f = getFunction();
    OpBuilder builder(f.getContext());
    processBlock(builder, &(f.body().front()));
}

std::unique_ptr<Pass> daphne::createInsertFreeOpPass()
{
    return std::make_unique<InsertFreeOpPass>();
}
