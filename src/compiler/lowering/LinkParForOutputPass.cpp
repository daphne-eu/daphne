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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include <cstdio>
#include <deque>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <set>

using namespace mlir;

namespace {
/**
 * @brief Exploits the canonical form of ParFor output handling to rewrite the ParFor loop body function, 
 * so that it conduct in-place updates on the shared buffer.  
*/
struct LinkParForOutputPass : public PassWrapper<LinkParForOutputPass, OperationPass<LLVM::LLVMFuncOp>> {
    void runOnOperation() override {
        LLVM::LLVMFuncOp func = getOperation();
        llvm::StringRef fName = func.getSymName();

        if (!fName.starts_with("parfor_body") || !func->hasAttr("parfor_inplace_rewrite_needed"))
            return;

        auto &blocks = func.getBody().getBlocks();
        if (blocks.empty())
            return;

        // find all GEPOps that have the return arg as their base pointer
        // output pointer is always passed first for kernels
        auto funcOutArg = blocks.front().getArgument(0);

        std::vector<LLVM::GEPOp> outGEPOps;
        for (auto *user : funcOutArg.getUsers()) {
            if (auto gep = dyn_cast<LLVM::GEPOp>(user)) {
                if (gep.getBase() == funcOutArg)
                    outGEPOps.push_back(gep);
            }
        }

        // find stores to the GEPOps, these are what currently represents returns
        std::vector<LLVM::StoreOp> returnStores;
        for (auto gepOp : outGEPOps) {
            for (auto user : gepOp->getUsers()) {
                if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
                    if (store.getOperand(1) == gepOp) {
                        returnStores.push_back(store);
                    }
                }
            }
        }
        OpBuilder b(&getContext());

        // rewire outputs of last kernel calls to the respective output of the function.
        SetVector<Operation *> toErase = {};
        for (auto store : returnStores) {
            auto retVal = store->getOperand(0);
            auto retValDef = retVal.getDefiningOp();
            mlir::Value operand = retValDef->getOperand(0);
            // load the output from the kernel
            if (auto retValDef2 = retValDef->getOperand(0).getDefiningOp()) {
                setInPlaceCalcKernelCall(retValDef2->getOperand(0).getDefiningOp(), store, b, &toErase);
            } else if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
                setInPlaceCalcKernelCallInPrevBlocks(blockArg, store, b, &toErase);
            }
        }

        // erase old operations after rewire
        for (auto opToErase : toErase) {
            opToErase->erase();
        }
        func->removeAttr("parfor_inplace_rewrite_needed");
    }

    /**
     * @brief Traverses blocks backwards to determinate the `LLVM::LoadOp`, which loads the result of the last kernel
     * call.
     */
    void setInPlaceCalcKernelCallInPrevBlocks(BlockArgument blockArg, mlir::Operation *store, OpBuilder b, SetVector<Operation *> *toErase) {
        mlir::Block *parentBlock = blockArg.getOwner();
        unsigned argIndex = blockArg.getArgNumber();

        for (mlir::Block *pred : parentBlock->getPredecessors()) {
            mlir::Operation &terminator = pred->back();

            if (auto branchInterface = mlir::dyn_cast<mlir::BranchOpInterface>(terminator)) {
                auto successors = terminator.getSuccessors();

                for (unsigned succIdx = 0; succIdx < successors.size(); ++succIdx) {
                    if (successors[succIdx] == parentBlock) {
                        auto succOperands = branchInterface.getSuccessorOperands(succIdx);
                        if (argIndex < succOperands.size()) {
                            setInPlaceCalcKernelCall(succOperands[argIndex].getDefiningOp(), store, b, toErase);
                        }
                    }
                }
            } else if (terminator.getNumSuccessors() == 1 && terminator.getSuccessor(0) == parentBlock) {
                auto succOperands = terminator.getOperands();
                if (argIndex < succOperands.size()) {
                    setInPlaceCalcKernelCall(succOperands[argIndex].getDefiningOp(), store, b, toErase);
                }
            }
        }
    }

    /**
     * @brief Replaces output argument of kernel CallOp with output argument of parfor body function to conduct in-place
     * updates in shared buffer
     */
    void setInPlaceCalcKernelCall(mlir::Operation *load, mlir::Operation *store, OpBuilder b,
                                  SetVector<Operation *> *toErase) {
        auto ptr = load->getOperand(0);
        mlir::Operation *lastUpdate = nullptr;
        for (auto usr : ptr.getUsers()) {
            if ((lastUpdate = llvm::dyn_cast<LLVM::CallOp>(usr))) {
                break;
            }
        }
        // Set the insertion point before lastUpdate
        b.setInsertionPoint(lastUpdate);
        // Get the defining operation of gep
        auto gep = store->getOperand(1);
        auto gepOp = gep.getDefiningOp();
        auto offset = gepOp->getOperand(1).getDefiningOp();

        // Insert a clone of gepOp at the new location (if moving, use move semantics if supported)
        auto *clonedGepOp = gepOp->clone();
        auto *clonedOffset = offset->clone();

        clonedGepOp->setOperand(1, clonedOffset->getResult(0));

        b.insert(clonedOffset);
        b.insert(clonedGepOp);
        // Update lastUpdate operand to use the result of the newly inserted gepOp
        lastUpdate->setOperand(0, clonedGepOp->getResult(0));
        // Load the in-place updated result
        load->setOperand(0, clonedGepOp->getResult(0));
        // Erase the old operations
        toErase->insert(store);
        toErase->insert(gepOp);
        toErase->insert(offset);
    }
};

} // end anonymous namespace

std::unique_ptr<Pass> daphne::createLinkParForOutputPass() { return std::make_unique<LinkParForOutputPass>(); }
