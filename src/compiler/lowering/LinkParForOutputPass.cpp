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
        auto blockArgs = blocks.front().getArguments();
        auto funcOutArg = blockArgs[0]; // output pointer is always passed first for kernels

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
        for (auto store : returnStores) {
            auto retVal = store->getOperand(0);
            // unrealized cast 1
            auto retValDef = retVal.getDefiningOp();
            // unrealized cast 2
            auto retValDef2 = retValDef->getOperand(0).getDefiningOp();
            // load
            auto load = retValDef2->getOperand(0).getDefiningOp();
            // pointer
            LLVM::CallOp lastUpdate = nullptr;
            auto ptr = load->getOperand(0);
            for (auto usr : ptr.getUsers()) {
                if (lastUpdate = llvm::dyn_cast<LLVM::CallOp>(usr)) {
                    break;
                }
            }

            auto gep = store->getOperand(1);
            gep.dump();

            // Set the insertion point before lastUpdate
            b.setInsertionPoint(lastUpdate);

            // Get the defining operation of gep
            auto gepOp = gep.getDefiningOp();
            auto offset = gepOp->getOperand(1).getDefiningOp();
            // Insert a clone of gepOp at the new location (if moving, use move semantics if supported)
            auto *clonedGepOp = gepOp->clone();
            auto *clonedOffset = offset->clone();
            b.insert(clonedOffset);
            clonedGepOp->setOperand(1, clonedOffset->getResult(0));
            b.insert(clonedGepOp);

            // Erase the old store operation
            store->erase();

            // Update lastUpdate operand to use the result of the newly inserted gepOp

            lastUpdate.setOperand(0, clonedGepOp->getResult(0));

            // Optionally erase the original gepOp if it is no longer needed
            gepOp->erase();

            //exit(-1);
            // TODO : can we somehow check whether its a kernel ?
            // if (!llvm::isa<daphne::CallKernelOp>(retValDef))
            //     llvm::report_fatal_error(
            //         "For parfor loops the defining op of the return value as of now is limited to CallKernelOps.");

            // TODO : the position of GEPOp relative to reValDef might be non-dominating

            // rewire output of the last kernel call to GEPOp of the output
        }
        func->removeAttr("parfor_inplace_rewrite_needed");
    }
};

} // end anonymous namespace

std::unique_ptr<Pass> daphne::createLinkParForOutputPass() { return std::make_unique<LinkParForOutputPass>(); }
