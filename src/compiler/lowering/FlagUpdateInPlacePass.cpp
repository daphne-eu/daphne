/*
 * Copyright 2023 The DAPHNE Consortium
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/IR/User.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include "compiler/utils/CompilerUtils.h"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Pass/Pass.h>

#include "ir/daphneir/DaphneUpdateInPlaceOpInterface.h"
#include "runtime/local/context/DaphneContext.h"

#include <mlir/Dialect/SCF/IR/SCF.h>

using namespace mlir;

#include <iostream>

/**
* @brief Check if an operand of an operation is used after the current operation.
*
* @param op The operation that is currently checked.
* @param operand_index The index of the operand that is checked.
* @return true if the operand is used after the current operation, false otherwise.
*/

bool hasAnyUseAfterCurrentOp(mlir::Operation *op, int operand_index) {

    //Check if operand is used after the current operation op
    mlir::Value arg = op->getOperand(operand_index);

    for (auto *userOp : arg.getUsers()) {

        //getDefiningOp is nullptr if arg is a block argument
        //TODO: Check for potential use cases, where the block argument could be used in-place 
        if (arg.getDefiningOp() == nullptr)
            return true;

        // If there is a loop, we assume that an argument outside the loop is used in the next iteration. 
        // Therefore, it is not safe to use it in-place. We need to check if the parent operation is an scf.for, 
        // as there is a possibility that op->getParentOp is scf.if and its parentOp is scf.for.
        mlir::Operation *parentOp = op->getParentOp();
        while (parentOp != nullptr) {
            if (isa<scf::ForOp, scf::WhileOp>(parentOp) &&
                arg.getDefiningOp()->getParentOp() != parentOp) {
                return true;
            }
            parentOp = parentOp->getParentOp();
        }

        // Check if userOp and op have the same parent block. 
        if (op->getBlock() == userOp->getBlock()) {

            // Check if userOp is after op
            if (op->isBeforeInBlock(userOp)) {
                return true;
            }
        }
        // Default case: userOp is in a different block than op
        else {
            return true;
        }
    }

    return false;
}

/**
* @brief Check if an operand of an operation is a valid type (matrix or frame).
*
* @param arg The operand that is checked.
* @return true if the operand is a valid type, false otherwise.
*/

template<typename T>
 bool isValidType(T arg) {
     return arg.getType().template isa<daphne::MatrixType>() || arg.getType().template isa<daphne::FrameType>();
 }

struct FlagUpdateInPlacePass: public PassWrapper<FlagUpdateInPlacePass, OperationPass<ModuleOp>>
{
    //explicit FlagInPlace() {}
    void runOnOperation() final;
};

void FlagUpdateInPlacePass::runOnOperation() {

    auto module = getOperation();

    // Traverse the operations in the module
    module.walk([&](mlir::Operation *op) {

        // Only apply to operations that implement the DaphneUpdateInPlaceOpInterface
        if (auto inPlaceOp = llvm::dyn_cast<daphne::InPlaceable>(op)) {
        
            // Fetches the operands that can be used in-place from the InPlaceable op
            auto inPlaceOperands = inPlaceOp.getInPlaceOperands();
            BoolAttr inPlaceFutureUse[inPlaceOperands.size()];

            for (auto inPlaceOperand : inPlaceOperands) {
                // TODO: Checking if the operand is valid type (matrix & frame) really necessary? 
                // We need to do it also in RewriteToCallKernelOpPass.cpp
                if (!isValidType(op->getOperand(inPlaceOperand)) || hasAnyUseAfterCurrentOp(op, inPlaceOperand))
                    inPlaceFutureUse[inPlaceOperand] = BoolAttr::get(op->getContext(), true);
                else
                    inPlaceFutureUse[inPlaceOperand] = BoolAttr::get(op->getContext(), false);
            }

            // inPlaceFutureUse is an array of bools, one for each operand e.g. [false, true]
            llvm::MutableArrayRef<mlir::Attribute> inPlaceFutureUseArray(inPlaceFutureUse, inPlaceOperands.size());
            op->setAttr("inPlaceFutureUse", mlir::ArrayAttr::get(op->getContext(), inPlaceFutureUseArray));

        }

    });
}

std::unique_ptr<Pass> daphne::createFlagUpdateInPlacePass() {
    return std::make_unique<FlagUpdateInPlacePass>();
}