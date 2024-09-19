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

#include <compiler/inference/TypeInferenceUtils.h>
#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <mlir/Pass/Pass.h>

#include <vector>

using namespace mlir;

/**
 * @brief Adapts an operation's input/output types such that it can be lowered
 * to an available pre-compiled kernel.
 *
 * While type inference propagates types through the IR, it is not guaranteed
 * that a pre-compiled kernel for each infered type combination is available.
 * Thus, the task of this pass is to adapt input and output types by casts,
 * where necessary, to ensure that an existing pre-compiled kernel can be used.
 *
 * At the moment, this pass is implemented in a very simple way. It supports
 * only two concrete actions:
 * - It harmonizes the value types of all inputs with those of the single
 * output, for certain operations. This is because so far we mainly pre-compile
 * our kernels for homogeneous combinations of input/output types.
 * - It harmonizes the value types of all inputs (independently of the output
 * type), for certain operations. This is because some kernels need to output a
 * different type than their inputs (e.g., comparisons on non-numeric value
 * types). In general, the affected operations are marked by traits.
 *
 * In the future, this pass should take the kernel registry and/or extension
 * catalog into account to find out for which type combinations there are
 * available kernels.
 */
struct AdaptTypesToKernelsPass
    : public PassWrapper<AdaptTypesToKernelsPass, OperationPass<func::FuncOp>> {
    void runOnOperation() final;
    StringRef getArgument() const final { return "adapt-types-to-kernels"; }
    StringRef getDescription() const final { return "TODO"; }
};

void AdaptTypesToKernelsPass::runOnOperation() {
    func::FuncOp f = getOperation();
    OpBuilder builder(f.getContext());
    f.getBody().front().walk([&](Operation *op) {
        const size_t numOperands = op->getNumOperands();

        // Depending on the related trait, determine which inputs to cast to
        // which value type.
        std::vector<size_t> operandIdxs; // the inputs to cast
        Type targetVTy;                  // the value type to cast to

        if (op->hasTrait<OpTrait::CastArgsToMostGeneralArgType>()) {
            // The only related trait that does not consider the result type.

            // TODO Support frame ops.
            // Skip frame ops, since we cannot easily cast the column types of
            // frames anyway.
            if (llvm::any_of(op->getOperands(), [](Value operand) {
                    return llvm::isa<daphne::FrameType>(operand.getType());
                }))
                return;

            // Cast all inputs to the most general input value type.
            for (size_t i = 0; i < numOperands; i++)
                operandIdxs.push_back(i);
            std::vector<Type> argVTys;
            for (size_t i = 0; i < numOperands; i++)
                argVTys.push_back(
                    CompilerUtils::getValueType(op->getOperand(i).getType()));
            targetVTy = mostGeneralVt(argVTys);
        } else {
            // All remaining related traits consider the result type.

            // Skip operations without results.
            if (!op->getNumResults())
                return;
            Type resTy = op->getResult(0).getType();
            // TODO Support frame ops.
            // Skip frame ops, since we cannot easily cast the column types of
            // frames anyway.
            if (llvm::isa<daphne::FrameType>(resTy) ||
                llvm::any_of(op->getOperands(), [](Value operand) {
                    return llvm::isa<daphne::FrameType>(operand.getType());
                }))
                return;
            Type resVTy = CompilerUtils::getValueType(resTy);

            if (op->hasTrait<OpTrait::CastArgsToResType>()) {
                // Cast all inputs to the result value type.
                for (size_t i = 0; i < numOperands; i++)
                    operandIdxs.push_back(i);
                targetVTy = resVTy;
            } else if (op->hasTrait<OpTrait::CastFirstTwoArgsToResType>()) {
                // Cast inputs 0 and 1 to the result value type.
                operandIdxs = {0, 1};
                targetVTy = resVTy;
            }
            // TODO Instead of such a non-reusable op-specific trait, we should
            // rather check for the concrete op here.
            else if (op->hasTrait<OpTrait::CastArgsToResTypeRandMatrixOp>()) {
                // Cast inputs 2 and 3 to the result value type.
                operandIdxs = {2, 3};
                targetVTy = resVTy;
            }
        }

        if (!operandIdxs.empty()) {
            // Insert casts where necessary.
            builder.setInsertionPoint(op);
            for (size_t i : operandIdxs) {
                Value argVal = op->getOperand(i);
                Type argTy = argVal.getType();
                if (CompilerUtils::getValueType(argTy) != targetVTy) {
                    op->setOperand(
                        i, builder.create<daphne::CastOp>(
                               argVal.getLoc(),
                               CompilerUtils::setValueType(argTy, targetVTy),
                               argVal));
                }
            }
        }
    });
}

std::unique_ptr<Pass> daphne::createAdaptTypesToKernelsPass() {
    return std::make_unique<AdaptTypesToKernelsPass>();
}
