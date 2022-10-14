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

#include <mlir/Pass/Pass.h>

using namespace mlir;

/**
 * @brief Adapts an operation's input/output types such that it can be lowered to an available pre-compiled kernel.
 * 
 * While type inference propagates types through the IR, it is not guaranteed that a pre-compiled kernel
 * for each infered type combination is available. Thus, the task of this pass is to adapt input and
 * output types by casts, where necessary, to ensure that an existing pre-compiled kernel can be used.
 * 
 * At the moment, this pass is implemented in a very simple way. It merely harmonizes the value types
 * of all inputs with those of the single output of certain operations. This is because so far we mainly
 * pre-compile our kernels for homogeneous combinations of input/output types. The operations are
 * marked by traits.
 * 
 * In the future, this pass should take the kernel registry and/or extension catalog into account to find
 * out for which type combinations there are available kernels.
 */
// TODO This is not always correct for idxMin() and idxMax(): while their output always has an integer value
// type, it is not always safe to cast their input to integers.
struct AdaptTypesToKernelsPass : public PassWrapper<AdaptTypesToKernelsPass, FunctionPass>
{
    void runOnFunction() final;
};

// TODO This should become a general utility.
Type getValueType(Type t) {
    if(auto mt = t.dyn_cast<daphne::MatrixType>())
        return mt.getElementType();
    if(auto ft = t.dyn_cast<daphne::FrameType>())
        throw std::runtime_error("getValueType() doesn't support frames yet"); // TODO
    else // TODO Check if this is really a scalar.
        return t;
}

// TODO This should become a general utility.
Type setValueType(Type t, Type vt) {
    if(auto mt = t.dyn_cast<daphne::MatrixType>())
        return mt.withElementType(vt);
    if(auto ft = t.dyn_cast<daphne::FrameType>())
        throw std::runtime_error("setValueType() doesn't support frames yet"); // TODO
    else // TODO Check if this is really a scalar.
        return vt;
}

void AdaptTypesToKernelsPass::runOnFunction()
{
    FuncOp f = getFunction();
    OpBuilder builder(f.getContext());
    f.body().front().walk([&](Operation* op) {
        const size_t numOperands = op->getNumOperands();

        // Depending on the related trait, detemine which inputs to harmonize with the output.
        std::vector<size_t> operandIdxs;
        if(op->hasTrait<OpTrait::CastArgsToResType>()) // all inputs
            for(size_t i = 0; i < numOperands; i++)
                operandIdxs.push_back(i);
        else if(op->hasTrait<OpTrait::CastFirstTwoArgsToResType>()) // inputs 0 and 1
            operandIdxs = {0, 1};
        else if(op->hasTrait<OpTrait::CastArgsToResTypeRandMatrixOp>()) // inputs 2 and 3
            operandIdxs = {2, 3};

        if(!operandIdxs.empty()) {
            Type resTy = op->getResult(0).getType();

            // TODO Support this.
            // Skip pure frame ops, since we cannot easily cast the column types of frames anyway.
            // TODO Adapt this to use operandIdxs.
            if(!(
                resTy.isa<daphne::FrameType>() &&
                llvm::all_of(op->getOperands(), [](Value operand){
                    return operand.getType().isa<daphne::FrameType>();
                })
            )) {
                // Insert casts where necessary.
                Type resVTy = getValueType(resTy);
                builder.setInsertionPoint(op);
                for(size_t i : operandIdxs) {
                    Value argVal = op->getOperand(i);
                    Type argTy = argVal.getType();
                    if(getValueType(argTy) != resVTy) {
                        op->setOperand(
                                i,
                                builder.create<daphne::CastOp>(
                                        argVal.getLoc(),
                                        setValueType(argTy, resVTy),
                                        argVal
                                )
                        );
                    }
                }
            }
        }
    });
}

std::unique_ptr<Pass> daphne::createAdaptTypesToKernelsPass()
{
    return std::make_unique<AdaptTypesToKernelsPass>();
}
