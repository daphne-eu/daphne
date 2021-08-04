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
#include <mlir/IR/Operation.h>

#include <stdexcept>
#include <memory>

#include <iostream>

using namespace mlir;

/**
 * @brief A compiler pass infering various properties of the data objects.
 * 
 * Rooted at a function, the pass walks the operations, and for each operation
 * it encounters, it infers all currently considered properties of the
 * operation's results based on the properties of the operation's arguments.
 * This approach can easily handle dependencies between different properties to
 * be infered without explicitly modeling them.
 * 
 * Note that the actual inference logic is outsourced to MLIR operation
 * interfaces.
 */
// TODO Currently, the properties to be infered are hardcoded, but we should
// make them configurable, whereby different instances of this pass should be
// able to infer different sets of properties.
class InferencePass : public PassWrapper<InferencePass, FunctionPass> {
public:
    void runOnFunction() override {
        getFunction().walk([&](Operation * op) {
            // Type inference.
            if(returnsUnknownType(op)) {
                if (auto inferTypesOp = llvm::dyn_cast<daphne::InferTypes>(op))
                    inferTypesOp.inferTypes();
                else
                    // TODO As soon as the run-time can handle unknown
                    // dat/value types, we do not need to throw here anymore.
                    throw std::runtime_error(
                            "some operation has an unknown result type, but "
                            "does not implement the type inference interface"
                    );
            }
            
            // Frame label inference.
            if (returnsFrameWithUnknownLabels(op)) {
                if (auto inferFrameLabelsOp = llvm::dyn_cast<daphne::InferFrameLabels>(op))
                    inferFrameLabelsOp.inferFrameLabels();
                // Else: Not a problem, since currently we use the frame labels
                // only to aid type inference, and for this purpose, we don't
                // need the labels in all cases.
            }
        });
    }

    static bool returnsUnknownType(Operation *op) {
        return llvm::any_of(op->getResultTypes(), [](Type resType) {
            if(resType.isa<daphne::UnknownType>())
                return true;
            if(auto mt = resType.dyn_cast<daphne::MatrixType>())
                return mt.getElementType().isa<daphne::UnknownType>();
            if(auto ft = resType.dyn_cast<daphne::FrameType>())
                for(Type ct : ft.getColumnTypes())
                    if(ct.isa<daphne::UnknownType>())
                        return true;
            return false;
        });
    }

    static bool returnsFrameWithUnknownLabels(Operation * op) {
        return llvm::any_of(op->getResultTypes(), [](Type resultType) {
            auto ft = resultType.dyn_cast<daphne::FrameType>();
            return ft && !ft.getLabels();
        });
    }
};

std::unique_ptr<Pass> daphne::createInferencePass() {
    return std::make_unique<InferencePass>();
}