/*
 * Copyright 2025 The DAPHNE Consortium
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

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferSymmetricOpInterface.cpp.inc>
}

#include <vector>

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Inference interface implementations
// ****************************************************************************

// (none so far)

// ****************************************************************************
// Inference function
// ****************************************************************************

std::vector<daphne::BoolOrUnknown> daphne::tryInferSymmetric(Operation *op) {
    if (auto inferSymmetricOp = llvm::dyn_cast<daphne::InferSymmetric>(op))
        // If the operation implements the inference interface, we apply that.
        return inferSymmetricOp.inferSymmetric();
    else {
        // If the operation does not implement the inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<BoolOrUnknown> symmetrics;
        for (size_t i = 0; i < op->getNumResults(); i++)
            symmetrics.push_back(daphne::BoolOrUnknown::Unknown);
        return symmetrics;
    }
}