/*
 *  Copyright 2021 The DAPHNE Consortium
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

#include <ir/daphneir/Daphne.h>

#include <string>
#include <vector>
#include <stdexcept>

namespace mlir::daphne {
#include <ir/daphneir/DaphneVectorizableOpInterface.cpp.inc>
}

using namespace mlir;

// ****************************************************************************
// Vector split and combine utility functions
// ****************************************************************************
// For families of operations.

template<class EwBinaryOp>
std::vector<daphne::VectorSplit> getVectorSplits_EwBinaryOp(EwBinaryOp * op) {
}
template<class EwBinaryOp>
std::vector<daphne::VectorCombine> getVectorCombines_EwBinaryOp(EwBinaryOp * op) {
}

// ****************************************************************************
// Vector split and combine implementations
// ****************************************************************************

std::vector<daphne::VectorSplit> daphne::EwAddOp::getVectorSplits() {
    return getVectorSplits_EwBinaryOp(this);
}
std::vector<daphne::VectorCombine> daphne::EwAddOp::getVectorCombines() {
    return getVectorCombines_EwBinaryOp(this);
}
