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
    return {daphne::VectorSplit::ROWS, daphne::VectorSplit::ROWS};
}
template<class EwBinaryOp>
std::vector<daphne::VectorCombine> getVectorCombines_EwBinaryOp(EwBinaryOp * op) {
    return {daphne::VectorCombine::ROWS};
}

// ****************************************************************************
// Vector split and combine implementations
// ****************************************************************************

#define IMPL_SPLIT_COMBINE_EWBINARYOP(OP) \
    std::vector<daphne::VectorSplit> daphne::OP::getVectorSplits() { \
        return getVectorSplits_EwBinaryOp(this); \
    } \
    std::vector<daphne::VectorCombine> daphne::OP::getVectorCombines() { \
        return getVectorCombines_EwBinaryOp(this); \
    }

// Arithmetic
IMPL_SPLIT_COMBINE_EWBINARYOP(EwAddOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwSubOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwMulOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwDivOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwPowOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwModOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwLogOp)

// Min/max
IMPL_SPLIT_COMBINE_EWBINARYOP(EwMinOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwMaxOp)

// Logical
IMPL_SPLIT_COMBINE_EWBINARYOP(EwAndOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwOrOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwXorOp)

// Strings
IMPL_SPLIT_COMBINE_EWBINARYOP(EwConcatOp)

// Comparisons
IMPL_SPLIT_COMBINE_EWBINARYOP(EwEqOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwNeqOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwLtOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwLeOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwGtOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwGeOp)