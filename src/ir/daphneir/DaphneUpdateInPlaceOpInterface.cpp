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

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>

#include <vector>

namespace mlir::daphne
{
#include <ir/daphneir/DaphneUpdateInPlaceOpInterface.cpp.inc>
}

using namespace mlir;

// ****************************************************************************
// Binary Ops {0, 1}
// ****************************************************************************

#define IMPL_IN_PLACE_OPERANDS_BINARYOP(OP) \
    std::vector<int> daphne::OP::getInPlaceOperands() { \
        return {0, 1}; \
    }

// Arithmetic
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwAddOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwSubOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwMulOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwDivOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwPowOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwModOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwLogOp)

// Min/max
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwMinOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwMaxOp)

// Logical
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwAndOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwOrOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwXorOp)

// Strings
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwConcatOp)

// Comparisons
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwEqOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwNeqOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwLtOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwLeOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwGtOp)
IMPL_IN_PLACE_OPERANDS_BINARYOP(EwGeOp)

#undef IMPL_IN_PLACE_OPERANDS_BINARYOP

// ****************************************************************************
// Unary Ops {0}
// ****************************************************************************

#define IMPL_IN_PLACE_OPERANDS_UNARYOP(OP) \
    std::vector<int> daphne::OP::getInPlaceOperands() { \
        return {0}; \
    }

// Elementwise Unary Ops
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwAbsOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwAcosOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwAsinOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwAtanOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwCeilOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwCosOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwCoshOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwExpOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwFloorOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwLnOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwMinusOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwNegOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwRoundOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwSignOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwSinOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwSinhOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwSqrtOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwTanOp)
IMPL_IN_PLACE_OPERANDS_UNARYOP(EwTanhOp)

// Transpose
IMPL_IN_PLACE_OPERANDS_UNARYOP(TransposeOp)
// Reverse
IMPL_IN_PLACE_OPERANDS_UNARYOP(ReverseOp)

#undef IMPL_IN_PLACE_OPERANDS_UNARYOP

// ****************************************************************************

/* Alternative way for defining with the extra class declaration inside DaphneOps.td.
 * This is not used, because it requires atleast one DeclareOpInterface in DaphneOps.td.
 * The information would be spread across multiple files.
 * It is kept here for reference.

let extraClassDeclaration = [{
    // InPlaceOpInterface:
    std::vector<int> getInPlaceOperands() { return {0, 1}; }
}];
*/