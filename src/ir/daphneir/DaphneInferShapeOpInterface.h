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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERSHAPEOPINTERFACE_H
#define SRC_IR_DAPHNEIR_DAPHNEINFERSHAPEOPINTERFACE_H

#include <utility>
#include <vector>

// ****************************************************************************
// Shape inference traits
// ****************************************************************************

// All of these traits address operations with **exactly one result**.
// Supporting multiple results would complicate the traits unnecessarily, given
// the low number of DaphneIR operations with multiple results. Thus,
// operations with multiple results should simply implement the shape inference
// interface instead of using traits.

namespace mlir::OpTrait {

// ============================================================================
// Traits determining #rows or #cols separately
// ============================================================================

// Constant one.

template <class ConcreteOp> class OneRow : public TraitBase<ConcreteOp, OneRow> {};

template <class ConcreteOp> class OneCol : public TraitBase<ConcreteOp, OneCol> {};

// Same as i-th scalar argument.

template <size_t i> struct NumRowsFromIthScalar {
    template <class ConcreteOp> class Impl : public TraitBase<ConcreteOp, Impl> {};
};

template <size_t i> struct NumColsFromIthScalar {
    template <class ConcreteOp> class Impl : public TraitBase<ConcreteOp, Impl> {};
};

// Same as i-th argument's same dimension.

template <size_t i> struct NumRowsFromIthArg {
    template <class ConcreteOp> class Impl : public TraitBase<ConcreteOp, Impl> {};
};

template <size_t i> struct NumColsFromIthArg {
    template <class ConcreteOp> class Impl : public TraitBase<ConcreteOp, Impl> {};
};

// Same as i-th argument's other dimension.

template <size_t i> struct NumRowsFromIthArgNumCols {
    template <class ConcreteOp> class Impl : public TraitBase<ConcreteOp, Impl> {};
};

template <size_t i> struct NumColsFromIthArgNumRows {
    template <class ConcreteOp> class Impl : public TraitBase<ConcreteOp, Impl> {};
};

// Same as all inputs' same dimension (they must all be the same).

template <class ConcreteOp> class NumRowsFromAllArgs : public TraitBase<ConcreteOp, NumRowsFromAllArgs> {};

template <class ConcreteOp> class NumColsFromAllArgs : public TraitBase<ConcreteOp, NumColsFromAllArgs> {};

// Sum of all inputs' same dimension.

template <class ConcreteOp> class NumRowsFromSumOfAllArgs : public TraitBase<ConcreteOp, NumRowsFromSumOfAllArgs> {};

template <class ConcreteOp> class NumColsFromSumOfAllArgs : public TraitBase<ConcreteOp, NumColsFromSumOfAllArgs> {};

// ============================================================================
// Traits determining #rows and #cols together
// ============================================================================

// Same shape as i-th argument.

template <size_t i> struct ShapeFromIthArg {
    template <class ConcreteOp> class Impl : public TraitBase<ConcreteOp, Impl> {};
};

// Broadcast-aware shape of elementwise binary operations.

template <class ConcreteOp> class ShapeEwBinary : public TraitBase<ConcreteOp, ShapeEwBinary> {};

} // namespace mlir::OpTrait

// ****************************************************************************
// Shape inference interfaces
// ****************************************************************************

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferShapeOpInterface.h.inc>
}

// ****************************************************************************
// Shape inference function
// ****************************************************************************

namespace mlir::daphne {
/**
 * @brief Tries to infer the shapes of all results of the given operation.
 *
 * If any shape inference traits are attached to the given operation, these are
 * applied to infer the result shape. If the operation implements any shape
 * inference interface, that implementation is invoked. If the shapes cannot be
 * inferred based on the available information, or if the operation does not
 * have any relevant traits or interfaces, -1 will be returned for all
 * dimensions.
 *
 * @param op The operation whose results' shapes shall be inferred.
 * @return A vector of pairs of (number of rows, number of columns). The i-th
 * pair in this vector represents the shape of the i-th result of the given
 * operation. A value of -1 for any of the numbers of rows or columns indicates
 * that this number is not known (yet).
 */
std::vector<std::pair<ssize_t, ssize_t>> tryInferShape(mlir::Operation *op);
} // namespace mlir::daphne

#endif // SRC_IR_DAPHNEIR_DAPHNEINFERSHAPEOPINTERFACE_H