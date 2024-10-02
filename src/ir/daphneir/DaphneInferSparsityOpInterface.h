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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERSPARSITYOPINTERFACE_H
#define SRC_IR_DAPHNEIR_DAPHNEINFERSPARSITYOPINTERFACE_H

#include <utility>
#include <vector>

// ****************************************************************************
// Sparsity inference traits
// ****************************************************************************

// All of these traits address operations with **exactly one result**.
// Supporting multiple results would complicate the traits unnecessarily, given
// the low number of DaphneIR operations with multiple results. Thus,
// operations with multiple results should simply implement the sparsity
// inference interface instead of using traits.

namespace mlir::OpTrait {

// ============================================================================
// Traits definitions
// ============================================================================

template <class ConcreteOp> class CompletelyDense : public TraitBase<ConcreteOp, CompletelyDense> {};

template <size_t i> struct SparsityFromIthScalar {
    template <class ConcreteOp> class Impl : public TraitBase<ConcreteOp, Impl> {};
};

template <size_t i> struct SparsityFromIthArg {
    template <class ConcreteOp> class Impl : public TraitBase<ConcreteOp, Impl> {};
};

template <class ConcreteOp> class EwSparseIfEither : public TraitBase<ConcreteOp, EwSparseIfEither> {};
template <class ConcreteOp> class EwSparseIfBoth : public TraitBase<ConcreteOp, EwSparseIfBoth> {};

} // namespace mlir::OpTrait

// ****************************************************************************
// Sparsity inference interfaces
// ****************************************************************************

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferSparsityOpInterface.h.inc>
}

// ****************************************************************************
// Sparsity inference function
// ****************************************************************************

namespace mlir::daphne {
// NOTE: we could replace this by instead using a default implementation of the
// interface method, which would check
//  the traits
/**
 * @brief Tries to infer the sparsities of all results of the given operation.
 *
 * If any sparsity inference traits are attached to the given operation, these
 * are applied to infer the result sparsity. If the operation implements any
 * sparsity inference interface, that implementation is invoked. If the sparsity
 * cannot be inferred based on the available information, or if the operation
 * does not have any relevant traits or interfaces, -1.0 (unknown) will be
 * returned for sparsity.
 *
 * @param op The operation whose results' sparsities shall be inferred.
 * @return A vector of sparsity. The i-th element in this vector represents the
 * sparsity of the i-th result of the given
 * operation. A value of -1.0 for any sparsity indicates
 * that this number is not known (yet).
 */
std::vector<double> tryInferSparsity(mlir::Operation *op);
} // namespace mlir::daphne

#endif // SRC_IR_DAPHNEIR_DAPHNEINFERSPARSITYOPINTERFACE_H
