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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERTYPESOPINTERFACE_H
#define SRC_IR_DAPHNEIR_DAPHNEINFERTYPESOPINTERFACE_H

#include <vector>

// ****************************************************************************
// Type inference traits
// ****************************************************************************

// All of these traits address operations with **exactly one result**.
// Supporting multiple results would complicate the traits unnecessarily, given
// the low number of DaphneIR operations with multiple results. Thus,
// operations with multiple results should simply implement the type inference
// interface instead of using traits.

namespace mlir::OpTrait {
    
// ============================================================================
// Traits determining data type and value type separately
// ============================================================================
    
// ----------------------------------------------------------------------------
// Data type
// ----------------------------------------------------------------------------

/**
 * @brief The data type (of the single result) is always the same as the data
 * type of the first argument.
 * 
 * Assumes that the operation has always exactly one result.
 */
template<class ConcreteOp>
class DataTypeFromFirstArg : public TraitBase<ConcreteOp, DataTypeFromFirstArg> {};

/**
 * @brief The data type (of the single result) is the most general of the data
 * types of all arguments.
 * 
 * In that context, `Frame` is more general than `Matrix` is more general than
 * scalar. In other words:
 * - If any argument is of `Frame` data type, the result will be a `Frame`.
 * - Otherwise, if any argument is of `Matrix` data type, the result will be a
 *   `Matrix`.
 * - Otherwise, the result will be a scalar.
 * 
 * If the type of any argument is unknown, the type of the result is unknown.
 * 
 * Assumes that the operation has always exactly one result.
 */
template<class ConcreteOp>
class DataTypeFromArgs : public TraitBase<ConcreteOp, DataTypeFromArgs> {};

/**
 * @brief The data type (of the single result) is always scalar.
 * 
 * Assumes that the operation has always exactly one result.
 */
template<class ConcreteOp>
class DataTypeSca : public TraitBase<ConcreteOp, DataTypeSca> {};

/**
 * @brief The data type (of the single result) is always `Matrix`.
 * 
 * Assumes that the operation has always exactly one result.
 */
template<class ConcreteOp>
class DataTypeMat : public TraitBase<ConcreteOp, DataTypeMat> {};

/**
 * @brief The data type (of the single result) is always `Frame`.
 * 
 * Assumes that the operation has always exactly one result.
 */
template<class ConcreteOp>
class DataTypeFrm : public TraitBase<ConcreteOp, DataTypeFrm> {};
    
// ----------------------------------------------------------------------------
// Value type
// ----------------------------------------------------------------------------

/**
 * @brief The value type (of the single result) is always the same as the value
 * type of the first argument.
 * 
 * Assumes that the operation has always exactly one result.
 */
template<class ConcreteOp>
class ValueTypeFromFirstArg : public TraitBase<ConcreteOp, ValueTypeFromFirstArg> {};

/**
 * @brief The value type (of the single result) is the most general of the
 * value types of all arguments.
 * 
 * If the result data type is `Frame`, the most general value type is
 * determined for each column separately.
 * 
 * In that context:
 * - `str` is the most general value type.
 * - Floating-point types are more general than integer types.
 * - Within floating-point or integer types, a type is the more general the
 *   more bits it has, irrespective of the (un)signedness of integers.
 * - The unsigned integer type of a certain bit width is more general than the
 *   signed integer type of the same bit width.
 * - `bool` is treated as an unsigned integer of 1 bit.
 * 
 * Examples:
 * - [argument value types...] -> result value type
 * - [`str`, `si64`] -> `str`
 * - [`si64`, `f32`] -> `f32`
 * - [`f64`, `f32`] -> `f64`
 * - [`ui64`, `si32`] -> `ui64`
 * - [`ui64`, `si64`] -> `ui64`
 * 
 * Assumes that the operation has always exactly one result.
 */
template<class ConcreteOp>
class ValueTypeFromArgs : public TraitBase<ConcreteOp, ValueTypeFromArgs> {};

/**
 * @brief Like `ValueTypeFromArgs`, but if the outcome is not a floating-point
 * type, it is replaced by the most general floating-point type.
 * 
 * If the data type of the single result is `Frame`, this replacement is
 * applied for each column value type separately.
 * 
 * Assumes that the operation has always exactly one result.
 */
template<class ConcreteOp>
class ValueTypeFromArgsFP : public TraitBase<ConcreteOp, ValueTypeFromArgsFP> {};

/**
 * @brief Like `ValueTypeFromArgs`, but if the outcome is not an integer
 * type, it is replaced by the most general integer type.
 * 
 * If the data type of the single result is `Frame`, this replacement is
 * applied for each column value type separately.
 * 
 * Assumes that the operation has always exactly one result.
 */
template<class ConcreteOp>
class ValueTypeFromArgsInt : public TraitBase<ConcreteOp, ValueTypeFromArgsInt> {};

/**
 * @brief The value type (of the single result) reflects a horizontal
 * concatenation of the first two arguments, i.e., column types are
 * concatenated.
 * 
 * If the data type of the single result is `Frame`, then:
 * - the column value types are obtained by concatenating the column value
 *   types of the first two arguments
 * - if any argument is of `Matrix` type, its number of columns must be known
 *   to be able to append that argument's value type the right number of times
 *   to the result's schema; if the number of columns is unknown, the result
 *   frame will have a single column of unknown type
 * Otherwise:
 * - this trait falls back to `ValueTypeFromArgs` limited to the first two
 *   arguments
 * 
 * Assumes that the operation has always exactly one result.
 */
template<class ConcreteOp>
class ValueTypesConcat : public TraitBase<ConcreteOp, ValueTypesConcat> {};

/**
 * @brief The value type (of the single result) is `Size`.
 */
template<class ConcreteOp>
class ValueTypeSize : public TraitBase<ConcreteOp, ValueTypeSize> {};
    
// ============================================================================
// Traits determining data type and value type together
// ============================================================================

/**
 * @brief The data type and value type (of the single result) are always the
 * same as the data type and value type of the first argument.
 * 
 * Assumes that the operation has always exactly one result.
 */
template<class ConcreteOp>
class TypeFromFirstArg : public TraitBase<ConcreteOp, TypeFromFirstArg> {};

}

// ****************************************************************************
// Type inference interface
// ****************************************************************************

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferTypesOpInterface.h.inc>
}

// ****************************************************************************
// Type inference function
// ****************************************************************************

namespace mlir::daphne {
/**
 * @brief Tries to infer the type of all results of the given operation.
 * 
 * If any type inference traits are attached to the given operation, these are
 * applied to infer the result type. If the operation implements the type
 * inference interface, that implementation is invoked. If the types cannot be
 * infered based on the available information, or if the operation does not
 * have any relevant traits or interfaces, `mlir::daphne::UnknownType` will be
 * returned.
 * 
 * @param op The operation whose results' types shall be infered.
 * @return A vector of `Type`s. The i-th pair in this vector represents the
 * type of the i-th result of the given operation. A value of
 * `mlir::daphne::UnknownType` indicates that this type is not known (yet).
 */
std::vector<mlir::Type> tryInferType(mlir::Operation* op);

/**
 * @brief Infers and sets the types of all results of the given operation.
 * 
 * @param op The operation whose results' types shall be infered and set.
 * @param partialInferenceAllowed If `true`, unknown will be allowed as an
 * infered type; if `false`, infering unknown will throw an exception.
*/
void setInferedTypes(mlir::Operation* op, bool partialInferenceAllowed = true);
}

#endif // SRC_IR_DAPHNEIR_DAPHNEINFERTYPESOPINTERFACE_H
