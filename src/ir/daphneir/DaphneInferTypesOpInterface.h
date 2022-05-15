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
}

#endif // SRC_IR_DAPHNEIR_DAPHNEINFERTYPESOPINTERFACE_H