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

#ifndef SRC_IR_DAPHNEIR_DAPHNE_H
#define SRC_IR_DAPHNEIR_DAPHNE_H

// The following includes are required by...
#include "llvm/ADT/StringRef.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/TypeID.h"

#include <ir/daphneir/DaphneAdaptTypesToKernelsTraits.h>
#include <ir/daphneir/DaphneOpsEnums.h.inc>
#include <ir/daphneir/DaphneDistributableOpInterface.h>
#include <ir/daphneir/DaphneInferFrameLabelsOpInterface.h>
#include <ir/daphneir/DaphneInferSparsityOpInterface.h>
#include <ir/daphneir/DaphneInferShapeOpInterface.h>
#include <ir/daphneir/DaphneInferTypesOpInterface.h>
#include <ir/daphneir/DaphneVectorizableOpInterface.h>

#include <string>
#include <utility>
#include <vector>

namespace mlir::OpTrait {
    template<class ConcreteOp>
    class FPGAOPENCLSupport : public TraitBase<ConcreteOp, FPGAOPENCLSupport> {
    };
}

namespace mlir::daphne {
    enum class MatrixRepresentation {
        Dense = 0,
        // default is dense
        Default = MatrixRepresentation::Dense,
        Sparse = 1,
    };

    std::string matrixRepresentationToString(MatrixRepresentation rep);

    MatrixRepresentation stringToMatrixRepresentation(const std::string &str);
}

// ... the following tablegen'erated headers.
#define GET_TYPEDEF_CLASSES
#include <ir/daphneir/DaphneOpsTypes.h.inc>
#include "ir/daphneir/DaphneOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "ir/daphneir/DaphneOps.h.inc"

#endif //SRC_IR_DAPHNEIR_DAPHNE_H
