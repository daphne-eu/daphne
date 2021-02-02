#ifndef INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNE_H
#define INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNE_H

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
#include "mlir/IR/Types.h"
#include "mlir/Support/TypeID.h"

#include <utility>

// ... the following tablegen'erated headers.
#define GET_TYPEDEF_CLASSES
#include <mlir/Dialect/daphne/DaphneOpsTypes.h.inc>
#include "mlir/Dialect/daphne/DaphneOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/daphne/DaphneOps.h.inc"

#endif //INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNE_H