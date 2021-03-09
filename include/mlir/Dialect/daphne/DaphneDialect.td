#ifndef INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNEDIALECT_H
#define INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNEDIALECT_H

include "mlir/IR/OpBase.td"

// ****************************************************************************
// Daphne dialect definition.
// ****************************************************************************

def Daphne_Dialect : Dialect {
    let name = "daphne";
    let summary = "An MLIR dialect for DAPHNE.";
    let description = [{
      This is an MLIR dialect for DAPHNE. Details coming soon.
    }];
    let cppNamespace = "::mlir::daphne";

    let hasConstantMaterializer = 1;
}

#endif //INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNEDIALECT_H