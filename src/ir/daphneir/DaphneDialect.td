#ifndef SRC_IR_DAPHNEIR_DAPHNEDIALECT_TD
#define SRC_IR_DAPHNEIR_DAPHNEDIALECT_TD

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

#endif //SRC_IR_DAPHNEIR_DAPHNEDIALECT_TD