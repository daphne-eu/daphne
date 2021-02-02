#ifndef INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNETYPES_H
#define INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNETYPES_H

//===----------------------------------------------------------------------===//
// Daphne types
//===----------------------------------------------------------------------===//

// Types

class Daphne_Type<string name> : TypeDef<Daphne_Dialect, name>;

def MatrixType : Daphne_Type<"Matrix"> {
    let summary = "A matrix for the daphne dialect";
}

def ScalarType : AnyTypeOf<[AnyFloat, AnyInteger]> {}
def MathType : TypeAlias<AnyTypeOf<[ScalarType, MatrixType]>> {}

#endif //INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNETYPES_H