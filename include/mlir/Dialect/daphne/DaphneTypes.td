#ifndef INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNETYPES_H
#define INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNETYPES_H

//===----------------------------------------------------------------------===//
// Daphne types
//===----------------------------------------------------------------------===//

// Types

class Daphne_Type<string name> : TypeDef<Daphne_Dialect, name>;

def ScalarType : AnyTypeOf<[SI64, F64, Index]> {}

def MatrixType : Daphne_Type<"Matrix"> {
    let summary = "a matrix for the daphne dialect";

    let parameters = (ins "Type":$elementType);
    let genVerifyInvariantsDecl = 1;
}

def MathType : TypeAlias<AnyTypeOf<[ScalarType, MatrixType]>> {}

#endif //INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNETYPES_H
