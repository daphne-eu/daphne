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

// TODO We could omit the "Of" and have it default to ScalarType.
// A type constraint checking if a type is a matrix whose value type is one of
// the given types.
// Reuses MLIR's ContainerType constraint in way inspired by MLIR's VectorOf.
class MatrixOf<list<Type> allowedTypes> : ContainerType<
        AnyTypeOf<allowedTypes>,
        MatrixType.predicate,
        "$_self.dyn_cast<::mlir::daphne::MatrixType>().getElementType()",
        "matrix"
>;

#endif //INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNETYPES_H
