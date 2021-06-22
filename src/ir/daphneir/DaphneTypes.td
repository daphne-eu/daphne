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

#ifndef SRC_IR_DAPHNEIR_DAPHNETYPES_TD
#define SRC_IR_DAPHNEIR_DAPHNETYPES_TD

// ****************************************************************************
// General
// ****************************************************************************

// The base for all DaphneIR-specific types.
class Daphne_Type<string name> : TypeDef<Daphne_Dialect, name>;

// ****************************************************************************
// Data types
// ****************************************************************************

// A matrix type.
def Matrix : Daphne_Type<"Matrix"> {
    let summary = "matrix";

    let parameters = (ins "::mlir::Type":$elementType);
    let genVerifyDecl = 1;
}

// TODO We could omit the "Of" and have it default to ScalarType.
// A type constraint checking if a type is a matrix whose value type is one of
// the given types.
// Reuses MLIR's ContainerType constraint in way inspired by MLIR's VectorOf.
class MatrixOf<list<Type> allowedTypes> : ContainerType<
        AnyTypeOf<allowedTypes>,
        Matrix.predicate,
        "$_self.dyn_cast<::mlir::daphne::MatrixType>().getElementType()",
        "matrix"
>;

def Frame : Daphne_Type<"Frame"> {
    let summary = "frame";

    let parameters = (ins "std::vector<::mlir::Type>":$columnTypes);
    let genVerifyDecl = 1;
}

def MatrixOrFrame : AnyTypeOf<[Matrix, Frame]>;

// ****************************************************************************
// Value types
// ****************************************************************************

// A character string type.
def StrScalar : Daphne_Type<"String">;

def SIntScalar : AnyTypeOf<[SI8, SI32, SI64], "signed integer">;
def UIntScalar : AnyTypeOf<[UI8, UI32, UI64], "unsigned integer">;
def IntScalar : AnyTypeOf<[SIntScalar, UIntScalar], "integer">;
def FloatScalar : AnyTypeOf<[F32, F64], "float">;
def NumScalar : AnyTypeOf<[IntScalar, FloatScalar], "numeric">;
def BoolScalar : TypeAlias<I1, "boolean">;

// TODO Exclude Index again.
def AnyScalar : AnyTypeOf<[NumScalar, BoolScalar, StrScalar, Index]>;

// ****************************************************************************
// Auxiliary types
// ****************************************************************************

// TODO Could also be Index or whatever, but it might be wise to define the
// type we use to express dimensions and numeric row/col indexes just once.
// TODO Rename, it's misleading. Sounds like a pair of height/width.
def Size : TypeAlias<Index, "size">;

def Seed : TypeAlias<SI64, "seed">;

// TODO Remove SI64 again, it was only added temporarily to make things work.
def Selection : AnyTypeOf<[Size, MatrixOf<[Size, SI64]>, MatrixOf<[BoolScalar]>]>;

#endif //SRC_IR_DAPHNEIR_DAPHNETYPES_TD
