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
// Unknown type
// ****************************************************************************

def Unknown : Daphne_Type<"Unknown"> {
    let summary = "placeholder for an unknown type";
    let description = [{
        This type is meant to be used in situations when a type is not (yet)
        known. In some cases, it could be replaced by a known type in a
        compiler pass. In other cases, it might not be known at compile-time at
        all and the run-time needs to deal with it.
    }];
}

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
// the given types (whereby Unknown is always allowed automatically).
// Reuses MLIR's ContainerType constraint in a way inspired by MLIR's VectorOf.
class MatrixOf<list<Type> allowedTypes> : ContainerType<
        AnyTypeOf<!listconcat(allowedTypes, [Unknown])>,
        Matrix.predicate,
        "$_self.dyn_cast<::mlir::daphne::MatrixType>().getElementType()",
        "matrix"
>;

def Frame : Daphne_Type<"Frame"> {
    let summary = "frame";

    let parameters = (ins
        "std::vector<::mlir::Type>":$columnTypes,
        "std::vector<std::string> *":$labels
    );

    let genVerifyDecl = 1;

    let builders = [
        TypeBuilder<(ins "std::vector<::mlir::Type>":$columnTypes), [{
            // By default, the label information is unknown (nullptr).
            return Base::get($_ctxt, columnTypes, nullptr);
        }]>,
    ];
}

def MatrixOrFrame : AnyTypeOf<[Matrix, Frame]>;

// ****************************************************************************
// Distributed types
// ****************************************************************************

def Handle : Daphne_Type<"Handle"> {
    let summary = "handle to distributed data";

    let parameters = (ins "::mlir::Type":$dataType);
    let genVerifyDecl = 1;
}

// ****************************************************************************
// Value types
// ****************************************************************************

// A character string type.
def StrScalar : Daphne_Type<"String"> {
    let summary = "string";
}

def SIntScalar : AnyTypeOf<[SI8, SI32, SI64], "signed integer">;
def UIntScalar : AnyTypeOf<[UI8, UI32, UI64], "unsigned integer">;
def IntScalar : AnyTypeOf<[SIntScalar, UIntScalar], "integer">;
def FloatScalar : AnyTypeOf<[F32, F64], "float">;
def NumScalar : AnyTypeOf<[IntScalar, FloatScalar], "numeric">;
def BoolScalar : TypeAlias<I1, "boolean">;

// TODO Exclude Index again.
def AnyScalar : AnyTypeOf<[NumScalar, BoolScalar, StrScalar, Index]>;

// ****************************************************************************
// IO types
// ****************************************************************************

def File : Daphne_Type<"File"> {
    let summary = "An open file.";
}

def Descriptor : Daphne_Type<"Descriptor"> {
    let summary = "An open file on a device.";
}

def Target : Daphne_Type<"Target"> {
    let summary = "An open device.";
}

// ****************************************************************************
// Auxiliary types
// ****************************************************************************

// TODO Could also be Index or whatever, but it might be wise to define the
// type we use to express dimensions and numeric row/col indexes just once.
// TODO Rename, it's misleading. Sounds like a pair of height/width.
def Size : TypeAlias<Index, "size">;

def Seed : TypeAlias<SI64, "seed">;

// TODO Remove SI64 again, it was only added temporarily to make things work.
def Selection : AnyTypeOf<[Size, MatrixOf<[Size, SI64]>]>;

def VariadicPack : Daphne_Type<"VariadicPack"> {
    let summary = "pack of variadic operands";
    let description = [{
        This type is used to represent a variadic operand of an operation as a
        single SSA value irrespective of the number of concrete operands. It
        is meant to exist only in the low levels of the Daphne compilation
        stack. It is introduced during the lowering to kernel calls and
        converted when lowering to the LLVM dialect.
    }];
    let parameters = (ins "::mlir::Type":$containedType);
}

def DaphneContext : Daphne_Type<"DaphneContext"> {
    let summary = "daphne context";
    let description = [{
        This type represents all kinds of run-time context information in one
        SSA value. This could be things like available hardware resources and
        global run-time facilities.
    }];
}

#endif //SRC_IR_DAPHNEIR_DAPHNETYPES_TD
