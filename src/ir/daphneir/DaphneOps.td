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

#ifndef SRC_IR_DAPHNEIR_DAPHNEOPS_TD
#define SRC_IR_DAPHNEIR_DAPHNEOPS_TD

// TODO Get rid of those ScalarType params.

include "ir/daphneir/DaphneDialect.td"
include "ir/daphneir/DaphneTypes.td"
include "ir/daphneir/DaphneDistributableOpInterface.td"
include "ir/daphneir/DaphneInferFrameLabelsOpInterface.td"
include "ir/daphneir/DaphneInferShapeOpInterface.td"
include "ir/daphneir/DaphneInferSparsityOpInterface.td"
include "ir/daphneir/DaphneInferSparsityTraits.td"
include "ir/daphneir/DaphneInferTypesOpInterface.td"
include "ir/daphneir/DaphneVectorizableOpInterface.td"
include "ir/daphneir/DaphneShapeInferenceTraits.td"
include "ir/daphneir/DaphneTypeAdaptationTraits.td"
include "ir/daphneir/DaphneTypeInferenceTraits.td"
include "ir/daphneir/CUDASupport.td"
include "ir/daphneir/FPGAOPENCLSupport.td"

include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

// ****************************************************************************
// Custom constraints
// ****************************************************************************

// The type of `mat` is a matrix whose value type is the type of `sca`.
class TypeIsMatrixOf<string mat, string sca> : And<[
    TypeIs<mat, Matrix>.predicate,
    TypesMatchWith<"", mat, sca, "$_self.dyn_cast<::mlir::daphne::MatrixType>().getElementType()">.predicate
]>;

// The types of `a` and `b` must either be the same (e.g. both are matrices of
// the same value type, or both are scalars of the same type), or the type of
// `a` must be a matrix whose value type is the type of `b`, or vice-versa.
class TypesMatchOrOneIsMatrixOfOther<string a, string b> : PredOpTrait<
    a # " and " # b # " must either be of the same type, or " # a # " must be a matrix whose value type is the type of " # b # ", or vice-versa",
    Or<[
        TypesMatchWith<"", a, b, "$_self">.predicate,
        TypeIsMatrixOf<a, b>,
        TypeIsMatrixOf<b, a>
    ]>
>;

// ****************************************************************************
// Base Daphne operation definition
// ****************************************************************************

class Daphne_Op<string mnemonic, list<Trait> traits = []> :
        Op<Daphne_Dialect, mnemonic, traits>;

// ****************************************************************************
// DAPHNE Runtime Interoperability
// ****************************************************************************

def Daphne_ConvertMemRefToDenseMatrix : Daphne_Op<"convertMemRefToDenseMatrix"> {
    let summary = "Return a DenseMatrix.";
    let description = [{ Constructs a DenseMatrix given a rank 2 StridedMemRefType. }];

    /* let arguments = (ins AnyMemRef:$arg); */
    let hasCanonicalizeMethod = 1;
    let arguments = (ins Size:$base, Size:$offset, Size:$size0, Size:$size1, Size:$stride0, Size:$stride1);
    let results = (outs MatrixOrU:$res);
}

def Daphne_ConvertDenseMatrixToMemRef : Daphne_Op<"convertDenseMatrixToMemRef", [Pure]> {
    let summary = "Given a DenseMatrix, return a StridedMemRefType.";
    let description = [{ Constructs a StridedMemRefType with rank 2 from a DenseMatrix* with already allocated memory. }];
    let hasCanonicalizeMethod = 1;
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs AnyMemRef:$output);
}

// ****************************************************************************
// Data generation
// ****************************************************************************

def Daphne_FillOp : Daphne_Op<"fill", [
    DataTypeMat, ValueTypeFromFirstArg,
    NumRowsFromIthScalar<1>, NumColsFromIthScalar<2>, CUDASupport
]> {
    let arguments = (ins AnyScalar:$arg, Size:$numRows, Size:$numCols);
    let results = (outs MatrixOrU:$res);
}

def Daphne_MatrixConstantOp : Daphne_Op<"matrixConstant", [
    DataTypeMat,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]>{
    let arguments = (ins UI64:$matrixAddr);
    let results = (outs MatrixOrU:$res);
}

def Daphne_CreateFrameOp : Daphne_Op<"createFrame", [
    SameVariadicOperandSize,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    let arguments = (ins Variadic<MatrixOrU>:$cols, Variadic<StrScalar>:$labels);
    let results = (outs FrameOrU:$res);
}

def Daphne_DiagMatrixOp : Daphne_Op<"diagMatrix", [
    TypeFromFirstArg,
    NumRowsFromArg, NumColsFromArgNumRows,
    DeclareOpInterfaceMethods<InferSparsityOpInterface>
]> {
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs MatrixOrU:$res);
}

def Daphne_RandMatrixOp : Daphne_Op<"randMatrix", [
    TypesMatchWith<"min and max arguments must be of the same type", "min", "max", "$_self">,
    NumRowsFromIthScalar<0>, NumColsFromIthScalar<1>, DeclareOpInterfaceMethods<InferTypesOpInterface>,
    SparsityFromIthScalar<4>, CastArgsToResTypeRandMatrixOp
]> {
    let arguments = (ins Size:$numRows, Size:$numCols, AnyScalar:$min, AnyScalar:$max, F64:$sparsity, IntScalar:$seed);
    let results = (outs MatrixOrU:$res);
}

def Daphne_SampleOp : Daphne_Op<"sample", [
    DataTypeMat, ValueTypeFromFirstArg,
    NumRowsFromIthScalar<1>, OneCol
]> {
    let arguments = (ins AnyScalar:$range, Size:$size, BoolScalar:$withReplacement, Seed:$seed);
    let results = (outs MatrixOrU:$res);
}

def Daphne_SeqOp : Daphne_Op<"seq", [
    DataTypeMat, ValueTypeFromArgs,
    OneCol, DeclareOpInterfaceMethods<InferNumRowsOpInterface>,
    CompletelyDense,
    CastArgsToResType
]> {
    let arguments = (ins NumScalar:$from, NumScalar:$to, NumScalar:$inc);
    let results = (outs MatrixOf<[NumScalar]>);
}

// ****************************************************************************
// Matrix/frame dimensions
// ****************************************************************************

class Daphne_NumOp<string name, list<Trait> traits = []> : Daphne_Op<name, !listconcat(traits, [
    DataTypeSca, ValueTypeSize, Pure
])> {
    let arguments = (ins MatrixOrFrame:$arg);
    let results = (outs Size:$res);

    let hasCanonicalizeMethod = 1;
}

def Daphne_NumRowsOp : Daphne_NumOp<"numRows">;
def Daphne_NumColsOp : Daphne_NumOp<"numCols">;
def Daphne_NumCellsOp : Daphne_NumOp<"numCells">;

// ****************************************************************************
// Matrix multiplication
// ****************************************************************************

def Daphne_MatMulOp : Daphne_Op<"matMul", [
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DataTypeMat, ValueTypeFromArgs,
    DeclareOpInterfaceMethods<InferShapeOpInterface>,
    DeclareOpInterfaceMethods<InferSparsityOpInterface>, CUDASupport, FPGAOPENCLSupport,
    CastFirstTwoArgsToResType
]> {
    let arguments = (ins MatrixOf<[NumScalar]>:$lhs, MatrixOf<[NumScalar]>:$rhs, BoolScalar:$transa, BoolScalar:$transb);
    let results = (outs MatrixOf<[NumScalar]>:$res);
    let hasCanonicalizeMethod = 1;
}

// ****************************************************************************
// Elementwise unary
// ****************************************************************************

class Daphne_EwUnaryOp<string name, Type scalarType, list<Trait> traits = []> : Daphne_Op<name, !listconcat(traits, [
    DataTypeFromFirstArg,
    ShapeFromArg,
    CastArgsToResType,
    NoMemoryEffect
])> {
    let arguments = (ins AnyTypeOf<[MatrixOf<[scalarType]>, scalarType, Unknown]>:$arg);
    let results = (outs AnyTypeOf<[MatrixOf<[scalarType]>, scalarType, Unknown]>:$res);
}

// ----------------------------------------------------------------------------
// Arithmetic/general math
// ----------------------------------------------------------------------------

def Daphne_EwMinusOp : Daphne_EwUnaryOp<"ewMinus", NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_EwAbsOp : Daphne_EwUnaryOp<"ewAbs", NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_EwSignOp : Daphne_EwUnaryOp<"ewSign", NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_EwExpOp : Daphne_EwUnaryOp<"ewExp", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwLnOp : Daphne_EwUnaryOp<"ewLn", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwSqrtOp : Daphne_EwUnaryOp<"ewSqrt", NumScalar, [ValueTypeFromArgsFP, DeclareOpInterfaceMethods<VectorizableOpInterface>]>;

// ----------------------------------------------------------------------------
// Logical
// ----------------------------------------------------------------------------

def Daphne_EwNegOp : Daphne_EwUnaryOp<"ewNeg", NumScalar, [ValueTypeFromFirstArg]>;

// ----------------------------------------------------------------------------
// Rounding
// ----------------------------------------------------------------------------

def Daphne_EwRoundOp : Daphne_EwUnaryOp<"ewRound", NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_EwFloorOp : Daphne_EwUnaryOp<"ewFloor", NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_EwCeilOp : Daphne_EwUnaryOp<"ewCeil", NumScalar, [ValueTypeFromFirstArg]>;

// ----------------------------------------------------------------------------
// Trigonometric
// ----------------------------------------------------------------------------

def Daphne_EwSinOp : Daphne_EwUnaryOp<"ewSin", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwCosOp : Daphne_EwUnaryOp<"ewCos", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwTanOp : Daphne_EwUnaryOp<"ewTan", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwSinhOp : Daphne_EwUnaryOp<"ewSinh", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwCoshOp : Daphne_EwUnaryOp<"ewCosh", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwTanhOp : Daphne_EwUnaryOp<"ewTanh", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwAsinOp : Daphne_EwUnaryOp<"ewAsin", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwAcosOp : Daphne_EwUnaryOp<"ewAcos", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwAtanOp : Daphne_EwUnaryOp<"ewAtan", NumScalar, [ValueTypeFromArgsFP]>;

// ****************************************************************************
// Elementwise binary
// ****************************************************************************

class Daphne_EwBinaryOp<string name, Type scalarType, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeFromArgs,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    ShapeEwBinary,
    CastArgsToResType,
    NoMemoryEffect
])> {
    let arguments = (ins AnyTypeOf<[MatrixOf<[scalarType]>, scalarType, Unknown]>:$lhs, AnyTypeOf<[MatrixOf<[scalarType]>, scalarType, Unknown]>:$rhs);
    let results = (outs AnyTypeOf<[MatrixOf<[scalarType]>, scalarType, Unknown]>:$res);

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            // TODO This is wrong if lhs is a scalar and rhs is a matrix.
            $_state.addTypes(lhs.getType());
            $_state.addOperands({lhs, rhs});
        }]>
    ];
    let hasFolder = 1;
}

// ----------------------------------------------------------------------------
// Arithmetic
// ----------------------------------------------------------------------------

// TODO Make EwAddOp Commutative again (see #449).
def Daphne_EwAddOp    : Daphne_EwBinaryOp<"ewAdd", NumScalar, [ValueTypeFromArgs/*, Commutative*/, EwSparseIfBoth, CUDASupport]> {
    let hasCanonicalizeMethod = 1;
}
def Daphne_EwSubOp    : Daphne_EwBinaryOp<"ewSub", NumScalar, [ValueTypeFromArgs, EwSparseIfBoth, CUDASupport]> {
    let hasCanonicalizeMethod = 1;
}
def Daphne_EwMulOp    : Daphne_EwBinaryOp<"ewMul", NumScalar, [ValueTypeFromArgs, Commutative, EwSparseIfEither, CUDASupport]> {
    let hasCanonicalizeMethod = 1;
}
def Daphne_EwDivOp    : Daphne_EwBinaryOp<"ewDiv", NumScalar, [ValueTypeFromArgs, CUDASupport]> {
    let hasCanonicalizeMethod = 1;
}
def Daphne_EwPowOp    : Daphne_EwBinaryOp<"ewPow", NumScalar, [ValueTypeFromArgs, CUDASupport]>;
def Daphne_EwModOp    : Daphne_EwBinaryOp<"ewMod", NumScalar, [ValueTypeFromArgs]>;
def Daphne_EwLogOp    : Daphne_EwBinaryOp<"ewLog", NumScalar, [ValueTypeFromArgsFP]>;

// ----------------------------------------------------------------------------
// Min/max
// ----------------------------------------------------------------------------

def Daphne_EwMinOp    : Daphne_EwBinaryOp<"ewMin", AnyScalar, [ValueTypeFromArgs, Commutative]>;
def Daphne_EwMaxOp    : Daphne_EwBinaryOp<"ewMax", AnyScalar, [ValueTypeFromArgs, Commutative, CUDASupport]>;

// ----------------------------------------------------------------------------
// Logical
// ----------------------------------------------------------------------------

def Daphne_EwAndOp    : Daphne_EwBinaryOp<"ewAnd", NumScalar, [Commutative, ValueTypeFromArgsInt]>;
def Daphne_EwOrOp     : Daphne_EwBinaryOp<"ewOr" , NumScalar, [Commutative, ValueTypeFromArgsInt]>;
def Daphne_EwXorOp    : Daphne_EwBinaryOp<"ewXor", NumScalar, [Commutative, ValueTypeFromArgsInt]>;

// ----------------------------------------------------------------------------
// Bitwise
// ----------------------------------------------------------------------------

def Daphne_EwBitwiseAndOp    : Daphne_EwBinaryOp<"ewBitwiseAnd", NumScalar, [Commutative, ValueTypeFromArgsInt]>;

// ----------------------------------------------------------------------------
// Strings
// ----------------------------------------------------------------------------

def Daphne_EwConcatOp : Daphne_EwBinaryOp<"ewConcat", StrScalar>;

// TODO This is just a quick and dirty solution that should be properly
// integrated with EwConcatOp above.
def Daphne_ConcatOp : Daphne_Op<"concat", [DataTypeSca, ValueTypeStr]> {
    let arguments = (ins StrScalar:$lhs, StrScalar:$rhs);
    let results = (outs StrScalar:$res);

    let hasFolder = 1;
}

def Daphne_StringEqOp : Daphne_Op<"stringEq", [ValueTypeStr]> {
    let arguments = (ins StrScalar:$lhs, StrScalar:$rhs);
    let results = (outs BoolScalar:$res);
    let hasFolder = 1;
}

// ----------------------------------------------------------------------------
// Comparisons
// ----------------------------------------------------------------------------

class Daphne_EwCmpOp<string name, Type inputScalarType, list<Trait> traits = []>
: Daphne_EwBinaryOp<name, inputScalarType, !listconcat(traits, [ValueTypeFromArgs])> {
    // TODO: We do not enforce (matrix of) boolean output any more, but should
    // think about that again.
    //let results = (outs AnyTypeOf<[MatrixOf<[BoolScalar]>, BoolScalar, Unknown]>:$res);
}

def Daphne_EwEqOp  : Daphne_EwCmpOp<"ewEq" , AnyScalar, [Commutative]> {
    let hasCanonicalizeMethod = 1;
}
def Daphne_EwNeqOp : Daphne_EwCmpOp<"ewNeq", AnyScalar, [Commutative, CUDASupport]>;
def Daphne_EwLtOp  : Daphne_EwCmpOp<"ewLt" , AnyScalar>;
def Daphne_EwLeOp  : Daphne_EwCmpOp<"ewLe" , AnyScalar>;
def Daphne_EwGtOp  : Daphne_EwCmpOp<"ewGt" , AnyScalar>;
def Daphne_EwGeOp  : Daphne_EwCmpOp<"ewGe" , AnyScalar>;

// ****************************************************************************
// Outer binary (generalized outer product)
// ****************************************************************************

class Daphne_OuterBinaryOp<string name, Type scalarType, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeMat,
    NumRowsFromIthArg<0>, NumColsFromIthArg<1>,
    CastArgsToResType
])> {
    let arguments = (ins MatrixOf<[scalarType]>:$lhs, MatrixOf<[scalarType]>:$rhs);
    let results = (outs MatrixOf<[scalarType]>:$res);
}

// ----------------------------------------------------------------------------
// Arithmetic
// ----------------------------------------------------------------------------

def Daphne_OuterAddOp : Daphne_OuterBinaryOp<"outerAdd", NumScalar, [ValueTypeFromArgs]>;
def Daphne_OuterSubOp : Daphne_OuterBinaryOp<"outerSub", NumScalar, [ValueTypeFromArgs]>;
def Daphne_OuterMulOp : Daphne_OuterBinaryOp<"outerMul", NumScalar, [ValueTypeFromArgs]>;
def Daphne_OuterDivOp : Daphne_OuterBinaryOp<"outerDiv", NumScalar, [ValueTypeFromArgs]>;
def Daphne_OuterPowOp : Daphne_OuterBinaryOp<"outerPow", NumScalar, [ValueTypeFromArgs]>;
def Daphne_OuterModOp : Daphne_OuterBinaryOp<"outerMod", NumScalar, [ValueTypeFromArgs]>;
def Daphne_OuterLogOp : Daphne_OuterBinaryOp<"outerLog", NumScalar, [ValueTypeFromArgsFP]>;

// ----------------------------------------------------------------------------
// Min/max
// ----------------------------------------------------------------------------

def Daphne_OuterMinOp : Daphne_OuterBinaryOp<"outerMin", AnyScalar, [ValueTypeFromArgs]>;
def Daphne_OuterMaxOp : Daphne_OuterBinaryOp<"outerMax", AnyScalar, [ValueTypeFromArgs]>;

// ----------------------------------------------------------------------------
// Logical
// ----------------------------------------------------------------------------

def Daphne_OuterAndOp : Daphne_OuterBinaryOp<"outerAnd", NumScalar, [ValueTypeFromArgsInt]>;
def Daphne_OuterOrOp  : Daphne_OuterBinaryOp<"outerOr" , NumScalar, [ValueTypeFromArgsInt]>;
def Daphne_OuterXorOp : Daphne_OuterBinaryOp<"outerXor", NumScalar, [ValueTypeFromArgsInt]>;

// ----------------------------------------------------------------------------
// Strings
// ----------------------------------------------------------------------------

def Daphne_OuterConcatOp : Daphne_OuterBinaryOp<"outerConcat", StrScalar>;

// ----------------------------------------------------------------------------
// Comparisons
// ----------------------------------------------------------------------------

class Daphne_OuterCmpOp<string name, Type inputScalarType, list<Trait> traits = []>
: Daphne_OuterBinaryOp<name, inputScalarType, !listconcat(traits, [ValueTypeFromArgs])> {
    // TODO: We do not enforce (matrix of) boolean output any more, but should
    // think about that again.
    //let results = (outs AnyTypeOf<[MatrixOf<[BoolScalar]>, BoolScalar, Unknown]>:$res);
}

def Daphne_OuterEqOp  : Daphne_OuterCmpOp<"outerEq" , AnyScalar>;
def Daphne_OuterNeqOp : Daphne_OuterCmpOp<"outerNeq", AnyScalar>;
def Daphne_OuterLtOp  : Daphne_OuterCmpOp<"outerLt" , AnyScalar>;
def Daphne_OuterLeOp  : Daphne_OuterCmpOp<"outerLe" , AnyScalar>;
def Daphne_OuterGtOp  : Daphne_OuterCmpOp<"outerGt" , AnyScalar>;
def Daphne_OuterGeOp  : Daphne_OuterCmpOp<"outerGe" , AnyScalar>;

// ****************************************************************************
// Elementwise ternary
// ****************************************************************************

def Daphne_CondOp : Daphne_Op<"cond", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    let arguments = (ins AnyTypeOf<[Matrix, AnyScalar, Unknown]>:$cond, AnyType:$thenVal, AnyType:$elseVal);
    let results = (outs AnyType:$res);

    let hasCanonicalizeMethod = 1;
}

// ****************************************************************************
// Aggregation and statistical
// ****************************************************************************

class Daphne_AggOp<string name, Type inScalarType, list<Trait> traits = []> : Daphne_Op<name, traits> {
    let arguments = (ins MatrixOf<[inScalarType]>:$arg);
}

// ----------------------------------------------------------------------------
// Full aggregation
// ----------------------------------------------------------------------------

class Daphne_AllAggOp<string name, Type scalarType, list<Trait> traits = []>
: Daphne_AggOp<name, scalarType, !listconcat(traits, [DataTypeSca, CastArgsToResType])> {
    let results = (outs scalarType:$res);
}

def Daphne_AllAggSumOp    : Daphne_AllAggOp<"sumAll", NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_AllAggMinOp    : Daphne_AllAggOp<"minAll", NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_AllAggMaxOp    : Daphne_AllAggOp<"maxAll", NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_AllAggMeanOp   : Daphne_AllAggOp<"meanAll", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_AllAggVarOp    : Daphne_AllAggOp<"varAll", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_AllAggStddevOp : Daphne_AllAggOp<"stddevAll", NumScalar, [ValueTypeFromArgsFP]>;

// ----------------------------------------------------------------------------
// Row/column-wise aggregation
// ----------------------------------------------------------------------------

class Daphne_DimAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<Trait> traits = []>
: Daphne_AggOp<name, inScalarType, !listconcat(traits, [DataTypeMat])> {
    let results = (outs MatrixOf<[outScalarType]>:$res);
}

class Daphne_RowAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<Trait> traits = []>
: Daphne_DimAggOp<name, inScalarType, outScalarType, !listconcat(traits, [
    NumRowsFromArg, OneCol
])>;
class Daphne_ColAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<Trait> traits = []>
: Daphne_DimAggOp<name, inScalarType, outScalarType, !listconcat(traits, [
    OneRow, NumColsFromArg
])>;

def Daphne_RowAggSumOp    : Daphne_RowAggOp<"sumRow"   , NumScalar, NumScalar, [ValueTypeFromFirstArg, CastArgsToResType,
        CUDASupport, DeclareOpInterfaceMethods<VectorizableOpInterface>]>;
def Daphne_RowAggMinOp    : Daphne_RowAggOp<"minRow"   , AnyScalar, AnyScalar, [ValueTypeFromFirstArg, CastArgsToResType, DeclareOpInterfaceMethods<VectorizableOpInterface>]>;
def Daphne_RowAggMaxOp    : Daphne_RowAggOp<"maxRow"   , AnyScalar, AnyScalar, [ValueTypeFromFirstArg, CastArgsToResType,
        CUDASupport, DeclareOpInterfaceMethods<VectorizableOpInterface>, DeclareOpInterfaceMethods<DistributableOpInterface>]>;
def Daphne_RowAggIdxMinOp : Daphne_RowAggOp<"idxminRow", NumScalar, Size, [ValueTypeSize]>;
def Daphne_RowAggIdxMaxOp : Daphne_RowAggOp<"idxmaxRow", NumScalar, Size, [ValueTypeSize]>;
def Daphne_RowAggMeanOp   : Daphne_RowAggOp<"meanRow"  , NumScalar, NumScalar, [ValueTypeFromArgsFP, CastArgsToResType]>;
def Daphne_RowAggVarOp    : Daphne_RowAggOp<"varRow"   , NumScalar, NumScalar, [ValueTypeFromArgsFP, CastArgsToResType]>;
def Daphne_RowAggStddevOp : Daphne_RowAggOp<"stddevRow", NumScalar, NumScalar, [ValueTypeFromArgsFP, CastArgsToResType]>;

def Daphne_ColAggSumOp    : Daphne_ColAggOp<"sumCol"   , NumScalar, NumScalar, [ValueTypeFromFirstArg, CastArgsToResType,
        CUDASupport, DeclareOpInterfaceMethods<VectorizableOpInterface>]>;
def Daphne_ColAggMinOp    : Daphne_ColAggOp<"minCol"   , AnyScalar, AnyScalar, [ValueTypeFromFirstArg, CastArgsToResType]>;
def Daphne_ColAggMaxOp    : Daphne_ColAggOp<"maxCol"   , AnyScalar, AnyScalar, [ValueTypeFromFirstArg, CastArgsToResType]>;
def Daphne_ColAggIdxMinOp : Daphne_ColAggOp<"idxminCol", NumScalar, Size, [ValueTypeSize]>;
def Daphne_ColAggIdxMaxOp : Daphne_ColAggOp<"idxmaxCol", NumScalar, Size, [ValueTypeSize]>;
def Daphne_ColAggMeanOp   : Daphne_ColAggOp<"meanCol"  , NumScalar, NumScalar, [ValueTypeFromArgsFP, CastArgsToResType]>;
def Daphne_ColAggVarOp    : Daphne_ColAggOp<"varCol"   , NumScalar, NumScalar, [ValueTypeFromArgsFP, CastArgsToResType]>;
def Daphne_ColAggStddevOp : Daphne_ColAggOp<"stddevCol", NumScalar, NumScalar, [ValueTypeFromArgsFP, CastArgsToResType]>;

// ----------------------------------------------------------------------------
// Cumulative aggregation
// ----------------------------------------------------------------------------

class Daphne_CumAggOp<string name, Type scalarType, list<Trait> traits = []>
: Daphne_AggOp<name, scalarType, !listconcat(traits, [
    DataTypeMat, ShapeFromArg
])> {
    let results = (outs MatrixOf<[scalarType]>);
}

def Daphne_CumAggSumOp  : Daphne_CumAggOp<"sumCum" , NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_CumAggProdOp : Daphne_CumAggOp<"prodCum", NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_CumAggMinOp  : Daphne_CumAggOp<"minCum" , AnyScalar, [ValueTypeFromFirstArg]>;
def Daphne_CumAggMaxOp  : Daphne_CumAggOp<"maxCum" , AnyScalar, [ValueTypeFromFirstArg]>;

// ----------------------------------------------------------------------------
// Grouped aggregation
// ----------------------------------------------------------------------------

class Daphne_GrpAggOp<string name, Type scalarType, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeMat,
    NumRowsFromIthScalar<2>, NumColsFromArg
])> {
    let arguments = (ins MatrixOf<[scalarType]>:$arg, MatrixOf<[Size]>:$groupIds, Size:$numGroups/*, Optional<MatrixOrU>:$weights*/);
    let results = (outs MatrixOf<[scalarType]>:$groupAggs);
}

def Daphne_GrpAggCountOp  : Daphne_GrpAggOp<"countGrp" , AnyScalar>;
def Daphne_GrpAggSumOp    : Daphne_GrpAggOp<"sumGrp"   , NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_GrpAggMinOp    : Daphne_GrpAggOp<"minGrp"   , AnyScalar, [ValueTypeFromFirstArg]>;
def Daphne_GrpAggMaxOp    : Daphne_GrpAggOp<"maxGrp"   , AnyScalar, [ValueTypeFromFirstArg]>;
def Daphne_GrpAggMeanOp   : Daphne_GrpAggOp<"meanGrp"  , NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_GrpAggVarOp    : Daphne_GrpAggOp<"varGrp"   , NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_GrpAggStddevOp : Daphne_GrpAggOp<"stddevGrp", NumScalar, [ValueTypeFromArgsFP]>;

// ----------------------------------------------------------------------------
// Statistical for column matrices
// ----------------------------------------------------------------------------

def Daphne_MedianOp : Daphne_Op<"median"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg, Optional<MatrixOf<[FloatScalar]>>:$weights);
    let results = (outs FloatScalar:$res);
}

def Daphne_QuantileOp : Daphne_Op<"quantile"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg, MatrixOf<[FloatScalar]>:$ps, Optional<MatrixOf<[FloatScalar]>>:$weights);
    let results = (outs MatrixOf<[FloatScalar]>:$res);
}

def Daphne_MomentOp : Daphne_Op<"moment"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg, Size:$k, Optional<MatrixOf<[FloatScalar]>>:$weights);
    let results = (outs FloatScalar:$res);
}

def Daphne_CovOp : Daphne_Op<"cov"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$lhs, MatrixOf<[FloatScalar]>:$rhs, Optional<MatrixOf<[FloatScalar]>>:$weights);
    let results = (outs FloatScalar:$res);
}

// ****************************************************************************
// Left and right indexing
// ****************************************************************************

def Daphne_ExtractOp : Daphne_Op<"extract", [
    NumRowsFromIthArg<1>, NumColsFromIthArgNumRows<2>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>
]> {
    let summary = "Copies the specified rows and columns from the argument to the result.";

    let description = [{
        This operation is effectively similar to a combination of `extractRow`
        and `extractCol`. Please see the documentation of those operations.
    }];

    let arguments = (ins MatrixOrFrame:$source, Selection:$selectedRows, AnyTypeOf<[Selection, StrScalar, Unknown]>:$selectedCols);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_ExtractRowOp : Daphne_Op<"extractRow", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    NumRowsFromIthArg<1>, NumColsFromArg
]> {
    let summary = "Copies the specified rows from the argument to the result.";

    let description = [{
        Copies the rows of a data object `source` (matrix or frame) at the
        positions specified by `selectedRows` to the result. `selectedRows`
        must be either a scalar or a single-column matrix of positions (row
        indexes) into `source`, i.e., the entries must be between zero
        (inclusive) and the number of rows in `source` (exclusive). Beyond that,
        there are no restrictions.

        Note that `FilterRowOp` is similar, but works with bit vectors instead
        of positions lists.
    }];

    let arguments = (ins MatrixOrFrame:$source, Selection:$selectedRows);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_SliceRowOp : Daphne_Op<"sliceRow", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    let summary = "Copies the specified rows from the argument to the result.";

    let description = [{
        Copies the rows of a data object `source` (matrix or frame) at the
        positions in `[lowerIncl, upperExcl)` to the result.
    }];

    let arguments = (ins MatrixOrFrame:$source, SI64:$lowerIncl, SI64:$upperExcl);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_ExtractColOp : Daphne_Op<"extractCol", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>, CUDASupport
]> {
    let summary = "Copies the specified columns from the argument to the result.";

    let description = [{
        Copies the columns of a data object `source` (matrix or frame) at the
        positions specified by `selectedCols` to the result. `selectedCols`
        must be either a scalar or a single-column matrix of positions (column
        indexes) into `source`, i.e., the entries must be between zero
        (inclusive) and the number of columns in `source` (exclusive). Beyond
        that, there are no restrictions.

        Alternatively, `selectedCols` can be a string, in which case the column
        with that label is extracted from `source` if it is a frame.

        Note that `FilterRowOp` is similar, but works with bit vectors instead
        of positions lists.
    }];

    let arguments = (ins MatrixOrFrame:$source, AnyTypeOf<[Selection, StrScalar, Unknown]>:$selectedCols);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_SliceColOp : Daphne_Op<"sliceCol", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    let summary = "Copies the specified columns from the argument to the result.";

    let description = [{
        Copies the columns of a data object `source` (matrix or frame) at the
        positions in `[lowerIncl, upperExcl)` to the result.
    }];

    let arguments = (ins MatrixOrFrame:$source, SI64:$lowerIncl, SI64:$upperExcl);
    let results = (outs MatrixOrFrame:$res);
}

// TODO Create combined InsertOp (see #238).

def Daphne_InsertRowOp : Daphne_Op<"insertRow", [
    TypeFromFirstArg, // this is debatable
    ShapeFromArg
]> {
    let arguments = (ins MatrixOrFrame:$arg, MatrixOrFrame:$ins, SI64:$rowLowerIncl, SI64:$rowUpperExcl);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_InsertColOp : Daphne_Op<"insertCol", [
    TypeFromFirstArg, // this is debatable
    ShapeFromArg
]> {
    let arguments = (ins MatrixOrFrame:$arg, MatrixOrFrame:$ins, SI64:$colLowerIncl, SI64:$colUpperExcl);
    let results = (outs MatrixOrFrame:$res);
}

// ****************************************************************************
// Reorganization
// ****************************************************************************

def Daphne_ReshapeOp : Daphne_Op<"reshape", [
    TypeFromFirstArg,
    NumRowsFromIthScalar<1>, NumColsFromIthScalar<2>, SparsityFromArg
]> {
    let arguments = (ins MatrixOrU:$arg, Size:$numRows, Size:$numCols);
    let results = (outs MatrixOrU:$res);
}

def Daphne_TransposeOp : Daphne_Op<"transpose", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    NumRowsFromArgNumCols, NumColsFromArgNumRows, SparsityFromArg, CUDASupport,
    Pure
]> {
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs MatrixOrU:$res);

    let builders = [
        OpBuilder<(ins "Value":$input), [{
            auto ty = input.getType();
            build($_builder, $_state, ty, input);
        }]>,
    ];
}

class Daphne_BindOp<string name, list<Trait> traits = []> : Daphne_Op<name, !listconcat(traits, [
    DataTypeFromArgs,
    CastArgsToResType
])> {
    let arguments = (ins MatrixOrFrame:$lhs, MatrixOrFrame:$rhs);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_ColBindOp : Daphne_BindOp<"colBind", [
    ValueTypesConcat,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    NumRowsFromAllArgs, NumColsFromSumOfAllArgs, CUDASupport
]>;

def Daphne_RowBindOp : Daphne_BindOp<"rowBind", [
    ValueTypeFromArgs,
    NumRowsFromSumOfAllArgs, NumColsFromAllArgs
]>;

def Daphne_ReverseOp : Daphne_Op<"reverse", [
    TypeFromFirstArg, ShapeFromArg
]> {
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs MatrixOrU:$res);
}

def Daphne_OrderOp : Daphne_Op<"order", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>, // due to possibility of returning indexes
    SameVariadicOperandSize,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    // TODO Maybe colIdxs and ascs should be attributes.
    let arguments = (ins MatrixOrFrame:$arg, Variadic<Size>:$colIdxs, Variadic<BoolScalar>:$ascs, BoolScalar:$returnIdxs);
    let results = (outs MatrixOrFrame:$res);
}

// ****************************************************************************
// Matrix decompositions & co
// ****************************************************************************

def Daphne_EigenOp : Daphne_Op<"eigenCal", [TypeFromFirstArg, DeclareOpInterfaceMethods<InferTypesOpInterface>,
        DeclareOpInterfaceMethods<InferShapeOpInterface>]> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg);
    let results = (outs MatrixOf<[FloatScalar]>:$eigenValues, MatrixOf<[FloatScalar]>:$eigenVectors);
}

def Daphne_LuOp : Daphne_Op<"lu"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg);
    let results = (outs MatrixOf<[FloatScalar]>:$p, MatrixOf<[FloatScalar]>:$l, MatrixOf<[FloatScalar]>:$u);
}

def Daphne_QrOp : Daphne_Op<"qr"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg);
    let results = (outs MatrixOf<[FloatScalar]>:$h, MatrixOf<[FloatScalar]>:$r);
}

def Daphne_SvdOp : Daphne_Op<"svd"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg);
    let results = (outs MatrixOf<[FloatScalar]>:$u, MatrixOf<[FloatScalar]>:$s, MatrixOf<[FloatScalar]>:$v);
}

// ****************************************************************************
// Deep neural network
// ****************************************************************************

// TODO The DNN-related operations share several arguments. Is there an elegant
// way to specify the *common* arguments just once?

// ----------------------------------------------------------------------------
// Activation
// ----------------------------------------------------------------------------

class Daphne_ActivationForwardOp<string name, list<Trait> traits = []> :
        Daphne_Op<name, traits> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input);

    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

def Daphne_ReluForwardOp : Daphne_ActivationForwardOp<"Relu_Forward", [ CUDASupport ]>;

// ----------------------------------------------------------------------------
// Affine
// ----------------------------------------------------------------------------

def Daphne_AffineForwardOp : Daphne_Op<"affineForward", [ CUDASupport ]> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$weights,
            MatrixOf<[FloatScalar]>:$bias);

    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

// ----------------------------------------------------------------------------
// Batch Normalization
// ----------------------------------------------------------------------------

def Daphne_BatchNorm2DTestForwardOp : Daphne_Op<"BatchNorm_Forward", [ CUDASupport ]> {
    let arguments = (ins
            MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$gamma, MatrixOf<[FloatScalar]>:$beta,
            MatrixOf<[FloatScalar]>:$emaMean, MatrixOf<[FloatScalar]>:$emaVar, FloatScalar:$eps);
    let results = (outs MatrixOf<[FloatScalar]>:$output);
}

// ----------------------------------------------------------------------------
// Bias Addition
// ----------------------------------------------------------------------------

def Daphne_BiasAddForwardOp : Daphne_Op<"biasAddForward", [ CUDASupport ]> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$bias);

    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

// ----------------------------------------------------------------------------
// Convolution
// ----------------------------------------------------------------------------

def Daphne_Conv2DForwardOp : Daphne_Op<"Convolution_Forward",
    [ DeclareOpInterfaceMethods<InferTypesOpInterface>,
      DeclareOpInterfaceMethods<InferShapeOpInterface>,
      CUDASupport ]> {
    let arguments = (ins
        MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$filter, MatrixOf<[FloatScalar]>:$bias,
        // input shape
        Size:$inputBatchSize, Size:$inputNumChannels, Size:$inputHeight, Size:$inputWidth,
        // filter shape (num filters & channels is encoded in dimensions of filter matrix (-> [F,C*H*W]))
        Size:$filterHeight, Size:$filterWidth, Size:$strideHeight, Size:$strideWidth, Size:$padHeight, Size:$padWidth);

    let results = (outs MatrixOf<[FloatScalar]>:$output, Size:$outHeight, Size:$outWidth);
}

def Daphne_Conv2DBackwardFilterOp : Daphne_Op<"conv2DBackwardFilter"> {
    let arguments = (ins
        MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$output,
        Size:$strideHeight, Size:$strideWidth, Size:$padHeight, Size:$padWidth,
        // input shape
        Size:$inputBatchSize, Size:$inputNumChannels, Size:$inputHeight, Size:$inputWidth,
        // filter shape
        Size:$filterNumFilters, Size:$filterNumChannels, Size:$filterHeight, Size:$filterWidth
    );
    let results = (outs MatrixOf<[FloatScalar]>:$filter);
}

def Daphne_Conv2DBackwardDataOp : Daphne_Op<"conv2DBackwardData"> {
    let arguments = (ins
        MatrixOf<[FloatScalar]>:$filter, MatrixOf<[FloatScalar]>:$output,
        Size:$strideHeight, Size:$strideWidth, Size:$padHeight, Size:$padWidth,
        // input shape
        Size:$inputBatchSize, Size:$inputNumChannels, Size:$inputHeight, Size:$inputWidth,
        // filter shape
        Size:$filterNumFilters, Size:$filterNumChannels, Size:$filterHeight, Size:$filterWidth
    );
    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

// ----------------------------------------------------------------------------
// Pooling
// ----------------------------------------------------------------------------

class Daphne_PoolForwardOp<string name, list<Trait> traits = []> : Daphne_Op<name, traits> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input,
        // input shape (NCHW)
        Size:$inputBatchSize, Size:$inputNumChannels, Size:$inputHeight, Size:$inputWidth,
        // pool shape (ToDo: optional, default filter size 2x2)
        Size:$poolHeight, Size:$poolWidth,
        // filter application properties (ToDo: optional, default stride=1x1, padding=0x0)
        Size:$strideHeight, Size:$strideWidth, Size:$padHeight, Size:$padWidth
    );

    let results = (outs MatrixOf<[FloatScalar]>:$data, Size:$outHeight, Size:$outWidth);
}

def Daphne_MaxPoolForwardOp : Daphne_PoolForwardOp<"NN_Max_Forward", [ CUDASupport ]>;
def Daphne_AvgPoolForwardOp : Daphne_PoolForwardOp<"NN_Avg_Forward", [ CUDASupport ]>;

// ----------------------------------------------------------------------------
// Softmax
// ----------------------------------------------------------------------------
def Daphne_SoftmaxForwardOp : Daphne_Op<"softmaxForward", [ CUDASupport ]> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input);

    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

// ****************************************************************************
// Other matrix operations
// ****************************************************************************

def Daphne_DiagVectorOp : Daphne_Op<"diagVector", [
    TypeFromFirstArg, NumRowsFromArg, OneCol
]> {
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs MatrixOrU:$res);
}

def Daphne_TriOp : Daphne_Op<"tri", [
    TypeFromFirstArg, ShapeFromArg, DeclareOpInterfaceMethods<InferSparsityOpInterface>
]> {
    let arguments = (ins MatrixOrU:$arg, BoolScalar:$upper, BoolScalar:$diag, BoolScalar:$values);
    let results = (outs MatrixOrU:$res);
}

def Daphne_SolveOp : Daphne_Op<"solve", [
    DataTypeMat, ValueTypeFromArgs, NumRowsFromArg, OneCol, CUDASupport, CastArgsToResType
]> {
    let arguments = (ins MatrixOf<[NumScalar]>:$a, MatrixOf<[NumScalar]>:$b);
    let results = (outs MatrixOf<[NumScalar]>:$x);
}

def Daphne_ReplaceOp : Daphne_Op<"replace", [
    DataTypeFromFirstArg, ValueTypeFromArgs, ShapeFromArg, CastArgsToResType
]> {
    let arguments = (ins MatrixOrU:$arg, AnyScalar:$pattern, AnyScalar:$replacement);
    let results = (outs MatrixOrU:$res);
}

def Daphne_CTableOp : Daphne_Op<"ctable", [
    DataTypeMat, ValueTypeFromThirdArg,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    let arguments = (ins MatrixOrU:$lhs, MatrixOrU:$rhs, NumScalar:$weight, SI64:$resNumRows, SI64:$resNumCols);
    let results = (outs MatrixOrU:$res);
}

def Daphne_SyrkOp : Daphne_Op<"syrk", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    NumRowsFromArgNumCols, NumColsFromArg, CUDASupport,FPGAOPENCLSupport, 
    CastArgsToResType
]> {
    // TODO: support `A @ t(A)` operation
    let summary = [{Performs the operation `t(A) @ A`}];
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs MatrixOrU:$res);
}

def Daphne_GemvOp : Daphne_Op<"gemv", [
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DataTypeMat, ValueTypeFromArgs,
    NumRowsFromArgNumCols, OneCol, CUDASupport,
    CastArgsToResType
]> {
    // TODO: support `A @ x` operation
    let summary = [{Performs the operation `t(A) @ x`}];
    let arguments = (ins MatrixOrU:$mat, MatrixOrU:$vec);
    let results = (outs MatrixOrU:$res);
}

// ****************************************************************************
// Extended relational algebra
// ****************************************************************************

// ----------------------------------------------------------------------------
// Entire SQL query
// ----------------------------------------------------------------------------

def Daphne_SqlOp : Daphne_Op<"sql"> {
    let arguments = (ins StrAttr:$sql);
    let results = (outs FrameOrU:$res);
}

def Daphne_RegisterViewOp : Daphne_Op<"registerView"> {
    let arguments = (ins StrAttr:$view, MatrixOrFrame:$arg);
    let results = (outs); // no results
}

// ----------------------------------------------------------------------------
// Set operations
// ----------------------------------------------------------------------------

class Daphne_SetOp<string name, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeFrm, ValueTypeFromArgs,
    NumColsFromAllArgs
])> {
    let arguments = (ins FrameOrU:$lhs, FrameOrU:$rhs);
    let results = (outs FrameOrU:$res);
}

def Daphne_IntersectOp : Daphne_SetOp<"intersect">;
def Daphne_MergeOp : Daphne_SetOp<"merge">;
def Daphne_ExceptOp : Daphne_SetOp<"except">;

// ----------------------------------------------------------------------------
// Cartesian product and joins
// ----------------------------------------------------------------------------

def Daphne_CartesianOp : Daphne_Op<"cartesian", [
    DataTypeFrm, ValueTypesConcat,
    NumColsFromSumOfAllArgs,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferNumRowsOpInterface>
]> {
    let arguments = (ins FrameOrU:$lhs, FrameOrU:$rhs); //let arguments = (ins Variadic<FrameOrU>:$args);
    let results = (outs FrameOrU:$res);
}

class Daphne_JoinOp<string name, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeFrm, ValueTypesConcat,
    SameVariadicOperandSize,
    NumColsFromSumOfAllArgs
])> {
    // TODO Maybe leftOn and rightOn should be attributes.
    let arguments = (ins FrameOrU:$lhs, FrameOrU:$rhs, Variadic<Size>:$leftOn, Variadic<Size>:$rightOn);
    let results = (outs FrameOrU:$res);
}

def Daphne_InnerJoinOp : Daphne_Op<"innerJoin", [
    DataTypeFrm, ValueTypesConcat,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
]> {
    let arguments = (ins FrameOrU:$lhs, FrameOrU:$rhs, StrScalar:$lhsOn, StrScalar:$rhsOn);
    let results = (outs FrameOrU:$res);
}


def Daphne_CompareOperation_Equal : I32EnumAttrCase<"Equal", 1>;
def Daphne_CompareOperation_LessThan : I32EnumAttrCase<"LessThan", 2>;
def Daphne_CompareOperation_LessEqual : I32EnumAttrCase<"LessEqual", 3>;
def Daphne_CompareOperation_GreaterThan : I32EnumAttrCase<"GreaterThan", 4>;
def Daphne_CompareOperation_GreaterEqual : I32EnumAttrCase<"GreaterEqual", 5>;
def Daphne_CompareOperation_NotEqual : I32EnumAttrCase<"NotEqual", 6>;

def Daphne_CompareEnum : I32EnumAttr<"CompareOperation", "", [
   Daphne_CompareOperation_Equal, Daphne_CompareOperation_LessThan, Daphne_CompareOperation_LessEqual,
   Daphne_CompareOperation_GreaterThan, Daphne_CompareOperation_GreaterEqual, Daphne_CompareOperation_NotEqual
   ]>{
    let cppNamespace = "::mlir::daphne";
}

def Daphne_ThetaJoinOp : Daphne_Op<"thetaJoin", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DataTypeFrm, ValueTypesConcat
]> {
    let arguments = (
        ins Frame:$lhs,
        Frame:$rhs,
        Variadic<StrScalar>:$lhsOn,
        Variadic<StrScalar>:$rhsOn,
        TypedArrayAttrBase<Daphne_CompareEnum, "enum">:$cmp);
    let results = (outs Frame:$res);
}


def Daphne_FullOuterJoinOp : Daphne_JoinOp<"fullOuterJoin">;
def Daphne_LeftOuterJoinOp : Daphne_JoinOp<"leftOuterJoin">;
def Daphne_AntiJoinOp : Daphne_JoinOp<"antiJoin">;

// TODO Reconcile this with the other join ops, but we need it to work quickly now.
def Daphne_SemiJoinOp : Daphne_Op<"semiJoin", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    NumColsFromArg
]> {
    let arguments = (ins FrameOrU:$lhs, FrameOrU:$rhs, StrScalar:$lhsOn, StrScalar:$rhsOn);
    let results = (outs FrameOrU:$res, MatrixOf<[Size]>:$lhsTids);
}

def Daphne_GroupJoinOp : Daphne_Op<"groupJoin", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    // TODO Support arbitrary aggregation functions.
    // TODO Support an arbitrary number of aggregates.
    // TODO Support an arbitrary number of join/group columns.

    let summary = [{
        Group-join of `lhs` and `rhs` on `lhs.lhsOn == rhs.rhsOn` with
        summation of `rhs.rhsAgg`.
    }];

    let description = [{
        Performs a group-join of the input frames `lhs` and `rhs` on
        `lhs.lhsOn == rhs.rhsOn`, where `lhsOn` and `rhsOn` are column labels
        in the respective frame, including a summation on `rhs.rhsAgg`, where
        `rhsAgg` is a column label in `rhs`.

        This is equivalent to an inner join of `lhs` and `rhs` on
        `lhs.lhsOn == rhs.rhsOn` followed by a grouping of the result on
        `lhsOn` including a summation on `rhsAgg`.

        `lhs.lhsOn` is assumed to be unique.

        The results are:
        - A two-column frame consisting of (1) the distinct values in
          `lhsOn`/`rhsOn` that pass the join and (2) the grouped aggregates in
          `rhsAgg`. The column labels are `lhsOn` and `rhsAgg`.
        - A column-matrix containing the positions of the tuples in `lhs`
          corresponding to the pairs in the first result.
    }];

    let arguments = (ins FrameOrU:$lhs, FrameOrU:$rhs, StrScalar:$lhsOn, StrScalar:$rhsOn, StrScalar:$rhsAgg);
    // TODO Result `lhsTids` could be made optional.
    let results = (outs FrameOrU:$res, MatrixOf<[Size]>:$lhsTids);
}

// ----------------------------------------------------------------------------
// Selection
// ----------------------------------------------------------------------------

def Daphne_QueryOp : Daphne_Op<"query", [
    TypeFromFirstArg, NumColsFromArg
]> {
    let arguments = (ins FrameOrU:$arg, StrAttr:$cond);
    let results = (outs FrameOrU:$res);
}

def Daphne_FilterRowOp : Daphne_Op<"filterRow", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    NumColsFromArg
]> {
    let summary = "Filters the rows of a data object according to a bit vector";

    let description = [{
        Filters the rows of a data object (matrix or frame) according to a bit
        vector. The argument `selectedRows` must be a single-column matrix. All
        its entries must be zero or one. The value type of `selectedRows` does
        not need to be boolean, it could also be an integer or floating-point
        type. `selectedRows` and `source` must have the same number of rows.
        The *i*-th row of `source` is contained in the result, if the *i*-th
        entry of `selectedRows` is a one (in its respective value type).

        Note that `ExtractRowOp` and `SliceRowOp` are similar, but work with
        positions instead of bit vectors.
    }];

    let arguments = (ins MatrixOrFrame:$source, MatrixOrU:$selectedRows);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_FilterColOp : Daphne_Op<"filterCol", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    // TODO Support frame label inference, (see #484).
    /*DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,*/
    NumRowsFromArg
]> {
    let summary = "Filters the columns of a data object according to a bit vector";

    let description = [{
        Filters the columns of a data object (matrix or frame) according to a bit
        vector. The argument `selectedCols` must be a single-column matrix. All
        its entries must be zero or one. The value type of `selectedCols` does
        not need to be boolean, it could also be an integer or floating-point
        type. The number of entries (rows) in `selectedCols` and the number of
        columns in `source` must be the same.
        The *i*-th column of `source` is contained in the result, if the *i*-th
        entry of `selectedCols` is a one (in its respective value type).

        Note that `ExtractColOp` and `SliceColOp` are similar, but work with
        positions instead of bit vectors.
    }];

    let arguments = (ins MatrixOrFrame:$source, MatrixOrU:$selectedCols);
    let results = (outs MatrixOrFrame:$res);
}

// Note that ExtractOp can be used to filter rows by a column of bool or a
// column of positions.

// ----------------------------------------------------------------------------
// Grouping
// ----------------------------------------------------------------------------

def Daphne_GroupAggCount : I32EnumAttrCase<"COUNT", 1>;
def Daphne_GroupAggSum : I32EnumAttrCase<"SUM", 2>;
def Daphne_GroupAggMin : I32EnumAttrCase<"MIN", 3>;
def Daphne_GroupAggMax : I32EnumAttrCase<"MAX", 4>;
def Daphne_GroupAggAvg : I32EnumAttrCase<"AVG", 5>;

def Daphne_GroupAggEnum : I32EnumAttr<"GroupEnum", "", [Daphne_GroupAggCount, Daphne_GroupAggSum, Daphne_GroupAggMin, Daphne_GroupAggMax, Daphne_GroupAggAvg]>{
    let cppNamespace = "::mlir::daphne";
}

def Daphne_GroupOp : Daphne_Op<"group", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>]>{
    let arguments = (
        ins FrameOrU:$frame,
        Variadic<StrScalar>:$keyCol,
        Variadic<StrScalar>:$aggCol,
        TypedArrayAttrBase<Daphne_GroupAggEnum, "enum">:$aggFuncs
    );
    let results = (outs FrameOrU:$res);
}

// ****************************************************************************
// Frame label manipulation
// ****************************************************************************

def Daphne_SetColLabelsOp : Daphne_Op<"setColLabels", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    ShapeFromArg
]> {
    let arguments = (ins FrameOrU:$arg, Variadic<StrScalar>:$labels);
    let results = (outs FrameOrU:$res);
}

def Daphne_SetColLabelsPrefixOp : Daphne_Op<"setColLabelsPrefix", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    ShapeFromArg
]> {
    let arguments = (ins FrameOrU:$arg, StrScalar:$prefix);
    let results = (outs FrameOrU:$res);
}

// ****************************************************************************
// Conversions, casts, and copying
// ****************************************************************************

def Daphne_ToStringOp : Daphne_Op<"toString", [DataTypeSca, ValueTypeStr]> {
    // TODO In the future, we might introduce more arguments, e.g. for
    // formatting a single value.
    let arguments = (ins MatrixOrFrame:$arg, StrScalar:$rowSep, StrScalar:$colSep);
    let results = (outs StrScalar:$res);
}

def Daphne_CastOp : Daphne_Op<"cast", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    ShapeFromArg
]> {
    // Note that the requested result type is not an argument, but should be
    // specified as the output type when creating a CastOp.
    let arguments = (ins AnyTypeOf<[MatrixOrFrame, AnyScalar, Unknown]>:$arg);
    let results = (outs AnyTypeOf<[MatrixOrFrame, AnyScalar, Unknown]>:$res);
    let hasFolder = 1;

    let extraClassDeclaration = [{
        bool isTrivialCast() {
            return getArg().getType() == getRes().getType();
        }
        /**
         * @brief Checks if this cast just removes detailed properties from a matrix/frame type,
         * to make them unknown instead.
         *
         * Usually casts like this need to exist to make the IR correct for SCF operations.
         * @return true if this cast just tags some known information unknown, false otherwise
         */
        bool isRemovePropertyCast() {
            Type argTy = getArg().getType();
            Type resTy = getRes().getType();
            auto argMatTy = argTy.dyn_cast<daphne::MatrixType>();
            auto resMatTy = resTy.dyn_cast<daphne::MatrixType>();
            auto argFrmTy = argTy.dyn_cast<daphne::FrameType>();
            auto resFrmTy = resTy.dyn_cast<daphne::FrameType>();
            return (argMatTy && resMatTy && argMatTy.isSpecializationOf(resMatTy))
                || (argFrmTy && resFrmTy && argFrmTy.isSpecializationOf(resFrmTy));
        }
    }];
}

def Daphne_CopyOp : Daphne_Op<"copy", [
    TypeFromFirstArg
]> {
    let arguments = (ins MatrixOrFrame:$arg);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_QuantizeOp : Daphne_Op<"quantize", [
    ShapeFromArg
]> {
    let arguments = (ins MatrixOrU:$arg, NumScalar:$min, NumScalar:$max);
    let results = (outs MatrixOrU:$res);
}

def Daphne_GetColIdxOp : Daphne_Op<"getColIdx", [DataTypeSca, ValueTypeSize]>{
    let arguments = (ins Frame:$frame, StrScalar:$columnName);
    let results = (outs Size:$res);
}
// ****************************************************************************
// Distributed Operations
// ****************************************************************************

def Daphne_DistributedReadOp : Daphne_Op<"distributedRead", [Pure]> {
    let arguments = (ins StrScalar:$fileName);
    let results = (outs Handle:$res);
}

def Daphne_DistributeOp : Daphne_Op<"distribute", [Pure]> {
    let arguments = (ins MatrixOrU:$mat);
    let results = (outs Handle:$res);

    let hasCanonicalizer = 1;
}

def Daphne_BroadcastOp : Daphne_Op<"broadcast", [Pure]> {
    let arguments = (ins MatrixOrU:$mat);
    let results = (outs Handle:$res);
}

def Daphne_DistributedComputeOp : Daphne_Op<"distributedCompute", [Pure, IsolatedFromAbove]> {
    let arguments = (ins Variadic<Handle>:$args);
    let results = (outs Variadic<Handle>:$res);
    let regions = (region SizedRegion<1>:$body);
}

// TODO: Map Keys Operation

def Daphne_DistributedCollectOp : Daphne_Op<"distributedCollect", [Pure]> {
    let arguments = (ins Handle:$arg);
    let results = (outs AnyType:$res);

    let builders = [
        OpBuilder<(ins "Value":$value), [{
            build($_builder, $_state, value.getType().cast<HandleType>().getDataType(), value);
        }]>,
    ];
}

// ****************************************************************************
// Input/output
// ****************************************************************************

// ----------------------------------------------------------------------------
// High-level
// ----------------------------------------------------------------------------

def Daphne_PrintOp : Daphne_Op<"print"> {
    // TODO We might change it to only accept scalars here and enforce toString
    // for matrices and frames. But currently, we need it like that for the
    // rest of the program.
    let arguments = (ins AnyTypeOf<[AnyScalar, MatrixOrFrame, AnyMemRef, Unknown]>:$arg, BoolScalar:$newline, BoolScalar:$err);
    let results = (outs); // no results
}

// TODO Take asynchronous read into account.
def Daphne_ReadOp : Daphne_Op<"read", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferSparsityOpInterface>
]> {
    // TODO We might add arguments for a UDF later.
    let arguments = (ins StrScalar:$fileName);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_WriteOp : Daphne_Op<"write"> {
    let arguments = (ins MatrixOrFrame:$arg, StrScalar:$fileName);
    let results = (outs); // no results
}

def Daphne_ReceiveFromNumpyOp: Daphne_Op<"receiveFromNumpy">{
    let arguments = (ins UI32:$upper, UI32:$lower, SI64:$rows, SI64:$cols);
    let results = (outs MatrixOrU:$res);
}

def Daphne_SaveDaphneLibResultOp : Daphne_Op<"saveDaphneLibResult"> {
    let arguments = (ins MatrixOrFrame:$arg);
    let results = (outs); // no results
}

// ----------------------------------------------------------------------------
// Low-level
// ----------------------------------------------------------------------------

def Daphne_OpenFileOp : Daphne_Op<"openFile"> {
    let arguments = (ins StrScalar:$filename);
    let results = (outs File:$res);
}

def Daphne_OpenDeviceOp : Daphne_Op<"openDevice"> {
    let arguments = (ins StrScalar:$device);
    let results = (outs Target:$res);
}

def Daphne_OpenFileOnTargetOp : Daphne_Op<"openFileOnTarget"> {
    let arguments = (ins Target:$target, StrScalar:$filename);
    let results = (outs Descriptor:$res);
}

def Daphne_CloseOp : Daphne_Op<"close"> {
    let arguments = (ins AnyTypeOf<[File, Target, Unknown]>);
    let results = (outs); // no results
}

def Daphne_ReadCsvOp : Daphne_Op<"readCsv"> {
    let arguments = (ins
        AnyTypeOf<[File, Descriptor, Unknown]>:$input,
        Size:$numRows, Size:$numCols,
        StrScalar:$delim
    );
    let results = (outs MatrixOrFrame:$res);
}

// ****************************************************************************
// Data preprocessing
// ****************************************************************************

def Daphne_OneHotOp : Daphne_Op<"oneHot", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    // TODO We can know the exact number of columns at compile-time if $info is
    // known at compile-time (e.g. if it is a matrix literal).
    NumRowsFromArg, // DeclareOpInterfaceMethods<InferNumColsOpInterface>
]> {
    let arguments = (ins MatrixOrFrame:$arg, MatrixOf<[SI64]>:$info);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_RecodeOp : Daphne_Op<"recode", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    let arguments = (ins MatrixOrFrame:$arg, BoolScalar:$orderPreserving);
    let results = (outs MatrixOrFrame:$res, MatrixOrFrame:$dict);
}

def Daphne_BinOp : Daphne_Op<"bin", [
    TypeFromFirstArg,
    ShapeFromArg
]> {
    let arguments = (ins MatrixOrFrame:$arg, NumScalar:$numBins, NumScalar:$min, NumScalar:$max);
    let results = (outs MatrixOrFrame:$res);
}

// ****************************************************************************
// Measurements
// ****************************************************************************

def Daphne_NowOp : Daphne_Op<"now", [DataTypeSca, ValueTypeSI64]> {
    let summary = "Returns the current time since the epoch in nano seconds.";
    let description = [{
        Take care if you use this for time measurements! This is currently just
        a quick and somewhat dirty approach. Once we have compiler passes that
        could reorder operations, you might not measure the time you expect
        anymore!
    }];

    let arguments = (ins); // no arguments
    let results = (outs SI64);
}

// ****************************************************************************
// Context handling
// ****************************************************************************

def Daphne_CreateDaphneContextOp : Daphne_Op<"createDaphneContext"> {
    let summary = "Creates an instance of a Daphne run-time context.";

    let description = [{
        The newly created instance might be partially initialized. The
        initialization of certain parts/aspects of the context (which are not
        always relevant) might require additional DaphneIR operations.
    }];

    let arguments = (ins UI64:$userConfigPtr, UI64:$dispatchMappingPtr);
    let results = (outs DaphneContext:$ctx);
}

def Daphne_DestroyDaphneContextOp : Daphne_Op<"destroyDaphneContext"> {
    let summary = "Destroys an instance of a Daphne run-time context.";

    // Note that the DaphneContext is automatically injected during the
    // lowering to kernel calls.
    let arguments = (ins); // no arguments
    let results = (outs); // no results
}


def Daphne_CreateCUDAContextOp : Daphne_Op<"createCUDAContext", [CUDASupport]> {
    let arguments = (ins);
    let results = (outs);
}

def Daphne_CreateDistributedContextOp : Daphne_Op<"createDistributedContext", []> {
    let arguments = (ins);
    let results = (outs);
}

def Daphne_CreateFPGAContextOp : Daphne_Op<"createFPGAContext", [FPGAOPENCLSupport]> {
    let arguments = (ins);
    let results = (outs);
}


// ****************************************************************************
// Vectorized operations
// ****************************************************************************

// TODO: should be recursive side effects
// TODO: use symbol handling (should be isolated from above, but needs reference to daphne context)
def Daphne_VectorizedPipelineOp : Daphne_Op<"vectorizedPipeline", [AttrSizedOperandSegments]> {
    let summary = "Performs multi-threaded vector-wise computations of the input.";

    let arguments = (ins Variadic<AnyTypeOf<[AnyScalar, MatrixOrFrame, Unknown]>>:$inputs,
            // TODO: both types should be SIZE, but we require `-1` for dynamic/unknown
            Variadic<AnyTypeOf<[SIntScalar, Size, Unknown]>>:$out_rows,
            Variadic<AnyTypeOf<[SIntScalar, Size, Unknown]>>:$out_cols,
            TypedArrayAttrBase<VectorSplitAttr, "Vector-Splits">:$splits,
            TypedArrayAttrBase<VectorCombineAttr, "Vector-Combines">:$combines,
            Optional<DaphneContext>:$ctx);
    let results = (outs Variadic<MatrixOrFrame>:$outputs);
    let regions = (region SizedRegion<1>:$body, AnyRegion:$cuda);

    let hasCanonicalizeMethod = 1;
}

def Daphne_DistributedPipelineOp : Daphne_Op<"distributedPipeline", [AttrSizedOperandSegments]> {
    let summary = "...";

    let arguments = (ins
            StrScalar:$ir,
            Variadic<AnyTypeOf<[AnyScalar, MatrixOrFrame]>>:$inputs,
            // TODO: both types should be SIZE, but we require `-1` for dynamic/unknown
            Variadic<AnyTypeOf<[SIntScalar, Size]>>:$out_rows,
            Variadic<AnyTypeOf<[SIntScalar, Size]>>:$out_cols,
            TypedArrayAttrBase<VectorSplitAttr, "Vector-Splits">:$splits,
            TypedArrayAttrBase<VectorCombineAttr, "Vector-Combines">:$combines
    );
    let results = (outs Variadic<MatrixOrFrame>:$outputs);
}

// ****************************************************************************
// Higher-order operations
// ****************************************************************************
def Daphne_MapOp : Daphne_Op<"map", [ShapeFromArg]> {
    let summary = "Applies a user defined function to elements of a matrix.";
    let arguments = (ins MatrixOrU:$arg, SymbolNameAttr:$func);
    let results = (outs MatrixOrU:$res);
}


// ****************************************************************************
// Low-level auxiliary operations
// ****************************************************************************

def Daphne_CreateVariadicPackOp : Daphne_Op<"createVariadicPack"> {
    let arguments = (ins I64Attr:$numElements);
    let results = (outs VariadicPack:$res);
}

def Daphne_StoreVariadicPackOp : Daphne_Op<"storeVariadicPack"> {
    let arguments = (ins VariadicPack:$pack, AnyType:$item, I64Attr:$pos);
    let results = (outs); // no results
}

def Daphne_IncRefOp : Daphne_Op<"incRef"> {
    let summary = "Increases the reference counter of the underlying runtime data object.";

    let arguments = (ins MatrixOrFrame:$arg);
    let results = (outs); // no results
}

def Daphne_DecRefOp : Daphne_Op<"decRef"> {
    let summary = "Decreases the reference counter of the underlying runtime data object and frees it if the reference counter becomes zero.";

    let arguments = (ins MatrixOrFrame:$arg);
    let results = (outs); // no results
}

// ****************************************************************************
// Profiling
// ****************************************************************************

def Daphne_StartProfilingOp : Daphne_Op<"startProfiling"> {
    let summary = "Starts profiling";

    let arguments = (ins); // no arguments
    let results = (outs); // no results
}

def Daphne_StopProfilingOp : Daphne_Op<"stopProfiling"> {
    let summary = "Stops profiling";

    let arguments = (ins); //no arguments
    let results = (outs); // no results
}

// ****************************************************************************
// Old operations
// ****************************************************************************
// These are currently still need for the prototype to work. They will be
// removed/replaced/changed in the future.

def Daphne_ConstantOp : Daphne_Op<"constant", [ConstantLike, Pure]> {
    let summary = "constant operation";

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyTypeOf<[AnyScalar, Matrix, Unknown]>:$result);

    let builders = [
        OpBuilder<(ins "double":$value), [{
            build($_builder, $_state, $_builder.getF64Type(), $_builder.getF64FloatAttr(value));
        }]>,
        OpBuilder<(ins "float":$value), [{
            build($_builder, $_state, $_builder.getF32Type(), $_builder.getF32FloatAttr(value));
        }]>,
        OpBuilder<(ins "int64_t":$value), [{
            Type t = $_builder.getIntegerType(64, true);
            build($_builder, $_state, t, $_builder.getIntegerAttr(t, value));
        }]>,
        OpBuilder<(ins "uint64_t":$value), [{
            Type t = $_builder.getIntegerType(64, false);
            build($_builder, $_state, t, $_builder.getIntegerAttr(t, value));
        }]>,
        OpBuilder<(ins "bool":$value), [{
            Type t = $_builder.getI1Type();
            // TODO Why not getBoolAttr?
            build($_builder, $_state, t, $_builder.getIntegerAttr(t, value));
        }]>,
        //OpBuilder<(ins "int":$value), [{
        //    build($_builder, $_state, $_builder.getI32IntegerAttr(value));
        //}]>,
        OpBuilder<(ins "std::string":$value), [{
            build($_builder, $_state, mlir::daphne::StringType::get($_builder.getContext()), $_builder.getStringAttr(value));
        }]>,
    ];

    let hasFolder = 1;
}

def Daphne_ReturnOp : Daphne_Op<"return", [Pure, Terminator, ReturnLike, ParentOneOf<["func::FuncOp", "DistributedComputeOp", "VectorizedPipelineOp"]>]> {
    let summary = "return operation";

    let arguments = (ins Variadic<AnyType>:$operands);

    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];

    let builders = [
        OpBuilder<(ins), [{
            build($_builder, $_state, std::nullopt);
        }]>
    ];
}

def Daphne_CallKernelOp : Daphne_Op<"call_kernel"> {
    let summary = "kernel operation for C";

    let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$inputs);
    let results = (outs Variadic<AnyType>);

    let builders = [
        OpBuilder<(ins "StringAttr":$callee, CArg<"ValueRange", "{}">:$operands, CArg<"TypeRange", "{}">:$results), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(results);
        }]>,
        OpBuilder<(ins "StringRef":$callee, CArg<"ValueRange", "{}">:$operands, CArg<"TypeRange", "{}">:$results), [{
            build($_builder, $_state, $_builder.getStringAttr(callee), operands, results);
        }]>,
    ];
}

def Daphne_GenericCallOp : Daphne_Op<"generic_call"> {
    let summary = "User defined function call";

    let arguments = (ins SymbolNameAttr:$callee, Variadic<AnyType>:$inputs);
    let results = (outs Variadic<AnyType>);

    let builders = [
        OpBuilder<(ins "StringAttr":$callee, CArg<"ValueRange", "{}">:$operands, CArg<"TypeRange", "{}">:$results), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(results);
        }]>,
        OpBuilder<(ins "StringRef":$callee, CArg<"ValueRange", "{}">:$operands, CArg<"TypeRange", "{}">:$results), [{
            build($_builder, $_state, $_builder.getStringAttr(callee), operands, results);
        }]>,
    ];
}

class Daphne_ElementwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
        Daphne_Op<mnemonic, !listconcat(traits, [Pure, TypesMatchOrOneIsMatrixOfOther<"lhs", "rhs">])> {
    let arguments = (ins AnyTypeOf<[AnyScalar, Matrix, Unknown]>:$lhs, AnyTypeOf<[AnyScalar, Matrix, Unknown]>:$rhs);
    let results = (outs AnyTypeOf<[AnyScalar, Matrix, Unknown]>:$result);

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            $_state.addTypes(lhs.getType());
            $_state.addOperands({lhs, rhs});
        }]>
    ];
}

def Daphne_AddOp : Daphne_ElementwiseBinaryOp<"addOld", [SameOperandsAndResultShape]> {
    let summary = "add operation";
}

def Daphne_SubOp : Daphne_ElementwiseBinaryOp<"subOld", [SameOperandsAndResultShape]> {
    let summary = "subtraction operation";
}

def Daphne_MulOp : Daphne_ElementwiseBinaryOp<"mulOld", [SameOperandsAndResultShape]> {
    let summary = "mul operation";
}

def Daphne_SetCellOp : Daphne_Op<"setCell",  []> {
    let summary = "set cell operation";

    let arguments = (ins MatrixOrU:$mat, Index:$row, Index:$col, AnyScalar:$val);
}

#endif //SRC_IR_DAPHNEIR_DAPHNEOPS_TD
