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
include "ir/daphneir/CUDASupport.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

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

class Daphne_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Daphne_Dialect, mnemonic, traits>;

// ****************************************************************************
// Data generation
// ****************************************************************************

def Daphne_FillOp : Daphne_Op<"fill", [
    NumRowsFromIthScalar<1>, NumColsFromIthScalar<2>
]> {
    let arguments = (ins AnyScalar:$arg, Size:$numRows, Size:$numCols);
    let results = (outs Matrix:$res);
}

def Daphne_MatrixConstantOp : Daphne_Op<"matrixConstant">{
    let arguments = (ins UI64:$matrixAddr);
    let results = (outs Matrix:$res);
}

def Daphne_CreateFrameOp : Daphne_Op<"createFrame", [
    SameVariadicOperandSize,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    let arguments = (ins Variadic<Matrix>:$cols, Variadic<StrScalar>:$labels);
    let results = (outs Frame:$res);
}

def Daphne_DiagMatrixOp : Daphne_Op<"diagMatrix", [
    NumRowsFromArg, NumColsFromArgNumRows,
    DeclareOpInterfaceMethods<InferSparsityOpInterface>
]> {
    let arguments = (ins Matrix:$arg);
    let results = (outs Matrix:$res);
}

def Daphne_RandMatrixOp : Daphne_Op<"randMatrix", [
    TypesMatchWith<"min and max arguments must be of the same type", "min", "max", "$_self">,
    NumRowsFromIthScalar<0>, NumColsFromIthScalar<1>, DeclareOpInterfaceMethods<InferTypesOpInterface>,
    SparsityFromIthScalar<4>
]> {
    //let arguments = (ins Size:$numRows, Size:$numCols, AnyScalar:$min, AnyScalar:$max, F64:$sparsity, Seed:$seed, StrScalar:$pdf);
    let arguments = (ins Size:$numRows, Size:$numCols, AnyScalar:$min, AnyScalar:$max, F64:$sparsity, IntScalar:$seed);
    let results = (outs Matrix:$res);
}

def Daphne_SampleOp : Daphne_Op<"sample", [
    NumRowsFromIthScalar<1>, OneCol
]> {
    let arguments = (ins AnyScalar:$range, Size:$size, BoolScalar:$withReplacement, Seed:$seed);
    let results = (outs Matrix:$res);
}

def Daphne_SeqOp : Daphne_Op<"seq", [
    OneCol, DeclareOpInterfaceMethods<InferNumRowsOpInterface>, CompletelyDense
]> {
    let arguments = (ins NumScalar:$from, NumScalar:$to, NumScalar:$inc);
    let results = (outs MatrixOf<[NumScalar]>);
}

// ****************************************************************************
// Matrix/frame dimensions
// ****************************************************************************

class Daphne_NumOp<string name, list<OpTrait> traits = []> : Daphne_Op<name, !listconcat(traits, [NoSideEffect])> {
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
    NumRowsFromIthArg<0>, NumColsFromIthArg<1>,
    DeclareOpInterfaceMethods<InferSparsityOpInterface>, CUDASupport
]> {
    let arguments = (ins MatrixOf<[NumScalar]>:$lhs, MatrixOf<[NumScalar]>:$rhs);
    let results = (outs MatrixOf<[NumScalar]>:$res);
}

// ****************************************************************************
// Elementwise unary
// ****************************************************************************

class Daphne_EwUnaryOp<string name, Type scalarType, list<OpTrait> traits = []> : Daphne_Op<name, !listconcat(traits, [
    ShapeFromArg
])> {
    let arguments = (ins AnyTypeOf<[MatrixOf<[scalarType]>, scalarType]>:$arg);
    let results = (outs AnyTypeOf<[MatrixOf<[scalarType]>, scalarType]>:$res);
}

// ----------------------------------------------------------------------------
// Arithmetic/general math
// ----------------------------------------------------------------------------

def Daphne_EwMinusOp : Daphne_EwUnaryOp<"ewMinus", NumScalar>;
def Daphne_EwAbsOp : Daphne_EwUnaryOp<"ewAbs", NumScalar>;
def Daphne_EwSignOp : Daphne_EwUnaryOp<"ewSign", NumScalar>;
def Daphne_EwExpOp : Daphne_EwUnaryOp<"ewExp", FloatScalar>;
def Daphne_EwLnOp : Daphne_EwUnaryOp<"ewLn", FloatScalar>;
def Daphne_EwSqrtOp : Daphne_EwUnaryOp<"ewSqrt", FloatScalar, [DeclareOpInterfaceMethods<VectorizableOpInterface>]>;

// ----------------------------------------------------------------------------
// Logical
// ----------------------------------------------------------------------------

def Daphne_EwNegOp : Daphne_EwUnaryOp<"ewNeg", AnyTypeOf<[BoolScalar, IntScalar]>>;

// ----------------------------------------------------------------------------
// Rounding
// ----------------------------------------------------------------------------

def Daphne_EwRoundOp : Daphne_EwUnaryOp<"ewRound", FloatScalar>;
def Daphne_EwFloorOp : Daphne_EwUnaryOp<"ewFloor", FloatScalar>;
def Daphne_EwCeilOp : Daphne_EwUnaryOp<"ewCeil", FloatScalar>;

// ----------------------------------------------------------------------------
// Trigonometric
// ----------------------------------------------------------------------------

def Daphne_EwSinOp : Daphne_EwUnaryOp<"ewSin", FloatScalar>;
def Daphne_EwCosOp : Daphne_EwUnaryOp<"ewCos", FloatScalar>;
def Daphne_EwTanOp : Daphne_EwUnaryOp<"ewTan", FloatScalar>;
def Daphne_EwSinhOp : Daphne_EwUnaryOp<"ewSinh", FloatScalar>;
def Daphne_EwCoshOp : Daphne_EwUnaryOp<"ewCosh", FloatScalar>;
def Daphne_EwTanhOp : Daphne_EwUnaryOp<"ewTanh", FloatScalar>;
def Daphne_EwAsinOp : Daphne_EwUnaryOp<"ewAsin", FloatScalar>;
def Daphne_EwAcosOp : Daphne_EwUnaryOp<"ewAcos", FloatScalar>;
def Daphne_EwAtanOp : Daphne_EwUnaryOp<"ewAtan", FloatScalar>;

// ****************************************************************************
// Elementwise binary
// ****************************************************************************

// TODO Temporarily commented, because otherwise it cannot handle UnknownType.
class Daphne_EwBinaryOp<string name, Type scalarType, list<OpTrait> traits = []>
//: Daphne_Op<name, !listconcat([TypesMatchOrOneIsMatrixOfOther<"lhs", "rhs">], traits)> {
: Daphne_Op<name, !listconcat(traits, [
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    ShapeEwBinary
])> {
    let arguments = (ins AnyTypeOf<[MatrixOf<[scalarType]>, scalarType]>:$lhs, AnyTypeOf<[MatrixOf<[scalarType]>, scalarType]>:$rhs);
    let results = (outs AnyTypeOf<[MatrixOf<[scalarType]>, scalarType]>:$res);

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

def Daphne_EwAddOp    : Daphne_EwBinaryOp<"ewAdd", NumScalar, [Commutative, EwSparseIfBoth, CUDASupport]> {
    let hasCanonicalizeMethod = 1;
}
def Daphne_EwSubOp    : Daphne_EwBinaryOp<"ewSub", NumScalar, [EwSparseIfBoth, CUDASupport]>;
def Daphne_EwMulOp    : Daphne_EwBinaryOp<"ewMul", NumScalar, [Commutative, EwSparseIfEither]>;
def Daphne_EwDivOp    : Daphne_EwBinaryOp<"ewDiv", NumScalar, [CUDASupport]>;
def Daphne_EwPowOp    : Daphne_EwBinaryOp<"ewPow", NumScalar, [CUDASupport]>;
def Daphne_EwModOp    : Daphne_EwBinaryOp<"ewMod", IntScalar>;
def Daphne_EwLogOp    : Daphne_EwBinaryOp<"ewLog", FloatScalar>;

// ----------------------------------------------------------------------------
// Min/max
// ----------------------------------------------------------------------------

def Daphne_EwMinOp    : Daphne_EwBinaryOp<"ewMin", AnyScalar, [Commutative]>;
def Daphne_EwMaxOp    : Daphne_EwBinaryOp<"ewMax", AnyScalar, [Commutative]>;

// ----------------------------------------------------------------------------
// Logical
// ----------------------------------------------------------------------------

def Daphne_EwAndOp    : Daphne_EwBinaryOp<"ewAnd", AnyTypeOf<[BoolScalar, IntScalar]>, [Commutative]>;
def Daphne_EwOrOp     : Daphne_EwBinaryOp<"ewOr" , AnyTypeOf<[BoolScalar, IntScalar]>, [Commutative]>;
def Daphne_EwXorOp    : Daphne_EwBinaryOp<"ewXor", AnyTypeOf<[BoolScalar, IntScalar]>, [Commutative]>;

// ----------------------------------------------------------------------------
// Strings
// ----------------------------------------------------------------------------

def Daphne_EwConcatOp : Daphne_EwBinaryOp<"ewConcat", StrScalar>;

// TODO This is just a quick and dirty solution that should be properly
// integrated with EwConcatOp above.
def Daphne_ConcatOp : Daphne_Op<"concat"> {
    let arguments = (ins StrScalar:$lhs, StrScalar:$rhs);
    let results = (outs StrScalar:$res);

    let hasFolder = 1;
}

// ----------------------------------------------------------------------------
// Comparisons
// ----------------------------------------------------------------------------

class Daphne_EwCmpOp<string name, Type inputScalarType, list<OpTrait> traits = []>
: Daphne_EwBinaryOp<name, inputScalarType, traits> {
    // TODO: We do not enforce (matrix of) boolean output any more, but should
    // think about that again.
    //let results = (outs AnyTypeOf<[MatrixOf<[BoolScalar]>, BoolScalar]>:$res);
}

def Daphne_EwEqOp  : Daphne_EwCmpOp<"ewEq" , AnyScalar, [Commutative]>;
def Daphne_EwNeqOp : Daphne_EwCmpOp<"ewNeq", AnyScalar, [Commutative]>;
def Daphne_EwLtOp  : Daphne_EwCmpOp<"ewLt" , AnyScalar>;
def Daphne_EwLeOp  : Daphne_EwCmpOp<"ewLe" , AnyScalar>;
def Daphne_EwGtOp  : Daphne_EwCmpOp<"ewGt" , AnyScalar>;
def Daphne_EwGeOp  : Daphne_EwCmpOp<"ewGe" , AnyScalar>;

// ****************************************************************************
// Aggregation and statistical
// ****************************************************************************

class Daphne_AggOp<string name, Type inScalarType, list<OpTrait> traits = []> : Daphne_Op<name, traits> {
    let arguments = (ins MatrixOf<[inScalarType]>:$arg);
}

// ----------------------------------------------------------------------------
// Full aggregation
// ----------------------------------------------------------------------------

class Daphne_AllAggOp<string name, Type scalarType, list<OpTrait> traits = []>
: Daphne_AggOp<name, scalarType, !listconcat(traits, [DeclareOpInterfaceMethods<InferTypesOpInterface>])> {
    let results = (outs scalarType:$res);
}

def Daphne_AllAggSumOp    : Daphne_AllAggOp<"sumAll", NumScalar>;
def Daphne_AllAggMinOp    : Daphne_AllAggOp<"minAll", NumScalar>;
def Daphne_AllAggMaxOp    : Daphne_AllAggOp<"maxAll", NumScalar>;
def Daphne_AllAggMeanOp   : Daphne_AllAggOp<"meanAll", FloatScalar>;
def Daphne_AllAggVarOp    : Daphne_AllAggOp<"varAll", FloatScalar>;
def Daphne_AllAggStddevOp : Daphne_AllAggOp<"stddevAll", FloatScalar>;

// ----------------------------------------------------------------------------
// Row/column-wise aggregation
// ----------------------------------------------------------------------------

class Daphne_DimAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<OpTrait> traits = []>
: Daphne_AggOp<name, inScalarType, traits> {
    let results = (outs MatrixOf<[outScalarType]>:$res);
}

class Daphne_RowAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<OpTrait> traits = []>
: Daphne_DimAggOp<name, inScalarType, outScalarType, !listconcat(traits, [
    NumRowsFromArg, OneCol
])>;
class Daphne_ColAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<OpTrait> traits = []>
: Daphne_DimAggOp<name, inScalarType, outScalarType, !listconcat(traits, [
    OneRow, NumColsFromArg
])>;

def Daphne_RowAggSumOp    : Daphne_RowAggOp<"sumRow"   , NumScalar, NumScalar, [DeclareOpInterfaceMethods<VectorizableOpInterface>]>;
def Daphne_RowAggMinOp    : Daphne_RowAggOp<"minRow"   , AnyScalar, AnyScalar, [DeclareOpInterfaceMethods<VectorizableOpInterface>]>;
def Daphne_RowAggMaxOp    : Daphne_RowAggOp<"maxRow"   , AnyScalar, AnyScalar, [DeclareOpInterfaceMethods<VectorizableOpInterface>, DeclareOpInterfaceMethods<DistributableOpInterface>]>;
def Daphne_RowAggIdxMinOp : Daphne_RowAggOp<"idxminRow", NumScalar>;
def Daphne_RowAggIdxMaxOp : Daphne_RowAggOp<"idxmaxRow", NumScalar>;
def Daphne_RowAggMeanOp   : Daphne_RowAggOp<"meanRow"  , FloatScalar>;
def Daphne_RowAggVarOp    : Daphne_RowAggOp<"varRow"   , FloatScalar>;
def Daphne_RowAggStddevOp : Daphne_RowAggOp<"stddevRow", FloatScalar>;

def Daphne_ColAggSumOp    : Daphne_ColAggOp<"sumCol"   , NumScalar, NumScalar, [CUDASupport,
        DeclareOpInterfaceMethods<VectorizableOpInterface>]>;
def Daphne_ColAggMinOp    : Daphne_ColAggOp<"minCol"   , AnyScalar>;
def Daphne_ColAggMaxOp    : Daphne_ColAggOp<"maxCol"   , AnyScalar>;
def Daphne_ColAggIdxMinOp : Daphne_ColAggOp<"idxminCol", NumScalar, Size>;
def Daphne_ColAggIdxMaxOp : Daphne_ColAggOp<"idxmaxCol", NumScalar, Size>;
def Daphne_ColAggMeanOp   : Daphne_ColAggOp<"meanCol"  , FloatScalar>;
def Daphne_ColAggVarOp    : Daphne_ColAggOp<"varCol"   , FloatScalar>;
def Daphne_ColAggStddevOp : Daphne_ColAggOp<"stddevCol", FloatScalar>;

// ----------------------------------------------------------------------------
// Cumulative aggregation
// ----------------------------------------------------------------------------

class Daphne_CumAggOp<string name, Type scalarType, list<OpTrait> traits = []>
: Daphne_AggOp<name, scalarType, !listconcat(traits, [
    ShapeFromArg
])> {
    let results = (outs MatrixOf<[scalarType]>);
}

def Daphne_CumAggSumOp  : Daphne_CumAggOp<"sumCum" , NumScalar>;
def Daphne_CumAggProdOp : Daphne_CumAggOp<"prodCum", NumScalar>;
def Daphne_CumAggMinOp  : Daphne_CumAggOp<"minCum" , AnyScalar>;
def Daphne_CumAggMaxOp  : Daphne_CumAggOp<"maxCum" , AnyScalar>;

// ----------------------------------------------------------------------------
// Grouped aggregation
// ----------------------------------------------------------------------------

class Daphne_GrpAggOp<string name, Type scalarType, list<OpTrait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    NumRowsFromIthScalar<2>, NumColsFromArg
])> {
    let arguments = (ins MatrixOf<[scalarType]>:$arg, MatrixOf<[Size]>:$groupIds, Size:$numGroups/*, Optional<Matrix>:$weights*/);
    let results = (outs MatrixOf<[scalarType]>:$groupAggs);
}

def Daphne_GrpAggCountOp  : Daphne_GrpAggOp<"countGrp" , AnyScalar>;
def Daphne_GrpAggSumOp    : Daphne_GrpAggOp<"sumGrp"   , NumScalar>;
def Daphne_GrpAggMinOp    : Daphne_GrpAggOp<"minGrp"   , AnyScalar>;
def Daphne_GrpAggMaxOp    : Daphne_GrpAggOp<"maxGrp"   , AnyScalar>;
def Daphne_GrpAggMeanOp   : Daphne_GrpAggOp<"meanGrp"  , FloatScalar>;
def Daphne_GrpAggVarOp    : Daphne_GrpAggOp<"varGrp"   , FloatScalar>;
def Daphne_GrpAggStddevOp : Daphne_GrpAggOp<"stddevGrp", FloatScalar>;

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
    NumRowsFromIthArg<1>, NumColsFromIthArgNumRows<2>
]> {
    let summary = "Copies the specified rows and columns from the argument to the result.";

    let description = [{
        This operation is effectively similar to a combination of `extractRow`
        and `extractCol`. Please see the documentation of those operations.
    }];

    let arguments = (ins MatrixOrFrame:$source, Selection:$selectedRows, AnyTypeOf<[Selection, StrScalar]>:$selectedCols);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_ExtractRowOp : Daphne_Op<"extractRow", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
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

def Daphne_SliceRowOp : Daphne_Op<"sliceRow"> {
    let summary = "Copies the specified rows from the argument to the result.";

    let description = [{
        Copies the rows of a data object `source` (matrix or frame) at the
        positions in `[lowerIncl, upperExcl)` to the result.
    }];

    let arguments = (ins MatrixOrFrame:$source, Size:$lowerIncl, Size:$upperExcl);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_ExtractColOp : Daphne_Op<"extractCol", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    NumRowsFromArg, NumColsFromIthArgNumRows<1>, CUDASupport
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

    let arguments = (ins MatrixOrFrame:$source, AnyTypeOf<[Selection, StrScalar]>:$selectedCols);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_SliceColOp : Daphne_Op<"sliceCol"> {
    let summary = "Copies the specified columns from the argument to the result.";

    let description = [{
        Copies the columns of a data object `source` (matrix or frame) at the
        positions in `[lowerIncl, upperExcl)` to the result.
    }];

    let arguments = (ins MatrixOrFrame:$source, Size:$lowerIncl, Size:$upperExcl);
    let results = (outs MatrixOrFrame:$res);
}

// TODO Create combined InsertOp (see #238).

def Daphne_InsertRowOp : Daphne_Op<"insertRow"> {
    let arguments = (ins MatrixOrFrame:$arg, MatrixOrFrame:$ins, Size:$rowLowerIncl, Size:$rowUpperExcl);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_InsertColOp : Daphne_Op<"insertCol"> {
    let arguments = (ins MatrixOrFrame:$arg, MatrixOrFrame:$ins, Size:$colLowerIncl, Size:$colUpperExcl);
    let results = (outs MatrixOrFrame:$res);
}

// ****************************************************************************
// Reorganization
// ****************************************************************************

def Daphne_ReshapeOp : Daphne_Op<"reshape", [
    NumRowsFromIthScalar<1>, NumColsFromIthScalar<2>, SparsityFromArg
]> {
    let arguments = (ins Matrix:$arg, Size:$numRows, Size:$numCols);
    let results = (outs Matrix:$res);
}

def Daphne_TransposeOp : Daphne_Op<"transpose", [
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    NumRowsFromArgNumCols, NumColsFromArgNumRows, SparsityFromArg, CUDASupport
]> {
    let arguments = (ins Matrix:$arg);
    let results = (outs Matrix:$res);

    let builders = [
        OpBuilder<(ins "Value":$input), [{
            auto ty = input.getType().cast<MatrixType>();
            build($_builder, $_state, ty, input);
        }]>,
    ];
}

class Daphne_BindOp<string name, list<OpTrait> traits = []> : Daphne_Op<name, traits> {
    let arguments = (ins MatrixOrFrame:$lhs, MatrixOrFrame:$rhs);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_ColBindOp : Daphne_BindOp<"colBind", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    NumRowsFromAllArgs, NumColsFromSumOfAllArgs, CUDASupport
]>;

def Daphne_RowBindOp : Daphne_BindOp<"rowBind", [
    NumRowsFromSumOfAllArgs, NumColsFromAllArgs
]>;

def Daphne_ReverseOp : Daphne_Op<"reverse", [
    ShapeFromArg
]> {
    let arguments = (ins Matrix:$arg);
    let results = (outs Matrix:$res);
}

def Daphne_OrderOp : Daphne_Op<"order", [
    SameVariadicOperandSize,
    ShapeFromArg
]> {
    // TODO Maybe colIdxs and ascs should be attributes.
    let arguments = (ins MatrixOrFrame:$arg, Variadic<Size>:$colIdxs, Variadic<BoolScalar>:$ascs, BoolScalar:$returnIdxs);
    let results = (outs MatrixOrFrame:$res);
}

// ****************************************************************************
// Matrix decompositions & co
// ****************************************************************************

def Daphne_EigenOp : Daphne_Op<"eigen"> {
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

class Daphne_ActivationForwardOp<string name, list<OpTrait> traits = []> :
        Daphne_Op<name, traits> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input);

    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

def Daphne_ReluForwardOp : Daphne_ActivationForwardOp<"reluForward">;

// ----------------------------------------------------------------------------
// Affine
// ----------------------------------------------------------------------------

def Daphne_AffineForwardOp : Daphne_Op<"affineForward"> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$weights,
            MatrixOf<[FloatScalar]>:$bias);

    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

// ----------------------------------------------------------------------------
// Batch Normalization
// ----------------------------------------------------------------------------

def Daphne_BatchNorm2DTestForwardOp : Daphne_Op<"batchNorm2DTestForward"> {
    let arguments = (ins
            MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$gamma, MatrixOf<[FloatScalar]>:$beta,
            MatrixOf<[FloatScalar]>:$emaMean, MatrixOf<[FloatScalar]>:$emaVar, FloatScalar:$eps);
    let results = (outs MatrixOf<[FloatScalar]>:$output);
}

// ----------------------------------------------------------------------------
// Bias Addition
// ----------------------------------------------------------------------------

def Daphne_BiasAddForwardOp : Daphne_Op<"biasAddForward"> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$bias);

    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

// ----------------------------------------------------------------------------
// Convolution
// ----------------------------------------------------------------------------

def Daphne_Conv2DForwardOp : Daphne_Op<"conv2DForward"> {
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

class Daphne_PoolForwardOp<string name, list<OpTrait> traits = []> : Daphne_Op<name, traits> {
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

def Daphne_MaxPoolForwardOp : Daphne_PoolForwardOp<"maxPoolForward">;
def Daphne_AvgPoolForwardOp : Daphne_PoolForwardOp<"avgPoolForward">;

// ----------------------------------------------------------------------------
// Softmax
// ----------------------------------------------------------------------------
def Daphne_SoftmaxForwardOp : Daphne_Op<"softmaxForward"> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input);

    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

// ****************************************************************************
// Other matrix operations
// ****************************************************************************

def Daphne_DiagVectorOp : Daphne_Op<"diagVector", [
    NumRowsFromArg, OneCol
]> {
    let arguments = (ins Matrix:$arg);
    let results = (outs Matrix:$res);
}

//class Daphne_TriOp<string name, list<OpTrait> traits = []> : Daphne_Op<name, traits> {
//    let arguments = (ins Matrix:$arg, BoolScalar:$diag, BoolScalar:$values);
//    let results = (outs Matrix:$res);
//}
//
//def Daphne_LowerTriOp : Daphne_TriOp<"lowerTri">;
//def Daphne_UpperTriOp : Daphne_TriOp<"upperTri">;
def Daphne_TriOp : Daphne_Op<"tri", [
    ShapeFromArg, DeclareOpInterfaceMethods<InferSparsityOpInterface>
]> {
    let arguments = (ins Matrix:$arg, BoolScalar:$upper, BoolScalar:$diag, BoolScalar:$values);
    let results = (outs Matrix:$res);
}

def Daphne_SolveOp : Daphne_Op<"solve", [
    NumRowsFromArg, OneCol, CUDASupport
]> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$a, MatrixOf<[FloatScalar]>:$b);
    let results = (outs MatrixOf<[FloatScalar]>:$x);
}

def Daphne_ReplaceOp : Daphne_Op<"replace", [
    ShapeFromArg
]> {
    let arguments = (ins Matrix:$arg, AnyScalar:$pattern, AnyScalar:$replacement);
    let results = (outs Matrix:$res);
}

def Daphne_CTableOp : Daphne_Op<"ctable"> {
    //// TODO The last three arguments should be optional.
    //let arguments = (ins Matrix:$lhs, AnyTypeOf<[Matrix, AnyScalar]>:$rhs, AnyTypeOf<[Matrix, AnyScalar]>:$weights, Size:$outHeight, Size:$outWidth);
    let arguments = (ins Matrix:$lhs, Matrix:$rhs);
    let results = (outs Matrix:$res);
}

def Daphne_SyrkOp : Daphne_Op<"syrk", [
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    NumRowsFromArgNumCols, NumColsFromArg, CUDASupport
]> {
    // TODO: support `A @ t(A)` operation
    let summary = [{Performs the operation `t(A) @ A`}];
    let arguments = (ins Matrix:$arg);
    let results = (outs Matrix:$res);
}

def Daphne_GemvOp : Daphne_Op<"gemv", [
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    NumRowsFromArgNumCols, OneCol, CUDASupport,
]> {
    // TODO: support `A @ x` operation
    let summary = [{Performs the operation `t(A) @ x`}];
    let arguments = (ins Matrix:$mat, Matrix:$vec);
    let results = (outs Matrix:$res);
}

// ****************************************************************************
// Extended relational algebra
// ****************************************************************************

// ----------------------------------------------------------------------------
// Entire SQL query
// ----------------------------------------------------------------------------

def Daphne_SqlOp : Daphne_Op<"sql"> {
    let arguments = (ins StrAttr:$sql);
    let results = (outs Frame:$res);
}

def Daphne_RegisterViewOp : Daphne_Op<"registerView"> {
    let arguments = (ins StrAttr:$view, MatrixOrFrame:$arg);
    let results = (outs); // no results
}

// ----------------------------------------------------------------------------
// Set operations
// ----------------------------------------------------------------------------

class Daphne_SetOp<string name, list<OpTrait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    NumColsFromAllArgs
])> {
    let arguments = (ins Frame:$lhs, Frame:$rhs);
    let results = (outs Frame:$res);
}

def Daphne_IntersectOp : Daphne_SetOp<"intersect">;
def Daphne_MergeOp : Daphne_SetOp<"merge">;
def Daphne_ExceptOp : Daphne_SetOp<"except">;

// ----------------------------------------------------------------------------
// Cartesian product and joins
// ----------------------------------------------------------------------------

// TODO Implement #row inference
def Daphne_CartesianOp : Daphne_Op<"cartesian", [
    NumColsFromSumOfAllArgs,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferNumRowsOpInterface>
]> {
    let arguments = (ins Frame:$lhs, Frame:$rhs); //let arguments = (ins Variadic<Frame>:$args);
    let results = (outs Frame:$res);
}

class Daphne_JoinOp<string name, list<OpTrait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    SameVariadicOperandSize,
    NumColsFromSumOfAllArgs
])> {
    // TODO Maybe leftOn and rightOn should be attributes.
    let arguments = (ins Frame:$lhs, Frame:$rhs, Variadic<Size>:$leftOn, Variadic<Size>:$rightOn);
    let results = (outs Frame:$res);
}

def Daphne_InnerJoinOp : Daphne_Op<"innerJoin", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>
]> {
    let arguments = (ins Frame:$lhs, Frame:$rhs, StrScalar:$lhsOn, StrScalar:$rhsOn);
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
    let arguments = (ins Frame:$lhs, Frame:$rhs, StrScalar:$lhsOn, StrScalar:$rhsOn);
    let results = (outs Frame:$res, MatrixOf<[Size]>:$lhsTids);
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

    let arguments = (ins Frame:$lhs, Frame:$rhs, StrScalar:$lhsOn, StrScalar:$rhsOn, StrScalar:$rhsAgg);
    // TODO Result `lhsTids` could be made optional.
    let results = (outs Frame:$res, MatrixOf<[Size]>:$lhsTids);
}

// ----------------------------------------------------------------------------
// Selection
// ----------------------------------------------------------------------------

def Daphne_QueryOp : Daphne_Op<"query", [
    NumColsFromArg
]> {
    let arguments = (ins Frame:$arg, StrAttr:$cond);
    let results = (outs Frame:$res);
}

def Daphne_FilterRowOp : Daphne_Op<"filterRow", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
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

        Note that `ExtractRowOp` is similar, but works with positions lists
        instead of bit vectors.
    }];

    let arguments = (ins MatrixOrFrame:$source, Matrix:$selectedRows);
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
        ins Frame:$frame,
        Variadic<StrScalar>:$keyCol,
        Variadic<StrScalar>:$aggCol,
        TypedArrayAttrBase<Daphne_GroupAggEnum, "enum">:$aggFuncs
    );
    let results = (outs Frame:$res);
}

// ****************************************************************************
// Frame label manipulation
// ****************************************************************************

def Daphne_SetColLabelsOp : Daphne_Op<"setColLabels", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    ShapeFromArg
]> {
    let arguments = (ins Frame:$arg, Variadic<StrScalar>:$labels);
    let results = (outs Frame:$res);
}

def Daphne_SetColLabelsPrefixOp : Daphne_Op<"setColLabelsPrefix", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    ShapeFromArg
]> {
    let arguments = (ins Frame:$arg, StrScalar:$prefix);
    let results = (outs Frame:$res);
}

// ****************************************************************************
// Conversions, casts, and copying
// ****************************************************************************

def Daphne_ToStringOp : Daphne_Op<"toString"> {
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
    let arguments = (ins AnyTypeOf<[MatrixOrFrame, AnyScalar]>:$arg);
    let results = (outs AnyTypeOf<[MatrixOrFrame, AnyScalar]>:$res);
    let hasFolder = 1;

    let extraClassDeclaration = [{
        bool isTrivialCast() {
            return arg().getType() == res().getType();
        }
        /**
         * @brief This cast just removes detailed properties from a matrix type, making them unknown instead.
         *
         * Usually casts like this need to exist to make the IR correct for SCF operations.
         * @return true if this cast just tags some known information unknown, false otherwise
         */
        bool isMatrixPropertyCast() {
            auto inTy = arg().getType().dyn_cast<daphne::MatrixType>();
            auto outTy = res().getType().dyn_cast<daphne::MatrixType>();
            if (inTy && outTy && inTy.isSpecializationOf(outTy)) {
                return true;
            }
            return false;
        }
    }];
}

def Daphne_CopyOp : Daphne_Op<"copy"> {
    let arguments = (ins MatrixOrFrame:$arg);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_QuantizeOp : Daphne_Op<"quantize", [
    ShapeFromArg
]> {
    let arguments = (ins Matrix:$arg, NumScalar:$min, NumScalar:$max);
    let results = (outs Matrix:$res);
}

// ****************************************************************************
// Distributed Operations
// ****************************************************************************

def Daphne_DistributedReadOp : Daphne_Op<"distributedRead", [NoSideEffect]> {
    let arguments = (ins StrScalar:$fileName);
    let results = (outs Handle:$res);
}

def Daphne_DistributeOp : Daphne_Op<"distribute", [NoSideEffect]> {
    let arguments = (ins Matrix:$mat);
    let results = (outs Handle:$res);

    let hasCanonicalizer = 1;
}

def Daphne_BroadcastOp : Daphne_Op<"broadcast", [NoSideEffect]> {
    let arguments = (ins Matrix:$mat);
    let results = (outs Handle:$res);
}

def Daphne_DistributedComputeOp : Daphne_Op<"distributedCompute", [NoSideEffect, IsolatedFromAbove]> {
    let arguments = (ins Variadic<Handle>:$args);
    let results = (outs Variadic<Handle>:$res);
    let regions = (region SizedRegion<1>:$body);
}

// TODO: Map Keys Operation

def Daphne_DistributedCollectOp : Daphne_Op<"distributedCollect", [NoSideEffect]> {
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
    let arguments = (ins AnyTypeOf<[AnyScalar, MatrixOrFrame]>:$arg, BoolScalar:$newline, BoolScalar:$err);
    let results = (outs); // no results
}

// TODO Take asynchronous read into account.
def Daphne_ReadOp : Daphne_Op<"read", [
    DeclareOpInterfaceMethods<InferShapeOpInterface>,
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
    let arguments = (ins AnyTypeOf<[File, Target]>);
    let results = (outs); // no results
}

def Daphne_ReadCsvOp : Daphne_Op<"readCsv"> {
    let arguments = (ins
        AnyTypeOf<[File, Descriptor]>:$input,
        Size:$numRows, Size:$numCols,
        StrScalar:$delim
    );
    let results = (outs MatrixOrFrame:$res);
}

// ****************************************************************************
// Data preprocessing
// ****************************************************************************

def Daphne_OneHotOp : Daphne_Op<"oneHot", [
    // TODO We can know the exact number of columns at compile-time if $info is
    // known at compile-time (e.g. if it is a matrix literal), but we don't
    // support that yet.
    NumRowsFromArg, // DeclareOpInterfaceMethods<InferNumColsOpInterface>
]> {
    let arguments = (ins MatrixOrFrame:$arg, MatrixOf<[SI64]>:$info);
    let results = (outs MatrixOrFrame:$res);
}

// ****************************************************************************
// Measurements
// ****************************************************************************

def Daphne_NowOp : Daphne_Op<"now"> {
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

    let arguments = (ins UI64:$userConfigPtr);
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

// ****************************************************************************
// Vectorized operations
// ****************************************************************************

// TODO: should be recursive side effects
// TODO: use symbol handling (should be isolated from above, but needs reference to daphne context)
def Daphne_VectorizedPipelineOp : Daphne_Op<"vectorizedPipeline", [AttrSizedOperandSegments]> {
    let summary = "Performs multi-threaded vector-wise computations of the input.";

    let arguments = (ins Variadic<AnyTypeOf<[AnyScalar, MatrixOrFrame]>>:$inputs,
            // TODO: both types should be SIZE, but we require `-1` for dynamic/unknown
            Variadic<AnyTypeOf<[SIntScalar, Size]>>:$out_rows,
            Variadic<AnyTypeOf<[SIntScalar, Size]>>:$out_cols,
            TypedArrayAttrBase<VectorSplitAttr, "Vector-Splits">:$splits,
            TypedArrayAttrBase<VectorCombineAttr, "Vector-Combines">:$combines,
            Optional<DaphneContext>:$ctx);
    let results = (outs Variadic<MatrixOrFrame>:$outputs);
    let regions = (region SizedRegion<1>:$body, AnyRegion:$cuda);

    let hasCanonicalizeMethod = 1;
}

// ****************************************************************************
// Low-level auxiliary operations
// ****************************************************************************

def Daphne_CreateVariadicPackOp : Daphne_Op<"createVariadicPack"> {
    let arguments = (ins IndexAttr:$numElements);
    let results = (outs VariadicPack:$res);
}

def Daphne_StoreVariadicPackOp : Daphne_Op<"storeVariadicPack"> {
    let arguments = (ins VariadicPack:$pack, AnyType:$item, IndexAttr:$pos);
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
// Old operations
// ****************************************************************************
// These are currently still need for the prototype to work. They will be
// removed/replaced/changed in the future.

def Daphne_ConstantOp : Daphne_Op<"constant", [ConstantLike, NoSideEffect]> {
    let summary = "constant operation";

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyTypeOf<[AnyScalar, Matrix]>:$result);

    let builders = [
        OpBuilder<(ins "Attribute":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>,
        OpBuilder<(ins "double":$value), [{
            build($_builder, $_state, $_builder.getF64FloatAttr(value));
        }]>,
        //OpBuilder<(ins "float":$value), [{
        //    build($_builder, $_state, $_builder.getF32FloatAttr(value));
        //}]>,
        OpBuilder<(ins "int64_t":$value), [{
            build($_builder, $_state, $_builder.getIntegerAttr($_builder.getIntegerType(64, true), value));
        }]>,
        OpBuilder<(ins "uint64_t":$value), [{
            build($_builder, $_state, $_builder.getIntegerAttr($_builder.getIntegerType(64, false), value));
        }]>,
        OpBuilder<(ins "bool":$value), [{
            build($_builder, $_state, $_builder.getIntegerAttr($_builder.getI1Type(), value));
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

def Daphne_ReturnOp : Daphne_Op<"return", [NoSideEffect, Terminator, ReturnLike, ParentOneOf<["FuncOp", "DistributedComputeOp", "VectorizedPipelineOp"]>]> {
    let summary = "return operation";

    let arguments = (ins Variadic<AnyType>:$operands);

    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];

    let builders = [
        OpBuilder<(ins), [{
            build($_builder, $_state, llvm::None);
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

    let extraClassDeclaration = [{
        StringAttr getCalleeAttr() { return (*this)->getAttrOfType<StringAttr>("callee"); }
    }];
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

class Daphne_ElementwiseBinaryOp<string mnemonic, list<OpTrait> traits = []> :
        Daphne_Op<mnemonic, !listconcat(traits, [NoSideEffect, TypesMatchOrOneIsMatrixOfOther<"lhs", "rhs">])> {
    let arguments = (ins AnyTypeOf<[AnyScalar, Matrix]>:$lhs, AnyTypeOf<[AnyScalar, Matrix]>:$rhs);
    let results = (outs AnyTypeOf<[AnyScalar, Matrix]>:$result);

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

    let arguments = (ins Matrix:$mat, Index:$row, Index:$col, AnyScalar:$val);
}

#endif //SRC_IR_DAPHNEIR_DAPHNEOPS_TD
