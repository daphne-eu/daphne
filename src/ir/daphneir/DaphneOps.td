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

def Daphne_FillOp : Daphne_Op<"fill"> {
    let arguments = (ins AnyScalar:$arg, Size:$numRows, Size:$numCols);
    let results = (outs Matrix:$res);
}

def Daphne_CreateFrameOp : Daphne_Op<"createFrame", [SameVariadicOperandSize]> {
    let arguments = (ins Variadic<Matrix>:$cols, Variadic<StrScalar>:$labels);
    let results = (outs Frame:$res);
}

def Daphne_DiagMatrixOp : Daphne_Op<"diagMatrix"> {
    let arguments = (ins Matrix:$arg);
    let results = (outs Matrix:$res);
}

def Daphne_RandMatrixOp : Daphne_Op<"randMatrix", [TypesMatchWith<"min and max arguments must be of the same type", "min", "max", "$_self">]> {
    //let arguments = (ins Size:$numRows, Size:$numCols, AnyScalar:$min, AnyScalar:$max, F64:$sparsity, Seed:$seed, StrScalar:$pdf);
    let arguments = (ins Size:$numRows, Size:$numCols, AnyScalar:$min, AnyScalar:$max, F64:$sparsity, IntScalar:$seed);
    let results = (outs Matrix:$res);
}

def Daphne_SampleOp : Daphne_Op<"sample"> {
    let arguments = (ins Size:$range, Size:$size, BoolScalar:$withReplacement, Seed:$seed);
    let results = (outs Matrix:$res);
}

def Daphne_SeqOp : Daphne_Op<"seq"> {
    let arguments = (ins NumScalar:$from, NumScalar:$to, NumScalar:$inc);
    let results = (outs MatrixOf<[NumScalar]>);
}

// ****************************************************************************
// Matrix/frame dimensions
// ****************************************************************************

class Daphne_NumOp<string name, list<OpTrait> traits = []> : Daphne_Op<name, traits> {
    let arguments = (ins MatrixOrFrame:$arg);
    let results = (outs Size:$res);
}

def Daphne_NumRowsOp : Daphne_NumOp<"numRows">;
def Daphne_NumColsOp : Daphne_NumOp<"numCols">;
def Daphne_NumCellsOp : Daphne_NumOp<"numCells">;

// ****************************************************************************
// Matrix multiplication
// ****************************************************************************

def Daphne_MatMulOp : Daphne_Op<"matMul"> {
    let arguments = (ins MatrixOf<[NumScalar]>:$lhs, MatrixOf<[NumScalar]>:$rhs);
    let results = (outs MatrixOf<[NumScalar]>:$res);
}

// ****************************************************************************
// Elementwise unary
// ****************************************************************************

class Daphne_EwUnaryOp<string name, Type scalarType, list<OpTrait> traits = []> : Daphne_Op<name, traits> {
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
def Daphne_EwSqrtOp : Daphne_EwUnaryOp<"ewSqrt", FloatScalar>;

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

class Daphne_EwBinaryOp<string name, Type scalarType, list<OpTrait> traits = []>
: Daphne_Op<name, !listconcat([TypesMatchOrOneIsMatrixOfOther<"lhs", "rhs">], traits)> {
    let arguments = (ins AnyTypeOf<[MatrixOf<[scalarType]>, scalarType]>:$lhs, AnyTypeOf<[MatrixOf<[scalarType]>, scalarType]>:$rhs);
    let results = (outs AnyTypeOf<[MatrixOf<[scalarType]>, scalarType]>:$res);

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            // TODO This is wrong if lhs is a scalar and rhs is a matrix.
            $_state.addTypes(lhs.getType());
            $_state.addOperands({lhs, rhs});
        }]>
    ];
}

// ----------------------------------------------------------------------------
// Arithmetic
// ----------------------------------------------------------------------------

def Daphne_EwAddOp    : Daphne_EwBinaryOp<"ewAdd", NumScalar, [DeclareOpInterfaceMethods<DistributableOpInterface>]>;
def Daphne_EwSubOp    : Daphne_EwBinaryOp<"ewSub", NumScalar>;
def Daphne_EwMulOp    : Daphne_EwBinaryOp<"ewMul", NumScalar>;
def Daphne_EwDivOp    : Daphne_EwBinaryOp<"ewDiv", NumScalar>;
def Daphne_EwPowOp    : Daphne_EwBinaryOp<"ewPow", NumScalar>;
def Daphne_EwModOp    : Daphne_EwBinaryOp<"ewMod", IntScalar>;
def Daphne_EwLogOp    : Daphne_EwBinaryOp<"ewLog", FloatScalar>;

// ----------------------------------------------------------------------------
// Min/max
// ----------------------------------------------------------------------------

def Daphne_EwMinOp    : Daphne_EwBinaryOp<"ewMin", AnyScalar>;
def Daphne_EwMaxOp    : Daphne_EwBinaryOp<"ewMax", AnyScalar>;

// ----------------------------------------------------------------------------
// Logical
// ----------------------------------------------------------------------------

def Daphne_EwAndOp    : Daphne_EwBinaryOp<"ewAnd", AnyTypeOf<[BoolScalar, IntScalar]>>;
def Daphne_EwOrOp     : Daphne_EwBinaryOp<"ewOr" , AnyTypeOf<[BoolScalar, IntScalar]>>;
def Daphne_EwXorOp    : Daphne_EwBinaryOp<"ewXor", AnyTypeOf<[BoolScalar, IntScalar]>>;

// ----------------------------------------------------------------------------
// Strings
// ----------------------------------------------------------------------------

def Daphne_EwConcatOp : Daphne_EwBinaryOp<"ewConcat", StrScalar>;

// ----------------------------------------------------------------------------
// Comparisons
// ----------------------------------------------------------------------------

class Daphne_EwCmpOp<string name, Type inputScalarType, list<OpTrait> traits = []> : Daphne_EwBinaryOp<name, inputScalarType, traits> {
    // TODO: We do not enforce (matrix of) boolean output any more, but should
    // think about that again.
    //let results = (outs AnyTypeOf<[MatrixOf<[BoolScalar]>, BoolScalar]>:$res);

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            if(lhs.getType().isa<::mlir::daphne::MatrixType>())
                $_state.addTypes(lhs.getType());
            else if(rhs.getType().isa<::mlir::daphne::MatrixType>())
                $_state.addTypes(rhs.getType());
            else
                $_state.addTypes($_builder.getI1Type()); // bool
            $_state.addOperands({lhs, rhs});
        }]>
    ];
}

def Daphne_EwEqOp  : Daphne_EwCmpOp<"ewEq" , AnyScalar>;
def Daphne_EwNeqOp : Daphne_EwCmpOp<"ewNeq", AnyScalar>;
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

class Daphne_AllAggOp<string name, Type scalarType, list<OpTrait> traits = []> : Daphne_AggOp<name, scalarType, traits> {
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

class Daphne_DimAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<OpTrait> traits = []> : Daphne_AggOp<name, inScalarType, traits> {
    let results = (outs MatrixOf<[outScalarType]>:$res);
}

class Daphne_RowAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<OpTrait> traits = []> : Daphne_DimAggOp<name, inScalarType, outScalarType, traits>;
class Daphne_ColAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<OpTrait> traits = []> : Daphne_DimAggOp<name, inScalarType, outScalarType, traits>;

def Daphne_RowAggSumOp    : Daphne_RowAggOp<"sumRow"   , NumScalar>;
def Daphne_RowAggMinOp    : Daphne_RowAggOp<"minRow"   , AnyScalar>;
def Daphne_RowAggMaxOp    : Daphne_RowAggOp<"maxRow"   , AnyScalar>;
def Daphne_RowAggIdxMinOp : Daphne_RowAggOp<"idxminRow", NumScalar, Size>;
def Daphne_RowAggIdxMaxOp : Daphne_RowAggOp<"idxmaxRow", NumScalar, Size>;
def Daphne_RowAggMeanOp   : Daphne_RowAggOp<"meanRow"  , FloatScalar>;
def Daphne_RowAggVarOp    : Daphne_RowAggOp<"varRow"   , FloatScalar>;
def Daphne_RowAggStddevOp : Daphne_RowAggOp<"stddevRow", FloatScalar>;

def Daphne_ColAggSumOp    : Daphne_ColAggOp<"sumCol"   , NumScalar>;
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

class Daphne_CumAggOp<string name, Type scalarType, list<OpTrait> traits = []> : Daphne_AggOp<name, scalarType, traits> {
    let results = (outs MatrixOf<[scalarType]>);
}

def Daphne_CumAggSumOp  : Daphne_CumAggOp<"sumCum" , NumScalar>;
def Daphne_CumAggProdOp : Daphne_CumAggOp<"prodCum", NumScalar>;
def Daphne_CumAggMinOp  : Daphne_CumAggOp<"minCum" , AnyScalar>;
def Daphne_CumAggMaxOp  : Daphne_CumAggOp<"maxCum" , AnyScalar>;

// ----------------------------------------------------------------------------
// Grouped aggregation
// ----------------------------------------------------------------------------

class Daphne_GrpAggOp<string name, Type scalarType, list<OpTrait> traits = []> : Daphne_Op<name, traits> {
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

def Daphne_ExtractOp : Daphne_Op<"extract"> {
    let summary = "Copies the specified rows and columns from the argument to the result.";

    let description = [{
        This operation is effectively similar to a combination of `extractRow`
        and `extractCol`. Please see the documentation of those operations.
    }];

    let arguments = (ins MatrixOrFrame:$source, Selection:$selectedRows, AnyTypeOf<[Selection, StrScalar]>:$selectedCols);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_ExtractRowOp : Daphne_Op<"extractRow"> {
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

def Daphne_ExtractColOp : Daphne_Op<"extractCol"> {
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

def Daphne_InsertOp : Daphne_Op<"insert"> {
    let arguments = (ins MatrixOrFrame:$target, MatrixOrFrame:$source, Selection:$selectedRows, Selection:$selectedCols);
    let results = (outs); // no results
}

// ****************************************************************************
// Reorganization
// ****************************************************************************

def Daphne_ReshapeOp : Daphne_Op<"reshape"> {
    let arguments = (ins Matrix:$arg, Size:$numRows, Size:$numCols);
    let results = (outs Matrix:$res);
}

def Daphne_TransposeOp : Daphne_Op<"transpose"> {
    let arguments = (ins Matrix:$arg);
    let results = (outs Matrix:$res);

    let builders = [
        OpBuilder<(ins "Value":$input), [{
            auto ty = input.getType().cast<MatrixType>();
            build($_builder, $_state, ty, input);
        }]>,
    ];
}

class Daphne_BindOp<string name> : Daphne_Op<name> {
    let arguments = (ins MatrixOrFrame:$lhs, MatrixOrFrame:$rhs);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_ColBindOp : Daphne_BindOp<"colBind">;
def Daphne_RowBindOp : Daphne_BindOp<"rowBind">;

def Daphne_ReverseOp : Daphne_Op<"reverse"> {
    let arguments = (ins Matrix:$arg);
    let results = (outs Matrix:$res);
}

def Daphne_OrderOp : Daphne_Op<"order", [SameVariadicOperandSize]> {
    // TODO Maybe colIdxs and ascs should be attributes.
    let arguments = (ins MatrixOrFrame:$arg, Variadic<Size>:$colIdxs, Variadic<BoolScalar>:$ascs, BoolAttr:$returnIdxs);
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
// Convolution
// ----------------------------------------------------------------------------

def Daphne_Conv2DForwardOp : Daphne_Op<"conv2DForward"> {
    let arguments = (ins
        MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$filter,
        Size:$strideHeight, Size:$strideWidth, Size:$padHeight, Size:$padWidth,
        // input shape
        Size:$inputBatchSize, Size:$inputNumChannels, Size:$inputHeight, Size:$inputWidth,
        // filter shape
        Size:$filterNumFilters, Size:$filterNumChannels, Size:$filterHeight, Size:$filterWidth
    );
    let results = (outs MatrixOf<[FloatScalar]>:$output);
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
    let arguments = (ins
        MatrixOf<[FloatScalar]>:$input,
        Size:$strideHeight, Size:$strideWidth, Size:$padHeight, Size:$padWidth,
        // input shape
        Size:$inputBatchSize, Size:$inputNumChannels, Size:$inputHeight, Size:$inputWidth,
        // pool shape
        Size:$poolHeight, Size:$poolWidth
    );
}

class Daphne_PoolBackwardOp<string name, list<OpTrait> traits = []> : Daphne_Op<name, traits> {
    let arguments = (ins
        MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$output,
        Size:$strideHeight, Size:$strideWidth, Size:$padHeight, Size:$padWidth,
        // input shape
        Size:$inputBatchSize, Size:$inputNumChannels, Size:$inputHeight, Size:$inputWidth,
        // pool shape
        Size:$poolHeight, Size:$poolWidth
    );
}

def Daphne_MaxPoolForwardOp : Daphne_PoolForwardOp<"maxPoolForward">;
def Daphne_AvgPoolForwardOp : Daphne_PoolForwardOp<"avgPoolForward">;

def Daphne_MaxPoolBackwardOp : Daphne_PoolBackwardOp<"maxPoolBackward">;
def Daphne_AvgPoolBackwardOp : Daphne_PoolBackwardOp<"avgPoolBackward">;

// ****************************************************************************
// Other matrix operations
// ****************************************************************************

def Daphne_DiagVectorOp : Daphne_Op<"diagVector"> {
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
def Daphne_TriOp : Daphne_Op<"tri", []> {
    let arguments = (ins Matrix:$arg, BoolScalar:$upper, BoolScalar:$diag, BoolScalar:$values);
    let results = (outs Matrix:$res);
}

def Daphne_SolveOp : Daphne_Op<"solve"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$a, MatrixOf<[FloatScalar]>:$b);
    let results = (outs MatrixOf<[FloatScalar]>:$x);
}

def Daphne_ReplaceOp : Daphne_Op<"replace"> {
    let arguments = (ins Matrix:$arg, AnyScalar:$pattern, AnyScalar:$replacement);
    let results = (outs Matrix:$res);
}

def Daphne_CTableOp : Daphne_Op<"ctable"> {
    // TODO The last three arguments should be optional.
    let arguments = (ins Matrix:$lhs, AnyTypeOf<[Matrix, AnyScalar]>:$rhs, AnyTypeOf<[Matrix, AnyScalar]>:$weights, Size:$outHeight, Size:$outWidth);
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

// ----------------------------------------------------------------------------
// Set operations
// ----------------------------------------------------------------------------

class Daphne_SetOp<string name, list<OpTrait> traits = []> : Daphne_Op<name, traits> {
    let arguments = (ins Frame:$lhs, Frame:$rhs);
    let results = (outs Frame:$res);
}

def Daphne_IntersectOp : Daphne_SetOp<"intersect">;
def Daphne_MergeOp : Daphne_SetOp<"merge">;
def Daphne_ExceptOp : Daphne_SetOp<"except">;

// ----------------------------------------------------------------------------
// Cartesian product and joins
// ----------------------------------------------------------------------------

def Daphne_CartesianOp : Daphne_Op<"cartesian"> {
    let arguments = (ins Variadic<Frame>:$args);
    let results = (outs Frame:$res);
}

class Daphne_JoinOp<string name, list<OpTrait> traits = []> : Daphne_Op<name, !listconcat([SameVariadicOperandSize], traits)> {
    // TODO Maybe leftOn and rightOn should be attributes.
    let arguments = (ins Frame:$lhs, Frame:$rhs, Variadic<Size>:$leftOn, Variadic<Size>:$rightOn);
    let results = (outs Frame:$res);
}

def Daphne_InnerJoinOp : Daphne_JoinOp<"innerJoin">;
def Daphne_FullOuterJoinOp : Daphne_JoinOp<"fullOuterJoin">;
def Daphne_LeftOuterJoinOp : Daphne_JoinOp<"leftOuterJoin">;
def Daphne_AntiJoinOp : Daphne_JoinOp<"antiJoin">;
def Daphne_SemiJoinOp : Daphne_JoinOp<"semiJoin">;

// ----------------------------------------------------------------------------
// Selection
// ----------------------------------------------------------------------------

def Daphne_QueryOp : Daphne_Op<"query"> {
    let arguments = (ins Frame:$arg, StrAttr:$cond);
    let results = (outs Frame:$res);
}

def Daphne_FilterRowOp : Daphne_Op<"filterRow"> {
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

def Daphne_GroupOp : Daphne_Op<"group"> {
    // TODO Maybe colIdxs should be an attribute.
    let arguments = (ins Frame:$arg, Variadic<Size>:$colIdxs);
    let results = (outs MatrixOf<[Size]>:$groupIds, MatrixOf<[Size]>:$reprIdxs);
}

// ****************************************************************************
// Frame label manipulation
// ****************************************************************************

def Daphne_SetColLabelsOp : Daphne_Op<"setColLabels"> {
    let arguments = (ins Frame:$arg, Variadic<StrScalar>:$labels);
    let results = (outs); // no results
}

def Daphne_SetColLabelsPrefixOp : Daphne_Op<"setColLabelsPrefix"> {
    let arguments = (ins Frame:$arg, StrScalar:$prefix);
    let results = (outs); // no results
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

def Daphne_CastOp : Daphne_Op<"cast"> {
    // Note that the requested result type is not an argument, but should be
    // specified as the output type when creating a CastOp.
    let arguments = (ins AnyTypeOf<[MatrixOrFrame, AnyScalar]>:$arg);
    let results = (outs AnyTypeOf<[MatrixOrFrame, AnyScalar]>:$res);
}

def Daphne_CopyOp : Daphne_Op<"copy"> {
    let arguments = (ins MatrixOrFrame:$arg);
    let results = (outs MatrixOrFrame:$res);
}

// ****************************************************************************
// Distributed Operations
// ****************************************************************************

def Daphne_DistributeOp : Daphne_Op<"distribute"> {
    let arguments = (ins Matrix:$mat);
    let results = (outs Handle:$res);
}

def Daphne_DistributedComputeOp : Daphne_Op<"distributedCompute", [NoSideEffect]> {
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

def Daphne_PrintOp : Daphne_Op<"print"> {
    // TODO We might change it to only accept scalars here and enforce toString
    // for matrices and frames. But currently, we need it like that for the
    // rest of the program.
    let arguments = (ins AnyTypeOf<[AnyScalar, MatrixOrFrame]>:$arg);
    let results = (outs); // no results
}

// TODO Take asynchronous read into account.
def Daphne_ReadOp : Daphne_Op<"read"> {
    // TODO We might add arguments for a UDF later.
    let arguments = (ins StrScalar:$fileName);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_WriteOp : Daphne_Op<"write"> {
    let arguments = (ins StrScalar:$fileName, MatrixOrFrame:$arg);
    let results = (outs); // no results
}

// ****************************************************************************
// Data preprocessing
// ****************************************************************************

def Daphne_OneHotOp : Daphne_Op<"oneHot"> {
    let arguments = (ins MatrixOrFrame:$arg, MatrixOf<[SI64]>:$info);
    let results = (outs MatrixOrFrame:$res);
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

    let arguments = (ins); // no arguments so far
    let results = (outs DaphneContext:$ctx);
}

def Daphne_DestroyDaphneContextOp : Daphne_Op<"destroyDaphneContext"> {
    let summary = "Destroys an instance of a Daphne run-time context.";

    // Note that the DaphneContext is automatically injected during the
    // lowering to kernel calls.
    let arguments = (ins); // no arguments
    let results = (outs); // no results
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
        //OpBuilder<(ins "int":$value), [{
        //    build($_builder, $_state, $_builder.getI32IntegerAttr(value));
        //}]>,
        OpBuilder<(ins "std::string":$value), [{
            build($_builder, $_state, mlir::daphne::StringType::get($_builder.getContext()), $_builder.getStringAttr(value));
        }]>,
    ];

    let hasFolder = 1;
}

def Daphne_ReturnOp : Daphne_Op<"return", [NoSideEffect, Terminator, ReturnLike, ParentOneOf<["FuncOp", "DistributedComputeOp"]>]> {
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
