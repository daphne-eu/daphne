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

def Daphne_MatrixOp : Daphne_Op<"matrix"> {
    let arguments = (ins AnyTypeOf<[AnyScalar, Matrix]>:$arg, Size:$numRows, Size:$numCols);
    let results = (outs Matrix:$res);
}

def Daphne_FrameOp : Daphne_Op<"frame"> {
    let arguments = (ins Variadic<Matrix>:$cols);
    let results = (outs Frame:$res);
}

def Daphne_DiagMatrixOp : Daphne_Op<"diagMatrix"> {
    let arguments = (ins Matrix:$arg);
    let results = (outs Matrix:$res);
}

def Daphne_RandMatrixOp : Daphne_Op<"randMatrix", [TypesMatchWith<"min and max arguments must be of the same type", "min", "max", "$_self">]> {
    //let arguments = (ins Size:$numRows, Size:$numCols, AnyScalar:$min, AnyScalar:$max, F64:$sparsity, Seed:$seed, StrScalar:$pdf);
    let arguments = (ins IntScalar:$numRows, IntScalar:$numCols, AnyScalar:$min, AnyScalar:$max, F64:$sparsity, IntScalar:$seed);
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

def Daphne_NumRowOp : Daphne_NumOp<"nrow">;
def Daphne_NumColOp : Daphne_NumOp<"ncol">;
def Daphne_NumCellOp : Daphne_NumOp<"ncell">;

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

def Daphne_EwAddOp    : Daphne_EwBinaryOp<"ewAdd", NumScalar>;
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

def Daphne_AllAggSumOp    : Daphne_AllAggOp<"allSum", NumScalar>;
def Daphne_AllAggMinOp    : Daphne_AllAggOp<"allMin", NumScalar>;
def Daphne_AllAggMaxOp    : Daphne_AllAggOp<"allMax", NumScalar>;
def Daphne_AllAggMeanOp   : Daphne_AllAggOp<"allMean", FloatScalar>;
def Daphne_AllAggVarOp    : Daphne_AllAggOp<"allVar", FloatScalar>;
def Daphne_AllAggStddevOp : Daphne_AllAggOp<"allStddev", FloatScalar>;

// ----------------------------------------------------------------------------
// Row/column-wise aggregation
// ----------------------------------------------------------------------------

class Daphne_DimAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<OpTrait> traits = []> : Daphne_AggOp<name, inScalarType, traits> {
    let results = (outs MatrixOf<[outScalarType]>:$res);
}

class Daphne_RowAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<OpTrait> traits = []> : Daphne_DimAggOp<name, inScalarType, outScalarType, traits>;
class Daphne_ColAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<OpTrait> traits = []> : Daphne_DimAggOp<name, inScalarType, outScalarType, traits>;

def Daphne_RowAggSumOp    : Daphne_RowAggOp<"rowSum"   , NumScalar>;
def Daphne_RowAggMinOp    : Daphne_RowAggOp<"rowMin"   , AnyScalar>;
def Daphne_RowAggMaxOp    : Daphne_RowAggOp<"rowMax"   , AnyScalar>;
def Daphne_RowAggIdxMinOp : Daphne_RowAggOp<"rowIdxMin", NumScalar, Size>;
def Daphne_RowAggIdxMaxOp : Daphne_RowAggOp<"rowIdxMax", NumScalar, Size>;
def Daphne_RowAggMeanOp   : Daphne_RowAggOp<"rowMean"  , FloatScalar>;
def Daphne_RowAggVarOp    : Daphne_RowAggOp<"rowVar"   , FloatScalar>;
def Daphne_RowAggStddevOp : Daphne_RowAggOp<"rowStddev", FloatScalar>;

def Daphne_ColAggSumOp    : Daphne_ColAggOp<"colSum"   , NumScalar>;
def Daphne_ColAggMinOp    : Daphne_ColAggOp<"colMin"   , AnyScalar>;
def Daphne_ColAggMaxOp    : Daphne_ColAggOp<"colMax"   , AnyScalar>;
def Daphne_ColAggIdxMinOp : Daphne_ColAggOp<"colIdxMin", NumScalar, Size>;
def Daphne_ColAggIdxMaxOp : Daphne_ColAggOp<"colIdxMax", NumScalar, Size>;
def Daphne_ColAggMeanOp   : Daphne_ColAggOp<"colMean"  , FloatScalar>;
def Daphne_ColAggVarOp    : Daphne_ColAggOp<"colVar"   , FloatScalar>;
def Daphne_ColAggStddevOp : Daphne_ColAggOp<"colStddev", FloatScalar>;

// ----------------------------------------------------------------------------
// Cumulative aggregation
// ----------------------------------------------------------------------------

class Daphne_CumAggOp<string name, Type scalarType, list<OpTrait> traits = []> : Daphne_AggOp<name, scalarType, traits> {
    let results = (outs MatrixOf<[scalarType]>);
}

def Daphne_CumAggSum  : Daphne_CumAggOp<"cumSum" , NumScalar>;
def Daphne_CumAggProd : Daphne_CumAggOp<"cumProd", NumScalar>;
def Daphne_CumAggMin  : Daphne_CumAggOp<"cumMin" , AnyScalar>;
def Daphne_CumAggMax  : Daphne_CumAggOp<"cumMax" , AnyScalar>;

// ----------------------------------------------------------------------------
// Grouped aggregation
// ----------------------------------------------------------------------------

class Daphne_GroupedAggOp<string name, Type scalarType, list<OpTrait> traits = []> : Daphne_Op<name, traits> {
    let arguments = (ins MatrixOf<[scalarType]>:$arg, MatrixOf<[Size]>:$groupIds, Size:$ngroups, Optional<Matrix>:$weights);
    let results = (outs MatrixOf<[scalarType]>:$groupAggs);
}

def Daphne_GroupedAggCountOp  : Daphne_GroupedAggOp<"grCount" , AnyScalar>;
def Daphne_GroupedAggMinOp    : Daphne_GroupedAggOp<"grMin"   , AnyScalar>;
def Daphne_GroupedAggMaxOp    : Daphne_GroupedAggOp<"grMax"   , AnyScalar>;
def Daphne_GroupedAggSumOp    : Daphne_GroupedAggOp<"grSum"   , NumScalar>;
def Daphne_GroupedAggMeanOp   : Daphne_GroupedAggOp<"grMean"  , FloatScalar>;
def Daphne_GroupedAggVarOp    : Daphne_GroupedAggOp<"grVar"   , FloatScalar>;
def Daphne_GroupedAggStddevOp : Daphne_GroupedAggOp<"grStddev", FloatScalar>;

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
    let arguments = (ins MatrixOrFrame:$source, Selection:$selectedRows, Selection:$selectedCols);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_InsertOp : Daphne_Op<"insert"> {
    let arguments = (ins MatrixOrFrame:$target, MatrixOrFrame:$source, Selection:$selectedRows, Selection:$selectedCols);
    let results = (outs); // no results
}

// ****************************************************************************
// Reorganization
// ****************************************************************************

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

def Daphne_ColBindOp : Daphne_BindOp<"cbind">;
def Daphne_RowBindOp : Daphne_BindOp<"rbind">;

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

def Daphne_LuOP : Daphne_Op<"lu"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg);
    let results = (outs MatrixOf<[FloatScalar]>:$p, MatrixOf<[FloatScalar]>:$l, MatrixOf<[FloatScalar]>:$u);
}

def Daphne_QrOP : Daphne_Op<"qr"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg);
    let results = (outs MatrixOf<[FloatScalar]>:$h, MatrixOf<[FloatScalar]>:$r);
}

def Daphne_SvdOP : Daphne_Op<"svd"> {
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

def Daphne_Conv2DForward : Daphne_Op<"conv2DForward"> {
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

def Daphne_Conv2DBackwardFilter : Daphne_Op<"conv2DNBackwardFilter"> {
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

def Daphne_Conv2DBackwardData : Daphne_Op<"conv2DNBackwardData"> {
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

class Daphne_TriOp<string name, list<OpTrait> traits = []> : Daphne_Op<name, traits> {
    let arguments = (ins Matrix:$arg, BoolScalar:$diag, BoolScalar:$values);
    let results = (outs Matrix:$res);
}

def Daphne_LowerTriOp : Daphne_TriOp<"lowerTri">;
def Daphne_UpperTriOp : Daphne_TriOp<"upperTri">;

def Daphne_SolveOp : Daphne_Op<"solve"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$a, MatrixOf<[FloatScalar]>:$b);
    let results = (outs MatrixOf<[FloatScalar]>:$x);
}

def Daphne_ReplaceOp : Daphne_Op<"replace"> {
    let arguments = (ins Matrix:$arg, AnyScalar:$pattern, AnyScalar:$replace);
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

def Daphne_FilterOp : Daphne_Op<"filter"> {
    let arguments = (ins Frame:$arg, MatrixOf<[BoolScalar]>:$selected);
    let results = (outs Frame:$res);
}

// ----------------------------------------------------------------------------
// Grouping
// ----------------------------------------------------------------------------

def Daphne_GroupOp : Daphne_Op<"group"> {
    // TODO Maybe colIdxs should be an attribute.
    let arguments = (ins Frame:$arg, Variadic<Size>:$colIdxs);
    let results = (outs MatrixOf<[Size]>:$groupIds, MatrixOf<[Size]>:$reprIdxs);
}

// ****************************************************************************
// Conversions and casts
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
// Old operations
// ****************************************************************************
// These are currently still need for the prototype to work. They will be
// removed/replaced/changed inthe future.

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
    ];

    let hasFolder = 1;
}

def Daphne_ReturnOp : Daphne_Op<"return", [NoSideEffect, Terminator, ReturnLike, HasParent<"FuncOp">]> {
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
