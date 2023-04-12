/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>

#include <vector>

namespace mlir::daphne
{
#include <ir/daphneir/DaphneVectorizableOpInterface.cpp.inc>
}

using namespace mlir;

// ****************************************************************************
// Vector split and combine utility functions
// ****************************************************************************
// For families of operations.

template<class EwBinaryOp>
std::vector<daphne::VectorSplit> getVectorSplits_EwBinaryOp(EwBinaryOp *op)
{
    // Matrix -> row-wise, Scalar -> none
    auto lhsSplit =
        op->getLhs().getType().template isa<daphne::MatrixType>() ? daphne::VectorSplit::ROWS : daphne::VectorSplit::NONE;
    auto rhsSplit =
        op->getRhs().getType().template isa<daphne::MatrixType>() ? daphne::VectorSplit::ROWS : daphne::VectorSplit::NONE;
    return {lhsSplit, rhsSplit};
}
template<class EwBinaryOp>
std::vector<daphne::VectorCombine> getVectorCombines_EwBinaryOp(EwBinaryOp *op)
{
    return {daphne::VectorCombine::ROWS};
}
template<class EwBinaryOp>
std::vector<std::pair<Value, Value>> createOpsOutputSizes_EwBinaryOp(EwBinaryOp *op, OpBuilder &builder)
{
    auto loc = op->getLoc();
    auto sizeTy = builder.getIndexType();
    auto lhsRows = builder.create<daphne::NumRowsOp>(loc, sizeTy, op->getLhs());
    auto lhsCols = builder.create<daphne::NumColsOp>(loc, sizeTy, op->getLhs());
    // TODO: do max on #rows/#cols of lhs and rhs for broadcasting
    return {{lhsRows, lhsCols}};
}
template<class EwUnaryOp>
std::vector<daphne::VectorSplit> getVectorSplits_EwUnaryOp(EwUnaryOp *op)
{
    return {daphne::VectorSplit::ROWS};
}
template<class EwUnaryOp>
std::vector<daphne::VectorCombine> getVectorCombines_EwUnaryOp(EwUnaryOp *op)
{
    return {daphne::VectorCombine::ROWS};
}
template<class EwUnaryOp>
std::vector<std::pair<Value, Value>> createOpsOutputSizes_EwUnaryOp(EwUnaryOp *op, OpBuilder &builder)
{
    auto loc = op->getLoc();
    auto sizeTy = builder.getIndexType();
    auto rows = builder.create<daphne::NumRowsOp>(loc, sizeTy, op->getArg());
    auto cols = builder.create<daphne::NumColsOp>(loc, sizeTy, op->getArg());
    // TODO: do max on #rows/#cols of lhs and rhs for broadcasting
    return {{rows, cols}};
}
template<class RowAggOp>
std::vector<daphne::VectorSplit> getVectorSplits_RowAggOp(RowAggOp *op)
{
    return {daphne::VectorSplit::ROWS};
}
template<class RowAggOp>
std::vector<daphne::VectorCombine> getVectorCombines_RowAggOp(RowAggOp *op)
{
    return {daphne::VectorCombine::ROWS};
}
template<class RowAggOp>
std::vector<std::pair<Value, Value>> createOpsOutputSizes_RowAggOp(RowAggOp *op, OpBuilder &builder)
{
    auto loc = op->getLoc();
    auto sizeTy = builder.getIndexType();
    auto rows = builder.create<daphne::NumRowsOp>(loc, sizeTy, op->getArg());
    auto cst1 = builder.create<daphne::ConstantOp>(loc, sizeTy, builder.getIndexAttr(1l));
    return {{rows, cst1}};
}
template<class ColAggOp>
std::vector<daphne::VectorSplit> getVectorSplits_ColAggOp(ColAggOp *op)
{
    return {daphne::VectorSplit::ROWS};
}
template<class ColAggOp>
std::vector<std::pair<Value, Value>> createOpsOutputSizes_ColAggOp(ColAggOp *op, OpBuilder &builder)
{
    auto loc = op->getLoc();
    auto sizeTy = builder.getIndexType();
    auto cst1 = builder.create<daphne::ConstantOp>(loc, sizeTy, builder.getIndexAttr(1l));
    auto cols = builder.create<daphne::NumColsOp>(loc, sizeTy, op->getArg());
    return {{cst1, cols}};
}

// ****************************************************************************
// Vector split and combine implementations
// ****************************************************************************

// ----------------------------------------------------------------------------
// Matrix multiplication
std::vector<daphne::VectorSplit> daphne::MatMulOp::getVectorSplits()
{
    return {
        daphne::VectorSplit::ROWS, // lhs
        daphne::VectorSplit::NONE, // rhs
        daphne::VectorSplit::NONE, // transa
        daphne::VectorSplit::NONE  // transb
    };
}
std::vector<daphne::VectorCombine> daphne::MatMulOp::getVectorCombines()
{
    return {daphne::VectorCombine::ROWS};
}
std::vector<std::pair<Value, Value>> daphne::MatMulOp::createOpsOutputSizes(OpBuilder &builder)
{
    auto loc = getLoc();
    auto sizeTy = builder.getIndexType();

    Value rows;
    bool ta = CompilerUtils::constantOrThrow<bool>(
            getTransa(),
            "VectorizableOpInterface::createOpsOutputSizes() for MatMulOp cannot know the number "
            "of rows of the result, because it is not known if the lhs input is transposed"
    );
    rows = ta
            ? builder.create<daphne::NumColsOp>(loc, sizeTy, getLhs()).getResult()
            : builder.create<daphne::NumRowsOp>(loc, sizeTy, getLhs()).getResult();
    
    Value cols;
    bool tb = CompilerUtils::constantOrThrow<bool>(
            getTransb(),
            "VectorizableOpInterface::createOpsOutputSizes() for MatMulOp cannot know the number "
            "of columns of the result, because it is not known if the rhs input is transposed"
    );
    cols = tb
            ? builder.create<daphne::NumRowsOp>(loc, sizeTy, getRhs()).getResult()
            : builder.create<daphne::NumColsOp>(loc, sizeTy, getRhs()).getResult();

    return {{rows, cols}};
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Binary
#define IMPL_SPLIT_COMBINE_EWBINARYOP(OP) \
    std::vector<daphne::VectorSplit> daphne::OP::getVectorSplits() { \
        return getVectorSplits_EwBinaryOp(this); \
    } \
    std::vector<daphne::VectorCombine> daphne::OP::getVectorCombines() { \
        return getVectorCombines_EwBinaryOp(this); \
    } \
    std::vector<std::pair<Value, Value>> daphne::OP::createOpsOutputSizes(OpBuilder &builder) { \
        return createOpsOutputSizes_EwBinaryOp(this, builder); \
    }

// Arithmetic
IMPL_SPLIT_COMBINE_EWBINARYOP(EwAddOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwSubOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwMulOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwDivOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwPowOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwModOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwLogOp)

// Min/max
IMPL_SPLIT_COMBINE_EWBINARYOP(EwMinOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwMaxOp)

// Logical
IMPL_SPLIT_COMBINE_EWBINARYOP(EwAndOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwOrOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwXorOp)

// Bitwise
IMPL_SPLIT_COMBINE_EWBINARYOP(EwBitwiseAndOp);

// Strings
IMPL_SPLIT_COMBINE_EWBINARYOP(EwConcatOp)

// Comparisons
IMPL_SPLIT_COMBINE_EWBINARYOP(EwEqOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwNeqOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwLtOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwLeOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwGtOp)
IMPL_SPLIT_COMBINE_EWBINARYOP(EwGeOp)
#undef IMPL_SPLIT_COMBINE_EWBINARYOP
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Unary
#define IMPL_SPLIT_COMBINE_EWUNARYOP(OP) \
    std::vector<daphne::VectorSplit> daphne::OP::getVectorSplits() { \
        return getVectorSplits_EwUnaryOp(this); \
    } \
    std::vector<daphne::VectorCombine> daphne::OP::getVectorCombines() { \
        return getVectorCombines_EwUnaryOp(this); \
    } \
    std::vector<std::pair<Value, Value>> daphne::OP::createOpsOutputSizes(OpBuilder &builder) { \
        return createOpsOutputSizes_EwUnaryOp(this, builder); \
    }

IMPL_SPLIT_COMBINE_EWUNARYOP(EwSqrtOp)

#undef IMPL_SPLIT_COMBINE_EWUNARYOP
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Aggregations
// TODO: splitting and combining by column probably makes more sense
#define IMPL_SPLIT_COMBINE_ROWAGG(OP) \
    std::vector<daphne::VectorSplit> daphne::OP::getVectorSplits() { \
        return getVectorSplits_RowAggOp(this); \
    } \
    std::vector<daphne::VectorCombine> daphne::OP::getVectorCombines() { \
        return getVectorCombines_RowAggOp(this); \
    } \
    std::vector<std::pair<Value, Value>> daphne::OP::createOpsOutputSizes(OpBuilder &builder) { \
        return createOpsOutputSizes_RowAggOp(this, builder); \
    }
#define IMPL_SPLIT_COMBINE_COLAGG(OP) \
    std::vector<daphne::VectorSplit> daphne::OP::getVectorSplits() { \
        return getVectorSplits_ColAggOp(this); \
    } \
    std::vector<std::pair<Value, Value>> daphne::OP::createOpsOutputSizes(OpBuilder &builder) { \
        return createOpsOutputSizes_ColAggOp(this, builder); \
    }

// RowAgg
IMPL_SPLIT_COMBINE_ROWAGG(RowAggMinOp)
IMPL_SPLIT_COMBINE_ROWAGG(RowAggMaxOp)
IMPL_SPLIT_COMBINE_ROWAGG(RowAggSumOp)

IMPL_SPLIT_COMBINE_COLAGG(ColAggSumOp)
std::vector<daphne::VectorCombine> daphne::ColAggSumOp::getVectorCombines()
{
    return {daphne::VectorCombine::ADD};
}

#undef IMPL_SPLIT_COMBINE_ROWAGG
#undef IMPL_SPLIT_COMBINE_COLAGG
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// Left and right indexing
std::vector<daphne::VectorSplit> daphne::ExtractColOp::getVectorSplits()
{
    return {daphne::VectorSplit::ROWS, daphne::VectorSplit::NONE};
}
std::vector<daphne::VectorCombine> daphne::ExtractColOp::getVectorCombines()
{
    return {daphne::VectorCombine::ROWS};
}
std::vector<std::pair<Value, Value>> daphne::ExtractColOp::createOpsOutputSizes(OpBuilder &builder)
{
    auto loc = getLoc();
    auto sizeTy = builder.getIndexType();
    auto rows = builder.create<daphne::NumRowsOp>(loc, sizeTy, getSource());
    // TODO: support scalar and maybe (based on definition of `ExtractColOp`) apply some kind of `unique()` op
    auto cols = builder.create<daphne::NumRowsOp>(loc, sizeTy, getSelectedCols());
    return {{rows, cols}};
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Reorganization
std::vector<daphne::VectorSplit> daphne::TransposeOp::getVectorSplits()
{
    return {daphne::VectorSplit::ROWS};
}
std::vector<daphne::VectorCombine> daphne::TransposeOp::getVectorCombines()
{
    return {daphne::VectorCombine::COLS};
}
std::vector<std::pair<Value, Value>> daphne::TransposeOp::createOpsOutputSizes(OpBuilder &builder)
{
    auto loc = getLoc();
    auto sizeTy = builder.getIndexType();
    auto rows = builder.create<daphne::NumRowsOp>(loc, sizeTy, getArg());
    auto cols = builder.create<daphne::NumColsOp>(loc, sizeTy, getArg());
    return {{cols, rows}};
}

std::vector<daphne::VectorSplit> daphne::ColBindOp::getVectorSplits()
{
    return {daphne::VectorSplit::ROWS, daphne::VectorSplit::ROWS};
}
std::vector<daphne::VectorCombine> daphne::ColBindOp::getVectorCombines()
{
    return {daphne::VectorCombine::ROWS};
}
std::vector<std::pair<Value, Value>> daphne::ColBindOp::createOpsOutputSizes(OpBuilder &builder)
{
    auto loc = getLoc();
    auto i64Ty = builder.getIntegerType(64, true);
    auto sizeTy = builder.getIndexType();
    auto rows = builder.create<daphne::NumRowsOp>(loc, sizeTy, getLhs());
    auto colsLhs = builder.create<daphne::NumColsOp>(loc, sizeTy, getLhs());
    auto colsRhs = builder.create<daphne::NumColsOp>(loc, sizeTy, getRhs());
    return {{rows, builder.create<daphne::CastOp>(loc,
        sizeTy,
        builder.create<daphne::EwAddOp>(loc,
            builder.create<daphne::CastOp>(loc, i64Ty, colsLhs),
            builder.create<daphne::CastOp>(loc, i64Ty, colsRhs)))}};
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Other
std::vector<daphne::VectorSplit> daphne::SyrkOp::getVectorSplits()
{
    return {daphne::VectorSplit::ROWS};
}
std::vector<daphne::VectorCombine> daphne::SyrkOp::getVectorCombines()
{
    return {daphne::VectorCombine::ADD};
}
std::vector<std::pair<Value, Value>> daphne::SyrkOp::createOpsOutputSizes(OpBuilder &builder)
{
    auto loc = getLoc();
    auto sizeTy = builder.getIndexType();
    auto cols = builder.create<daphne::NumColsOp>(loc, sizeTy, getArg());
    // TODO: do max on #rows/#cols of lhs and rhs for broadcasting
    return {{cols, cols}};
}

std::vector<daphne::VectorSplit> daphne::GemvOp::getVectorSplits()
{
    return {daphne::VectorSplit::ROWS, daphne::VectorSplit::ROWS};
}
std::vector<daphne::VectorCombine> daphne::GemvOp::getVectorCombines()
{
    return {daphne::VectorCombine::ADD};
}
std::vector<std::pair<Value, Value>> daphne::GemvOp::createOpsOutputSizes(OpBuilder &builder)
{
    auto loc = getLoc();
    auto sizeTy = builder.getIndexType();
    auto cols = builder.create<daphne::NumColsOp>(loc, sizeTy, getMat());
    auto one = builder.create<daphne::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(1));
    return {{cols, one}};
}
// ----------------------------------------------------------------------------
