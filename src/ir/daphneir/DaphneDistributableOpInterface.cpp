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

#include <ir/daphneir/Daphne.h>
#include <util/ErrorHandler.h>

#include <string>
#include <vector>
#include <stdexcept>

namespace mlir::daphne
{
#include <ir/daphneir/DaphneDistributableOpInterface.cpp.inc>
}

using namespace mlir;

// ****************************************************************************
// DistributableOpInterface utilities
// ****************************************************************************
// For families of operations.

Type getWrappedType(Value v) {
    // Get the type wrapped into this distributed handle.
    Type wrappedType = v.getType().cast<daphne::HandleType>().getDataType();
    // Remove all information on interesting properties except for the type.
    // This is necessary since these properties do not necessarily hold for a
    // distributed partition of the whole data object.
    return wrappedType.dyn_cast<daphne::MatrixType>().withSameElementTypeAndRepr();
}

template<class EwBinaryOp>
std::vector<mlir::Value> createEquivalentDistributedDAG_EwBinaryOp(EwBinaryOp *op, mlir::OpBuilder &builder,
                                                                   mlir::ValueRange distributedInputs)
{
    auto loc = op->getLoc();
    auto compute = builder.create<daphne::DistributedComputeOp>(loc,
        ArrayRef<Type>{daphne::HandleType::get(op->getContext(), op->getType())},
        distributedInputs);
    auto &block = compute.getBody().emplaceBlock();
    auto argLhs = block.addArgument(getWrappedType(distributedInputs[0]), builder.getUnknownLoc());
    auto argRhs = block.addArgument(getWrappedType(distributedInputs[1]), builder.getUnknownLoc());

    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(&block, block.begin());

        mlir::Type resTyOrig = op->getType();
        mlir::Type resTy = resTyOrig.dyn_cast<mlir::daphne::MatrixType>().withSameElementTypeAndRepr();
        auto addOp = builder.create<EwBinaryOp>(loc, resTy, argLhs, argRhs);
        builder.create<daphne::ReturnOp>(loc, ArrayRef<Value>{addOp});
    }

    std::vector<Value> ret({builder.create<daphne::DistributedCollectOp>(loc, compute.getResult(0))});
    return ret;
}

template<class EwBinaryOp>
std::vector<bool> getOperandDistrPrimitives_EwBinaryOp(EwBinaryOp *op) {
    Type tL0 = op->getLhs().getType();
    auto tL  = tL0.dyn_cast<daphne::MatrixType>();
    Type tR0 = op->getRhs().getType();
    auto tR  = tR0.dyn_cast<daphne::MatrixType>();
    const ssize_t nrL = tL.getNumRows();
    const ssize_t ncL = tL.getNumCols();
    const ssize_t nrR = tR.getNumRows();
    const ssize_t ncR = tR.getNumCols();

    if (nrL == -1 || nrR == -1 || ncL == -1 || ncR == -1)
        throw ErrorHandler::compilerError(
            op->getLoc(), "DistributableOpInterface",
            "unknown shapes of left and/or right operand to elementwise "
            "binary operation are not supported while deciding "
            "distribute/broadcast");

    if(nrL == nrR && ncL == ncR) // matrix-matrix
        return {false, false}; // distribute both inputs
    else if(nrR == 1 && ncL == ncR) // matrix-row
        return {false, true}; // distribute lhs, broadcast rhs
    else if(nrL == nrR && ncR == 1) // matrix-col
        return {false, true}; // distribute lhs, broadcast rhs
    else
        throw ErrorHandler::compilerError(
            op->getLoc(), "DistributableOpInterface",
            "mismatching shapes of left and right operand to elementwise "
            "binary operation while deciding distribute/broadcast");
}

// ****************************************************************************
// DistributableOpInterface implementations
// ****************************************************************************

#define IMPL_EWBINARYOP(OP) \
    std::vector<mlir::Value> mlir::daphne::OP::createEquivalentDistributedDAG(mlir::OpBuilder &builder, \
        mlir::ValueRange distributedInputs) \
    { \
        return createEquivalentDistributedDAG_EwBinaryOp(this, builder, distributedInputs); \
    } \
    \
    std::vector<bool> mlir::daphne::OP::getOperandDistrPrimitives() { \
        return getOperandDistrPrimitives_EwBinaryOp(this); \
    }

// TODO We should use traits (like for shape inference) so that we don't need
// to repeat here.

// Arithmetic
IMPL_EWBINARYOP(EwAddOp)
IMPL_EWBINARYOP(EwSubOp)
IMPL_EWBINARYOP(EwMulOp)
IMPL_EWBINARYOP(EwDivOp)
IMPL_EWBINARYOP(EwPowOp)
IMPL_EWBINARYOP(EwModOp)
IMPL_EWBINARYOP(EwLogOp)

// Min/max
IMPL_EWBINARYOP(EwMinOp)
IMPL_EWBINARYOP(EwMaxOp)

// Logical
IMPL_EWBINARYOP(EwAndOp)
IMPL_EWBINARYOP(EwOrOp)
IMPL_EWBINARYOP(EwXorOp)

// Bitwise
IMPL_EWBINARYOP(EwBitwiseAndOp);

// Strings
IMPL_EWBINARYOP(EwConcatOp)

// Comparisons
IMPL_EWBINARYOP(EwEqOp)
IMPL_EWBINARYOP(EwNeqOp)
IMPL_EWBINARYOP(EwLtOp)
IMPL_EWBINARYOP(EwLeOp)
IMPL_EWBINARYOP(EwGtOp)
IMPL_EWBINARYOP(EwGeOp)

std::vector<mlir::Value> daphne::RowAggMaxOp::createEquivalentDistributedDAG(
        OpBuilder &builder, ValueRange distributedInputs
) {
    auto loc = getLoc();
    auto compute = builder.create<daphne::DistributedComputeOp>(loc,
        ArrayRef<Type>{daphne::HandleType::get(getContext(), getType())},
        distributedInputs);
    auto &block = compute.getBody().emplaceBlock();
    auto arg = block.addArgument(getWrappedType(distributedInputs[0]), builder.getUnknownLoc());

    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(&block, block.begin());

        mlir::Type resTy = getType().dyn_cast<mlir::daphne::MatrixType>().withSameElementTypeAndRepr();
        auto aggOp = builder.create<RowAggMaxOp>(loc, resTy, arg);
        builder.create<daphne::ReturnOp>(loc, ArrayRef<Value>{aggOp});
    }

    std::vector<Value> ret({builder.create<daphne::DistributedCollectOp>(loc, compute.getResult(0))});
    return ret;
}

std::vector<bool> daphne::RowAggMaxOp::getOperandDistrPrimitives() {
    return {false};
}
