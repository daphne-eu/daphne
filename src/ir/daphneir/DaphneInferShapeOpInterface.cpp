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

#include <ir/daphneir/Daphne.h>

#include <mlir/IR/Value.h>

#include <runtime/local/io/FileMetaData.h>

#include <vector>
#include <stdexcept>
#include <utility>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferShapeOpInterface.cpp.inc>
}

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Utilities
// ****************************************************************************

std::pair<ssize_t, ssize_t> getShape(Value v) {
    Type t = v.getType();
    if(auto mt = t.dyn_cast<daphne::MatrixType>())
        return std::make_pair(mt.getNumRows(), mt.getNumCols());
    if(auto ft = t.dyn_cast<daphne::FrameType>())
        return std::make_pair(ft.getNumRows(), ft.getNumCols());
    // TODO Maybe check if it is really a scalar type.
    else // scalar
        return std::make_pair(1, 1);
}

ssize_t getSizeOrUnknown(Value v) {
    if (!v.getDefiningOp()) // check if block argument
        return -1;
    if(auto co = llvm::dyn_cast<daphne::ConstantOp>(v.getDefiningOp()))
        if(auto intAttr = co.value().dyn_cast<IntegerAttr>())
            return intAttr.getValue().getLimitedValue();
    // TODO Remove this once we support constant propagation (see #151).
    if(auto co = llvm::dyn_cast<daphne::CastOp>(v.getDefiningOp()))
        return getSizeOrUnknown(co.arg());
    return -1; // the value of the scalar is unknown at the moment
}

// TODO This is just a quick and dirty workaround. Make this a central utility.
int64_t getConstantInt(Value v) {
    if(auto co = llvm::dyn_cast<daphne::ConstantOp>(v.getDefiningOp()))
        if(auto intAttr = co.value().dyn_cast<IntegerAttr>())
            return intAttr.getValue().getLimitedValue();
    // TODO Remove this once we support constant propagation (see #151).
    if(auto co = llvm::dyn_cast<daphne::CastOp>(v.getDefiningOp()))
        return getConstantInt(co.arg());
    if(auto co = llvm::dyn_cast<daphne::EwSubOp>(v.getDefiningOp()))
        return getConstantInt(co.lhs()) - getConstantInt(co.rhs());
    if(auto co = llvm::dyn_cast<daphne::EwDivOp>(v.getDefiningOp()))
        return getConstantInt(co.lhs()) / getConstantInt(co.rhs());
    if(auto co = llvm::dyn_cast<daphne::NumColsOp>(v.getDefiningOp()))
        return getSizeOrUnknown(co.arg());
    throw std::runtime_error("expected an integer constant");
}

// TODO This is just a quick and dirty workaround. Make this a central utility.
double getConstantDouble(Value v) {
    if(auto co = llvm::dyn_cast<daphne::ConstantOp>(v.getDefiningOp()))
        if(auto floatAttr = co.value().dyn_cast<FloatAttr>())
            return floatAttr.getValue().convertToDouble();
    // TODO Remove this once we support constant propagation (see #151).
    if(auto co = llvm::dyn_cast<daphne::CastOp>(v.getDefiningOp())) {
        if(co.arg().getType().isF64())
            return getConstantDouble(co.arg());
        else
            return static_cast<double>(getConstantInt(co.arg()));
    }
    throw std::runtime_error("expected a floating-point constant");
}

ssize_t inferNumRowsFromArgs(ValueRange vs) {
    // If the #rows of all arguments is known and matches, then this is the
    // infered #rows. If the known #rows of any two arguments mismatch, an
    // exception is thrown. Otherwise, if the #rows of any argument is unknown,
    // the infered #rows is unknown.
    ssize_t numRows = getShape(vs[0]).first;
    bool someUnknown = false;
    if(numRows == -1)
        someUnknown = true;
    for(size_t i = 1; i < vs.size(); i++) {
        const ssize_t nextNumRows = getShape(vs[i]).first;
        if(nextNumRows == -1)
            someUnknown = true;
        else if(numRows == -1)
            numRows = nextNumRows;
        else if(nextNumRows != numRows)
            throw std::runtime_error(
                    "shape inference: inferNumRowsFromArgs() requires that "
                    "arguments have the same number of rows, but there is "
                    "one with " + std::to_string(numRows) + " and one with " +
                    std::to_string(nextNumRows) + " rows"
            );
    }
    return someUnknown ? -1 : numRows;
}

ssize_t inferNumColsFromArgs(ValueRange vs) {
    // If the #cols of all arguments is known and matches, then this is the
    // infered #cols. If the known #cols of any two arguments mismatch, an
    // exception is thrown. Otherwise, if the #cols of any argument is unknown,
    // the infered #cols is unknown.
    ssize_t numCols = getShape(vs[0]).first;
    bool someUnknown = false;
    if(numCols == -1)
        someUnknown = true;
    for(size_t i = 1; i < vs.size(); i++) {
        const ssize_t nextNumCols = getShape(vs[i]).first;
        if(nextNumCols == -1)
            someUnknown = true;
        else if(numCols == -1)
            numCols = nextNumCols;
        else if(nextNumCols != numCols)
            throw std::runtime_error(
                    "shape inference: inferNumColsFromArgs() requires that "
                    "arguments have the same number of columns, but there is "
                    "one with " + std::to_string(numCols) + " and one with " +
                    std::to_string(nextNumCols) + " columns"
            );
    }
    return someUnknown ? -1 : numCols;
}

ssize_t inferNumRowsFromSumOfArgs(ValueRange vs) {
    ssize_t sumNumRows = 0;
    for(Value v : vs) {
        const ssize_t numRows = getShape(v).first;
        if(numRows == -1)
            return -1;
        sumNumRows += numRows;
    }
    return sumNumRows;
}

ssize_t inferNumColsFromSumOfArgs(ValueRange vs) {
    ssize_t sumNumCols = 0;
    for(Value v : vs) {
        const ssize_t numCols = getShape(v).second;
        if(numCols == -1)
            return -1;
        sumNumCols += numCols;
    }
    return sumNumCols;
}

// ****************************************************************************
// Shape inference interface implementations
// ****************************************************************************

ssize_t daphne::CartesianOp::inferNumRows() {
    auto ftLhs = lhs().getType().dyn_cast<daphne::FrameType>();
    auto ftRhs = rhs().getType().dyn_cast<daphne::FrameType>();
    return ftLhs.getNumRows() * ftRhs.getNumRows();
}

ssize_t daphne::SeqOp::inferNumRows() {
    if(from().getType().isF64()) {
        double vFrom = getConstantDouble(from());
        double vTo = getConstantDouble(to());
        double vInc = getConstantDouble(inc());
        return floor(vTo / vInc - vFrom / vInc) + 1;
    }
    else if(from().getType().isSignedInteger(64)) {
        int64_t vFrom = getConstantInt(from());
        int64_t vTo = getConstantInt(to());
        int64_t vInc = getConstantInt(inc());
        return abs(vTo - vFrom) / abs(vInc) + 1;
    }
    throw std::runtime_error(
            "at the moment, shape inference for SeqOp supports only F64 and "
            "SI64 value types"
    );
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::CreateFrameOp::inferShape() {
    return {{inferNumRowsFromArgs(cols()), inferNumColsFromSumOfArgs(cols())}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::GroupJoinOp::inferShape() {
    // We don't know the exact numbers of rows here, but we know the numbers of
    // columns.
    return {{-1, 2}, {-1, 1}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::GroupOp::inferShape() {
    // We don't know the exact number of groups here.
    const size_t numRows = arg().getType().dyn_cast<daphne::MatrixType>().getNumRows();
    return {{numRows, 1}, {-1, 1}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::ReadOp::inferShape() {
    // We don't know the exact number of groups here.
    if(auto co = llvm::dyn_cast<mlir::daphne::ConstantOp>(fileName().getDefiningOp())) {
        if(auto strAttr = co.value().dyn_cast<mlir::StringAttr>()) {
            auto filename = strAttr.getValue().str();
            FileMetaData fmd = FileMetaData::ofFile(filename);
            return {{fmd.numRows, fmd.numCols}};
        }
    }
    return {{-1, -1}};
}

// ****************************************************************************
// Shape inference trait implementations
// ****************************************************************************
         
/**
 * @brief Utility for trying a parametric trait for all values of the parameter
 * from 0 to some upper bound.
 */
template<size_t upper, template<size_t> class tryParametricTrait>
struct tryParamTraitUntil {
    static void apply(ssize_t& numRows, ssize_t& numCols, Operation * op) {
        tryParametricTrait<upper>::apply(numRows, numCols, op);
        tryParamTraitUntil<upper - 1, tryParametricTrait>::apply(numRows, numCols, op);
    }
};
template<template<size_t> class tryParametricTrait>
struct tryParamTraitUntil<0, tryParametricTrait> {
    static void apply(ssize_t& numRows, ssize_t& numCols, Operation * op) {
        tryParametricTrait<0>::apply(numRows, numCols, op);
    }
};

template<size_t i>
struct tryNumRowsFromIthScalar {
    static void apply(ssize_t& numRows, ssize_t& numCols, Operation* op) {
        if(op->hasTrait<NumRowsFromIthScalar<i>::template Impl>())
            numRows = getSizeOrUnknown(op->getOperand(i));
    }
};
template<size_t i>
struct tryNumColsFromIthScalar {
    static void apply(ssize_t& numRows, ssize_t& numCols, Operation* op) {
        if(op->hasTrait<NumColsFromIthScalar<i>::template Impl>())
            numCols = getSizeOrUnknown(op->getOperand(i));
    }
};

template<size_t i>
struct tryNumRowsFromIthArg {
    static void apply(ssize_t& numRows, ssize_t& numCols, Operation* op) {
        if(op->hasTrait<NumRowsFromIthArg<i>::template Impl>())
            numRows = getShape(op->getOperand(i)).first;
    }
};
template<size_t i>
struct tryNumColsFromIthArg {
    static void apply(ssize_t& numRows, ssize_t& numCols, Operation* op) {
        if(op->hasTrait<NumColsFromIthArg<i>::template Impl>())
            numCols = getShape(op->getOperand(i)).second;
    }
};

template<size_t i>
struct tryNumRowsFromIthArgNumCols {
    static void apply(ssize_t& numRows, ssize_t& numCols, Operation* op) {
        if(op->hasTrait<NumRowsFromIthArgNumCols<i>::template Impl>())
            numRows = getShape(op->getOperand(i)).second;
    }
};
template<size_t i>
struct tryNumColsFromIthArgNumRows {
    static void apply(ssize_t& numRows, ssize_t& numCols, Operation* op) {
        if(op->hasTrait<NumColsFromIthArgNumRows<i>::template Impl>())
            numCols = getShape(op->getOperand(i)).first;
    }
};

template<size_t i>
struct tryShapeFromIthArg {
    static void apply(ssize_t& numRows, ssize_t& numCols, Operation* op) {
        if(op->hasTrait<ShapeFromIthArg<i>::template Impl>()) {
            auto shape = getShape(op->getOperand(i));
            numRows = shape.first;
            numCols = shape.second;
        }
    }
};

// ****************************************************************************
// Shape inference function
// ****************************************************************************

std::vector<std::pair<ssize_t, ssize_t>> daphne::tryInferShape(Operation* op) {
    if(auto inferShapeOp = llvm::dyn_cast<daphne::InferShape>(op))
        // If the operation implement the shape inference interface, we apply
        // that.
        return inferShapeOp.inferShape();
    else {
        // If the operation does not implement the shape inference interface,
        // we utilize its shape inference traits, or the inference interfaces
        // for the number of rows and columns (separately).
        
        ssize_t numRows = -1;
        ssize_t numCols = -1;

        if(op->hasTrait<OneRow>())
            numRows = 1;
        if(op->hasTrait<OneCol>())
            numCols = 1;
        // Our parametric traits addressing a certain argument are supported
        // for up to 10 arguments (this can easily be changed here).
        // There does not seem to be a way in MLIR do it more generically,
        // since the parameters of parametric traits are template parameters.
        const size_t u = 9;
        tryParamTraitUntil<u, tryNumRowsFromIthScalar>::apply(numRows, numCols, op);
        tryParamTraitUntil<u, tryNumColsFromIthScalar>::apply(numRows, numCols, op);
        tryParamTraitUntil<u, tryNumRowsFromIthArg>::apply(numRows, numCols, op);
        tryParamTraitUntil<u, tryNumColsFromIthArg>::apply(numRows, numCols, op);
        tryParamTraitUntil<u, tryNumRowsFromIthArgNumCols>::apply(numRows, numCols, op);
        tryParamTraitUntil<u, tryNumColsFromIthArgNumRows>::apply(numRows, numCols, op);
        if(op->hasTrait<NumRowsFromAllArgs>())
            numRows = inferNumRowsFromArgs(op->getOperands());
        if(op->hasTrait<NumColsFromAllArgs>())
            numCols = inferNumColsFromArgs(op->getOperands());
        if(op->hasTrait<NumRowsFromSumOfAllArgs>())
            numRows = inferNumRowsFromSumOfArgs(op->getOperands());
        if(op->hasTrait<NumColsFromSumOfAllArgs>())
            numCols = inferNumColsFromSumOfArgs(op->getOperands());
        tryParamTraitUntil<u, tryShapeFromIthArg>::apply(numRows, numCols, op);
        if(op->hasTrait<ShapeEwBinary>()) {
            // The output has the shape of the left-hand-side operand. This is
            // consistent with the kernel, but in the future, we should extend
            // this to support broadcasting of vectors and scalars from left
            // and right.
            auto shapeLhs = getShape(op->getOperand(0));
            auto shapeRhs = getShape(op->getOperand(1));
            if(shapeRhs.first == -1 || shapeRhs.second == -1) {
                numRows = -1;
                numCols = -1;
            }
            else {
                numRows = shapeLhs.first;
                numCols = shapeLhs.second;
            }
            // TODO Throw if lhs and rhs don't agree.
        }
        
        if(auto inferNumRowsOp = llvm::dyn_cast<daphne::InferNumRows>(op))
            numRows = inferNumRowsOp.inferNumRows();
        if(auto inferNumColsOp = llvm::dyn_cast<daphne::InferNumCols>(op))
            numCols = inferNumColsOp.inferNumCols();
    
        // Note that all our shape inference traits assume that the operation
        // has exactly one result (which is the case for most DaphneIR ops).
        return {{numRows, numCols}};
    }
}