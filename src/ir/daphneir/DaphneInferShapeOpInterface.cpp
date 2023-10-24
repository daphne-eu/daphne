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

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>
#include <runtime/local/datastructures/Structure.h>

#include <mlir/IR/Value.h>

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

ssize_t inferNumRowsFromArgs(ValueRange vs) {
    // If the #rows of all arguments is known and matches, then this is the
    // inferred #rows. If the known #rows of any two arguments mismatch, an
    // exception is thrown. Otherwise, if the #rows of any argument is unknown,
    // the inferred #rows is unknown.
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
    ssize_t numCols = getShape(vs[0]).second;
    bool someUnknown = false;
    if(numCols == -1)
        someUnknown = true;
    for(size_t i = 1; i < vs.size(); i++) {
        const ssize_t nextNumCols = getShape(vs[i]).second;
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
    auto ftLhs = getLhs().getType().dyn_cast<daphne::FrameType>();
    auto ftRhs = getRhs().getType().dyn_cast<daphne::FrameType>();
    return ftLhs.getNumRows() * ftRhs.getNumRows();
}

ssize_t daphne::SeqOp::inferNumRows() {
    Type fromTy = getFrom().getType();
    if(fromTy.isF64()) {
        try {
            double vFrom = CompilerUtils::constantOrThrow<double>(getFrom());
            double vTo = CompilerUtils::constantOrThrow<double>(getTo());
            double vInc = CompilerUtils::constantOrThrow<double>(getInc());
            return floor(vTo / vInc - vFrom / vInc) + 1;
        }
        catch(const std::runtime_error & e) {
            return -1;
        }
    }
    if(fromTy.isF32()) {
        try {
            float vFrom = CompilerUtils::constantOrThrow<float>(getFrom());
            float vTo = CompilerUtils::constantOrThrow<float>(getTo());
            float vInc = CompilerUtils::constantOrThrow<float>(getInc());
            return floor(vTo / vInc - vFrom / vInc) + 1;
        }
        catch(const std::runtime_error & e) {
            return -1;
        }
    }
    else if(fromTy.isSignedInteger(64)) {
        try {
            int64_t vFrom = CompilerUtils::constantOrThrow<int64_t>(getFrom());
            int64_t vTo = CompilerUtils::constantOrThrow<int64_t>(getTo());
            int64_t vInc = CompilerUtils::constantOrThrow<int64_t>(getInc());
            return abs(vTo - vFrom) / abs(vInc) + 1;
        }
        catch(const std::runtime_error & e) {
            return -1;
        }
    }
    throw std::runtime_error(
            "at the moment, shape inference for SeqOp supports only F64 and "
            "SI64 value types"
    );
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::CreateFrameOp::inferShape() {
    return {{inferNumRowsFromArgs(getCols()), inferNumColsFromSumOfArgs(getCols())}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::GroupJoinOp::inferShape() {
    // We don't know the exact numbers of rows here, but we know the numbers of
    // columns.
    return {{-1, 2}, {-1, 1}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::GroupOp::inferShape() {
    // We don't know the exact number of groups here.
    const size_t numRows = -1;
    const size_t numCols = getKeyCol().size() + getAggCol().size();
    return {{numRows, numCols}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::MatMulOp::inferShape() {
    auto shapeLhs = getShape(getLhs());
    auto shapeRhs = getShape(getRhs());

    ssize_t numRows = -1;
    std::pair<bool, bool> pr = CompilerUtils::isConstant<bool>(getTransa());
    if(pr.first)
        numRows = pr.second ? shapeLhs.second : shapeLhs.first;
    
    ssize_t numCols = -1;
    std::pair<bool, bool> pc = CompilerUtils::isConstant<bool>(getTransb());
    if(pc.first)
        numCols = pc.second ? shapeRhs.first : shapeRhs.second;

    return {{numRows, numCols}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::ReadOp::inferShape() {
    auto p = CompilerUtils::isConstant<std::string>(getFileName());
    if (p.first) {
        FileMetaData fmd = CompilerUtils::getFileMetaData(getFileName());
        return {{fmd.numRows, fmd.numCols}};
    } else {
        return {{-1, -1}};
    }
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::OrderOp::inferShape() {
    size_t numRows = -1;
    size_t numCols = -1;

    Type t = getArg().getType();
    if(auto mt = t.dyn_cast<daphne::MatrixType>()){
        numRows = mt.getNumRows();
        numCols = mt.getNumCols();
    }
    if(auto ft = t.dyn_cast<daphne::FrameType>()){
        numRows = ft.getNumRows();
        numCols = ft.getNumCols();
    }
    std::pair<bool, bool> p = CompilerUtils::isConstant<bool>(getReturnIdxs());
    if(p.first) {
        if(p.second)
            numCols = 1;
    }
    else
        numCols = -1;

    return {{numRows, numCols}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::CondOp::inferShape() {
    Type condTy = getCond().getType();
    if(condTy.isa<daphne::UnknownType>())
        // Actually, this should not happen, because if the type of the
        // condition is unknown, the type of the result should be unknown
        // too per type inference, such that shape inference should not
        // even get called. Nevertheless, returning unknown will probably
        // not hurt in case anyone ever calls this from somewhere else.
        return {{-1, -1}};
    if(auto condMatTy = condTy.dyn_cast<daphne::MatrixType>())
        return {{condMatTy.getNumRows(), condMatTy.getNumCols()}};
    else if(auto condFrmTy = condTy.dyn_cast<daphne::FrameType>())
        throw std::runtime_error("CondOp does not support frames for the condition yet");
    else { // cond is a scalar // TODO check if it is really a scalar
        Type thenTy = getThenVal().getType();
        Type elseTy = getElseVal().getType();
        
        ssize_t thenNumRows = -1;
        ssize_t thenNumCols = -1;
        ssize_t elseNumRows = -1;
        ssize_t elseNumCols = -1;
        auto thenMatTy = thenTy.dyn_cast<daphne::MatrixType>();
        auto thenFrmTy = thenTy.dyn_cast<daphne::FrameType>();
        auto elseMatTy = elseTy.dyn_cast<daphne::MatrixType>();
        auto elseFrmTy = elseTy.dyn_cast<daphne::FrameType>();
        if(thenMatTy) {
            thenNumRows = thenMatTy.getNumRows();
            thenNumCols = thenMatTy.getNumCols();
        }
        else if(thenFrmTy) {
            thenNumRows = thenFrmTy.getNumRows();
            thenNumCols = thenFrmTy.getNumCols();
        }
        if(elseMatTy) {
            elseNumRows = elseMatTy.getNumRows();
            elseNumCols = elseMatTy.getNumCols();
        }
        else if(elseFrmTy) {
            elseNumRows = elseFrmTy.getNumRows();
            elseNumCols = elseFrmTy.getNumCols();
        }

        if((thenMatTy || thenFrmTy) && (elseMatTy || elseFrmTy))
            return {{
                (thenNumRows == elseNumRows) ? thenNumRows : -1,
                (thenNumCols == elseNumCols) ? thenNumCols : -1
            }};
        else
            // Then-value or else-value is a scalar.
            return {{-1, -1}};
    }
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::CTableOp::inferShape() {
    // If the result shape is given as arguments, then we know it.
    // Otherwise, we don't.
    // TODO In case resNumRows/resNumCols are known to be -1 (i.e., if
    // the output shape shall be determined depending on the values in
    // the lhs and rhs input matrices) and the lhs/rhs input matrices
    // are compile-time constants, then we could determine the number
    // of rows/columns here.
    return {{
        CompilerUtils::constantOrDefault<ssize_t>(getResNumRows(), -1),
        CompilerUtils::constantOrDefault<ssize_t>(getResNumCols(), -1)
    }};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::MatrixConstantOp::inferShape() {
    const Structure* mat = reinterpret_cast<const Structure*>(CompilerUtils::constantOrThrow<uint64_t>(getMatrixAddr()));
    return {{mat->getNumRows(), mat->getNumCols()}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::SliceRowOp::inferShape() {
    Type srcTy = getSource().getType();
    ssize_t srcNumRows;
    ssize_t srcNumCols;
    if(auto srcMatTy = srcTy.dyn_cast<daphne::MatrixType>()) {
        srcNumRows = srcMatTy.getNumRows();
        srcNumCols = srcMatTy.getNumCols();
    }
    else if(auto srcFrmTy = srcTy.dyn_cast<daphne::FrameType>()) {
        srcNumRows = srcFrmTy.getNumRows();
        srcNumCols = srcFrmTy.getNumCols();
    }
    else
        // If this is the case, shape inference shouldn't have been called.
        throw std::runtime_error(
                "SliceRowOp shape inference does only support matrix and frame inputs"
        );
    
    auto loIn = CompilerUtils::isConstant<int64_t>(getLowerIncl());
    auto upEx = CompilerUtils::isConstant<int64_t>(getUpperExcl());

    ssize_t resNumRows = -1;
    if(srcNumRows != -1 && loIn.first && upEx.first) {
        ssize_t loInPos = loIn.second;
        ssize_t upExPos = upEx.second;
        if(loInPos < 0 || loInPos >= srcNumRows)
            throw std::runtime_error(
                "SliceRowOp shape inference: lowerIncl must be in [0, numRows), "
                "but is " + std::to_string(loInPos) +
                " with " + std::to_string(srcNumRows) + " rows"
            );
        if(upExPos < 0 || upExPos > srcNumRows)
            throw std::runtime_error(
                "SliceRowOp shape inference: upperExcl must be in [0, numRows], "
                "but is " + std::to_string(upExPos) +
                " with " + std::to_string(srcNumRows) + " rows"
            );
        if(loInPos > upExPos)
            throw std::runtime_error(
                "SliceRowOp shape inference: lowerIncl must not be greater than upperExcl"
                " (found " + std::to_string(loInPos) + " and " + std::to_string(upExPos) + ")"
            );
        resNumRows = upExPos - loInPos;
    }

    return {{resNumRows, srcNumCols}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::SliceColOp::inferShape() {
    Type srcTy = getSource().getType();
    ssize_t srcNumRows;
    ssize_t srcNumCols;
    if(auto srcMatTy = srcTy.dyn_cast<daphne::MatrixType>()) {
        srcNumRows = srcMatTy.getNumRows();
        srcNumCols = srcMatTy.getNumCols();
    }
    else if(auto srcFrmTy = srcTy.dyn_cast<daphne::FrameType>()) {
        srcNumRows = srcFrmTy.getNumRows();
        srcNumCols = srcFrmTy.getNumCols();
    }
    else
        // If this is the case, shape inference shouldn't have been called.
        throw std::runtime_error(
                "SliceColOp shape inference does only support matrix and frame inputs"
        );
    
    auto loIn = CompilerUtils::isConstant<int64_t>(getLowerIncl());
    auto upEx = CompilerUtils::isConstant<int64_t>(getUpperExcl());

    ssize_t resNumCols = -1;
    if(srcNumCols != -1 && loIn.first && upEx.first) {
        ssize_t loInPos = loIn.second;
        ssize_t upExPos = upEx.second;
        if(loInPos < 0 || loInPos >= srcNumCols)
            throw std::runtime_error(
                "SliceColOp shape inference: lowerIncl must be in [0, numCols), "
                "but is " + std::to_string(loInPos) +
                " with " + std::to_string(srcNumCols) + " cols"
            );
        if(upExPos < 0 || upExPos > srcNumCols)
            throw std::runtime_error(
                "SliceColOp shape inference: upperExcl must be in [0, numCols], "
                "but is " + std::to_string(upExPos) +
                " with " + std::to_string(srcNumCols) + " cols"
            );
        if(loInPos > upExPos)
            throw std::runtime_error(
                "SliceColOp shape inference: lowerIncl must not be greater than upperExcl"
                " (found " + std::to_string(loInPos) + " and " + std::to_string(upExPos) + ")"
            );
        resNumCols = upEx.second - loIn.second;
    }

    return {{srcNumRows, resNumCols}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::EigenOp::inferShape() {
    auto shape = getShape(getOperand());
    return {{shape.first, 1}, {shape.first, shape.first}};
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
            numRows = CompilerUtils::constantOrDefault<int64_t>(op->getOperand(i), -1);
    }
};
template<size_t i>
struct tryNumColsFromIthScalar {
    static void apply(ssize_t& numRows, ssize_t& numCols, Operation* op) {
        if(op->hasTrait<NumColsFromIthScalar<i>::template Impl>())
            numCols = CompilerUtils::constantOrDefault<int64_t>(op->getOperand(i), -1);
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
        // If the operation implements the shape inference interface,
        // we apply that.
        return inferShapeOp.inferShape();
    else if(op->getNumResults() == 1) {
        // If the operation does not implement the shape inference interface
        // and has exactly one result, we utilize its shape inference traits,
        // or the inference interfaces for the number of rows and columns
        // (separately).

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
            // This first case is just a workaround, we should decide later how
            // to treat incomplete knowledge of the shapes.
            if(shapeLhs.first == -1 && shapeLhs.second == 1 && shapeRhs.first == -1 && shapeRhs.second == 1) {
                numRows = -1;
                numCols = 1;
            }
            else if(shapeRhs.first == -1 || shapeRhs.second == -1) {
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
    else {
        // If the operation does not implement the shape inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<std::pair<ssize_t, ssize_t>> shapes;
        for(size_t i = 0; i < op->getNumResults(); i++)
            shapes.push_back({-1, -1});
        return shapes;
    }
}
