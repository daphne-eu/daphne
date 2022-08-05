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
#include <ir/daphneir/DaphneInferSparsityOpInterface.cpp.inc>
}

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Utilities
// ****************************************************************************

double getSparsityOrUnknownFromType(Value v) {
    Type t = v.getType();
    if(auto mt = t.dyn_cast<daphne::MatrixType>())
        return mt.getSparsity();
    else // scalar or frame
        // TODO: read scalar value (if 0 -> sparsity 0.0)
        return -1.0;
}

double getSparsityOrUnknownFromScalar(Value v) {
    if(!v.getDefiningOp()) // check if block argument
        return -1.0;
    if(auto co = llvm::dyn_cast<daphne::ConstantOp>(v.getDefiningOp()))
        if(auto doubleAttr = co.value().dyn_cast<FloatAttr>())
            return doubleAttr.getValue().convertToDouble();
    return -1.0; // the value of the scalar is unknown at the moment
}

// ****************************************************************************
// Sparsity inference interface implementations
// ****************************************************************************

std::vector<double> daphne::DiagMatrixOp::inferSparsity() {
    auto argTy = arg().getType().dyn_cast<daphne::MatrixType>();
    auto k = argTy.getNumRows();
    auto sparsity = argTy.getSparsity();

    if(argTy.getSparsity() == -1.0) {
        sparsity = 1;
    }

    return {sparsity / k};
}

std::vector<double> daphne::MatMulOp::inferSparsity() {
    auto lhsTy = lhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhsTy = rhs().getType().dyn_cast<daphne::MatrixType>();
    if(lhsTy.getSparsity() == -1.0 || rhsTy.getSparsity() == -1.0) {
        return {-1.0};
    }
    auto k = lhsTy.getNumCols();
    if(k == -1) {
        k = rhsTy.getNumRows();
    }
    if(k == -1)
        return {-1.0};
    else
        // unbiased estimate
        return {1.0 - std::pow(1.0 - lhsTy.getSparsity() * rhsTy.getSparsity(), k)};
}

std::vector<double> daphne::TriOp::inferSparsity() {
    auto argTy = arg().getType().dyn_cast<daphne::MatrixType>();
    if(argTy.getSparsity() == -1.0) {
        return {-1.0};
    }
    // TODO: remove diagonal
    return {argTy.getSparsity() / 2.0};
}

std::vector<double> daphne::ReadOp::inferSparsity() {
    // TODO Use CompilerUtils::getFileMetaData() here, but it throws if the
    // file name is not a constant (after constant propagation).
    if(auto co = llvm::dyn_cast<mlir::daphne::ConstantOp>(fileName().getDefiningOp())) {
        if(auto strAttr = co.value().dyn_cast<mlir::StringAttr>()) {
            auto filename = strAttr.getValue().str();
            FileMetaData fmd = FileMetaData::ofFile(filename);
            if (fmd.numNonZeros == -1)
                return {-1.0};
            // TODO: maybe use type shape info instead of file? (would require correct order of optimization passes)
            return {(static_cast<double>(fmd.numNonZeros) / fmd.numRows) / fmd.numCols};
        }
    }
    return {-1.0};
}

// ****************************************************************************
// Sparsity inference trait implementations
// ****************************************************************************

// TODO This is also used in DaphneInferShapeOpInterface.cpp, make it a central
// utility.
/**
 * @brief Utility for trying a parametric trait for all values of the parameter
 * from 0 to some upper bound.
 */
template<size_t upper, template<size_t> class tryParametricTrait>
struct tryParamTraitUntil {
    static void apply(double &sparsity, Operation *op) {
        tryParametricTrait<upper>::apply(sparsity, op);
        tryParamTraitUntil<upper - 1, tryParametricTrait>::apply(sparsity, op);
    }
};
template<template<size_t> class tryParametricTrait>
struct tryParamTraitUntil<0, tryParametricTrait> {
    static void apply(double &sparsity, Operation *op) {
        tryParametricTrait<0>::apply(sparsity, op);
    }
};

template<size_t i>
struct trySparsityFromIthScalar {
    static void apply(double &sparsity, Operation *op) {
        if(op->hasTrait<SparsityFromIthScalar<i>::template Impl>())
            sparsity = getSparsityOrUnknownFromScalar(op->getOperand(i));
    }
};

template<size_t i>
struct trySparsityFromIthArg {
    static void apply(double &sparsity, Operation *op) {
        if(op->hasTrait<SparsityFromIthArg<i>::template Impl>())
            sparsity = getSparsityOrUnknownFromType(op->getOperand(i));
    }
};

// ****************************************************************************
// Sparsity inference function
// ****************************************************************************

std::vector<double> daphne::tryInferSparsity(Operation *op) {
    if(auto inferSparsityOp = llvm::dyn_cast<daphne::InferSparsity>(op))
        // If the operation implements the sparsity inference interface,
        // we apply that.
        return inferSparsityOp.inferSparsity();
    else if(op->getNumResults() == 1) {
        // If the operation does not implement the sparsity inference interface
        // and has exactly one result, we utilize its sparsity inference traits.
        double sparsity = -1.0;

        if(op->hasTrait<CompletelyDense>()) {
            sparsity = 1.0;
        }

        if(op->hasTrait<EwSparseIfBoth>()) {
            auto spLhs = getSparsityOrUnknownFromType(op->getOperand(0));
            auto spRhs = getSparsityOrUnknownFromType(op->getOperand(1));
            if(spLhs != -1.0 && spRhs != -1.0)
                // unbiased estimate
                sparsity = spLhs + spRhs - spLhs * spRhs;
        }

        if(op->hasTrait<EwSparseIfEither>()) {
            auto spLhs = getSparsityOrUnknownFromType(op->getOperand(0));
            auto spRhs = getSparsityOrUnknownFromType(op->getOperand(1));
            if(spLhs != -1.0 && spRhs != -1.0)
                // unbiased estimate
                sparsity = spLhs * spRhs;
            else if (spLhs != -1.0)
                sparsity = spLhs;
            else if (spRhs != -1.0)
                sparsity = spRhs;
        }

        // Our parametric traits addressing a certain argument are supported
        // for up to 10 arguments (this can easily be changed here).
        // There does not seem to be a way in MLIR do it more generically,
        // since the parameters of parametric traits are template parameters.
        const size_t u = 9;
        tryParamTraitUntil<u, trySparsityFromIthScalar>::apply(sparsity, op);
        tryParamTraitUntil<u, trySparsityFromIthArg>::apply(sparsity, op);

        return {sparsity};
    }
    else {
        // If the operation does not implement the sparsity inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<double> sparsities;
        for(size_t i = 0; i < op->getNumResults(); i++)
            sparsities.push_back(-1);
        return sparsities;
    }
}
