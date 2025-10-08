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

#include <compiler/inference/TypeInferenceUtils.h>
#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <mlir/IR/Value.h>

#include <parser/metadata/MetaDataParser.h>

#include <stdexcept>
#include <utility>
#include <vector>

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
    if (auto mt = t.dyn_cast<daphne::MatrixType>()) // matrix
        return mt.getSparsity();
    if (CompilerUtils::isObjType(t) || CompilerUtils::isScaType(t)) // other data type (e.g., frame) or value type
        // TODO: read scalar value (if 0 -> sparsity 0.0)
        return -1.0;

    std::stringstream s;
    s << "getSparsityOrUnknownFromType(): the given value has neither a supported data type nor a supported value "
         "type: `"
      << t << '`';
    throw std::runtime_error(s.str());
}

// ****************************************************************************
// Sparsity inference interface implementations
// ****************************************************************************

template <typename VT> double getSparsity(uint64_t matrixAddr) {
    const DenseMatrix<VT> *arg = reinterpret_cast<DenseMatrix<VT> *>(matrixAddr);
    const size_t numRows = arg->getNumRows();
    const size_t numCols = arg->getNumCols();
    const VT *values = arg->getValues();
    size_t nnz = 0;
    for (size_t r = 0; r < numRows; r++) {
        for (size_t c = 0; c < numCols; c++)
            if (values[c] != 0)
                nnz++;
        values += arg->getRowSkip();
    }
    return static_cast<double>(nnz) / (numRows * numCols);
}

std::vector<double> daphne::MatrixConstantOp::inferSparsity() {
    std::pair<bool, uint64_t> p = CompilerUtils::isConstant<uint64_t>(getMatrixAddr());
    if (p.first) { // if the address is a compile-time constant (it should be)
        const uint64_t matrixAddr = p.second;
        Type vt = CompilerUtils::getValueType(getResult().getType());
        if (vt.isF64())
            return {getSparsity<double>(matrixAddr)};
        if (vt.isF32())
            return {getSparsity<float>(matrixAddr)};
        if (vt.isSignedInteger(64))
            return {getSparsity<int64_t>(matrixAddr)};
        if (vt.isSignedInteger(32))
            return {getSparsity<int32_t>(matrixAddr)};
        if (vt.isSignedInteger(8))
            return {getSparsity<int8_t>(matrixAddr)};
        if (vt.isUnsignedInteger(64))
            return {getSparsity<uint64_t>(matrixAddr)};
        if (vt.isUnsignedInteger(32))
            return {getSparsity<uint32_t>(matrixAddr)};
        if (vt.isUnsignedInteger(8))
            return {getSparsity<uint8_t>(matrixAddr)};
    }
    // For (so far) non-supported value types (or in the unexpected case that the address was not a compile-time
    // constant), return an unknown sparsity.
    return {-1};
}

std::vector<double> daphne::DiagMatrixOp::inferSparsity() {
    auto argTy = getArg().getType().dyn_cast<daphne::MatrixType>();
    auto k = argTy.getNumRows();
    auto sparsity = argTy.getSparsity();

    if (argTy.getSparsity() == -1.0) {
        sparsity = 1;
    }

    return {sparsity / k};
}

std::vector<double> daphne::MatMulOp::inferSparsity() {
    auto lhsTy = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhsTy = getRhs().getType().dyn_cast<daphne::MatrixType>();
    if (lhsTy.getSparsity() == -1.0 || rhsTy.getSparsity() == -1.0) {
        return {-1.0};
    }
    auto k = lhsTy.getNumCols();
    if (k == -1) {
        k = rhsTy.getNumRows();
    }
    if (k == -1)
        return {-1.0};
    else
        // unbiased estimate
        return {1.0 - std::pow(1.0 - lhsTy.getSparsity() * rhsTy.getSparsity(), k)};
}

std::vector<double> daphne::TriOp::inferSparsity() {
    auto argTy = getArg().getType().dyn_cast<daphne::MatrixType>();
    if (argTy.getSparsity() == -1.0) {
        return {-1.0};
    }
    // TODO: remove diagonal
    return {argTy.getSparsity() / 2.0};
}

std::vector<double> daphne::ReadOp::inferSparsity() {
    std::pair<bool, std::string> p = CompilerUtils::isConstant<std::string>(getFileName());
    if (p.first) {
        FileMetaData fmd = MetaDataParser::readMetaData(p.second);
        if (fmd.numNonZeros == -1)
            return {-1.0};
        // TODO: maybe use type shape info instead of file? (would require
        // correct order of optimization passes)
        return {(static_cast<double>(fmd.numNonZeros) / fmd.numRows) / fmd.numCols};
    } else
        return {-1.0};
}

// ****************************************************************************
// Sparsity inference trait implementations
// ****************************************************************************

template <size_t i> struct trySparsityFromIthScalar {
    using T = double;
    static void apply(T &sparsity, Operation *op) {
        if (op->hasTrait<SparsityFromIthScalar<i>::template Impl>())
            sparsity = CompilerUtils::constantOrDefault<T>(op->getOperand(i), -1);
    }
};

template <size_t i> struct trySparsityFromIthArg {
    using T = double;
    static void apply(T &sparsity, Operation *op) {
        if (op->hasTrait<SparsityFromIthArg<i>::template Impl>())
            sparsity = getSparsityOrUnknownFromType(op->getOperand(i));
    }
};

// ****************************************************************************
// Sparsity inference function
// ****************************************************************************

std::vector<double> daphne::tryInferSparsity(Operation *op) {
    if (auto inferSparsityOp = llvm::dyn_cast<daphne::InferSparsity>(op))
        // If the operation implements the sparsity inference interface,
        // we apply that.
        return inferSparsityOp.inferSparsity();
    else if (op->getNumResults() == 1) {
        // If the operation does not implement the sparsity inference interface
        // and has exactly one result, we utilize its sparsity inference traits.
        double sparsity = -1.0;

        if (op->hasTrait<CompletelyDense>()) {
            sparsity = 1.0;
        }

        if (op->hasTrait<EwSparseIfBoth>()) {
            auto spLhs = getSparsityOrUnknownFromType(op->getOperand(0));
            auto spRhs = getSparsityOrUnknownFromType(op->getOperand(1));
            if (spLhs != -1.0 && spRhs != -1.0)
                // unbiased estimate
                sparsity = spLhs + spRhs - spLhs * spRhs;
        }

        if (op->hasTrait<EwSparseIfEither>()) {
            auto spLhs = getSparsityOrUnknownFromType(op->getOperand(0));
            auto spRhs = getSparsityOrUnknownFromType(op->getOperand(1));
            if (spLhs != -1.0 && spRhs != -1.0)
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
    } else {
        // If the operation does not implement the sparsity inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<double> sparsities;
        for (size_t i = 0; i < op->getNumResults(); i++)
            sparsities.push_back(-1);
        return sparsities;
    }
}
