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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_AGGALL_H
#define SRC_RUNTIME_LOCAL_KERNELS_AGGALL_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <cmath>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <typename VTRes, class DTArg> struct AggAll {
    static VTRes apply(AggOpCode opCode, const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <typename VTRes, class DTArg> VTRes aggAll(AggOpCode opCode, const DTArg *arg, DCTX(ctx)) {
    return AggAll<VTRes, DTArg>::apply(opCode, arg, ctx);
}

// ****************************************************************************
// Helper functions
// ****************************************************************************

/**
 * @brief Aggregates the given array using the given binary operation.
 *
 * @tparam BinOp The binary operation to apply to a pair of values; must be an instantiation of `EwBinarySca`
 * or a custom class providing a similar `apply()`-function as `EwBinarySca`.
 * @tparam VTRes The value type of the result.
 * @tparam VTArg The value type of the argument array.
 * @param values The array to aggregate.
 * @param numValues The number of data elements in `values`.
 * @param includeExtraZero Iff `true`, apply the given binary operation to the aggregation result and a single zero
 * before returning.
 * @param neutral The neutral element of the given binary operation; will be returned if `numValues` is zero.
 */
template <class BinOp, typename VTRes, typename VTArg>
VTRes aggAllArray(const VTArg *values, size_t numValues, bool includeExtraZero, VTRes neutral, DCTX(ctx)) {
    VTRes agg;

    // Note that BinOp::apply() will usually be inlined.

    if (numValues) {
        // Non-empty array.
        // We perform manual loop unrolling by a factor of 4 (turned out to be a good value) to reduce data
        // dependencies between loop iterations and, thereby, improve runtime.
        VTRes agg0 = neutral;
        VTRes agg1 = neutral;
        VTRes agg2 = neutral;
        VTRes agg3 = neutral;
        for (size_t i = 0; i < numValues / 4 * 4; i += 4) {
            agg0 = BinOp::apply(agg0, static_cast<VTRes>(values[i]), ctx);
            agg1 = BinOp::apply(agg1, static_cast<VTRes>(values[i + 1]), ctx);
            agg2 = BinOp::apply(agg2, static_cast<VTRes>(values[i + 2]), ctx);
            agg3 = BinOp::apply(agg3, static_cast<VTRes>(values[i + 3]), ctx);
        }
        agg = BinOp::apply(BinOp::apply(agg0, agg1, ctx), BinOp::apply(agg2, agg3, ctx), ctx);
        for (size_t i = 0; i < numValues % 4; i++)
            agg = BinOp::apply(agg, static_cast<VTRes>(values[numValues / 4 * 4 + i]), ctx);
    } else
        // Empty array, return the neutral element of the binary op.
        agg = neutral;

    // Process an extra zero if specified.
    if (includeExtraZero)
        agg = BinOp::apply(agg, 0, ctx);

    return agg;
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// scalar <- DenseMatrix
// ----------------------------------------------------------------------------

#define MAKE_CASE_1(aggOpCode, binOpCode)                                                                              \
    case aggOpCode:                                                                                                    \
        agg = aggAllArray<EwBinarySca<binOpCode, VTRes, VTArg, VTArg>, VTRes>(valuesArg, numRows * numCols, false,     \
                                                                              neutral, ctx);                           \
        break;

#define MAKE_CASE_2(aggOpCode, binOpCode)                                                                              \
    case aggOpCode:                                                                                                    \
        for (size_t r = 0; r < numRows; r++) {                                                                         \
            agg = EwBinarySca<binOpCode, VTRes, VTArg, VTArg>::apply(                                                  \
                agg,                                                                                                   \
                aggAllArray<EwBinarySca<binOpCode, VTRes, VTArg, VTArg>, VTRes>(valuesArg, numCols, false, neutral,    \
                                                                                ctx),                                  \
                ctx);                                                                                                  \
            valuesArg += arg->getRowSkip();                                                                            \
        }                                                                                                              \
        break;

template <typename VTRes, typename VTArg> struct AggAll<VTRes, DenseMatrix<VTArg>> {
    static VTRes apply(AggOpCode opCode, const DenseMatrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        const VTArg *valuesArg = arg->getValues();

        VTRes agg, stddev;
        VTRes neutral =
            AggOpCodeUtils::isPureBinaryReduction(opCode) ? AggOpCodeUtils::template getNeutral<VTRes>(opCode) : 0;
        if (numCols == arg->getRowSkip() || numRows == 1)
            // contiguous memory, call aggAllArray() just once
            switch (opCode) {
                MAKE_CASE_1(AggOpCode::SUM, BinaryOpCode::ADD)
                MAKE_CASE_1(AggOpCode::PROD, BinaryOpCode::MUL)
                MAKE_CASE_1(AggOpCode::MIN, BinaryOpCode::MIN)
                MAKE_CASE_1(AggOpCode::MAX, BinaryOpCode::MAX)
                MAKE_CASE_1(AggOpCode::MEAN, BinaryOpCode::ADD)
                MAKE_CASE_1(AggOpCode::STDDEV, BinaryOpCode::ADD)
                MAKE_CASE_1(AggOpCode::VAR, BinaryOpCode::ADD)
            default:
                throw std::runtime_error("unsupported AggOpCode");
            }
        else {
            // non-contiguous memory, call aggAllArray() for each row
            agg = neutral;
            switch (opCode) {
                MAKE_CASE_2(AggOpCode::SUM, BinaryOpCode::ADD)
                MAKE_CASE_2(AggOpCode::PROD, BinaryOpCode::MUL)
                MAKE_CASE_2(AggOpCode::MIN, BinaryOpCode::MIN)
                MAKE_CASE_2(AggOpCode::MAX, BinaryOpCode::MAX)
                MAKE_CASE_2(AggOpCode::MEAN, BinaryOpCode::ADD)
                MAKE_CASE_2(AggOpCode::STDDEV, BinaryOpCode::ADD)
                MAKE_CASE_2(AggOpCode::VAR, BinaryOpCode::ADD)
            default:
                throw std::runtime_error("unsupported AggOpCode");
            }
        }
        if (AggOpCodeUtils::isPureBinaryReduction(opCode))
            return agg;

        agg /= arg->getNumCols() * arg->getNumRows();
        // The op-code is either MEAN or STDDEV or VAR.
        if (opCode == AggOpCode::MEAN) {
            return agg;
        }
        // else op-code is STDDEV or VAR
        // TODO We could use aggAllArray() here as well (with a custom EwBinarySca-like class).
        stddev = 0;
        valuesArg = arg->getValues();
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                VTRes val = static_cast<VTRes>(valuesArg[c]) - agg;
                stddev = stddev + val * val;
            }
            valuesArg += arg->getRowSkip();
        }

        stddev /= arg->getNumCols() * arg->getNumRows();

        // Variance --> stddev before sqrt() is variance
        if (opCode == AggOpCode::VAR) {
            VTRes var = stddev;
            return var;
        }

        stddev = sqrt(stddev);
        return stddev;
    }
};

#undef MAKE_CASE_1
#undef MAKE_CASE_2

// ----------------------------------------------------------------------------
// scalar <- CSRMatrix
// ----------------------------------------------------------------------------

#define MAKE_CASE(aggOpCode, binOpCode)                                                                                \
    case aggOpCode:                                                                                                    \
        return aggAllArray<EwBinarySca<binOpCode, VTRes, VTArg, VTArg>, VTRes>(                                        \
            arg->getValues(0), arg->getNumNonZeros(), includeExtraZero,                                                \
            AggOpCodeUtils::template getNeutral<VTRes>(opCode), ctx);

template <typename VTRes, typename VTArg> struct AggAll<VTRes, CSRMatrix<VTArg>> {
    static VTRes apply(AggOpCode opCode, const CSRMatrix<VTArg> *arg, DCTX(ctx)) {
        if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {
            bool includeExtraZero =
                !AggOpCodeUtils::isSparseSafe(opCode) && arg->getNumNonZeros() < arg->getNumRows() * arg->getNumCols();
            switch (opCode) {
                MAKE_CASE(AggOpCode::SUM, BinaryOpCode::ADD)
                MAKE_CASE(AggOpCode::PROD, BinaryOpCode::MUL)
                MAKE_CASE(AggOpCode::MIN, BinaryOpCode::MIN)
                MAKE_CASE(AggOpCode::MAX, BinaryOpCode::MAX)
                MAKE_CASE(AggOpCode::MEAN, BinaryOpCode::ADD)
                MAKE_CASE(AggOpCode::STDDEV, BinaryOpCode::ADD)
                MAKE_CASE(AggOpCode::VAR, BinaryOpCode::ADD)
            default:
                throw std::runtime_error("unsupported AggOpCode");
            }
        } else { // The op-code is either MEAN or STDDEV or VAR.
            bool includeExtraZero =
                !AggOpCodeUtils::isSparseSafe(opCode) && arg->getNumNonZeros() < arg->getNumRows() * arg->getNumCols();
            auto agg = aggAllArray<EwBinarySca<BinaryOpCode::ADD, VTRes, VTArg, VTArg>, VTRes>(
                arg->getValues(0), arg->getNumNonZeros(), includeExtraZero, 0, ctx);
            agg = agg / (arg->getNumRows() * arg->getNumCols());
            if (opCode == AggOpCode::MEAN)
                return agg;
            else {
                // STDDEV-VAR
                VTRes stddev = 0;

                const VTArg *valuesArg = arg->getValues(0);
                for (size_t i = 0; i < arg->getNumNonZeros(); i++) {
                    VTRes val = static_cast<VTRes>((valuesArg[i])) - agg;
                    stddev = stddev + val * val;
                }
                stddev += ((arg->getNumRows() * arg->getNumCols()) - arg->getNumNonZeros()) * agg * agg;
                stddev /= (arg->getNumRows() * arg->getNumCols());

                // Variance --> stddev before sqrt() is variance
                if (opCode == AggOpCode::VAR) {
                    VTRes var = stddev;
                    return var;
                }

                stddev = sqrt(stddev);
                return stddev;
            }
        }
    }
};

#undef MAKE_CASE

// ----------------------------------------------------------------------------
// scalar <- Matrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> struct AggAll<VTRes, Matrix<VTArg>> {
    static VTRes apply(AggOpCode opCode, const Matrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        EwBinaryScaFuncPtr<VTRes, VTRes, VTRes> func;
        VTRes agg, stddev;
        if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {
            func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(opCode));
            agg = AggOpCodeUtils::template getNeutral<VTRes>(opCode);
        } else {
            // TODO Setting the function pointer yields the correct result.
            // However, since MEAN, VAR, and STDDEV are not sparse-safe, the
            // program does not take the same path for doing the summation, and
            // is less efficient. for MEAN, VAR, and STDDEV, we need to sum
            func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));
            agg = VTRes(0);
        }

        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                agg = func(agg, static_cast<VTRes>(arg->get(r, c)), ctx);

        if (AggOpCodeUtils::isPureBinaryReduction(opCode))
            return agg;

        agg /= numCols * numRows;
        // The op-code is either MEAN or STDDEV or VAR.
        if (opCode == AggOpCode::MEAN)
            return agg;

        // else op-code is STDDEV or VAR
        stddev = 0;
        for (size_t r = 0; r < numRows; ++r) {
            for (size_t c = 0; c < numCols; ++c) {
                VTRes val = static_cast<VTRes>(arg->get(r, c)) - agg;
                stddev = stddev + val * val;
            }
        }

        stddev /= numCols * numRows;

        // VAR --> stddev before sqrt() is variance
        if (opCode == AggOpCode::VAR)
            return stddev;

        // STDDEV
        stddev = sqrt(stddev);
        return stddev;
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_AGGALL_H
