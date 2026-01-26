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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/CSRStats.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/CastObj.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTLhs, class DTRhs> struct MatMul {
    static void apply(DTRes *&res, const DTLhs *lhs, const DTRhs *rhs, bool transa, bool transb, DCTX(ctx)) = delete;
};

template <typename T> struct MatMul<DenseMatrix<T>, DenseMatrix<T>, DenseMatrix<T>> {
    static void apply(DenseMatrix<T> *&res, const DenseMatrix<T> *lhs, const DenseMatrix<T> *rhs, bool transa,
                      bool transb, DCTX(dctx));
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTLhs, class DTRhs>
void matMul(DTRes *&res, const DTLhs *lhs, const DTRhs *rhs, bool transa, bool transb, DCTX(ctx)) {
    MatMul<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, transa, transb, ctx);
}

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix, DenseMatrix
// ----------------------------------------------------------------------------

enum class SpMatMulVariant { ROWWISE, PARALLEL, BLOCKED };

template <typename VT> struct MatMul<DenseMatrix<VT>, CSRMatrix<VT>, DenseMatrix<VT>> {
  private:
    // Naive Row-wise serial implementation
    static void applyRowWise(DenseMatrix<VT> *res, const CSRMatrix<VT> *lhs, const DenseMatrix<VT> *rhs) {
        const size_t nr1 = lhs->getNumRows();
        const size_t nc2 = rhs->getNumCols();

        const VT *valuesRhs = rhs->getValues();
        VT *valuesRes = res->getValues();

        const size_t rowSkipRhs = rhs->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();

        memset(valuesRes, VT(0), sizeof(VT) * nr1 * nc2);
        for (size_t r = 0; r < nr1; r++) {
            const size_t rowNumNonZeros = lhs->getNumNonZeros(r);
            const size_t *rowColIdxs = lhs->getColIdxs(r);
            const VT *rowValues = lhs->getValues(r);

            const size_t rowIdxRes = r * rowSkipRes;
            for (size_t i = 0; i < rowNumNonZeros; i++) {
                const size_t c = rowColIdxs[i];
                const size_t rowIdxRhs = c * rowSkipRhs;

                for (size_t j = 0; j < nc2; j++) {
                    valuesRes[rowIdxRes + j] += rowValues[i] * valuesRhs[rowIdxRhs + j];
                }
            }
        }
    }

    // Parallel implementation using OpenMP
    static void applyParallel(DenseMatrix<VT> *res, const CSRMatrix<VT> *lhs, const DenseMatrix<VT> *rhs) {
        const size_t nr1 = lhs->getNumRows();
        const size_t nc2 = rhs->getNumCols();

        const VT *valuesRhs = rhs->getValues();
        VT *valuesRes = res->getValues();

        const size_t rowSkipRhs = rhs->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();

        memset(valuesRes, VT(0), sizeof(VT) * nr1 * nc2);

#pragma omp parallel for schedule(dynamic)
        for (size_t r = 0; r < nr1; r++) {
            const size_t rowNumNonZeros = lhs->getNumNonZeros(r);
            const size_t *rowColIdxs = lhs->getColIdxs(r);
            const VT *rowValues = lhs->getValues(r);

            const size_t rowIdxRes = r * rowSkipRes;
            for (size_t i = 0; i < rowNumNonZeros; i++) {
                const size_t c = rowColIdxs[i];
                const size_t rowIdxRhs = c * rowSkipRhs;

                for (size_t j = 0; j < nc2; j++) {
                    valuesRes[rowIdxRes + j] += rowValues[i] * valuesRhs[rowIdxRhs + j];
                }
            }
        }
    }

    // Blocked implementation for cache efficiency
    static void applyBlocked(DenseMatrix<VT> *res, const CSRMatrix<VT> *lhs, const DenseMatrix<VT> *rhs,
                             size_t blockSize = 64) {
        const size_t nr1 = lhs->getNumRows();
        const size_t nc2 = rhs->getNumCols();

        const VT *valuesRhs = rhs->getValues();
        VT *valuesRes = res->getValues();

        const size_t rowSkipRhs = rhs->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();

        memset(valuesRes, VT(0), sizeof(VT) * nr1 * nc2);

// Process rows in blocks for better cache locality
#pragma omp parallel for schedule(dynamic)
        for (size_t blockStart = 0; blockStart < nr1; blockStart += blockSize) {
            const size_t blockEnd = std::min(blockStart + blockSize, nr1);

            for (size_t r = blockStart; r < blockEnd; r++) {
                const size_t rowNumNonZeros = lhs->getNumNonZeros(r);
                const size_t *rowColIdxs = lhs->getColIdxs(r);
                const VT *rowValues = lhs->getValues(r);

                const size_t rowIdxRes = r * rowSkipRes;
                for (size_t i = 0; i < rowNumNonZeros; i++) {
                    const size_t c = rowColIdxs[i];
                    const size_t rowIdxRhs = c * rowSkipRhs;

                    for (size_t j = 0; j < nc2; j++) {
                        valuesRes[rowIdxRes + j] += rowValues[i] * valuesRhs[rowIdxRhs + j];
                    }
                }
            }
        }
    }

    // Algorithm selection heuristic based on matrix characteristics
    static SpMatMulVariant selectVariant(const CSRStats &stats, size_t rhsCols) {
        // Estimate work: each non-zero contributes rhsCols multiply-adds
        const size_t totalWork = stats.nnz * rhsCols;

        // Small workloads: serial is faster (avoid OpenMP overhead)
        // Threshold ~100K ops empirically balances overhead vs parallelism
        if (totalWork < 100000 || stats.numRows < 500) {
            return SpMatMulVariant::ROWWISE;
        }

        // Large matrices with moderate-to-high density benefit from blocking
        if (stats.sparsity > 0.05 && stats.numRows >= 1000 && rhsCols >= 50) {
            return SpMatMulVariant::BLOCKED;
        }

        // Large sparse matrices with many rows: parallelize over rows
        if (stats.numRows >= 1000 && totalWork >= 500000) {
            return SpMatMulVariant::PARALLEL;
        }

        return SpMatMulVariant::ROWWISE;
    }

    // Get adaptive level from environment variable
    // Returns: 0 = disabled, >= 1 = enabled
    // enabled via DAPHNE_ADAPTIVE env var
    static int getAdaptiveLevel() {
        const char *env = std::getenv("DAPHNE_ADAPTIVE");
        if (env == nullptr)
            return 0;
        try {
            return std::atoi(env);
        } catch (...) {
            return 0;
        }
    }

    static const char *variantToString(SpMatMulVariant var) {
        switch (var) {
        case SpMatMulVariant::ROWWISE:
            return "ROWWISE";
        case SpMatMulVariant::PARALLEL:
            return "PARALLEL";
        case SpMatMulVariant::BLOCKED:
            return "BLOCKED";
        default:
            return "UNKNOWN";
        }
    }

  public:
    static void apply(DenseMatrix<VT> *&res, const CSRMatrix<VT> *lhs, const DenseMatrix<VT> *rhs, bool transa,
                      bool transb, DCTX(ctx)) {
        const size_t nr1 = lhs->getNumRows();
        [[maybe_unused]] const size_t nc1 = lhs->getNumCols();

        [[maybe_unused]] const size_t nr2 = rhs->getNumRows();
        const size_t nc2 = rhs->getNumCols();

        if (nc1 != nr2) {
            throw std::runtime_error("MatMul - #cols of lhs and #rows of rhs must be the same");
        }
        // FIXME: transpose isn't supported atm

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(nr1, nc2, false);

        SpMatMulVariant var = SpMatMulVariant::ROWWISE; // default
        const int adaptiveLevel = getAdaptiveLevel();

        if (adaptiveLevel >= 1) {
            CSRStats stats = CSRStats::compute(lhs);
            var = selectVariant(stats, nc2);

            ctx->logger->debug("MatMul<Dense,CSR,Dense> rows={} cols={} nnz={} sparsity={:.4f} rhsCols={} -> {}",
                               stats.numRows, stats.numCols, stats.nnz, stats.sparsity, nc2, variantToString(var));
        }

        switch (var) {
        case SpMatMulVariant::ROWWISE:
            applyRowWise(res, lhs, rhs);
            break;
        case SpMatMulVariant::PARALLEL:
            applyParallel(res, lhs, rhs);
            break;
        case SpMatMulVariant::BLOCKED:
            applyBlocked(res, lhs, rhs);
            break;
        }
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct MatMul<DenseMatrix<VT>, DenseMatrix<VT>, CSRMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const DenseMatrix<VT> *lhs, const CSRMatrix<VT> *rhs, bool transa,
                      bool transb, DCTX(ctx)) {
        const size_t nr1 = lhs->getNumRows();
        const size_t nc1 = lhs->getNumCols();
        const size_t nr2 = rhs->getNumRows();
        const size_t nc2 = rhs->getNumCols();

        if (nc1 != nr2) {
            throw std::runtime_error("MatMul - #cols of lhs and #rows of rhs must be the same");
        }
        // FIXME: transpose isn't supported atm

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(nr1, nc2, /*zero=*/true);

        const VT *valuesLhs = lhs->getValues();
        VT *valuesRes = res->getValues();

        const size_t rowSkipLhs = lhs->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();

        // For each row m of lhs
        for (size_t m = 0; m < nr1; m++) {
            const size_t rowIdxLhs = m * rowSkipLhs;
            const size_t rowIdxRes = m * rowSkipRes;

            // For each row n of rhs
            for (size_t n = 0; n < nr2; n++) {
                const VT lhsVal = valuesLhs[rowIdxLhs + n];

                // Get non-zeros in row k of rhs
                const size_t rowNumNonZeros = rhs->getNumNonZeros(n);
                const size_t *rowColIdxs = rhs->getColIdxs(n);
                const VT *rowValues = rhs->getValues(n);

                // For each non-zero in row k of rhs
                for (size_t i = 0; i < rowNumNonZeros; i++) {
                    const size_t c = rowColIdxs[i];
                    valuesRes[rowIdxRes + c] += lhsVal * rowValues[i];
                }
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix, Matrix
// ----------------------------------------------------------------------------

template <typename VT> struct MatMul<Matrix<VT>, Matrix<VT>, Matrix<VT>> {
    static void apply(Matrix<VT> *&res, const Matrix<VT> *lhs, const Matrix<VT> *rhs, bool transa, bool transb,
                      DCTX(ctx)) {
        const size_t lhsRows = transa ? lhs->getNumCols() : lhs->getNumRows();
        const size_t lhsCols = transa ? lhs->getNumRows() : lhs->getNumCols();
        const size_t rhsRows = transb ? rhs->getNumCols() : rhs->getNumRows();
        const size_t rhsCols = transb ? rhs->getNumRows() : rhs->getNumCols();

        if (lhsCols != rhsRows)
            throw std::runtime_error("MatMul: #cols of lhs and #rows of rhs must be the same");

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(lhsRows, rhsCols, false);

        res->prepareAppend();
        for (size_t rowRes = 0; rowRes < lhsRows; ++rowRes) {
            for (size_t colRes = 0; colRes < rhsCols; ++colRes) {
                VT resVal = 0;
                for (size_t cell = 0; cell < lhsCols; ++cell) {
                    VT lhsVal = transa ? lhs->get(cell, rowRes) : lhs->get(rowRes, cell);
                    VT rhsVal = transb ? rhs->get(colRes, cell) : rhs->get(cell, colRes);
                    resVal += lhsVal * rhsVal;
                }
                res->append(rowRes, colRes, resVal);
            }
        }
        res->finishAppend();
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix, CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct MatMul<CSRMatrix<VT>, CSRMatrix<VT>,
                                     CSRMatrix<VT>> { // ToDo: support transpose
    static void apply(CSRMatrix<VT> *&res, const CSRMatrix<VT> *lhs, const CSRMatrix<VT> *rhs, bool transa, bool transb,
                      DCTX(ctx)) {
        const size_t nr1 = lhs->getNumRows();
        const size_t nc1 = lhs->getNumCols();
        const size_t nr2 = rhs->getNumRows();
        const size_t nc2 = rhs->getNumCols();

        if (nc1 != nr2)
            throw std::runtime_error("#cols of lhs and #rows of rhs must be the same");

        // TODO: Better estimation of the number of non-zeros
        size_t estimationNumNonZeros = lhs->getNumNonZeros() * rhs->getNumNonZeros();
        if (res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(nr1, nc2, estimationNumNonZeros, true);

        const VT *valuesLhs = lhs->getValues();
        const size_t *colIdxsLhs = lhs->getColIdxs();
        const size_t *rowOffsetsLhs = lhs->getRowOffsets();

        const VT *valuesRhs = rhs->getValues();
        const size_t *colIdxsRhs = rhs->getColIdxs();
        const size_t *rowOffsetsRhs = rhs->getRowOffsets();

        for (size_t row = 0; row < nr1; row++) {
            for (size_t col = 0; col < nc2; col++) {
                VT sum = VT(0);
                // Dot product between the row `row` of Lhs and the col `col` of
                // Rhs
                for (size_t j = rowOffsetsLhs[row]; j < rowOffsetsLhs[row + 1]; j++) {
                    size_t k = colIdxsLhs[j];
                    // For this we need to find the values Rhs[k, col]
                    // (we already have Lhs[row, k])
                    size_t i = rowOffsetsRhs[k];
                    size_t endRhsRow = rowOffsetsRhs[k + 1];
                    // We are scanning the k^{th} row of Rhs to find a value at
                    // the col `col`
                    while (i < endRhsRow && colIdxsRhs[i] < col)
                        i++;
                    if (i < endRhsRow && colIdxsRhs[i] == col)
                        sum += valuesLhs[j] * valuesRhs[i];
                }
                res->set(row, col, sum);
            }
        }
    }
};
